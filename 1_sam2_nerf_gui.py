import os
import random
import cv2
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, TextBox
from sam2.build_sam import build_sam2_video_predictor
from prepare_images import prepare_images

# ========== Config ==========
mydir = r"Laptop\9912"
video_dir = os.path.join(os.getcwd(), r"data\new\sapien", mydir, "images")
checkpoint_path = "checkpoints/sam2.1_hiera_large.pt"
config_path = "configs/sam2.1/sam2.1_hiera_l.yaml"
output_mask_dir = os.path.join(os.getcwd(), r"data\new\sapien", mydir, "masks")
os.makedirs(output_mask_dir, exist_ok=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========== Run Prepare Images ==========
print("Preparing images...")
prepare_images(mydir)

# ========== Load SAM2 ==========
print("Loading SAM2...")
predictor = build_sam2_video_predictor(config_path, checkpoint_path, device=device)
inference_state = predictor.init_state(video_path=video_dir)
predictor.reset_state(inference_state)

# ========== Load Images ==========

frame_names = sorted(
    [f for f in os.listdir(video_dir) if f.lower().endswith((".jpg", ".jpeg"))],
    key=lambda x: int(os.path.splitext(x)[0]),
)

frames = [Image.open(os.path.join(video_dir, f)) for f in frame_names]
num_frames = len(frames)
# ========== GUI State ==========
current_frame = 0
clicks = {}  # obj_id -> list of (x, y, label)
current_obj_id = 1
video_segments = {}

# ========== GUI Functions ==========


def update_display():
    ax.clear()
    ax.imshow(frames[current_frame])
    # Show clicks
    for obj_id, pts in clicks.items():
        for x, y, label in pts:
            color = "green" if label == 1 else "red"
            ax.scatter(x, y, c=color, s=100, marker="*")
    ax.set_title(f"Frame {current_frame} | Obj ID: {current_obj_id}")
    fig.canvas.draw_idle()


def on_click(event):
    global clicks
    if event.inaxes != ax:
        return

    x, y = event.xdata, event.ydata
    if x is None or y is None:
        return

    # Determine label from mouse button
    if event.button == 1:  # Left click = positive
        label = 1
        print(f"Positive click at ({x:.1f}, {y:.1f})")
    elif event.button == 3:  # Right click = negative
        label = 0
        print(f"Negative click at ({x:.1f}, {y:.1f})")
    else:
        return  # Ignore other buttons

    clicks.setdefault(current_obj_id, []).append((x, y, label))
    update_display()


def clear_clicks(event=None):
    global clicks
    clicks = {}
    update_display()


def segment_current():
    print("Segmenting...")
    for obj_id, pts in clicks.items():
        coords = np.array([[x, y] for x, y, _ in pts], dtype=np.float32)
        labels = np.array([label for _, _, label in pts], dtype=np.int32)

        _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=current_frame,
            obj_id=obj_id,
            points=coords,
            labels=labels,
        )

        for i, out_obj_id in enumerate(out_obj_ids):
            mask = (out_mask_logits[i] > 0).cpu().numpy()

            # Use object ID to assign a unique color intensity
            colored_mask = mask.astype(np.float32) * out_obj_id

            # Display with consistent colormap
            ax.imshow(colored_mask.squeeze(), alpha=0.5, cmap="jet")
            print(f"Displayed mask for obj_id {out_obj_id}")

    fig.canvas.draw_idle()


def propagate_all(event=None):
    global video_segments
    print("Propagating masklets...")
    video_segments = {}
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(
        inference_state
    ):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }
    print("Propagation done.")
    update_display()


def generate_color_map(num_classes):
    """
    Generate a color map for visualizing instance masks.
    Returns a dict: {id: (B, G, R)}
    """
    random.seed(42)  # For reproducibility
    color_map = {0: (0, 0, 0)}  # background = black
    for i in range(1, num_classes + 1):
        color_map[i] = tuple(random.randint(0, 255) for _ in range(3))
    return color_map


def save_masks(event=None):
    print("Saving masks as .npy and colored .png files...")

    for frame_idx, masks in video_segments.items():
        # Create an empty mask
        first_mask = next(iter(masks.values()))
        mask_shape = np.squeeze(first_mask).shape
        full_mask = np.zeros(mask_shape, dtype=np.uint8)

        object_ids = list(masks.keys())

        for obj_id, mask in masks.items():
            mask = np.squeeze(mask)
            if mask.ndim != 2:
                raise ValueError(f"Expected 2D mask, got shape {mask.shape}")
            full_mask[mask > 0] = obj_id

        # === Save .npy file ===
        npy_filename = f"{frame_idx:05d}.npy"
        npy_path = os.path.join(output_mask_dir, npy_filename)
        np.save(npy_path, full_mask)

        # === Generate and save color visualization ===
        color_map = generate_color_map(max(object_ids))
        color_mask = np.zeros((*mask_shape, 3), dtype=np.uint8)

        for obj_id, color in color_map.items():
            color_mask[full_mask == obj_id] = color

        png_filename = f"{frame_idx:05d}.png"
        png_path = os.path.join(output_mask_dir, png_filename)
        cv2.imwrite(png_path, color_mask)

        print(f"Saved: {npy_path}, {png_path}")

    print(f"All masks and visualizations saved to {output_mask_dir}")


def next_frame(event=None):
    global current_frame
    current_frame = min(current_frame + 1, num_frames - 1)
    update_display()


def prev_frame(event=None):
    global current_frame
    current_frame = max(current_frame - 1, 0)
    update_display()


def set_obj_id(text):
    global current_obj_id
    try:
        current_obj_id = int(text)
        update_display()
    except ValueError:
        print("Invalid object ID")


def restart(event=None):
    global video_segments, clicks, current_frame, current_obj_id

    print("🔄 Restarting session...")

    # Clear all segmentation data and clicks
    video_segments.clear()
    clicks.clear()

    # Reset frame and object ID
    current_frame = 0
    current_obj_id = 1

    # Reset text box value
    text_box.set_val(str(current_obj_id))

    # Update display
    update_display()

    print("✅ Session restarted.")


# ========== GUI Setup ==========
fig, ax = plt.subplots(figsize=(10, 7))
plt.subplots_adjust(bottom=0.38)  # More space for 2 rows

fig.canvas.mpl_connect("button_press_event", on_click)

# === Row 1: Main Actions ===
ax_seg = plt.axes([0.01, 0.22, 0.1, 0.05])
btn_seg = Button(ax_seg, "Segment")
btn_seg.on_clicked(lambda e: segment_current())

ax_prop = plt.axes([0.12, 0.22, 0.15, 0.05])
btn_prop = Button(ax_prop, "Propagate")
btn_prop.on_clicked(propagate_all)

ax_save = plt.axes([0.28, 0.22, 0.1, 0.05])
btn_save = Button(ax_save, "Save")
btn_save.on_clicked(save_masks)

ax_clear = plt.axes([0.39, 0.22, 0.12, 0.05])
btn_clear = Button(ax_clear, "Clear Clicks")
btn_clear.on_clicked(clear_clicks)

ax_restart = plt.axes([0.52, 0.22, 0.1, 0.05])
btn_restart = Button(ax_restart, "Restart")
btn_restart.on_clicked(restart)

# === Row 2: Navigation + ID + Jump ===
ax_prev = plt.axes([0.01, 0.12, 0.1, 0.05])
btn_prev = Button(ax_prev, "Prev")
btn_prev.on_clicked(prev_frame)

ax_next = plt.axes([0.12, 0.12, 0.1, 0.05])
btn_next = Button(ax_next, "Next")
btn_next.on_clicked(next_frame)

ax_obj_id = plt.axes([0.23, 0.12, 0.12, 0.05])
text_box = TextBox(ax_obj_id, "Obj ID", initial=str(current_obj_id))
text_box.on_submit(set_obj_id)


# === Jump to frame ===
def jump_to_frame(text):
    global current_frame
    try:
        idx = int(text)
        if 0 <= idx < num_frames:
            current_frame = idx
            update_display()
        else:
            print(f"❌ Frame {idx} out of range (0–{num_frames - 1})")
    except ValueError:
        print("❌ Invalid frame index")


ax_jump_box = plt.axes([0.36, 0.12, 0.12, 0.05])
jump_box = TextBox(ax_jump_box, "Jump to", initial=str(current_frame))
jump_box.on_submit(jump_to_frame)

ax_jump_btn = plt.axes([0.49, 0.12, 0.1, 0.05])
btn_jump = Button(ax_jump_btn, "Jump")
btn_jump.on_clicked(lambda e: jump_to_frame(jump_box.text))


print("✅ GUI ready.")
print("🖱️ Left-click = Positive (green), Right-click = Negative (red)")
update_display()
plt.show()
