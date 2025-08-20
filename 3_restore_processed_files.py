import os
import shutil
import json

# Configuration
mydir = r"oven\101917"
input(f"WARNING: {mydir}!!!!")
processed_dir = os.path.join(r"C:\Users\Douge\Workspace\sam2\nerf_masks", mydir)
record_file = 'mapping.json'
input_root = os.path.join(r'D:\Datasets\dataset_paris\load\sapien', mydir)  # Base path to dataset


# Load mapping
with open(record_file, 'r') as f:
    mapping = json.load(f)

for key, info in mapping.items():
    npy_file = os.path.join(processed_dir, f"{key}.npy")
    png_file = os.path.join(processed_dir, f"{key}.png")

    if not os.path.exists(npy_file) or not os.path.exists(png_file):
        print(f"Warning: Missing files for {key}")
        continue

    # Destination folder
    dest_folder = os.path.join(input_root, info['domain'], info['split'], 'mask')
    os.makedirs(dest_folder, exist_ok=True)

    # Restore PNG
    dest_png = os.path.join(dest_folder, info['original_name'])
    shutil.copy2(png_file, dest_png)

    # Restore NPY with same base name
    npy_name = os.path.splitext(info['original_name'])[0] + '.npy'
    dest_npy = os.path.join(dest_folder, npy_name)
    shutil.copy2(npy_file, dest_npy)

print("Restoration complete.")
