import os
import shutil
import json

# Configuration
mydir = r"Laptop\9912"
input(f"WARNING: {mydir}!!!!")
processed_dir = os.path.join(os.getcwd(), r"data\new\sapien", mydir, "masks")
record_file = "mapping.json"
input_root = os.path.join(os.getcwd(), r"data\load\sapien", mydir)

# Load mapping
with open(record_file, "r") as f:
    mapping = json.load(f)

# Ask for bad sample ids:
bad_samples = input("Enter bad sample IDs (comma-separated): ").split(",")
bad_samples = ["{:05d}".format(int(s.strip())) for s in bad_samples]

for key, info in mapping.items():
    npy_file = os.path.join(processed_dir, f"{key}.npy")
    png_file = os.path.join(processed_dir, f"{key}.png")

    if not os.path.exists(npy_file) or not os.path.exists(png_file):
        print(f"Warning: Missing files for {key}")
        continue

    # Destination folder
    dest_folder = os.path.join(input_root, info["domain"], info["split"], "mask")
    os.makedirs(dest_folder, exist_ok=True)

    if key in bad_samples:
        with open(
            os.path.join(input_root, info["domain"], f"{info['split']}_badmask.txt"), "a"
        ) as f:
            f.write(f"{os.path.splitext(info['original_name'])[0]}\n")

    # Restore PNG
    dest_png = os.path.join(dest_folder, info["original_name"])
    shutil.copy2(png_file, dest_png)

    # Restore NPY with same base name
    npy_name = os.path.splitext(info["original_name"])[0] + ".npy"
    dest_npy = os.path.join(dest_folder, npy_name)
    shutil.copy2(npy_file, dest_npy)

print("Restoration complete.")
