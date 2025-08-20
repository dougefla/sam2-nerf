import os
from PIL import Image
import json


def prepare_images(mydir):
    input_root = os.path.join(os.getcwd(), r"data\load\sapien", mydir)
    output_dir = os.path.join(os.getcwd(), r"data\new\sapien", mydir, "images")
    record_file = os.path.join(output_dir, "mapping.json")

    os.makedirs(output_dir, exist_ok=True)

    mapping = {}
    counter = 0

    for domain in ["start", "end"]:
        for split in ["train", "val", "test"]:
            folder = os.path.join(input_root, domain, split)
            for fname in sorted(os.listdir(folder)):
                if fname.endswith(".png"):
                    orig_path = os.path.join(folder, fname)
                    new_name = f"{counter:05d}.jpg"
                    new_path = os.path.join(output_dir, new_name)

                    # Convert PNG to JPG with white background
                    with Image.open(orig_path) as im:
                        if im.mode in ("RGBA", "LA") or (
                            im.mode == "P" and "transparency" in im.info
                        ):
                            # Create white background image
                            background = Image.new("RGB", im.size, (255, 255, 255))
                            background.paste(
                                im.convert("RGBA"), mask=im.convert("RGBA").split()[-1]
                            )
                            background.save(new_path, "JPEG")
                        else:
                            rgb_im = im.convert("RGB")
                            rgb_im.save(new_path, "JPEG")

                    # Record mapping
                    mapping[f"{counter:05d}"] = {
                        "original_path": orig_path,
                        "original_name": fname,
                        "domain": domain,
                        "split": split,
                    }

                    counter += 1

    # Save mapping
    with open(record_file, "w") as f:
        json.dump(mapping, f, indent=2)

    print(f"✅ Processed {counter} images. Mapping saved to '{record_file}'.")


if __name__ == "__main__":
    mydir = r"Laptop\9912"
    prepare_images(mydir)
