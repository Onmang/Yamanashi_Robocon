import os

target_dir = "."

for filename in os.listdir(target_dir):
    old_path = os.path.join(target_dir, filename)

    if not os.path.isfile(old_path):
        continue

    # "image _" → "image_"
    new_name = filename.replace("image _", "image_")

    new_path = os.path.join(target_dir, new_name)

    if old_path != new_path:
        os.rename(old_path, new_path)
        print(f"Renamed: {filename} -> {new_name}")
