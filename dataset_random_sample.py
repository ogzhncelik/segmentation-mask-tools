"""
Randomly samples a fixed number of GeoTIFF tiles from each class folder
and copies them into a new folder, keeping the same class structure.

Useful for building balanced train/test subsets for segmentation models.

Expected input structure:
    SOURCE_DIR/
        lake/
        terrestrial/
        boundaries/

Output structure:
    OUTPUT_DIR/
        lake/
        terrestrial/
        boundaries/
"""

import os
import random
import shutil

# ============================ SETTINGS ============================
SOURCE_DIR = r"path\to\input"     # folder that contains the class folders
OUTPUT_DIR = r"path\to\output"    # folder where the samples will be copied

# Number of tiles to sample from each class folder
SAMPLE_COUNTS = {
    "lake": 3437, # number of lake tiles to sample
    "terrestrial": 3500, # number of terrestrial tiles to sample
    "boundaries": 936, # number of boundary tiles to sample
}

RANDOM_SEED = 42                   # fixed seed for a reproducible selection
# ==================================================================


def select_and_copy_images(src_folder, dest_folder, count):
    """Randomly select `count` .tif files from src_folder (including
    subfolders) and copy them to dest_folder. Returns the copied file names."""
    if not os.path.isdir(src_folder):
        print(f"Warning: folder not found, skipped -> {src_folder}")
        return []

    # Collect the full paths of .tif files from all subfolders
    tif_files = []
    for root, _, files in os.walk(src_folder):
        for f in files:
            if f.lower().endswith((".tif", ".tiff")):
                tif_files.append(os.path.join(root, f))

    selected_files = random.sample(tif_files, min(count, len(tif_files)))

    os.makedirs(dest_folder, exist_ok=True)

    for src_path in selected_files:
        dest_path = os.path.join(dest_folder, os.path.basename(src_path))
        if os.path.exists(dest_path):
            print(f"Warning: file name already exists, overwritten -> {dest_path}")
        shutil.copy2(src_path, dest_path)

    return [os.path.basename(f) for f in selected_files]


def main():
    random.seed(RANDOM_SEED)

    for class_name, count in SAMPLE_COUNTS.items():
        selected = select_and_copy_images(
            src_folder=os.path.join(SOURCE_DIR, class_name),
            dest_folder=os.path.join(OUTPUT_DIR, class_name),
            count=count,
        )
        print(f"{class_name}: {len(selected)} / {count} files copied")


if __name__ == "__main__":
    main()