"""
This Python script processes raster images (TIF) and vector data (SHP) to create the **binary masks** required for segmentation models. It produces masks for three different image categories: `lake`, `terrestrial`, and `boundaries` (boundary regions).

🎯 Purpose:
- Create masks with completely **white (255)** pixel values for lake images
- Create masks with completely **black (0)** pixel values for terrestrial images
- Produce masks for boundary regions using the geometry in the SHP file

📥 Input Data:
- `lake/`, `terrestrial/`, `boundaries/`: 256x256 raster images in TIF format
- `vector.shp`: Shapefile containing the boundary regions (contains polygon geometry)

📤 Output:
- Inside the `binary_masks/` folder:
  - `lake/`: Masks with all values 255
  - `terrestrial/`: Masks with all values 0
  - `boundaries/`: Masks rasterized from the shapefile (areas containing 255)

🧪 Processing:
1. For each `lake` image, a completely 255 (white) mask is created.
2. For each `terrestrial` image, a completely 0 (black) mask is created.
3. Each `boundaries` image is converted into a mask by rasterizing it with the geometry in the SHP file.
4. Each mask is written with `rasterio`; boundary masks are saved with `cv2.imwrite()`.

📌 Note:
- If the coordinate system of the `vector.shp` file does not match the raster files, it is automatically reprojected.

"""


import os
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
import rasterio
from rasterio import features
import geopandas as gpd
from tqdm import tqdm

# 📌 Data folders
lake_folder = r"path\to\input\lake"
terrestrial_folder = r"path\to\input\terrestrial"
boundaries_folder = r"path\to\input\boundaries"
vector_path = r"path\to\input\shapefile.shp"
river_folder = r"path\to\input\river"
mask_folder = r"path\to\output"


# 📌 Create the output folder (left untouched if it exists)
os.makedirs(mask_folder, exist_ok=True)

mask_lake_folder = os.path.join(mask_folder, "lake")
mask_terrestrial_folder = os.path.join(mask_folder, "terrestrial")
mask_boundaries_folder = os.path.join(mask_folder, "boundaries")
mask_river_folder = os.path.join(mask_folder, "river")

os.makedirs(mask_lake_folder, exist_ok=True)
os.makedirs(mask_terrestrial_folder, exist_ok=True)
os.makedirs(mask_boundaries_folder, exist_ok=True)
os.makedirs(mask_river_folder, exist_ok=True)

# 📌 Read the files


lake_files = [f for f in os.listdir(lake_folder) if f.endswith(".tif")] if os.path.exists(lake_folder) else []
terrestrial_files = [f for f in os.listdir(terrestrial_folder) if f.endswith(".tif")] if os.path.exists(terrestrial_folder) else []
boundaries_files = [f for f in os.listdir(boundaries_folder) if f.endswith(".tif")] if os.path.exists(boundaries_folder) else []
river_files = [f for f in os.listdir(river_folder) if f.endswith(".tif")] if os.path.exists(river_folder) else []

lake_missing = sum(1 for f in lake_files if not os.path.exists(os.path.join(mask_lake_folder, f)))
terrestrial_missing = sum(1 for f in terrestrial_files if not os.path.exists(os.path.join(mask_terrestrial_folder, f)))
boundaries_missing = sum(1 for f in boundaries_files
                         if not os.path.exists(os.path.join(mask_boundaries_folder, os.path.splitext(f)[0] + ".tif")))
river_missing = sum(1 for f in river_files
                    if not os.path.exists(os.path.join(mask_river_folder, os.path.splitext(f)[0] + ".tif")))
#total_files = lake_missing + terrestrial_missing + boundaries_missing + river_missing

total_files = len(lake_files) + len(terrestrial_files) + len(boundaries_files) + len(river_files)  # Total of all files

# 📌 Mask generation function for lake and terrestrial
def create_binary_mask(image_path, is_water, save_folder):
    with rasterio.open(image_path) as src:
        shape = (src.height, src.width)
        mask = np.full(shape, 255, dtype=np.uint8) if is_water else np.zeros(shape, dtype=np.uint8)

        file_name = os.path.basename(image_path)
        mask_path = os.path.join(save_folder, file_name)

        meta = src.meta.copy()
        if is_water:
            # 3-band RGB white mask
            meta.update(dtype=rasterio.uint8, count=3, nodata=None)
            mask_rgb = np.stack([mask, mask, mask])  # R=255, G=255, B=255
            with rasterio.open(mask_path, "w", **meta) as dst:
                dst.write(mask_rgb)
        else:
            # Single band 0 (black) for terrestrial
            meta.update(dtype=rasterio.uint8, count=1, nodata=None)
            with rasterio.open(mask_path, "w", **meta) as dst:
                dst.write(mask, 1)

    pbar.update(1)


# 📌 Read the shapefile
vector_data = gpd.read_file(vector_path)

# 📌 Mask generation
with tqdm(total=total_files, desc="⏳ Mask Generation", unit="file") as pbar:

    # Lake images
    for file in lake_files:
        if file.endswith(".tif"):
            mask_out = os.path.join(mask_lake_folder, file)
            if os.path.exists(mask_out):
                pbar.update(1)  # Mask already exists → skip
                continue
            create_binary_mask(os.path.join(lake_folder, file), is_water=True, save_folder=mask_lake_folder)

    # Terrestrial images
    for file in terrestrial_files:
        if file.endswith(".tif"):
            mask_out = os.path.join(mask_terrestrial_folder, file)
            if os.path.exists(mask_out):
                pbar.update(1)
                continue
            create_binary_mask(os.path.join(terrestrial_folder, file), is_water=False, save_folder=mask_terrestrial_folder)

    # Boundaries (with shapefile)
    # Boundaries (with shapefile) — WRITE WITH COORDINATES
    for file in boundaries_files:
        if file.endswith(".tif"):
            tif_path = os.path.join(boundaries_folder, file)

            mask_filename = os.path.splitext(file)[0] + ".tif"
            mask_path = os.path.join(mask_boundaries_folder, mask_filename)
            if os.path.exists(mask_path):
                pbar.update(1)  # Mask already exists → skip
                continue

            with rasterio.open(tif_path) as src:
                # Spatial information of the raster
                transform = src.transform
                crs = src.crs
                height, width = src.height, src.width

                # Convert the vector to the raster's CRS (without modifying the original)
                vector_reproj = vector_data.to_crs(crs) if vector_data.crs != crs else vector_data

                # Rasterize (0 background, 255 for areas that fall on the boundary)
                mask = features.rasterize(
                    [(geom, 255) for geom in vector_reproj.geometry],
                    out_shape=(height, width),
                    transform=transform,
                    fill=0,
                    dtype=np.uint8
                )

                # Output path and metadata
                mask_filename = os.path.splitext(file)[0] + ".tif"
                mask_path = os.path.join(mask_boundaries_folder, mask_filename)

                meta = src.meta.copy()
                meta.update({
                    "count": 1,  # single band (mask)
                    "dtype": "uint8",
                    "compress": "lzw",
                    #"nodata": 0
                })

                # Write as GeoTIFF
                with rasterio.open(mask_path, "w", **meta) as dst:
                    dst.write(mask, 1)

            pbar.update(1)

    # River (with shapefile) — WRITE WITH COORDINATES
    for file in river_files:
        if file.endswith(".tif"):
            tif_path = os.path.join(river_folder, file)

            mask_filename = os.path.splitext(file)[0] + ".tif"
            mask_path = os.path.join(mask_river_folder, mask_filename)
            if os.path.exists(mask_path):
                pbar.update(1)  # Mask already exists → skip
                continue

            with rasterio.open(tif_path) as src:
                transform = src.transform
                crs = src.crs
                height, width = src.height, src.width

                # Convert the vector to the raster CRS
                vector_reproj = vector_data.to_crs(crs) if vector_data.crs != crs else vector_data

                # (OPTIONAL) If the vector is a line (LineString), you can apply a buffer to make the river thicker:
                # buffer_size = 2   # not in pixels; if the CRS is in meters, it is in meters!
                # vector_reproj = vector_reproj.copy()
                # vector_reproj["geometry"] = vector_reproj.geometry.buffer(buffer_size)

                # Rasterize: areas covered by the vector are 255 (white), other areas are 0 (black)
                mask = features.rasterize(
                    [(geom, 255) for geom in vector_reproj.geometry],
                    out_shape=(height, width),
                    transform=transform,
                    fill=0,
                    dtype=np.uint8
                )



                meta = src.meta.copy()
                meta.update({
                    "count": 1,
                    "dtype": "uint8",
                    "compress": "lzw",
                })

                with rasterio.open(mask_path, "w", **meta) as dst:
                    dst.write(mask, 1)

            pbar.update(1)

print("✅ All masks were created successfully.")