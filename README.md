# segmentation-mask-tools

Scripts for preparing ground-truth data for semantic segmentation of satellite imagery (e.g. water / shoreline detection).
They create binary masks for GeoTIFF tiles, extract and visualize mask boundaries, and build balanced training/test subsets.
Each script is standalone: set the paths at the top and run it.

## Scripts

| Script | What it does |
|---|---|
| `mask_binary_by_class.py` | Creates a binary mask for every tile based on its class folder: lake tiles become fully white (255), terrestrial tiles fully black (0), and boundary/river tiles are rasterized from a shapefile. Masks keep the georeferencing of the source tile, and existing masks are skipped. |
| `mask_morphological_boundary.py` | Extracts the boundary of a binary mask with a morphological gradient (dilation − erosion) and shows the mask, the boundary and an overlay side by side. |
| `mask_boundary_xiaolin_wu_antialiased.py` | Traces the mask boundary with sub-pixel contours and draws it as an anti-aliased line using Xiaolin Wu's algorithm, then saves the result as a GeoTIFF. |
| `dataset_random_sample.py` | Randomly samples a fixed number of tiles from each class folder and copies them to a new folder, using a fixed seed for reproducible train/test subsets. |

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python mask_binary_by_class.py
```

Edit the input/output paths at the top of each script before running it.

## License

MIT
