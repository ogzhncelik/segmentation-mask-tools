import numpy as np
import rasterio
from skimage import measure  # pip install scikit-image

# ---------- Wu line helpers ----------
def _ipart(x):  return int(np.floor(x))
def _round(x):  return int(np.floor(x + 0.5))
def _fpart(x):  return x - np.floor(x)
def _rfpart(x): return 1.0 - _fpart(x)

def wu_line_blend(img, x0, y0, x1, y1, line_val=180):
    """
    img: uint8 [H,W] (0..255). Draws a Wu line on it using alpha blending.
    line_val: target gray value of the line (e.g. 180).
              Applied with alpha as (1-a)*img + a*line_val.
    """
    H, W = img.shape
    def plot(px, py, a):
        if 0 <= px < W and 0 <= py < H and a > 0:
            base = img[py, px].astype(np.float32)
            img[py, px] = np.clip((1 - a) * base + a * line_val, 0, 255).astype(np.uint8)

    steep = abs(y1 - y0) > abs(x1 - x0)
    if steep:
        x0, y0 = y0, x0
        x1, y1 = y1, x1
    if x0 > x1:
        x0, x1, y0, y1 = x1, x0, y1, y0

    dx = x1 - x0
    dy = y1 - y0
    gradient = dy / dx if dx != 0 else 0.0

    # first endpoint
    xend = _round(x0)
    yend = y0 + gradient * (xend - x0)
    xgap = _rfpart(x0 + 0.5)
    xpxl1 = xend
    ypxl1 = _ipart(yend)

    if steep:
        plot(ypxl1,     xpxl1, _rfpart(yend) * xgap)
        plot(ypxl1 + 1, xpxl1, _fpart(yend)  * xgap)
    else:
        plot(xpxl1, ypxl1,     _rfpart(yend) * xgap)
        plot(xpxl1, ypxl1 + 1, _fpart(yend)  * xgap)

    intery = yend + gradient

    # last endpoint
    xend = _round(x1)
    yend = y1 + gradient * (xend - x1)
    xgap = _fpart(x1 + 0.5)
    xpxl2 = xend
    ypxl2 = _ipart(yend)

    # main body
    if steep:
        for x in range(xpxl1 + 1, xpxl2):
            plot(_ipart(intery),     x, _rfpart(intery))
            plot(_ipart(intery) + 1, x, _fpart(intery))
            intery += gradient
        plot(ypxl2,     xpxl2, _rfpart(yend) * xgap)
        plot(ypxl2 + 1, xpxl2, _fpart(yend)  * xgap)
    else:
        for x in range(xpxl1 + 1, xpxl2):
            plot(x, _ipart(intery),     _rfpart(intery))
            plot(x, _ipart(intery) + 1, _fpart(intery))
            intery += gradient
        plot(xpxl2, ypxl2,     _rfpart(yend) * xgap)
        plot(xpxl2, ypxl2 + 1, _fpart(yend)  * xgap)

# ---------- Main flow ----------
in_tif  = r"path\to\input\boundary_mask.tif"
out_tif = in_tif.replace(".tif", "_with_wu.tif")

# 1) Read the raster
with rasterio.open(in_tif) as src:
    prof = src.profile.copy()
    a = src.read(1)  # we expect 0/255
    transform = src.transform

# 2) Base image: keep left-black / right-white as is
base = a.copy()  # uint8 0/255

# 3) Extract the boundary (0↔1 transition). level=0.5 → sub-pixel contour
binary = (a >= 128).astype(np.uint8)
contours = measure.find_contours(binary.astype(float), level=0.5)  # each contour: (N, 2) (row, col)

# 4) Draw each contour with Wu (between neighboring points)
#    Note: find_contours returns (row, col) → (y, x)
for cnt in contours:
    if len(cnt) < 2:
        continue
    # Wu expects (x,y) relative to pixel centers
    # cnt[:,1]=x, cnt[:,0]=y
    xs = cnt[:, 1]
    ys = cnt[:, 0]
    for (x0, y0), (x1, y1) in zip(zip(xs[:-1], ys[:-1]), zip(xs[1:], ys[1:])):
        wu_line_blend(base, x0, y0, x1, y1, line_val=180)  # 180 → line color; increase/decrease

# 5) Write
prof.update(dtype=rasterio.uint8, count=1, nodata=None, photometric="minisblack")
with rasterio.open(out_tif, "w", **prof) as dst:
    dst.write(base.astype(np.uint8), 1)

print("Done ✅ ->", out_tif)