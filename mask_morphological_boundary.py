import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def boundary_extraction(mask_path, kernel_size=(3, 3)):
    # 1. Read the TIF mask
    # Note: write the path of a real mask on your computer here
    mask_pil = Image.open(mask_path).convert("L")
    mask = np.array(mask_pil, dtype=np.float32)

    # 2. Threshold the mask (make it 0 and 1)
    # Logic: pixels with value 255 become 1
    mask_binary = np.zeros_like(mask)
    mask_binary[mask == 255.0] = 1.0

    # 3. Morphological gradient operation (dilate - erode)
    # Convert the mask back to uint8 format (for OpenCV)
    mask_uint8 = (mask_binary * 255).astype(np.uint8)

    kernel = np.ones(kernel_size, np.uint8)

    dilated = cv2.dilate(mask_uint8, kernel, iterations=1)
    eroded = cv2.erode(mask_uint8, kernel, iterations=1)

    # Calculate the boundary
    boundary = dilated - eroded

    # 4. Visualization
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.title("Original Mask")
    plt.imshow(mask_binary, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title(f"Extracted Boundary (Kernel: {kernel_size})")
    plt.imshow(boundary, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title("Overlay")
    plt.imshow(mask_binary, cmap='gray')
    plt.imshow(boundary, cmap='Reds', alpha=0.5)  # Show the boundary in red
    plt.axis('off')

    plt.show()


# --- FOR TESTING ---
# Copy the path of one of your real mask files here
test_path = r"path\to\input\mask.tif"
boundary_extraction(test_path, kernel_size=(3, 3))