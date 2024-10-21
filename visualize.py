import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2

# Load the .tif label file (assuming it's a single-channel grayscale label map)
label_tif = Image.open('/home/yueming/data/DDR-dataset/lesion_segmentation/valid/segmentation label/HE/007-5846-300.tif')
label_np = np.array(label_tif)

# Load the corresponding image
image = Image.open('/home/yueming/data/DDR-dataset/lesion_segmentation/valid/image/007-3036-100.jpg')  # or png, etc.
image_np = np.array(image)

# Normalize the label values to be between 0 and 1 (to apply a heatmap)
label_normalized = (label_np - np.min(label_np)) / (np.max(label_np) - np.min(label_np))

# Apply a colormap to the normalized label (convert to heatmap)
heatmap = cv2.applyColorMap(np.uint8(255 * label_normalized), cv2.COLORMAP_JET)

# Resize the heatmap to match the size of the image (if needed)
if heatmap.shape[:2] != image_np.shape[:2]:
    heatmap = cv2.resize(heatmap, (image_np.shape[1], image_np.shape[0]))

# If the image is grayscale, convert it to BGR to match the heatmap format
if len(image_np.shape) == 2:
    image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2BGR)

# Overlay the heatmap on the image
alpha = 0.6  # Transparency for the heatmap
overlay_image = cv2.addWeighted(image_np, 1 - alpha, heatmap, alpha, 0)

# Display the result
plt.figure(figsize=(10, 10))
plt.imshow(cv2.cvtColor(overlay_image, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for displaying
plt.axis('off')
plt.show()

# Optionally save the result
cv2.imwrite('overlayed_image_with_heatmap.png', overlay_image)

