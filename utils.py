import os
import numpy as np
import torch
import cv2
from PIL import Image
import matplotlib.pyplot as plt

# ==============================
# 📌 Image Loading & Saving
# ==============================
def load_image(image_path, to_rgb=True):
    """
    Loads an image from the given path.
    Converts it to RGB format if required.
    
    Args:
        image_path (str): Path to the image file.
        to_rgb (bool): Convert to RGB format if True.

    Returns:
        np.ndarray: Loaded image.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Error: Unable to load image {image_path}")
    
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if to_rgb else image

def save_image(image, output_folder, filename):
    """
    Saves an image from a NumPy array or Tensor.

    Args:
        image (np.ndarray or torch.Tensor): Image array to save.
        output_folder (str): Directory to save the image.
        filename (str): Filename for the output image.
    """
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, filename)

    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()  # Convert tensor to NumPy

    image = Image.fromarray((image * 255).astype('uint8')) if image.dtype != np.uint8 else Image.fromarray(image)
    image.save(output_path)
    
    print(f"✅ Image saved at: {output_path}")

# ==============================
# 📌 Watermark Processing
# ==============================
def preprocess_watermark(watermark_str):
    """
    Converts a watermark string to a tensor of bits.

    Args:
        watermark_str (str): Input string to be converted.

    Returns:
        torch.Tensor: A tensor representing watermark bits.
    """
    watermark_bits = np.array([int(b) for b in watermark_str], dtype=np.float32)
    return torch.tensor(watermark_bits).unsqueeze(0)

# ==============================
# 📌 Image Display
# ==============================
def show_image(image, title="Image"):
    """
    Displays an image using matplotlib.

    Args:
        image (np.ndarray): Image to be displayed.
        title (str): Title of the image display.
    """
    plt.imshow(image)
    plt.title(title)
    plt.axis("off")
    plt.show()
