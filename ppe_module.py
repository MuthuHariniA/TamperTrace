import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2

class PPEM(nn.Module):
    def __init__(self):
        super(PPEM, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 3, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.conv2(x)
        return x

def load_image(image_path):
    """ Load image as NumPy array (H, W, C) """
    image = cv2.imread(image_path)  # Read as BGR
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
    image = image.astype(np.float32) / 255.0  # Normalize
    return image

def save_image(image_np, output_path):
    """ Save NumPy image to file """
    image_np = (image_np * 255).astype(np.uint8)  # Convert back to uint8
    image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)  # Convert to BGR
    cv2.imwrite(output_path, image_np)
