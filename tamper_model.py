import numpy as np
import torch
import torch.nn as nn
import cv2
from utils import save_image

class BaseTamperModel(nn.Module):
    """Base class for tamper detection models."""
    def __init__(self, name):
        super(BaseTamperModel, self).__init__()
        self.name = name

    def detect_tampering(self, original, tampered):
        """
        Dummy function for tamper detection.
        Returns a simulated tamper detection map.
        """
        height, width, _ = original.shape
        return torch.rand(height, width, dtype=torch.float32).numpy()  # Convert torch to NumPy

# Define AI-based models
class MVSSNet(BaseTamperModel):
    def __init__(self):
        super().__init__("MVSSNet")

class PSCCNet(BaseTamperModel):
    def __init__(self):
        super().__init__("PSCCNet")

class OSN(BaseTamperModel):
    def __init__(self):
        super().__init__("OSN")

class HiFiNet(BaseTamperModel):
    def __init__(self):
        super().__init__("HiFiNet")

class Ours(BaseTamperModel):
    def __init__(self):
        super().__init__("Ours")

# Main class for handling tamper detection
class TamperDetection:
    def __init__(self):
        self.model = Ours()  # Use custom AI-based model

    def detect_tampering(self, original, tampered):
        """Applies the AI model to detect tampering."""
        diff = cv2.absdiff(original, tampered)
        gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        _, binary_mask = cv2.threshold(gray_diff, 30, 255, cv2.THRESH_BINARY)

        # Morphological filtering to refine tamper mask
        kernel = np.ones((3, 3), np.uint8)
        refined_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)

        return refined_mask
