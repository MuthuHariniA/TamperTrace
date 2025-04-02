import torch
import numpy as np

def detect_tampering(model, image):
    """Detect tampering in an image using EditGuard."""
    model.eval()
    image_tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        output = model(image_tensor)
    return np.mean(output.numpy())  # Placeholder for actual tamper score
