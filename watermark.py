import cv2
import numpy as np

def embed_watermark(image, watermark_text, position=(50, 50), font_scale=1, color=(255, 0, 0), thickness=2):
    """Embed a watermark into an image."""
    watermarked_image = image.copy()
    cv2.putText(watermarked_image, watermark_text, position, cv2.FONT_HERSHEY_SIMPLEX, 
                font_scale, color, thickness, cv2.LINE_AA)
    return watermarked_image

def extract_watermark(original_image, watermarked_image):
    """Extract watermark by subtracting the original image from the watermarked image."""
    extracted = cv2.absdiff(watermarked_image, original_image)
    
    # Convert to grayscale to highlight watermark regions
    gray = cv2.cvtColor(extracted, cv2.COLOR_BGR2GRAY)
    
    # Apply thresholding to isolate the watermark
    _, thresholded = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)

    return thresholded
