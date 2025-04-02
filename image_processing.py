import cv2
import numpy as np
import os

def process_images(original, tampered, output_folder):
    """Process images to generate difference and thresholded images."""

    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Convert to grayscale
    original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    tampered_gray = cv2.cvtColor(tampered, cv2.COLOR_BGR2GRAY)

    # Save grayscale images
    cv2.imwrite(os.path.join(output_folder, "03_Grayscale_Original.png"), original_gray)
    cv2.imwrite(os.path.join(output_folder, "04_Grayscale_Tampered.png"), tampered_gray)

    # Compute pixel-wise absolute difference
    diff = cv2.absdiff(original_gray, tampered_gray)

    # Ensure the image is properly converted for saving
    diff = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    success_diff = cv2.imwrite(os.path.join(output_folder, "06_Difference.png"), diff)

    if not success_diff:
        print("Error saving Difference Image")

    # Apply thresholding
    _, threshold_diff = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
    success_threshold = cv2.imwrite(os.path.join(output_folder, "07_Threshold.png"), threshold_diff)

    if not success_threshold:
        print("Error saving Threshold Image")

    # Edge Detection (Canny)
    edges = cv2.Canny(diff, 50, 150)
    success_edges = cv2.imwrite(os.path.join(output_folder, "08_Edges.png"), edges)

    if not success_edges:
        print("Error saving Edges Image")

    print("All images processed and saved successfully!")
