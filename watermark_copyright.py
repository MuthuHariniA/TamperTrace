import cv2
import numpy as np

def embed_copyright_watermark(image_path, output_path):
    """Embed a copyright watermark (binary pattern) into the image"""
    image = cv2.imread(image_path)

    # Get image dimensions
    height, width, _ = image.shape

    # Generate a binary watermark and resize it to match the image
    binary_watermark = np.random.randint(0, 2, (32, 32), dtype=np.uint8) * 255  # Random 32x32 watermark
    watermark = cv2.resize(binary_watermark, (width, height))  # Resize to match image
    watermark = cv2.cvtColor(watermark, cv2.COLOR_GRAY2BGR)  # Convert to 3 channels

    # Blend watermark with image
    watermarked_image = cv2.addWeighted(image, 0.9, watermark, 0.1, 0)

    # Save output image
    cv2.imwrite(output_path, watermarked_image)
    return output_path

def extract_copyright_watermark(original_path, watermarked_path, output_path):
    """Extract a copyright watermark by comparing original & watermarked images"""
    original = cv2.imread(original_path)
    watermarked = cv2.imread(watermarked_path)

    # Ensure both images are the same size
    if original.shape != watermarked.shape:
        watermarked = cv2.resize(watermarked, (original.shape[1], original.shape[0]))

    # Extract difference
    diff = cv2.absdiff(original, watermarked)

    # Save extracted watermark
    cv2.imwrite(output_path, diff)
    return output_path
