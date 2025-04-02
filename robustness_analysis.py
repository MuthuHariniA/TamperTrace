# robustness_analysis.py
import cv2
import numpy as np

def add_gaussian_noise(image, sigma=5):
    noise = np.random.normal(0, sigma, image.shape).astype(np.uint8)
    return cv2.add(image, noise)

def apply_jpeg_compression(image, quality=80):
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, encoded_image = cv2.imencode('.jpg', image, encode_param)
    return cv2.imdecode(encoded_image, cv2.IMREAD_UNCHANGED)

def test_robustness(image):
    degraded_image = add_gaussian_noise(image, sigma=5)
    compressed_image = apply_jpeg_compression(image, quality=80)
    
    return degraded_image, compressed_image
