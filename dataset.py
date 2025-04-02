import cv2
import glob
import os
import logging

def load_dataset(dataset_name, image_formats=["*.jpg", "*.png", "*.jpeg"], preprocess=None):
    """Load dataset (e.g., COCO) with support for multiple image formats and optional preprocessing."""
    try:
        # Create the dataset folder path
        dataset_path = os.path.join("datasets", dataset_name)
        
        # Check if the dataset exists
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset folder '{dataset_path}' not found.")
        
        # Initialize an empty list for images
        images = []
        
        # Collect image file paths for each format
        image_paths = []
        for fmt in image_formats:
            image_paths.extend(glob.glob(os.path.join(dataset_path, fmt)))
        
        # Check if images were found
        if not image_paths:
            raise FileNotFoundError(f"No image files found in '{dataset_path}' for formats: {image_formats}")
        
        # Read images and optionally apply preprocessing
        for img_path in image_paths:
            img = cv2.imread(img_path)
            if img is None:
                logging.warning(f"Failed to read image: {img_path}")
                continue
            
            # Apply preprocessing if provided
            if preprocess:
                img = preprocess(img)
            
            images.append(img)
        
        logging.info(f"Loaded {len(images)} images from '{dataset_name}' dataset.")
        return images
    
    except Exception as e:
        logging.error(f"Error loading dataset '{dataset_name}': {e}")
        return []

