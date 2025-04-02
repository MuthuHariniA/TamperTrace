import os
from turtle import pd
import cv2
from matplotlib import pyplot as plt
import torch
import logging
import numpy as np
from utils import load_image, save_image, preprocess_watermark
from watermark_comparison import WatermarkComparison
from watermark_localization import embed_localization_watermark, extract_localization_watermark
from watermark_copyright import embed_copyright_watermark, extract_copyright_watermark
from ppe_module import PPEM  
from bit_encryption import BitEncryptionModule, encrypt_watermark
from recovery_module import RecoveryModule, recover_watermark
from tamper_detection import detect_tampering
from aigc_editing_methods import AIGCEditingMethods 
from tamper_localization import TamperLocalization
from copyright_protection import CopyrightProtection
from tamper_models import MVSSNet, PSCCNet, OSN, HiFiNet, Ours, GT
import os
from skimage.metrics import structural_similarity as ssim
from matplotlib import pyplot as plt

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Define output folder
OUTPUT_FOLDER = "output_images"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def process_image(image_path):
    """Processes an image by embedding watermarks, applying PPEM, and performing encryption & recovery."""
    try:
        logging.info(f"Processing image: {image_path}")

        # Check if the image file exists
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file '{image_path}' not found.")
        
        # Embed Localization Watermark
        loc_watermarked = os.path.join(OUTPUT_FOLDER, "localization_watermarked.png")
        embed_localization_watermark(image_path, "TamperCheck123", loc_watermarked)

        # Embed Copyright Watermark
        copyright_watermarked = os.path.join(OUTPUT_FOLDER, "copyright_watermarked.png")
        embed_copyright_watermark(image_path, copyright_watermarked)

        # Extract Localization Watermark
        extracted_text = extract_localization_watermark(loc_watermarked)
        logging.info(f"Extracted Localization Watermark: {extracted_text}")

        # Extract Copyright Watermark
        extracted_copyright = os.path.join(OUTPUT_FOLDER, "extracted_copyright.png")
        extract_copyright_watermark(image_path, copyright_watermarked, extracted_copyright)

        # Apply PPEM for Enhancement
        ppe_model = PPEM()

        # Load and check the image
        image_np = load_image(extracted_copyright)
        if image_np is None:
            raise ValueError(f"Failed to load image: {extracted_copyright}")
        
        # Check the shape of the image (e.g., it should be 3-channel RGB)
        if len(image_np.shape) != 3 or image_np.shape[2] != 3:
            raise ValueError(f"Invalid image shape: {image_np.shape}, expected (H, W, 3) for RGB image.")
        
        image_tensor = torch.from_numpy(image_np).float().permute(2, 0, 1).unsqueeze(0)
        enhanced_output = ppe_model(image_tensor).detach().cpu().squeeze(0).permute(1, 2, 0).numpy()

        # Save Enhanced Image
        enhanced_output_path = os.path.join(OUTPUT_FOLDER, "enhanced_copyright.png")
        save_image(enhanced_output, enhanced_output_path)
        logging.info(f"Enhanced watermark saved as {enhanced_output_path}")

        # Encrypt and Recover Watermark
        encrypt_and_recover_watermark()

    except Exception as e:
        logging.error(f"Error processing image '{image_path}': {e}")

def encrypt_and_recover_watermark():
    """Encrypts and recovers a watermark using neural network models."""
    try:
        input_size, hidden_size, output_size = 10, 20, 10

        # Initialize Models
        encryption_model = BitEncryptionModule(input_size, hidden_size, output_size)
        recovery_model = RecoveryModule(output_size, hidden_size, input_size)

        # Sample Watermark
        watermark_str = "1010101010"
        watermark_tensor = preprocess_watermark(watermark_str)

        # Encrypt and Recover
        encrypted = encrypt_watermark(watermark_tensor, encryption_model)
        recovered = recover_watermark(encrypted, recovery_model)

        logging.info(f"Original Watermark: {watermark_str}")
        logging.info(f"Recovered Watermark: {recovered.detach().numpy()}")

    except Exception as e:
        logging.error(f"Error in encryption & recovery: {e}")
        
def compare_images(original_path, tampered_path, output_folder="tamper_results"):
    """
    Enhanced image comparison with better difference visualization
    """
    try:
        # Load images
        original = cv2.imread(original_path)
        tampered = cv2.imread(tampered_path)
        
        if original is None or tampered is None:
            raise ValueError("Failed to load one or both images")
            
        # Ensure same dimensions
        if original.shape != tampered.shape:
            tampered = cv2.resize(tampered, (original.shape[1], original.shape[0]))

        # Convert to grayscale
        original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        tampered_gray = cv2.cvtColor(tampered, cv2.COLOR_BGR2GRAY)

        # Calculate absolute difference
        diff = cv2.absdiff(original_gray, tampered_gray)
        
        # Enhanced difference visualization
        # 1. Normalize to 0-255
        diff_norm = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
        
        # 2. Apply contrast stretching
        min_val = np.min(diff_norm)
        max_val = np.max(diff_norm)
        stretched = ((diff_norm - min_val) / (max_val - min_val) * 255).astype(np.uint8)
        
        # 3. Apply color mapping for better visualization
        colored_diff = cv2.applyColorMap(stretched, cv2.COLORMAP_JET)
        
        # 4. Create highlighted version (red differences on original)
        highlighted = original.copy()
        mask = stretched > 10  # Threshold for significant differences
        highlighted[mask] = [0, 0, 255]  # Mark differences in red

        # Save results
        os.makedirs(output_folder, exist_ok=True)
        cv2.imwrite(f"{output_folder}/01_original.png", original)
        cv2.imwrite(f"{output_folder}/02_tampered.png", tampered)
        cv2.imwrite(f"{output_folder}/03_raw_difference.png", diff)
        cv2.imwrite(f"{output_folder}/04_normalized_difference.png", diff_norm)
        cv2.imwrite(f"{output_folder}/05_stretched_difference.png", stretched)
        cv2.imwrite(f"{output_folder}/06_colored_difference.png", colored_diff)
        cv2.imwrite(f"{output_folder}/07_highlighted.png", highlighted)
        
        print(f"Results saved to {output_folder}")
        return {
            'original': original,
            'tampered': tampered,
            'raw_diff': diff,
            'norm_diff': diff_norm,
            'stretched': stretched,
            'colored': colored_diff,
            'highlighted': highlighted
        }
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return None

# Example usage:
results = compare_images("org.png", "tamp.png")

def result_compare_images(original_path, tampered_path):
    """Compares original and tampered images using AI-based tamper detection."""
    result_paths = {}
    try:
        logging.info("Comparing images for tamper detection...")
        original = cv2.imread(original_path)
        tampered = cv2.imread(tampered_path)

        if original is None or tampered is None:
            raise ValueError("Failed to load images")
        
        if original.shape != tampered.shape:
            tampered = cv2.resize(tampered, (original.shape[1], original.shape[0]))

        # Basic difference methods
        gray_original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        gray_tampered = cv2.cvtColor(tampered, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(gray_original, gray_tampered)
        _, threshold_diff = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

        # Save outputs
        save_image(original, os.path.join(OUTPUT_FOLDER, "01_Original.png"))
        save_image(tampered, os.path.join(OUTPUT_FOLDER, "02_Tampered.png"))
        save_image(threshold_diff, os.path.join(OUTPUT_FOLDER, "04_Threshold_Difference.png"))

        # Process with advanced methods
        models = {
            'IML_VIT': detect_iml_vit,
            'MVSS_Net': detect_mvss_net,
            'PSCC_Net': detect_pscc_net,
            'OSN': detect_osn,
            'HiFi_Net': detect_hifi_net,
            'Custom': detect_custom_method
        }

        for model_name, detector in models.items():
            logging.info(f"Running {model_name}...")
            tamper_map = detector(original.copy(), tampered.copy())
            if tamper_map is not None:
                path = os.path.join(OUTPUT_FOLDER, f"05_{model_name}.png")
                save_image((tamper_map * 255).astype(np.uint8), path)
                result_paths[model_name] = path

        logging.info("Tamper detection completed successfully.")
        return result_paths

    except Exception as e:
        logging.error(f"Error comparing images: {e}")
        return None

def generate_comparison_collage(result_paths):
    """Creates a visual comparison collage of all results"""
    try:
        valid_images = {k:v for k,v in result_paths.items() if v and os.path.exists(v)}
        if not valid_images:
            raise ValueError("No valid images to create collage")
            
        images = []
        titles = []
        for name, path in valid_images.items():
            img = cv2.imread(path)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                images.append(img)
                titles.append(name.replace('_', ' ').title())

        n_cols = min(4, len(images))
        n_rows = (len(images) + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5*n_rows))
        
        if n_rows == 1:
            axes = np.array([axes])
        
        for ax, img, title in zip(axes.flatten(), images, titles):
            ax.imshow(img)
            ax.set_title(title)
            ax.axis('off')

        for ax in axes.flatten()[len(images):]:
            ax.axis('off')
            
        plt.tight_layout()
        collage_path = os.path.join(OUTPUT_FOLDER, "comparison_collage.png")
        plt.savefig(collage_path, bbox_inches='tight', dpi=300)
        plt.close()
        logging.info(f"Saved comparison collage to: {collage_path}")
        return collage_path
    except Exception as e:
        logging.error(f"Error generating collage: {str(e)}")
        return None


def calculate_precision_recall(pred, gt):
    """Calculate precision and recall metrics"""
    true_pos = np.sum((pred == 255) & (gt == 255))
    false_pos = np.sum((pred == 255) & (gt == 0))
    false_neg = np.sum((pred == 0) & (gt == 255))
    
    precision = true_pos / (true_pos + false_pos + 1e-6)
    recall = true_pos / (true_pos + false_neg + 1e-6)
    
    return precision, recall

def generate_comparison_report(results, OUTPUT_FOLDER):
    """Generate visual comparison and metrics report"""
    # Create figure
    fig = plt.figure(figsize=(20, 15))
    
    # Plot all method results
    methods = list(results['methods'].keys())
    n_methods = len(methods)
    
    for i, method in enumerate(methods, 1):
        ax = fig.add_subplot(3, (n_methods+2)//3, i)
        ax.imshow(cv2.cvtColor(results['methods'][method], cv2.COLOR_BGR2RGB))
        ax.set_title(method.replace('_', ' '))
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_FOLDER}/method_comparison.png")
    plt.close()
    
    # Create metrics table if available
    if results['metrics']:
        metrics_df = pd.DataFrame.from_dict(results['metrics'], orient='index')
        metrics_df.to_csv(f"{OUTPUT_FOLDER}/metrics_report.csv")
        
        # Plot metrics comparison
        metrics_df.plot(kind='bar', figsize=(15, 8))
        plt.title('Detection Method Performance Comparison')
        plt.ylabel('Score')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_FOLDER}/metrics_comparison.png")
        plt.close()

# Example detection functions (implement according to your specific methods)
def detect_iml_vit(original, tampered):
    """
    IML-VIT (Image Manipulation Localization using Vision Transformer) detection
    Placeholder implementation - replace with actual model inference
    """
    try:
        # Convert images to grayscale
        gray_original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        gray_tampered = cv2.cvtColor(tampered, cv2.COLOR_BGR2GRAY)
        
        # Calculate absolute difference as placeholder
        diff = cv2.absdiff(gray_original, gray_tampered)
        
        # Normalize to 0-1 range
        diff_normalized = diff.astype(np.float32) / 255.0
        
        # Apply threshold to create binary mask (placeholder)
        _, binary_mask = cv2.threshold(diff, 30, 1, cv2.THRESH_BINARY)
        
        return binary_mask
    
    except Exception as e:
        print(f"Error in IML-VIT detection: {str(e)}")
        return None

def detect_mvss_net(original, tampered):
    """
    MVSS-Net detection placeholder
    Replace with actual MVSS-Net implementation
    """
    try:
        # Placeholder using structural similarity
        gray_original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        gray_tampered = cv2.cvtColor(tampered, cv2.COLOR_BGR2GRAY)
        
        # Calculate SSIM map
        score, diff = structural_similarity(
            gray_original, gray_tampered, full=True)
        diff = (diff * 255).astype(np.uint8)
        
        # Threshold difference map
        _, thresh = cv2.threshold(diff, 0, 1, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        
        return thresh
    
    except Exception as e:
        print(f"Error in MVSS-Net detection: {str(e)}")
        return None

def detect_pscc_net(original, tampered):
    """
    PSCC-Net detection placeholder
    Replace with actual PSCC-Net implementation
    """
    try:
        # Placeholder using Canny edge differences
        edges_original = cv2.Canny(original, 100, 200)
        edges_tampered = cv2.Canny(tampered, 100, 200)
        
        diff = cv2.absdiff(edges_original, edges_tampered)
        diff_normalized = diff.astype(np.float32) / 255.0
        
        return diff_normalized
    
    except Exception as e:
        print(f"Error in PSCC-Net detection: {str(e)}")
        return None

def detect_osn(original, tampered):
    """
    OSN (Object Scene Network) detection placeholder
    Replace with actual OSN implementation
    """
    try:
        # Placeholder using color histogram differences
        hist_original = cv2.calcHist([original], [0,1,2], None, [8,8,8], [0,256,0,256,0,256])
        hist_tampered = cv2.calcHist([tampered], [0,1,2], None, [8,8,8], [0,256,0,256,0,256])
        
        diff = cv2.compareHist(hist_original, hist_tampered, cv2.HISTCMP_CHISQR)
        diff_map = np.ones_like(original[:,:,0]) * diff
        
        # Normalize
        diff_map = (diff_map - diff_map.min()) / (diff_map.max() - diff_map.min())
        
        return diff_map
    
    except Exception as e:
        print(f"Error in OSN detection: {str(e)}")
        return None

def detect_hifi_net(original, tampered):
    """
    HiFi-Net detection placeholder
    Replace with actual HiFi-Net implementation
    """
    try:
        # Placeholder using frequency domain analysis
        gray_original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        gray_tampered = cv2.cvtColor(tampered, cv2.COLOR_BGR2GRAY)
        
        # Fourier transform
        f_original = np.fft.fft2(gray_original)
        f_tampered = np.fft.fft2(gray_tampered)
        
        # Magnitude spectrum difference
        diff = np.abs(np.abs(f_original) - np.abs(f_tampered))
        diff = np.fft.fftshift(diff)
        
        # Log scale and normalize
        diff = np.log(diff + 1)
        diff = (diff - diff.min()) / (diff.max() - diff.min())
        
        return diff
    
    except Exception as e:
        print(f"Error in HiFi-Net detection: {str(e)}")
        return None

def detect_custom_method(original, tampered):
    """
    Your custom detection method
    Implement your actual method here
    """
    try:
        # Example: Simple difference with noise reduction
        gray_original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        gray_tampered = cv2.cvtColor(tampered, cv2.COLOR_BGR2GRAY)
        
        diff = cv2.absdiff(gray_original, gray_tampered)
        
        # Apply Gaussian blur to reduce noise
        diff = cv2.GaussianBlur(diff, (5,5), 0)
        
        # Adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            diff, 1, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2)
        
        return thresh
    
    except Exception as e:
        print(f"Error in custom method detection: {str(e)}")
        return None

# Utility function for SSIM calculation
def structural_similarity(im1, im2):
    """
    Compute SSIM map between two images
    """
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(im1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(im2, -1, window)[5:-5, 5:-5]
    
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = cv2.filter2D(im1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(im2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(im1*im2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2)) / ((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean(), ssim_map
# ... implement other detection functions similarly

def save_image(image, path):
    """
    Enhanced image saving with validation and conversion
    
    Args:
        image: numpy array of the image
        path: full output path
        
    Returns:
        str: absolute path where image was saved
    """
    try:
        # Convert to BGR if needed
        if len(image.shape) == 3 and image.shape[2] == 3:  # RGB
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        elif len(image.shape) == 2:  # Grayscale
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            
        # Ensure directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        if cv2.imwrite(path, image):
            logging.info(f"Saved image to: {os.path.abspath(path)}")
            return os.path.abspath(path)
        else:
            raise IOError(f"Failed to write image to {path}")
            
    except Exception as e:
        logging.error(f"Error saving image: {str(e)}")
        return None


def generate_comparison_collage(result_paths, OUTPUT_FOLDER):
    """Creates a visual comparison collage of all results"""
    try:
        # Only include images that exist
        valid_images = {k:v for k,v in result_paths.items() if v and os.path.exists(v)}
        
        if not valid_images:
            logging.warning("No valid images to create collage")
            return
            
        # Read all images
        images = []
        titles = []
        for name, path in valid_images.items():
            img = cv2.imread(path)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                images.append(img)
                titles.append(name.replace('_', ' ').title())
        
        # Create figure
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        if len(images) > 8:  # Handle more than 8 images
            fig, axes = plt.subplots(3, 4, figsize=(20, 15))
            
        axes = axes.flatten()
        
        for ax, img, title in zip(axes, images, titles):
            ax.imshow(img)
            ax.set_title(title)
            ax.axis('off')
            
        plt.tight_layout()
        collage_path = os.path.join(OUTPUT_FOLDER, "comparison_collage.png")
        fig.savefig(collage_path, bbox_inches='tight', dpi=300)
        plt.close()
        logging.info(f"Saved comparison collage to: {collage_path}")
        
    except Exception as e:
        logging.error(f"Error generating collage: {str(e)}")
        
def run_watermark_analysis():
    """Runs watermark comparison, AIGC editing methods, tamper localization, and copyright protection analysis."""
    try:
        logging.info("Starting watermark analysis...")

        # Watermark Comparison
        comparator = WatermarkComparison()
        logging.info(f"Fidelity Scores: {comparator.compare_fidelity()}")
        logging.info(f"Perceptual Quality Scores: {comparator.evaluate_perceptual_quality()}")

        # AIGC Editing Methods
        editor = AIGCEditingMethods()
        editor.prepare_dataset()
        logging.info(f"Editing Method Performance: {editor.evaluate_methods()}")

        # Tamper Localization
        localizer = TamperLocalization()
        logging.info(f"Tamper Localization Accuracy: {localizer.analyze_tampering()}")

        # Copyright Protection
        protector = CopyrightProtection()
        logging.info(protector.recover_copyright())

    except Exception as e:
        logging.error(f"Error in watermark analysis: {e}")



def main():
    """Main function to execute tamper detection and watermark analysis."""
    original_path = "org.jpg"   # Change to actual original image path
    tampered_path = "tamp.jpg"  # Change to actual tampered image path

    logging.info("Starting Image Processing Pipeline...")

    # Step 1: Process Image (Watermarking and Enhancement)
    process_image(original_path)
    
    result_compare_images(original_path, tampered_path)
    
    # Step 2: Compare Images (Tamper Detection)
    compare_images(original_path, tampered_path)

    # Step 3: Perform Watermark Analysis
    run_watermark_analysis()

    logging.info("Process completed successfully!")

if __name__ == "__main__":
    main()
