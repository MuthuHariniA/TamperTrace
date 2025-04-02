from stegano import lsb

def embed_localization_watermark(image_path, watermark_text, output_path):
    """Embed a localization watermark (text) using LSB steganography"""
    encoded_image = lsb.hide(image_path, watermark_text)
    encoded_image.save(output_path)
    return output_path

def extract_localization_watermark(image_path):
    """Extract a localization watermark from an image"""
    hidden_message = lsb.reveal(image_path)
    return hidden_message
