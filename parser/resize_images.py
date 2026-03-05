import cv2
import os
from tqdm import tqdm
from pathlib import Path

# Importing paths from your new central config
from utils.config import DATA_RAW, DATA_PROCESSED

def resize_dataset():
    """
    Resizes raw 1280x720 images to 640x640 for high-throughput training.
    Uses INTER_AREA interpolation for high-quality downscaling.
    """
    # Source: bdd100k-object-detection/data/raw/images (adjust if necessary)
    source_img_dir = DATA_RAW / "images" 
    
    # Ensure processed directory exists
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    
    if not source_img_dir.exists():
        print(f"[ERROR] Source folder not found at: {source_img_dir}")
        return

    image_files = [f for f in os.listdir(source_img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    print(f"Found {len(image_files)} images. Resizing to 640x640...")
    
    for img_name in tqdm(image_files):
        target_path = DATA_PROCESSED / img_name
        
        # Skip if already exists to allow for process resume
        if target_path.exists():
            continue
            
        img = cv2.imread(str(source_img_dir / img_name))
        if img is not None:
            # INTER_AREA is preferred for shrinking images
            resized_img = cv2.resize(img, (640, 640), interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(target_path), resized_img)

    print(f"\nResize complete. Images saved to: {DATA_PROCESSED}")

if __name__ == "__main__":
    resize_dataset()