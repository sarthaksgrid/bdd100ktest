import json
import os
import cv2
from tqdm import tqdm
from pathlib import Path

def visualize_ground_truth():
    # --- PATH SETUP ---
    CURRENT_DIR = Path(__file__).resolve().parent
    JSON_FILE = CURRENT_DIR / "raw" / "samples.json"
    
    # Pointing to your ALREADY RESIZED images
    PROCESSED_IMAGES_DIR = CURRENT_DIR / "processed" 
    
    # Where to save the images with boxes drawn on them
    OUTPUT_DIR = CURRENT_DIR / "detected_samples"

    if not JSON_FILE.exists():
        print(f"[ERROR] Could not find JSON at: {JSON_FILE}")
        return
    if not PROCESSED_IMAGES_DIR.exists():
        print(f"[ERROR] Could not find processed images at: {PROCESSED_IMAGES_DIR}")
        return

    # Load annotations
    print(f"Loading annotations from {JSON_FILE}...")
    with open(JSON_FILE, 'r') as f:
        data = json.load(f)
    
    samples = data.get("samples", [])
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Drawing boxes on images from data/processed...")

    # Processing the first 100 images as a quick sanity check
    for sample in tqdm(samples[:100]): 
        file_path = sample.get("filepath", "")
        img_name = os.path.basename(file_path)
        img_full_path = PROCESSED_IMAGES_DIR / img_name
        
        if not img_full_path.exists():
            continue

        # Load the 640x640 image
        img = cv2.imread(str(img_full_path))
        if img is None: continue
        
        # Hardcoding to your processed dimension
        h, w = 640, 640 

        # Extract Detections
        detections = sample.get("detections", {}).get("detections", [])
        
        for det in detections:
            label = det.get("label", "unknown")
            bbox = det.get("bounding_box", [])
            if len(bbox) != 4: continue
            
            # Convert normalized 0.0-1.0 to pixel coordinates (using 640x640)
            x1 = int(bbox[0] * w)
            y1 = int(bbox[1] * h)
            bw = int(bbox[2] * w)
            bh = int(bbox[3] * h)
            x2, y2 = x1 + bw, y1 + bh

            # Draw the Bounding Box (Green color, thickness 2)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw the Label text
            cv2.putText(img, label, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Save the full image with the boxes drawn on it
        cv2.imwrite(str(OUTPUT_DIR / img_name), img)

    print(f"\n[SUCCESS] Visualized images saved in: {OUTPUT_DIR}")

if __name__ == "__main__":
    visualize_ground_truth()