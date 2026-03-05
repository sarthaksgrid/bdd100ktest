import os
import sys
import torch
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from transformers import DetrImageProcessor, DetrForObjectDetection

# --- PATH MATH (FIXES FILENOTFOUND) ---
SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_DIR = SCRIPT_DIR / "inputs"
OUTPUT_DIR = SCRIPT_DIR / "outputs"

# Ensure directories exist
INPUT_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

def fix_detr_inference():
    print(f"\n{'='*60}\n 🚀 DETR MANUAL OPTIMIZATION & BOX FIX\n{'='*60}")

    # 1. Load Model and Processor
    # Using local weights if available, else standard pretrained
    model_name = "facebook/detr-resnet-50"
    processor = DetrImageProcessor.from_pretrained(model_name)
    model = DetrForObjectDetection.from_pretrained(model_name)
    
    # Stability: Force CPU for manual post-processing to avoid M4 shape mismatches
    device = torch.device("cpu") 
    model.to(device)
    model.eval()

    # 2. Process Images
    image_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.jpg', '.png', '.jpeg', '.webp'))]
    
    if not image_files:
        print(f"⚠️  No images found in: {INPUT_DIR}")
        return

    for img_name in image_files:
        print(f"🔍 Fixing: {img_name}")
        img_path = INPUT_DIR / img_name
        image = Image.open(img_path).convert("RGB")
        
        # Keep original CV2 image for drawing
        img_cv = cv2.imread(str(img_path))
        h_orig, w_orig = img_cv.shape[:2]

        # 3. Inference
        inputs = processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)

        # 4. MANUAL COORDINATE SCALING & GHOST BOX FILTERING
        results = processor.post_process_object_detection(outputs, target_sizes=[(h_orig, w_orig)], threshold=0.5)[0]

        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            x1, y1, x2, y2 = map(int, box.tolist())
            
            # --- THE MANUAL FIXES ---
            
            # A. Vertical Ghost Filter: Ignore boxes taller than 70% of image (rain effects)
            if (y2 - y1) > (h_orig * 0.7):
                continue
                
            # B. Area Filter: Ignore boxes spanning more than 80% width
            if (x2 - x1) > (w_orig * 0.8):
                continue

            label_name = model.config.id2label[label.item()]
            conf_val = round(score.item(), 2)

            # Draw high-confidence detections
            color = (0, 255, 0) # Green for stabilized boxes
            cv2.rectangle(img_cv, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img_cv, f"{label_name} {conf_val}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # 5. Save standardized 640x640 output
        final_output = cv2.resize(img_cv, (640, 640))
        save_path = OUTPUT_DIR / f"fixed_{img_name}"
        cv2.imwrite(str(save_path), final_output)
        print(f"✅ Saved to: {save_path}")

if __name__ == "__main__":
    fix_detr_inference()