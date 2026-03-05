import os
import sys
import cv2  # OpenCV for manual resizing
from pathlib import Path
from ultralytics import YOLO

# --- Path Math ---
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent.parent
sys.path.append(str(ROOT_DIR))

INPUT_DIR = SCRIPT_DIR / "input"
OUTPUT_DIR = SCRIPT_DIR / "output"
RUNS_DIR = SCRIPT_DIR / "runs"

def get_latest_weights():
    all_runs = sorted(list(RUNS_DIR.glob("bdd100k_run*")), key=os.path.getmtime, reverse=True)
    if not all_runs: return None
    best_weights = all_runs[0] / "weights" / "best.pt"
    return str(best_weights) if best_weights.exists() else str(all_runs[0] / "weights" / "last.pt")

def run_inference():
    weights_path = get_latest_weights()
    if not weights_path:
        print("❌ Error: Weights file not found.")
        return

    image_files = list(INPUT_DIR.glob("*.[jJ][pP]*[gG]")) + list(INPUT_DIR.glob("*.webp"))
    if not image_files:
        print(f"❌ Error: No images found in {INPUT_DIR}")
        return
    
    # 1. LOAD AND MANUALLY RESIZE IMAGE
    original_img = cv2.imread(str(image_files[0]))
    # Resize to 640x640 (Width, Height)
    resized_img = cv2.resize(original_img, (640, 640), interpolation=cv2.INTER_LINEAR)
    
    # 2. LOAD MODEL
    model = YOLO(weights_path)
    
    # 3. RUN PREDICTION ON RESIZED IMAGE
    results = model.predict(
        source=resized_img, # Pass the pre-resized image array
        conf=0.45,        
        iou=0.35,         
        agnostic_nms=True, 
        augment=True,
        save=True,        
        project=str(OUTPUT_DIR),
        name="manual_resize_output",
        exist_ok=True,    
        device="mps"      
    )

    print(f"\n✅ Detection complete! Results saved in: {OUTPUT_DIR}/manual_resize_output")

if __name__ == "__main__":
    run_inference()