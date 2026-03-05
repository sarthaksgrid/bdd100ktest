import os
import sys
import torch
from pathlib import Path
from ultralytics import YOLO

# --- STEP 1: SILENCE LOGGING ---
os.environ["YOLO_VERBOSE"] = "False" 

# --- Path Math ---
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent.parent
sys.path.append(str(ROOT_DIR))

INPUT_DIR = SCRIPT_DIR / "input"
OUTPUT_DIR = SCRIPT_DIR / "output"
RUNS_DIR = SCRIPT_DIR / "runs"

def get_latest_weights():
    """Finds the 'best.pt' weights from the most recent training run."""
    all_runs = sorted(list(RUNS_DIR.glob("bdd100k_run*")), key=os.path.getmtime, reverse=True)
    if not all_runs: 
        return None
    best_weights = all_runs[0] / "weights" / "best.pt"
    return str(best_weights) if best_weights.exists() else str(all_runs[0] / "weights" / "last.pt")

def run_boosted_inference():
    print(f"\n{'='*60}\n 🚀 YOLO BOOSTED INFERENCE MODE (run_yolo2)\n{'='*60}")
    
    weights_path = get_latest_weights()
    if not weights_path:
        print("❌ Error: Weights file not found.")
        return

    # Find images in the input folder
    image_files = list(INPUT_DIR.glob("*.[jJ][pP]*[gG]")) + list(INPUT_DIR.glob("*.webp"))
    if not image_files:
        print(f"❌ Error: No images found in {INPUT_DIR}")
        return
    
    input_source = str(image_files[0])
    model = YOLO(weights_path)
    
    print(f"[LOAD] Using weights: {Path(weights_path).name}")
    print(f"[PROC] Image: {image_files[0].name}")

    # --- BOOSTED INFERENCE PARAMS ---
    # These tweaks are designed to push your mAP higher by capturing more objects
    results = model.predict(
        source=input_source,
        imgsz=640,           # Match training resolution
        conf=0.25,           # Lowered from 0.45 to find more objects (higher recall)
        iou=0.45,            # Balanced NMS to prevent excessive box deletion
        agnostic_nms=True,   # Resolve class overlaps (Car vs Truck)
        augment=True,        # Enable TTA (Test-Time Augmentation) for higher accuracy
        save=True,        
        project=str(OUTPUT_DIR),
        name="boosted_output",
        exist_ok=True,    
        device="mps",        # Accelerated on M4 Pro
        verbose=False        # Keep terminal output clean
    )

    print(f"\n✅ Done! Boosted result saved in: {OUTPUT_DIR}/models/YOLO/boosted_output")

if __name__ == "__main__":
    run_boosted_inference()