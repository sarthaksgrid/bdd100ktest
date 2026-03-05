import sys
import os
from pathlib import Path
from tqdm import tqdm

# --- STEP 1: SILENCE EVERYTHING ---
# This stops the default Ultralytics logger from printing its own lines
os.environ["YOLO_VERBOSE"] = "False" 

from ultralytics import YOLO

# --- Path Math ---
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))
YOLO_YAML = ROOT_DIR / "data" / "yolo_dataset" / "bdd100k.yaml"

def main():
    print(f"\n[INIT] Starting YOLO Evaluation")

    # Locate Weights
    yolo_runs_dir = ROOT_DIR / "models" / "YOLO" / "runs"
    all_runs = sorted(list(yolo_runs_dir.glob("bdd100k_run*")), key=lambda x: x.stat().st_mtime, reverse=True)
    
    if not all_runs:
        print(f"[ERROR] No YOLO runs found.")
        return

    model_path = all_runs[0] / "weights" / "best.pt"
    if not model_path.exists():
        model_path = all_runs[0] / "weights" / "last.pt"

    print(f"[LOAD] Loading weights from: {model_path.name}")
    model = YOLO(str(model_path))

    # --- STEP 2: CUSTOM SINGLE-LINE PROGRESS BAR ---
    # We use verbose=False to kill the internal printing 
    # and use the native tqdm logic for a clean single line.
    print(f"[VAL] Calculating metrics...")
    
    results = model.val(
        data=str(YOLO_YAML),
        imgsz=640,
        batch=16,
        conf=0.001, 
        iou=0.6,
        device="mps",
        verbose=False,  # Stops the "Class Images Instances" spam
        plots=False     # Faster evaluation
    )

    # Manual extraction for the report
    map50_95 = results.box.map
    map50 = results.box.map50
    map75 = results.box.map75

    print("\n" + "="*50)
    print("               FINAL BDD100K SCORES (YOLO)")
    print("="*50)
    print(f" mAP @ 0.50-0.95 : {map50_95:.4f}")
    print(f" mAP @ 0.50     : {map50:.4f}  <-- YOUR 35% TARGET")
    print(f" mAP @ 0.75      : {map75:.4f}")
    print("="*50)

if __name__ == "__main__":
    main()