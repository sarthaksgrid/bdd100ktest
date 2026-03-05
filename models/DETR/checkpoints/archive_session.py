import os
import shutil
import glob
from pathlib import Path

def archive_current_run(model_name="DETR", run_number=1):
    # --- PATH CALCULATIONS ---
    # Current folder is models/checkpoints
    CURRENT_DIR = Path(__file__).resolve().parent
    # Root folder is 2 levels up (for visualization access)
    ROOT_DIR = CURRENT_DIR.parent.parent
    
    # NEW: Archive is now INSIDE models/checkpoints/
    ARCHIVE_BASE_DIR = CURRENT_DIR / "experiments"
    
    VIS_DIR = ROOT_DIR / "outputs" / "visualizations"
    GRAPH_FILE = CURRENT_DIR / "loss_metric_curve.png"

    # 1. Create the new destination folder
    run_folder_name = f"{model_name}_Run_{run_number}"
    target_dir = ARCHIVE_BASE_DIR / run_folder_name
    
    # Auto-increment run number if folder already exists
    if target_dir.exists():
        return archive_current_run(model_name, run_number + 1)

    os.makedirs(target_dir, exist_ok=True)
    print(f"[STATUS] Archiving session to: {target_dir}")

    # 2. Identify Checkpoints (.pt files)
    # Exclude the script itself and only grab .pt files
    checkpoints = glob.glob(str(CURRENT_DIR / "*.pt"))

    # 3. Execution: Move Checkpoints
    for ckpt in checkpoints:
        shutil.move(ckpt, target_dir / os.path.basename(ckpt))
    
    # 4. Execution: Move Graph
    if GRAPH_FILE.exists():
        shutil.move(str(GRAPH_FILE), target_dir / "loss_metric_curve.png")
        print("[INFO] Moved loss curve graph.")

    # 5. Execution: Move Visualizations
    if VIS_DIR.exists():
        # Check if there is anything to move
        if any(os.scandir(VIS_DIR)):
            shutil.move(str(VIS_DIR), target_dir / "visualizations")
            # Recreate for next run
            os.makedirs(VIS_DIR, exist_ok=True)
            print("[INFO] Moved visualization snapshots.")

    print(f"\n{'='*50}")
    print(f"SUCCESS: Session Archived inside Checkpoints!")
    print(f"Location: {target_dir}")
    print(f"{'='*50}\n")

if __name__ == "__main__":
    archive_current_run(model_name="DETR", run_number=1)