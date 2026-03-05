import os
import sys
import glob
from pathlib import Path

# --- PATH RESOLUTION ---
# This looks "up" one folder to the project root and adds it to Python's system path.
# This ensures 'from utils.config import...' works perfectly no matter where you run this script from.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from utils.config import CHECKPOINT_DIR

def delete_old_checkpoints():
    """
    Deletes all DETR epoch checkpoint files in the checkpoint directory.
    Safely keeps the final_m4_pro_model.pt and loss_curve.png if present.
    """
    print(f"Scanning for checkpoints in: {CHECKPOINT_DIR}")
    
    # We use pathlib to securely build the path
    pattern = str(CHECKPOINT_DIR / "detr_epoch_*.pt")
    files = glob.glob(pattern)
    
    if not files:
        print("\n[INFO] No old epoch checkpoints found to delete.")
        return

    print(f"Found {len(files)} epoch checkpoints. Deleting...")
    
    for f in files:
        try:
            os.remove(f)
            print(f" [DELETED] {Path(f).name}")
        except Exception as e:
            print(f" [ERROR] Failed to delete {Path(f).name}: {e}")
            
    print("\n[SUCCESS] Workspace cleared. Old checkpoints deleted.")

if __name__ == "__main__":
    delete_old_checkpoints()