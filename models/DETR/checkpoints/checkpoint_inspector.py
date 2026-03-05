import torch
import glob
import os
import sys
from pathlib import Path

# --- BOOTSTRAP PATHING ---
# Since this script lives in models/checkpoints, we add the root 
# directory to sys.path so we can still import utils.config
CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent.parent
sys.path.append(str(ROOT_DIR))

try:
    from utils.config import CHECKPOINT_DIR
except ImportError:
    # Fallback if config is moved: use the current folder as checkpoint dir
    CHECKPOINT_DIR = CURRENT_DIR

def inspect_session_metadata():
    """
    Analyzes serialized model states within the current directory 
    to provide a professional summary of training progress.
    """
    print(f"\n{'='*70}")
    print(f"{'PROJECT CHECKPOINT ANALYTICS':^70}")
    print(f"{'='*70}\n")

    # Locate all .pt files in the same folder as this script
    checkpoint_pattern = os.path.join(CURRENT_DIR, "detr_epoch_*.pt")
    checkpoint_files = glob.glob(checkpoint_pattern)

    if not checkpoint_files:
        print(f"[ERROR] No serialized states found in: {CURRENT_DIR}")
        return

    # Numerical sort for proper epoch ordering
    checkpoint_files.sort(key=lambda x: int(os.path.basename(x).split('_')[-1].split('.')[0]))

    print(f"{'Epoch':<10} | {'Mean Loss':<15} | {'Delta':<12} | {'Filename':<20}")
    print("-" * 70)

    historical_losses = []

    for i, file_path in enumerate(checkpoint_files):
        try:
            # Map to CPU to ensure no interference with active GPU training sessions
            checkpoint = torch.load(file_path, map_location='cpu', weights_only=False)
            
            epoch = checkpoint.get('epoch', 'N/A')
            loss = checkpoint.get('loss', 'N/A')
            fname = os.path.basename(file_path)

            if isinstance(loss, (float, int)):
                historical_losses.append(loss)
                display_loss = f"{loss:.6f}"
                # Calculate improvement from previous epoch
                delta = f"{(historical_losses[i-1] - loss):+.4f}" if i > 0 else "---"
            else:
                display_loss = "Unknown"
                delta = "---"

            print(f"{epoch:<10} | {display_loss:<15} | {delta:<12} | {fname:<20}")

        except Exception as e:
            print(f"[WARN] Integrity check failed for {os.path.basename(file_path)}: {e}")

    # Summary Statistics
    if historical_losses:
        print(f"\n{'='*70}")
        print(f"Session Summary:")
        print(f" - Completed Training Cycles: {len(historical_losses)}")
        print(f" - Peak Optimization (Min Loss): {min(historical_losses):.6f}")
        print(f" - Final Session Convergence:    {historical_losses[-1]:.6f}")
        print(f"{'='*70}\n")

if __name__ == "__main__":
    inspect_session_metadata()