import os
import re
from pathlib import Path

# --- Path Math ---
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
# Target 1: models/DETR/checkpoints/experiments/ (Nested folders)
EXP_DIR = ROOT_DIR / "models" / "DETR" / "checkpoints" / "experiments"
# Target 2: models/DETR/weights/ (Flat folder)
WEIGHTS_DIR = ROOT_DIR / "models" / "DETR" / "weights"

def extract_epoch(filename):
    """Parses 'detr_epoch_12.pt' to return integer 12."""
    match = re.search(r'detr_epoch_(\d+)\.pt', filename)
    return int(match.group(1)) if match else None

def get_files_to_prune(directory, is_nested=False):
    """Logic to determine which files to keep/delete in a given directory."""
    prune_data = {'keep': [], 'delete': [], 'folders_scanned': 0}
    
    # Handle nested experiment folders (Run_1, Run_2...)
    if is_nested:
        subdirs = sorted([d for d in directory.iterdir() if d.is_dir()])
        for folder in subdirs:
            files = list(folder.glob("detr_epoch_*.pt"))
            if not files: continue
            
            _process_file_list(files, prune_data)
            prune_data['folders_scanned'] += 1
    # Handle flat weights folder
    else:
        files = list(directory.glob("detr_epoch_*.pt"))
        if files:
            _process_file_list(files, prune_data)
            prune_data['folders_scanned'] += 1
            
    return prune_data

def _process_file_list(file_list, prune_data):
    """Internal helper to apply the 10th-epoch + last-epoch logic."""
    file_map = {extract_epoch(f.name): f for f in file_list if extract_epoch(f.name) is not None}
    sorted_epochs = sorted(file_map.keys())
    
    if not sorted_epochs: return

    last_epoch = sorted_epochs[-1]
    for ep in sorted_epochs:
        if ep % 10 == 0 or ep == last_epoch:
            prune_data['keep'].append(file_map[ep])
        else:
            prune_data['delete'].append(file_map[ep])

def run_cleanup():
    print(f"🔍 Starting Deep Cleanup...")
    
    # Gather data from both locations
    exp_results = get_files_to_prune(EXP_DIR, is_nested=True) if EXP_DIR.exists() else None
    weights_results = get_files_to_prune(WEIGHTS_DIR, is_nested=False) if WEIGHTS_DIR.exists() else None

    all_delete = (exp_results['delete'] if exp_results else []) + (weights_results['delete'] if weights_results else [])
    
    if not all_delete:
        print("✨ No redundant checkpoints found in experiments or weights folders.")
        return

    # --- Summary Report ---
    print("\n📋 SCAN SUMMARY:")
    if exp_results:
        print(f"📂 Experiments: Scanned {exp_results['folders_scanned']} run folders.")
    if weights_results:
        print(f"📂 Weights: Scanned global weights directory.")
    
    print("-" * 60)
    print(f"🗑️  TOTAL FILES TO DELETE: {len(all_delete)}")
    # DETR-L is approx 160MB, DETR-ResNet50 is approx 160MB.
    est_gb = (len(all_delete) * 160) / 1024
    print(f"📦 ESTIMATED RECLAIMABLE SPACE: ~{est_gb:.2f} GB")
    print("-" * 60)

    # List a few examples of what is being deleted for safety
    print(f"Example deletions: {[f.name for f in all_delete[:3]]} ...")

    confirm = input("\n⚠️ Proceed with mass deletion? (y/n): ").strip().lower()
    
    if confirm == 'y':
        for f_path in all_delete:
            try:
                f_path.unlink()
            except Exception as e:
                print(f"   [ERROR] Skipping {f_path.name}: {e}")
        print("\n✅ Cleanup successful. Your models are now lean.")
    else:
        print("\n❌ Action cancelled.")

if __name__ == "__main__":
    run_cleanup()