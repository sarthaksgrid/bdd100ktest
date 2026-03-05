import sys
import shutil
from pathlib import Path

def main():
    print(f"\n{'='*50}")
    print(" DETR BULK MODEL ARCHIVER")
    print(f"{'='*50}")

    # --- Path Math Updated for models/DETR ---
    script_dir = Path(__file__).resolve().parent    # bdd100k/models/DETR
    root_dir = script_dir.parent.parent             # bdd100k/
    
    checkpoints_dir = root_dir / "models" / "checkpoints"
    weights_dir = script_dir / "weights"
    
    weights_dir.mkdir(parents=True, exist_ok=True)

    pt_files = list(checkpoints_dir.glob("*.pt"))
    if not pt_files:
        print(f"\n[ERROR] No checkpoints found in {checkpoints_dir}")
        print("-> Please train the model before archiving!")
        sys.exit(1)

    print(f"[INFO] Found {len(pt_files)} checkpoints. Archiving to: {weights_dir.name}/\n")
    
    archived_count = 0
    for pt_file in sorted(pt_files):
        dest_path = weights_dir / pt_file.name
        shutil.copy2(pt_file, dest_path)
        print(f" -> Archived: {pt_file.name}")
        archived_count += 1
    
    print(f"\n[SUCCESS] {archived_count} models safely archived!")
    print(f"-> You can now safely delete the original checkpoints in {checkpoints_dir.name}/ to start a new training run.")

if __name__ == "__main__":
    main()