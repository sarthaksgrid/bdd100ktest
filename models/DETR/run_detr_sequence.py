import subprocess
import sys
import os
from pathlib import Path

def print_header(text):
    print(f"\n{'='*60}")
    print(f" {text}")
    print(f"{'='*60}")

def main():
    SCRIPT_DIR = Path(__file__).resolve().parent    # bdd100k/models/DETR
    ROOT_DIR = SCRIPT_DIR.parent.parent             # bdd100k/
    
    print_header("STANDARD DETR TRAINING SEQUENCE")
    
    try:
        epochs = input("Enter number of epochs [Default 50]: ") or "50"
        batch_size = input("Enter batch size [Recommended 16 for M4 Pro]: ") or "16"
        lr = input("Enter learning rate [Default 1e-4]: ") or "1e-4"
        
        print(f"\n[SUMMARY] Training Standard DETR with:")
        print(f" - Epochs: {epochs}")
        print(f" - Batch Size: {batch_size}")
        print(f" - Learning Rate: {lr}")

        confirm = input("\n[CRITICAL] Ready to start training? (y/n): ")
        if confirm.lower() == 'y':
            env = os.environ.copy()
            env["CUSTOM_EPOCHS"] = epochs
            env["CUSTOM_BATCH"] = batch_size
            env["CUSTOM_LR"] = lr
            
            # --- STEP 1: Execute train_detr.py (LOCALLY in models/DETR) ---
            print_header("STEP 1: TRAINING MODEL (train_detr.py)")
            train_script = SCRIPT_DIR / "train_detr.py"
            subprocess.run([sys.executable, str(train_script)], env=env, check=True)
            
            # --- STEP 2 & 3: EVALUATION (ROOT DIR) ---
            print_header("STEP 2: VALIDATION EVALUATION (evaluate.py)")
            subprocess.run([sys.executable, str(ROOT_DIR / "evaluation" / "evaluate.py")], env=env, check=True)
            
            print_header("STEP 3: UNSEEN TEST EVALUATION (test_detr.py)")
            subprocess.run([sys.executable, str(ROOT_DIR / "evaluation" / "test_detr.py")], env=env, check=True)
            
            print_header("🎉 STANDARD DETR PIPELINE COMPLETED SUCCESSFULLY! 🎉")
            
        else:
            print("[ABORT] Sequence ended by user.")

    except KeyboardInterrupt:
        print("\n[STOP] Sequence interrupted.")
    except subprocess.CalledProcessError as e:
        print(f"\n[FATAL ERROR] A script in the pipeline failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()