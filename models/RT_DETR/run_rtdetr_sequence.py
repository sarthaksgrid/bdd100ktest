import subprocess
import sys
import os
from pathlib import Path

def print_header(text):
    print(f"\n{'='*60}")
    print(f" {text}")
    print(f"{'='*60}")

def main():
    # Local paths for this specific model folder
    SCRIPT_DIR = Path(__file__).resolve().parent
    
    print_header("RT-DETR TRAINING SEQUENCE")
    
    try:
        epochs = input("Enter number of epochs [Default 50]: ") or "50"
        batch_size = input("Enter batch size [Recommended 8 for M4 Pro]: ") or "8"
        lr = input("Enter learning rate [Default 1e-4]: ") or "1e-4"
        
        print(f"\n[SUMMARY] Training RT-DETR with:")
        print(f" - Epochs: {epochs}")
        print(f" - Batch Size: {batch_size}")
        print(f" - Learning Rate: {lr}")

        if input("\n[CRITICAL] Ready to start RT-DETR training? (y/n): ").lower() == 'y':
            env = os.environ.copy()
            env["CUSTOM_EPOCHS"] = epochs
            env["CUSTOM_BATCH"] = batch_size
            env["CUSTOM_LR"] = lr
            
            # Target the train script right next to this sequence file
            train_script = SCRIPT_DIR / "train_rtdetr.py"
            subprocess.run([sys.executable, str(train_script)], env=env, check=True)
            
            print_header("🎉 RT-DETR TRAINING COMPLETED 🎉")
        else:
            print("[ABORT] Sequence ended.")

    except KeyboardInterrupt:
        print("\n[STOP] Sequence interrupted.")
    except subprocess.CalledProcessError as e:
        print(f"\n[FATAL ERROR] RT-DETR pipeline failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()