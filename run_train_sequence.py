import subprocess
import sys
import os
from pathlib import Path

# --- STEP 1: PATH RESOLUTION ---
ROOT_DIR = Path(__file__).resolve().parent
sys.path.append(str(ROOT_DIR))

def print_header(text):
    print(f"\n{'='*60}")
    print(f" {text}")
    print(f"{'='*60}")

def ask_to_run(step_name):
    """Asks the user if they want to execute a specific step."""
    choice = input(f"\n[PROMPT] Do you want to run {step_name}? (y/n): ").strip().lower()
    return choice == 'y'

def get_input_with_default(prompt, default_val):
    """Helper to handle the 'd' for default logic."""
    val = input(f"{prompt} [Default: {default_val}, or 'd']: ").strip().lower()
    if val == 'd' or val == '':
        return default_val
    return val

def check_data_exists(folder_relative_path):
    """Checks if a data folder exists and has files."""
    path = ROOT_DIR / folder_relative_path
    if path.exists() and any(path.iterdir()):
        return True
    return False

def run_script(script_relative_path, env, cwd=None):
    """Helper to execute sub-scripts via subprocess."""
    script_path = ROOT_DIR / script_relative_path
    print(f"[EXEC] Running: {script_relative_path}...")
    subprocess.run([sys.executable, str(script_path)], env=env, cwd=cwd, check=True)

def main():
    print_header("BDD100K MASTER TRAINING PIPELINE")
    
    # --- MODEL SELECTION ---
    print("Select the architecture to train/evaluate:")
    print("  1. Standard DETR (models/DETR/)")
    print("  2. YOLOv8        (models/YOLO/)")
    
    choice = input("\nEnter choice (1-2): ").strip()
    
    configs = {
        "1": {
            "name": "DETR", 
            "dir": "models/DETR", 
            "train": "train_detr.py", 
            "eval": "evaluate_detr.py",
            "data_folder": "data/processed",
            "data_script": "parser/resize_images.py",
            "def_epochs": "50",
            "def_lr": "1e-4",
            "def_batch": "16"
        },
        "2": {
            "name": "YOLO", 
            "dir": "models/YOLO", 
            "train": "train_yolo.py", 
            "eval": "evaluate_yolo.py",
            "data_folder": "data/yolo_dataset",
            "data_script": "data/convert_to_yolo.py",
            "def_epochs": "50",
            "def_lr": "0.01",
            "def_batch": "16"
        }
    }
    
    if choice not in configs:
        print("[ERROR] Invalid choice. Exiting.")
        return

    cfg = configs[choice]
    print(f"\n[INFO] Initializing {cfg['name']} Pipeline...")

    # --- STEP 1: CONFIGURATION ---
    print_header("PIPELINE CONFIGURATION")
    epochs = get_input_with_default("Enter Epochs", cfg['def_epochs'])
    lr = get_input_with_default("Enter Learning Rate", cfg['def_lr'])
    batch = get_input_with_default("Enter Batch Size", cfg['def_batch'])

    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT_DIR)
    env["CUSTOM_EPOCHS"] = epochs
    env["CUSTOM_LR"] = lr
    env["CUSTOM_BATCH"] = batch

    # --- STEP 2: DATA PREPARATION ---
    print_header("STEP 1: DATA PREPARATION")
    
    processed_exists = check_data_exists("data/processed")
    target_exists = check_data_exists(cfg['data_folder'])

    # Logic for Data Generation
    if not target_exists:
        print(f"⚠️  MISSING: {cfg['data_folder']} not found.")
        if ask_to_run(f"Generate {cfg['name']} Data"):
            # SILENT DEPENDENCY CHECK: YOLO needs 'processed' to exist
            if cfg['name'] == "YOLO" and not processed_exists:
                print("[PIPELINE] 'data/processed' missing. Resizing raw images first...")
                run_script("parser/resize_images.py", env)
            
            run_script(cfg['data_script'], env)
    else:
        print(f"✅ Found existing data in: {cfg['data_folder']}")
        if ask_to_run(f"RE-GENERATE {cfg['data_folder']} (Overwrite?)"):
            # Even on overwrite, ensure the source exists
            if cfg['name'] == "YOLO" and not processed_exists:
                run_script("parser/resize_images.py", env)
            
            run_script(cfg['data_script'], env)

    if ask_to_run("Data Stratification (Re-generate balanced split)"):
        run_script("data/create_stratified_split.py", env)

    # --- STEP 3: TRAINING ---
    print_header("STEP 2: MODEL TRAINING")
    if ask_to_run(f"Start Training ({cfg['name']})"):
        train_path = cfg['dir'] + "/" + cfg['train']
        run_script(train_path, env, cwd=str(ROOT_DIR / cfg['dir']))

    # --- STEP 4: EVALUATION ---
    print_header("STEP 3: PERFORMANCE EVALUATION")
    if ask_to_run(f"Run Evaluation (mAP Calculation)"):
        eval_path = "evaluation/" + cfg['eval']
        run_script(eval_path, env)

    print_header("🎉 PIPELINE COMPLETE 🎉")

if __name__ == "__main__":
    main()