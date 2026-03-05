import os
import json
from pathlib import Path

# ANSI color codes for terminal output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
CYAN = '\033[96m'
RESET = '\033[0m'

def print_header(title):
    print(f"\n{CYAN}{'='*60}")
    print(f" {title}")
    print(f"{'='*60}{RESET}")

def check_file_status(file_path, purpose, required_strings=None):
    """Checks if a file exists, isn't empty, and contains specific required code."""
    path = Path(file_path)
    if not path.exists():
        return f"{RED}[MISSING]{RESET} {path.name} - {purpose}"
    
    if path.stat().st_size < 10:
        return f"{RED}[USELESS]{RESET} {path.name} is empty or too small!"

    if required_strings:
        try:
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
                missing = [s for s in required_strings if s not in content]
                if missing:
                    return f"{YELLOW}[WARNING]{RESET} {path.name} is missing crucial code: {missing}"
        except Exception as e:
            return f"{RED}[ERROR]{RESET} Could not read {path.name}: {e}"

    return f"{GREEN}[ACTIVE]{RESET} {path.name} - Doing its job perfectly."

def audit_project():
    # UPDATED: Looks up one directory level to find the project root
    root = Path(__file__).parent.parent.resolve()
    
    print_header("1. DATA DIRECTORY AUDIT")
    # Raw Data
    raw_img_dir = root / "data" / "raw" / "images"
    raw_json = root / "data" / "raw" / "samples.json"
    print(f"Raw Images : {GREEN}Found {len(list(raw_img_dir.glob('*')))} files{RESET}" if raw_img_dir.exists() else f"{RED}Missing Raw Images{RESET}")
    print(f"Raw Labels : {GREEN}Found{RESET}" if raw_json.exists() else f"{RED}Missing samples.json{RESET}")

    # Processed Data
    processed_dir = root / "data" / "processed"
    if processed_dir.exists():
        count = len(list(processed_dir.glob('*.jpg')))
        if count > 0:
            print(f"Pre-sized  : {GREEN}Found {count} 640x640 images (M4 Pro CPU optimization active){RESET}")
        else:
            print(f"Pre-sized  : {RED}Folder exists but is empty! Run parser/resize_images.py{RESET}")
    else:
        print(f"Pre-sized  : {RED}Missing processed folder! You will have severe CPU bottlenecks.{RESET}")

    # COCO Labels
    coco_json = root / "data" / "splits" / "labels.json"
    if coco_json.exists():
        try:
            with open(coco_json, 'r') as f:
                data = json.load(f)
                print(f"COCO JSON  : {GREEN}Valid format. Contains {len(data.get('images', []))} images & {len(data.get('annotations', []))} boxes.{RESET}")
        except json.JSONDecodeError:
            print(f"COCO JSON  : {RED}Corrupted! Re-run parser/bdd_to_coco.py{RESET}")
    else:
        print(f"COCO JSON  : {RED}Missing! Run parser/bdd_to_coco.py{RESET}")


    print_header("2. SCRIPT INTEGRITY & USEFULNESS")
    # Formatted as: (Relative Path, Purpose, Required Strings inside the code)
    scripts_to_check = [
        ("utils/config.py", "Central hub for variables. Prevents hardcoding errors.", None),
        ("utils/data_utils.py", "Feeds data to GPU.", ["do_resize=False", "collate_fn", "640"]),
        ("parser/bdd_to_coco.py", "Converts BDD normalized coords to 640x640 pixels.", ["640"]),
        ("parser/resize_images.py", "Shrinks 1280x720 to 640x640.", ["cv2.resize"]),
        ("main.py", "Core training loop.", ["num_workers=0", "DetrForObjectDetection"]),
        ("run_train_sequence.py", "Orchestrator to prevent running steps out of order.", ["subprocess.run"]),
        ("outputs/checkpoints/plot_loss.py", "Generates your loss curve for the report.", ["matplotlib"]),
    ]
    
    for relative_path, purpose, required in scripts_to_check:
        print(check_file_status(root / relative_path, purpose, required))


    print_header("3. TRAINING PROGRESS")
    ckpt_dir = root / "outputs" / "checkpoints"
    if ckpt_dir.exists():
        ckpts = list(ckpt_dir.glob("detr_epoch_*.pt"))
        if ckpts:
            latest = max(ckpts, key=lambda p: int(p.stem.split('_')[-1]))
            print(f"{GREEN}[PROGRESS]{RESET} Model has trained up to: {latest.name}")
        else:
            print(f"{YELLOW}[STANDBY]{RESET} No checkpoints found. Training hasn't started or saved yet.")
        
        if (ckpt_dir / "loss_curve.png").exists():
            print(f"{GREEN}[SUCCESS]{RESET} Loss curve generated.")
        else:
            print(f"{YELLOW}[PENDING]{RESET} No loss_curve.png found. Run outputs/checkpoints/plot_loss.py")
    else:
        print(f"{YELLOW}[STANDBY]{RESET} outputs/checkpoints folder not created yet.")


    print_header("4. FINAL SUBMISSION: WHAT IS LEFT?")
    
    # Check for Evaluation Script
    eval_script = root / "evaluation" / "evaluate.py"
    if not eval_script.exists():
        print(f"{RED}[MISSING]{RESET} evaluation/evaluate.py")
        print("  -> You need this to test your model on the 10% test set and calculate the final 35% mAP score.")
    else:
        print(f"{GREEN}[READY]{RESET} evaluation/evaluate.py is present.")

    # Next steps checklist
    print("\nTo-Do List:")
    print(" [ ] 1. Finish the 50 epochs of training (Wait for main.py to finish).")
    print(" [ ] 2. Run outputs/checkpoints/plot_loss.py to get the final training graph.")
    if not eval_script.exists():
        print(" [ ] 3. Ask for the `evaluation/evaluate.py` script to calculate mAP.")
    print(" [ ] 4. Run the evaluation script on your final_m4_pro_model.pt.")
    print(" [ ] 5. Write your final report citing your M4 Pro optimizations, 300-query experiment, and mAP score.\n")

if __name__ == "__main__":
    audit_project()