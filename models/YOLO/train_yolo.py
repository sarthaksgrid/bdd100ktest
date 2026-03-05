import os
import sys
import gc
import torch
from pathlib import Path
from tqdm import tqdm

# --- STEP 1: SILENCE DEFAULT LOGGER ---
os.environ["YOLO_VERBOSE"] = "False" 

from ultralytics import YOLO
from ultralytics.engine.trainer import BaseTrainer
from ultralytics.utils import tal

# --- STEP 2: THE M4 PRO "SHAPE MISMATCH" FIX ---
original_get_box_metrics = tal.TaskAlignedAssigner.get_box_metrics

def patched_get_box_metrics(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_gt):
    device = pd_scores.device
    pd_scores_cpu = pd_scores.detach().cpu()
    pd_bboxes_cpu = pd_bboxes.detach().cpu()
    gt_labels_cpu = gt_labels.detach().cpu()
    gt_bboxes_cpu = gt_bboxes.detach().cpu()
    mask_gt_cpu = mask_gt.detach().cpu()
    
    align_metric, overlaps = original_get_box_metrics(
        self, pd_scores_cpu, pd_bboxes_cpu, gt_labels_cpu, gt_bboxes_cpu, mask_gt_cpu
    )
    return align_metric.to(device), overlaps.to(device)

tal.TaskAlignedAssigner.get_box_metrics = patched_get_box_metrics

# --- Path Math ---
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent.parent
sys.path.append(str(ROOT_DIR))

YOLO_WEIGHTS_DIR = SCRIPT_DIR / "weights"
YOLO_WEIGHTS_DIR.mkdir(exist_ok=True)

pbar = None

# --- STEP 3: STABILITY HACKS ---
BaseTrainer.validate = lambda x: ({}, 0) 
BaseTrainer.final_eval = lambda x: None

# --- STEP 4: PERFORMANCE CALLBACKS ---
def on_train_batch_end(trainer):
    global pbar
    if pbar:
        try:
            loss_items = [float(x) for x in trainer.tloss]
            desc = (f"🚀 Ep:{trainer.epoch+1}/{trainer.epochs} | "
                    f"Box:{loss_items[0]:.3f} | Cls:{loss_items[1]:.3f}")
            pbar.set_description(desc)
            pbar.update(1)
        except: 
            pbar.update(1)

def on_train_epoch_start(trainer):
    global pbar
    if pbar: pbar.close()
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache() 
    pbar = tqdm(total=len(trainer.train_loader), unit="batch", leave=False, dynamic_ncols=True)

def train_yolo():
    print(f"\n{'='*60}")
    print(" 🚀 M4 PRO AUTO-RESUME & COUNTER-FIX")
    print(f"{'='*60}")

    yaml_path = ROOT_DIR / "data" / "yolo_dataset" / "bdd100k.yaml"
    runs_dir = SCRIPT_DIR / "runs"
    
    # Logic to find the run folder that actually contains weights
    all_runs = list(runs_dir.glob("bdd100k_run*"))
    valid_runs = [r for r in all_runs if (r / "weights" / "last.pt").exists()]
    valid_runs.sort(key=lambda x: x.stat().st_mtime, reverse=True)

    model = None
    resume_flag = False
    active_run_name = "bdd100k_run"
    last_epoch_completed = 0

    if valid_runs:
        last_ckpt = valid_runs[0] / "weights" / "last.pt"
        
        # FIX FOR PYTORCH 2.6+ SECURITY ERROR
        ckpt = torch.load(last_ckpt, map_location='cpu', weights_only=False)
        last_epoch_completed = ckpt.get('epoch', 0)
        
        print(f"\n✅ SUCCESS: Detected {last_epoch_completed} completed epochs in {valid_runs[0].name}")
        
        # --- THIS IS THE TERMINAL PROMPT ---
        try:
            print(f"\n{'*'*40}")
            user_input = input(f"👉 How many ADDITIONAL epochs to perform? ")
            print(f"{'*'*40}\n")
            additional_epochs = int(user_input)
        except ValueError:
            print("❌ Invalid input. Defaulting to 1 additional epoch.")
            additional_epochs = 1
            
        target_epochs = last_epoch_completed + additional_epochs
        
        # Sync metadata
        ckpt['epoch'] = last_epoch_completed 
        if 'train_args' in ckpt:
            ckpt['train_args']['epochs'] = target_epochs
        
        torch.save(ckpt, last_ckpt)
        print(f"✅ Metadata updated. Resuming from {last_epoch_completed + 1} to {target_epochs}.")

        model = YOLO(str(last_ckpt))
        resume_flag = True
        active_run_name = valid_runs[0].name
    else:
        print("[INFO] No valid history found in models/YOLO/runs/. Starting fresh.")
        model = YOLO("models/YOLO/yolov8s.pt")
        target_epochs = int(input("👉 How many epochs for this new run? ") or 1)

    # Register Callbacks
    model.add_callback("on_train_epoch_start", on_train_epoch_start)
    model.add_callback("on_train_batch_end", on_train_batch_end)

    batch_size = int(os.environ.get("CUSTOM_BATCH", 16))

    # Training Loop
    model.train(
        data=str(yaml_path),
        epochs=target_epochs, 
        batch=batch_size,
        device="mps",
        project=str(runs_dir),
        name=active_run_name,
        resume=resume_flag,
        exist_ok=True,        
        workers=0,            
        imgsz=640,
        amp=False,            
        cache=False,          
        plots=False,        
        verbose=False,
        val=False,            
        overlap_mask=False    
    )
    
    if pbar: pbar.close()

if __name__ == "__main__":
    train_yolo()