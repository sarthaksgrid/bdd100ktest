import sys
import os
import json
import re
from pathlib import Path

# --- STEP 1: DYNAMIC PATH RESOLUTION ---
# SCRIPT_DIR is bdd100k-object-detection/evaluation/
SCRIPT_DIR = Path(__file__).resolve().parent
# ROOT_DIR is the project root (where 'utils' folder lives)
ROOT_DIR = SCRIPT_DIR.parent

# Force Python to see the root directory BEFORE we import utils
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# --- STEP 2: IMPORTS ---
import torch
from torch.utils.data import DataLoader, Subset
from transformers import DetrConfig, DetrForObjectDetection, DetrImageProcessor
from tqdm import tqdm
from torchmetrics.detection.mean_ap import MeanAveragePrecision

# Import central configuration from ROOT/utils/
from utils.config import (
    DEVICE, BATCH_SIZE, NUM_LABELS, NUM_QUERIES, ANN_FILE, IMG_DIR
)
from utils.data_utils import BDD100KGPUDataset, collate_fn

# Define path to the DETR subfolder for checkpoints
DETR_BASE_DIR = ROOT_DIR / "models" / "DETR"
CHECKPOINTS_BASE_DIR = DETR_BASE_DIR / "checkpoints"

def convert_boxes_for_metrics(boxes, img_size=640):
    """
    DETR targets are saved as normalized [center_x, center_y, width, height].
    TorchMetrics expects absolute pixel coordinates [xmin, ymin, xmax, ymax].
    """
    if len(boxes) == 0:
        return torch.empty((0, 4), device=boxes.device)
        
    # 1. Un-normalize by multiplying by image size (640)
    boxes = boxes * img_size
    
    # 2. Convert cxcywh to xyxy
    cx, cy, w, h = boxes.unbind(1)
    xmin = cx - 0.5 * w
    ymin = cy - 0.5 * h
    xmax = cx + 0.5 * w
    ymax = cy + 0.5 * h
    
    return torch.stack((xmin, ymin, xmax, ymax), dim=1)

def main():
    print(f"\n[INIT] Starting DETR Evaluation on {DEVICE}")

    # 1. Setup the Evaluation Metric
    metric = MeanAveragePrecision(box_format='xyxy', iou_type='bbox')
    metric.warn_on_many_detections = False 

    # 2. Load the exact Validation Data
    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", do_resize=False)
    dataset = BDD100KGPUDataset(ANN_FILE, IMG_DIR, processor, is_training=False)
    
    split_path = Path(ANN_FILE).parent / "stratified_split_indices.json"
    if not split_path.exists():
        print(f"\n[ERROR] Split file not found at {split_path}")
        return
        
    with open(split_path, 'r') as f:
        splits = json.load(f)
        
    img_id_to_idx = {img_id: idx for idx, img_id in enumerate(dataset.coco.getImgIds())}
    val_indices = [img_id_to_idx[i] for i in splits["val"] if i in img_id_to_idx]
    
    val_ds = Subset(dataset, val_indices)
    print(f"[DATA] Loaded {len(val_ds)} perfectly stratified validation images.")

    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, collate_fn=collate_fn, num_workers=0)

    # 3. ADVANCED CHECKPOINT DISCOVERY (Sort by Run Number then Epoch)
    CHECKPOINTS_BASE_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_files = list(CHECKPOINTS_BASE_DIR.rglob("detr_epoch_*.pt"))
    
    if not checkpoint_files:
        print(f"\n[ERROR] No checkpoints found in {CHECKPOINTS_BASE_DIR}!")
        return
        
    def get_run_and_epoch(p):
        """
        Extracts Run ID and Epoch ID for hierarchical sorting.
        e.g., .../DETR_Run_7/detr_epoch_50.pt -> (7, 50)
        """
        # Search path string for 'DETR_Run_X'
        run_match = re.search(r'DETR_Run_(\d+)', str(p))
        run_num = int(run_match.group(1)) if run_match else -1
        
        # Extract Epoch Number from filename 'detr_epoch_Y.pt'
        try:
            epoch_num = int(p.stem.split('_')[-1])
        except (ValueError, IndexError):
            epoch_num = -1
            
        return (run_num, epoch_num)

    # Selection logic: Primarily the latest Run, secondarily the latest Epoch
    latest_ckpt = max(checkpoint_files, key=get_run_and_epoch)
    
    run_id, epoch_id = get_run_and_epoch(latest_ckpt)
    print(f"[LOAD] Targeted Experiment: DETR_Run_{run_id} | Epoch: {epoch_id}")
    print(f"[PATH] {latest_ckpt.relative_to(ROOT_DIR)}")
    
    # --- DYNAMIC CONFIGURATION SYNC ---
    checkpoint = torch.load(latest_ckpt, map_location=DEVICE)
    state_dict = checkpoint['model_state_dict']

    # Determine dimensions from weights to avoid size mismatches
    ckpt_num_labels = state_dict['class_labels_classifier.bias'].shape[0] - 1
    ckpt_num_queries = state_dict['model.query_position_embeddings.weight'].shape[0]

    print(f"[SYNC] Detected Checkpoint Specs: {ckpt_num_labels} Classes | {ckpt_num_queries} Queries")

    config = DetrConfig.from_pretrained(
        "facebook/detr-resnet-50", 
        num_labels=ckpt_num_labels, 
        num_queries=ckpt_num_queries
    )
    model = DetrForObjectDetection(config).to(DEVICE)
    model.load_state_dict(state_dict)
    model.eval()

    # 4. The Evaluation Loop
    pbar = tqdm(val_loader, desc="  Evaluating")
    
    with torch.no_grad():
        for batch in pbar:
            pixel_values = batch['pixel_values'].to(DEVICE)
            labels = [{k: v.to(DEVICE) for k, v in t.items()} for t in batch['labels']]
            
            outputs = model(pixel_values=pixel_values)
            
            target_sizes = torch.tensor([[640, 640]] * pixel_values.shape[0]).to(DEVICE)
            results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.0)
            
            preds = []
            for res in results:
                preds.append({
                    "boxes": res["boxes"], 
                    "scores": res["scores"], 
                    "labels": res["labels"]
                })
            
            targets = []
            for t in labels:
                targets.append({
                    "boxes": convert_boxes_for_metrics(t["boxes"], 640),
                    "labels": t["class_labels"]
                })
            
            metric.update(preds, targets)

    # 5. Compute Final Score
    print("\n[INFO] Calculating COCO Metrics...")
    metrics = metric.compute()
    
    print("\n" + "="*50)
    print(f" FINAL SCORES (RUN {run_id} - EPOCH {epoch_id})")
    print("="*50)
    print(f" mAP (Overall) : {metrics['map'].item():.4f}")
    print(f" mAP @ 0.50    : {metrics['map_50'].item():.4f}")
    print(f" mAP @ 0.75    : {metrics['map_75'].item():.4f}")
    print("="*50)

if __name__ == "__main__":
    main()