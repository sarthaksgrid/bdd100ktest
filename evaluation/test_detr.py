import sys
import os
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Subset
from transformers import DetrConfig, DetrForObjectDetection, DetrImageProcessor
from tqdm import tqdm
from torchmetrics.detection.mean_ap import MeanAveragePrecision

# --- Force Python to see the root directory ---
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

# Import your central configuration
from utils.config import (
    DEVICE, BATCH_SIZE, NUM_LABELS, NUM_QUERIES, ANN_FILE, IMG_DIR, CHECKPOINT_DIR
)
from utils.data_utils import BDD100KGPUDataset, collate_fn

def convert_boxes_for_metrics(boxes, img_size=640):
    """
    Converts DETR's normalized [cx, cy, w, h] to absolute [xmin, ymin, xmax, ymax]
    """
    if len(boxes) == 0:
        return torch.empty((0, 4), device=boxes.device)
        
    boxes = boxes * img_size
    cx, cy, w, h = boxes.unbind(1)
    xmin = cx - 0.5 * w
    ymin = cy - 0.5 * h
    xmax = cx + 0.5 * w
    ymax = cy + 0.5 * h
    
    return torch.stack((xmin, ymin, xmax, ymax), dim=1)

def main():
    print(f"\n[INIT] Starting DETR Final Test Set Evaluation on {DEVICE}")

    # 1. Setup the Evaluation Metric
    metric = MeanAveragePrecision(box_format='xyxy', iou_type='bbox')
    metric.warn_on_many_detections = False 

    # 2. Load the Data
    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", do_resize=False)
    dataset = BDD100KGPUDataset(ANN_FILE, IMG_DIR, processor, is_training=False)
    
    # --- NEW: Load Stratified Test Split ---
    split_path = Path(ANN_FILE).parent / "stratified_split_indices.json"
    if not split_path.exists():
        print(f"\n[ERROR] Split file not found at {split_path}")
        print("Run 'python data/create_stratified_split.py' first!")
        return
        
    with open(split_path, 'r') as f:
        splits = json.load(f)
        
    # Map raw COCO image IDs back to their PyTorch Dataset list index
    img_id_to_idx = {img_id: idx for idx, img_id in enumerate(dataset.coco.getImgIds())}
    
    # Extract only the Test indices (The Final Blind Exam)
    test_indices = [img_id_to_idx[i] for i in splits["test"] if i in img_id_to_idx]
    
    # Create the perfectly balanced subset
    test_ds = Subset(dataset, test_indices)
    print(f"[DATA] Isolated {len(test_ds)} perfectly stratified images for the final blind test.")
    
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, collate_fn=collate_fn, num_workers=0)

    # 3. Load the Model & Latest Checkpoint
    config = DetrConfig.from_pretrained("facebook/detr-resnet-50", num_labels=NUM_LABELS, num_queries=NUM_QUERIES)
    model = DetrForObjectDetection(config).to(DEVICE)
    
    checkpoint_files = list(CHECKPOINT_DIR.glob("detr_epoch_*.pt"))
    if not checkpoint_files:
        print("\n[ERROR] No checkpoints found! Train the model first.")
        return
        
    latest_ckpt = max(checkpoint_files, key=lambda p: int(p.stem.split('_')[-1]))
    print(f"[LOAD] Restoring weights from: {latest_ckpt.name}...")
    
    checkpoint = torch.load(latest_ckpt, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # 4. The Evaluation Loop
    pbar = tqdm(test_loader, desc="  Testing")
    
    with torch.no_grad():
        for batch in pbar:
            pixel_values = batch['pixel_values'].to(DEVICE)
            labels = [{k: v.to(DEVICE) for k, v in t.items()} for t in batch['labels']]
            
            outputs = model(pixel_values=pixel_values)
            
            # Post-processing targets 640x640 size
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
    print("\n[INFO] Calculating Final Test Metrics... (This can take a minute)")
    metrics = metric.compute()
    
    print("\n" + "="*50)
    print("           FINAL UNSEEN TEST SET SCORES")
    print("="*50)
    print(f" mAP (Overall) : {metrics['map'].item():.4f}")
    print(f" mAP @ 0.50    : {metrics['map_50'].item():.4f}  <-- THE FINAL VERDICT")
    print(f" mAP @ 0.75    : {metrics['map_75'].item():.4f}")
    print("="*50)

if __name__ == "__main__":
    main()