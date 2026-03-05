import sys
import os

# --- CRITICAL MPS FIX FOR RT-DETR ---
# RT-DETR uses 'grid_sampler_2d_backward', which Apple Silicon (MPS) doesn't fully support yet.
# This forces PyTorch to temporarily route that specific math operation through your M4 Pro's CPU.
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import json
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset
from torch.optim import AdamW
from transformers import (
    RTDetrConfig, 
    RTDetrForObjectDetection, 
    RTDetrImageProcessor,
    get_scheduler
)

# --- Path Math (Two levels up to reach root) ---
SCRIPT_DIR = Path(__file__).resolve().parent         # models/RT_DETR
ROOT_DIR = SCRIPT_DIR.parent.parent                  # bdd100k-object-detection
sys.path.append(str(ROOT_DIR))

# Import your existing config and dataset logic
from utils.config import DEVICE, ANN_FILE, IMG_DIR, NUM_LABELS
from utils.data_utils import BDD100KGPUDataset, collate_fn

# RT-DETR Specific Settings
RT_DETR_MODEL = "PekingU/rtdetr_r50vd"
BATCH_SIZE = int(os.environ.get("CUSTOM_BATCH", 8)) 
EPOCHS = int(os.environ.get("CUSTOM_EPOCHS", 50))
LR = float(os.environ.get("CUSTOM_LR", 1e-4))
WEIGHTS_DIR = SCRIPT_DIR / "weights"

def train_rtdetr():
    print(f"\n{'='*50}")
    print(f" INITIALIZING RT-DETR TRAINING ON {DEVICE}")
    print(f"{'='*50}")

    # Ensure weights directory exists
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Setup Processor and Data
    processor = RTDetrImageProcessor.from_pretrained(RT_DETR_MODEL, do_resize=False)
    full_dataset = BDD100KGPUDataset(ANN_FILE, IMG_DIR, processor, is_training=True)
    
    # 2. Load Stratified Split
    split_path = ROOT_DIR / "data" / "splits" / "stratified_split_indices.json"
    if not split_path.exists():
        print(f"\n[ERROR] Stratified split not found at {split_path}")
        return

    with open(split_path, 'r') as f:
        splits = json.load(f)

    img_id_to_idx = {img_id: idx for idx, img_id in enumerate(full_dataset.coco.getImgIds())}
    train_indices = [img_id_to_idx[i] for i in splits["train"] if i in img_id_to_idx]
    val_indices = [img_id_to_idx[i] for i in splits["val"] if i in img_id_to_idx]

    train_ds = Subset(full_dataset, train_indices)
    val_ds = Subset(full_dataset, val_indices)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, collate_fn=collate_fn, num_workers=0)

    print(f"[DATA] Train: {len(train_ds)} | Val: {len(val_ds)} | Batch: {BATCH_SIZE}")

    # 3. Load Model
    config = RTDetrConfig.from_pretrained(RT_DETR_MODEL, num_labels=NUM_LABELS)
    model = RTDetrForObjectDetection.from_pretrained(
        RT_DETR_MODEL, 
        config=config, 
        ignore_mismatched_sizes=True
    ).to(DEVICE)

    # 4. Optimizer and Scheduler
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_scheduler("cosine", optimizer=optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    train_losses = []
    val_losses = []

    # 5. Training Loop
    print("\n[TRAIN] Beginning Training Loop...")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_train_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} [Train]")

        for batch in pbar:
            pixel_values = batch['pixel_values'].to(DEVICE)
            labels = [{k: v.to(DEVICE) for k, v in t.items()} for t in batch['labels']]

            optimizer.zero_grad()
            outputs = model(pixel_values=pixel_values, labels=labels)
            
            loss = outputs.loss
            loss.backward()
            
            # Gradient clipping is highly recommended for RT-DETR
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
            
            optimizer.step()
            scheduler.step()

            total_train_loss += loss.item()
            pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # 6. Validation Loop
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                pixel_values = batch['pixel_values'].to(DEVICE)
                labels = [{k: v.to(DEVICE) for k, v in t.items()} for t in batch['labels']]
                outputs = model(pixel_values=pixel_values, labels=labels)
                total_val_loss += outputs.loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        print(f"[EPOCH {epoch}] Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # Save Checkpoint
        ckpt_path = WEIGHTS_DIR / f"rt_detr_epoch_{epoch}.pt"
        torch.save({'model_state_dict': model.state_dict()}, ckpt_path)

    # 7. Plot Loss Curve
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, EPOCHS + 1), train_losses, label="Train Loss", marker='o', color='blue')
    plt.plot(range(1, EPOCHS + 1), val_losses, label="Val Loss", marker='s', color='red', linestyle='--')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("RT-DETR Training & Validation Loss")
    plt.legend()
    plt.grid()
    plt.savefig(SCRIPT_DIR / "rt_detr_loss_curve.png")
    
    print(f"\n[SUCCESS] RT-DETR Training Complete! Model saved to {WEIGHTS_DIR.name}/")

if __name__ == "__main__":
    train_rtdetr()