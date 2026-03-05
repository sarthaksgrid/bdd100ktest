import torch
import os
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from transformers import DetrConfig, DetrForObjectDetection, DetrImageProcessor
from tqdm import tqdm
from pathlib import Path
from torch.optim.lr_scheduler import StepLR
import json
from torch.utils.data import Subset

# Import central configuration (Our Source of Truth)
from utils.config import (
    DEVICE, BATCH_SIZE, NUM_EPOCHS, LR, WEIGHT_DECAY, 
    GRAD_CLIP, NUM_LABELS, NUM_QUERIES, ANN_FILE, IMG_DIR, 
    CHECKPOINT_DIR
)
from utils.data_utils import BDD100KGPUDataset, collate_fn

def train_one_epoch(model, loader, optimizer, device, accumulation_steps=1):
    """
    Standard Training Loop.
    accumulation_steps is set to 1 because M4 Pro natively handles Batch Size 16.
    """
    model.train()
    total_loss = 0
    optimizer.zero_grad() 
    
    pbar = tqdm(loader, desc="  Training (AdamW)")
    
    for i, batch in enumerate(pbar):
        pixel_values = batch['pixel_values'].to(device)
        labels = [{k: v.to(device) for k, v in t.items()} for t in batch['labels']]
        
        # Forward pass (FP32 precision required for DETR on MPS)
        outputs = model(pixel_values=pixel_values, labels=labels)
        
        # Scale loss (no-op when accumulation_steps=1, but keeps math safe)
        loss = outputs.loss / accumulation_steps
        loss.backward()
        
        if (i + 1) % accumulation_steps == 0 or (i + 1) == len(loader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            optimizer.zero_grad()
        
        actual_loss = loss.item() * accumulation_steps
        total_loss += actual_loss
        pbar.set_postfix(loss=f"{actual_loss:.4f}")
        
    return total_loss / len(loader)

def validate_one_epoch(model, loader, device):
    """Calculates true generalization loss without updating weights."""
    model.eval()
    total_loss = 0
    pbar = tqdm(loader, desc="  Validating")
    
    with torch.no_grad():
        for batch in pbar:
            pixel_values = batch['pixel_values'].to(device)
            labels = [{k: v.to(device) for k, v in t.items()} for t in batch['labels']]
            
            outputs = model(pixel_values=pixel_values, labels=labels)
            loss = outputs.loss
            
            total_loss += loss.item()
            pbar.set_postfix(val_loss=f"{loss.item():.4f}")
            
    return total_loss / len(loader)

def main():
    target_epochs = int(os.getenv("CUSTOM_EPOCHS", NUM_EPOCHS))
    batch_size = int(os.getenv("CUSTOM_BATCH", BATCH_SIZE))
    lr = float(os.getenv("CUSTOM_LR", LR))

    print(f"\n[INIT] Starting Optimized DETR Training on {DEVICE}")
    print(f"[CONFIG] Target Epochs: {target_epochs} | Batch Size: {batch_size} | LR: {lr}")

    # 1. Initialize Processor & Data
    # CRITICAL: do_resize=False because the images and the JSON are natively 640x640!
    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", do_resize=False)
    
    # We pass is_training=True to enable the ColorJitter data augmentation 
    dataset = BDD100KGPUDataset(ANN_FILE, IMG_DIR, processor, is_training=True)
    
    split_path = Path(ANN_FILE).parent / "stratified_split_indices.json"
    with open(split_path, 'r') as f:
        splits = json.load(f)
    
    # We need to map the raw COCO image IDs back to their index position in the PyTorch Dataset list
    img_id_to_idx = {img_id: idx for idx, img_id in enumerate(dataset.coco.getImgIds())}
    
    train_indices = [img_id_to_idx[i] for i in splits["train"] if i in img_id_to_idx]
    val_indices = [img_id_to_idx[i] for i in splits["val"] if i in img_id_to_idx]
    
    # Create PyTorch Subsets using our perfect indices
    train_ds = Subset(dataset, train_indices)
    val_ds = Subset(dataset, val_indices)
    # num_workers=0 to prevent Apple Silicon memory bottlenecks
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, collate_fn=collate_fn, num_workers=0)

    # 2. Build Model & Optimizer
    config = DetrConfig.from_pretrained("facebook/detr-resnet-50", num_labels=NUM_LABELS, num_queries=NUM_QUERIES)
    model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", config=config, ignore_mismatched_sizes=True).to(DEVICE)
    
    # The crucial AdamW optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    
    # The Learning Rate Scheduler (Drops LR by 90% at Epoch 30)
    lr_scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

    # 3. Checkpoint Resumption Logic
    start_epoch = 0
    epoch_train_losses = []
    epoch_val_losses = []
    recorded_epochs = []
    
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_files = list(CHECKPOINT_DIR.glob("detr_epoch_*.pt"))
    
    if checkpoint_files:
        latest_ckpt = max(checkpoint_files, key=lambda p: int(p.stem.split('_')[-1]))
        print(f"[RESUME] Found checkpoint: {latest_ckpt.name}. Loading...")
        
        checkpoint = torch.load(latest_ckpt, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        
        epoch_train_losses = checkpoint.get('epoch_train_losses', [])
        epoch_val_losses = checkpoint.get('epoch_val_losses', [])
        recorded_epochs = checkpoint.get('recorded_epochs', [])
        
        if 'scheduler_state_dict' in checkpoint:
            lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
        print(f"[RESUME] Successfully restored state. Resuming from Epoch {start_epoch + 1}.")

    if start_epoch >= target_epochs:
        print(f"\n[INFO] Model has already been trained for {start_epoch} epochs.")
        return

    # 4. The Training & Validation Loop
    for epoch in range(start_epoch, target_epochs):
        current_epoch = epoch + 1
        current_lr = lr_scheduler.get_last_lr()[0]
        
        print(f"\n{'='*40}\n Epoch {current_epoch} / {target_epochs} | LR: {current_lr:.2e}\n{'='*40}")
        
        # Train & Validate (accumulation_steps=1 since native batch is 16)
        train_loss = train_one_epoch(model, train_loader, optimizer, DEVICE, accumulation_steps=1)
        val_loss = validate_one_epoch(model, val_loader, DEVICE)
        
        lr_scheduler.step()
        
        print(f" -> Result: Train Loss = {train_loss:.4f} | Val Loss = {val_loss:.4f}")
        
        # Track metrics for the graph
        epoch_train_losses.append(train_loss)
        epoch_val_losses.append(val_loss)
        recorded_epochs.append(current_epoch)
        
        # Generate the Dual-Line Loss Curve
        plt.figure(figsize=(10, 5))
        plt.plot(recorded_epochs, epoch_train_losses, marker='o', linestyle='-', color='b', label='Train Loss')
        plt.plot(recorded_epochs, epoch_val_losses, marker='s', linestyle='--', color='r', label='Val Loss')
        plt.title("DETR Training & Validation Loss Curve (AdamW)")
        plt.xlabel("Epoch")
        plt.ylabel("Average Loss")
        
        if recorded_epochs:
            plt.xticks(range(min(recorded_epochs), max(recorded_epochs) + 1))
            
        plt.legend()
        plt.grid(True)
        plt.savefig(CHECKPOINT_DIR.parent / "loss_curve.png") 
        plt.close() 

        # Save Checkpoint
        ckpt_path = CHECKPOINT_DIR / f"detr_epoch_{current_epoch}.pt"
        torch.save({
            'epoch': current_epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': lr_scheduler.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'epoch_train_losses': epoch_train_losses,
            'epoch_val_losses': epoch_val_losses,
            'recorded_epochs': recorded_epochs 
        }, ckpt_path)
        
        if current_epoch == target_epochs:
            torch.save(model.state_dict(), CHECKPOINT_DIR / "final_m4_pro_model.pt")
            print(f"\n[SUCCESS] Final model saved to {CHECKPOINT_DIR / 'final_m4_pro_model.pt'}")

if __name__ == "__main__":
    main()