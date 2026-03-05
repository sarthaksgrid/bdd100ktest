import torch
import os
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["WANDB_DISABLED"] = "true"
# Connects to your organized structure
from utils.config import DEVICE, BATCH_SIZE, NUM_EPOCHS, LR, CHECKPOINT_DIR, DATA_PROCESSED, ANN_FILE
from features.feature_engineer import BDD100KDETRDataset, collate_fn, get_train_transform
from models.detr_model import get_model

def save_average_loss_plot(epoch_losses):
    """Generates and saves a graph of the Average Loss per Epoch."""
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(epoch_losses) + 1), epoch_losses, color='#e74c3c', marker='o', linewidth=2, label='Avg Epoch Loss')
    plt.title('M4 Pro: BDD100K Average Loss per Epoch', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss Value', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    
    plot_path = os.path.join(CHECKPOINT_DIR, 'average_loss_curve.png')
    plt.savefig(plot_path)
    plt.close()

def train_m4_pro():
    # --- M4 Pro Hardware Saturation ---
    device = DEVICE

    # --- Check for existing checkpoints ---
    checkpoint_files = [f for f in os.listdir(CHECKPOINT_DIR) if f.startswith("detr_epoch_") and f.endswith(".pt")]
    def extract_epoch(fname):
        try:
            return int(fname.split("_")[-1].split(".")[0])
        except Exception:
            return -1
    checkpoint_files = sorted(checkpoint_files, key=extract_epoch)
    start_epoch = 0
    epoch_avg_losses = []
    total_run_loss = 0.0
    total_batches_processed = 0
    model = get_model(num_labels=7, freeze_backbone=True).to(device)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=LR, 
        weight_decay=1e-4,
        foreach=True 
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    if checkpoint_files:
        latest_ckpt = checkpoint_files[-1]
        checkpoint_path = os.path.join(CHECKPOINT_DIR, latest_ckpt)
        print(f"Resuming from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch']
        # Optionally restore optimizer/scheduler state if saved
        # optimizer.load_state_dict(checkpoint.get('optimizer_state_dict', optimizer.state_dict()))
        # scheduler.load_state_dict(checkpoint.get('scheduler_state_dict', scheduler.state_dict()))
        # Restore loss history if available
        if 'epoch_avg_losses' in checkpoint:
            epoch_avg_losses = checkpoint['epoch_avg_losses']
        print(f"Continuing from epoch {start_epoch+1}")

    print(f"--- M4 Pro Speed Run Initialized ---")
    print(f"Device: {device} | Batch Size: {BATCH_SIZE} | Total Epochs: {NUM_EPOCHS}")

    # 1. Setup DataLoader
    dataset = BDD100KDETRDataset(DATA_PROCESSED, ANN_FILE, get_train_transform())
    loader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=os.cpu_count(), 
        persistent_workers=True,
        prefetch_factor=2
    )

    # ...existing code...

    # 5. Training Loop
    for epoch in range(start_epoch, NUM_EPOCHS):
        model.train()
        running_epoch_loss = 0.0
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
        
        optimizer.zero_grad(set_to_none=True) # Optimized memory clearing
        
        for imgs, targs in pbar:
            imgs = torch.stack(imgs).to(device)
            # MIGRATION: Convert any 'class_labels' to 'labels' in targets (for old checkpoints or data)
            for t in targs:
                if 'class_labels' in t and 'labels' not in t:
                    t['labels'] = t.pop('class_labels')
            targs = [{k: v.to(device) for k, v in t.items()} for t in targs]

            # 6. FP16 Mixed Precision for M4 GPU acceleration
            with torch.autocast(device_type="mps", dtype=torch.float16):
                outputs = model(pixel_values=imgs, labels=targs)
                loss = outputs.loss

            loss.backward()

            # Gradient clip to fix "exponential" spikes
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            batch_loss = loss.item()
            running_epoch_loss += batch_loss
            total_run_loss += batch_loss
            total_batches_processed += 1

            pbar.set_postfix(batch_loss=f"{batch_loss:.4f}")

        # --- Epoch End Metrics ---
        avg_epoch_loss = running_epoch_loss / len(loader)
        epoch_avg_losses.append(avg_epoch_loss)
        avg_run_loss = total_run_loss / total_batches_processed
        
        print(f"\n[REPORT] Epoch {epoch+1}: Avg Loss = {avg_epoch_loss:.4f} | Total Run Avg = {avg_run_loss:.4f}")

        # Save visualization and scheduler step
        save_average_loss_plot(epoch_avg_losses)
        scheduler.step()
        
        # FIXED: Saving checkpoint after EVERY epoch as requested
        checkpoint_path = os.path.join(CHECKPOINT_DIR, f"detr_epoch_{epoch+1}.pt")
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'avg_loss': avg_epoch_loss,
            'run_avg': avg_run_loss,
            'epoch_avg_losses': epoch_avg_losses
            # Optionally add optimizer/scheduler state_dicts here
        }, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")

    # Final Save
    final_path = os.path.join(CHECKPOINT_DIR, "final_m4_pro_model_50e.pt")
    torch.save(model.state_dict(), final_path)
    print(f"Training Complete. Final model at: {final_path}")

if __name__ == "__main__":
    train_m4_pro()