import torch
import matplotlib.pyplot as plt
from pathlib import Path

def generate_loss_curve():
    current_dir = Path(__file__).resolve().parent.parent / "outputs" / "checkpoints"
    checkpoint_files = list(current_dir.glob("detr_epoch_*.pt"))

    if not checkpoint_files:
        print(f"[ERROR] No checkpoint files found in {current_dir}")
        return

    print(f"Found {len(checkpoint_files)} checkpoints. Extracting dual loss data...")

    data = []
    for ckpt_path in checkpoint_files:
        try:
            checkpoint = torch.load(ckpt_path, map_location='cpu')
            if 'epoch' in checkpoint and 'train_loss' in checkpoint and 'val_loss' in checkpoint:
                data.append((checkpoint['epoch'], checkpoint['train_loss'], checkpoint['val_loss']))
        except Exception as e:
            pass

    if not data:
        print("[ERROR] No valid train/val data found in the checkpoints.")
        return

    data.sort(key=lambda x: x[0])
    epochs = [x[0] for x in data]
    train_losses = [x[1] for x in data]
    val_losses = [x[2] for x in data]

    
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_losses, marker='o', linestyle='-', color='b', linewidth=2, label='Train Loss')
    plt.plot(epochs, val_losses, marker='s', linestyle='--', color='r', linewidth=2, label='Val Loss')
    plt.title("DETR Training & Validation Loss Curve", fontsize=14)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Average Loss", fontsize=12)
    plt.xticks(range(min(epochs), max(epochs) + 1))
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    output_path = current_dir.parent / "loss_curve.png"
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

    print(f"[SUCCESS] Dual loss curve saved to: {output_path}")

if __name__ == "__main__":
    generate_loss_curve()