import matplotlib.pyplot as plt
import torch
import glob

def rebuild_training_history(checkpoint_dir):
    history = []
    files = sorted(glob.glob(str(checkpoint_dir / "detr_epoch_*.pt")), 
                   key=lambda x: int(x.split('_')[-1].split('.')[0]))
    for f in files:
        # UPDATED LINE: Added weights_only=False to allow loading numpy indices/metadata
        ckpt = torch.load(f, map_location='cpu', weights_only=False)
        if 'loss' in ckpt:
            history.append(ckpt['loss'])
    return history

def export_metric_visualization(losses, checkpoint_dir):
    if not losses: return
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(losses) + 1), losses, color='#2c3e50', marker='o', linewidth=1.5)
    plt.title("Model Convergence: Training Loss History")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(checkpoint_dir / "loss_metric_curve.png")
    plt.close()