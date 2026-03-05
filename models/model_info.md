# 🧠 Model Zoo: BDD100K Detection Framework

This directory contains the modular laboratory for benchmarking object detection architectures. It is primarily focused on the comparison between **Standard DETR** and **YOLOv8**, with a legacy section for **RT-DETR**.

---

## 🏗️ 1. DETR (DEtection TRansformer)
**Logic:** DETR treats object detection as a direct set prediction problem. It utilizes a Convolutional backbone (ResNet-50) followed by a Transformer Encoder-Decoder. 
* **Key Innovation:** It removes the need for hand-crafted components like anchors and Non-Maximum Suppression (NMS) by using **Bipartite Matching Loss** to uniquely assign predictions.

### 📂 Key Files & Implementation:
* **`train_detr.py`**: The primary training script using HuggingFace `transformers`.
    * *MPS Optimization:* Specifically tuned to handle Transformer attention blocks on the M4 Pro GPU.
* **`run_detr_sequence.py`**: A centralized orchestrator managing the training lifecycle.
* **`checkpoints/experiments/`**: Stores periodic weights and convergence logs (Loss Curves).

---

## 🚀 2. YOLO (You Only Look Once)
**Logic:** YOLOv8 is an anchor-free, convolutional-first architecture. It is the industry benchmark for real-time speed, relying on a highly optimized C2f (Cross Stage Partial Bottleneck with two convolutions) backbone.

### 📂 Key Files & Implementation:
* **`train_yolo.py`**: Fine-tunes `yolov8s.pt` on BDD100K data.
* **`runs/`**: Automated directory created by the Ultralytics engine containing `best.pt` and detailed validation metrics (Confusion Matrices, P-R Curves).

---

## ⚠️ 3. RT-DETR (Archived / Not Recommended for MPS)
**Logic:** RT-DETR was designed to bridge the gap between Transformer accuracy and YOLO speed. 

### 🛑 Why RT-DETR is NOT used in this Framework:
While the code remains in `models/RT_DETR/`, it is currently **disabled** for active research on Apple Silicon due to the following technical blockers:

1.  **Unsupported Operators (CPU Backtrack):** Many core RT-DETR operators (e.g., `aten::_upsample_bicubic2d_aa.out`) are currently not implemented in the PyTorch MPS backend.
2.  **Performance Degradation:** When these operators are encountered, the system triggers a **CPU Fallback**. This forces the M4 Pro to move massive tensors from the GPU back to the CPU for a single operation and then back to the GPU.
3.  **The Bottleneck:** This "backtracking" creates a massive I/O bottleneck, making training significantly slower than even basic CPU-only training. 



---

## 🛠️ Root Model Utilities

* **`detr_model.py`**: A modular script to initialize the DETR architecture with custom label counts.
* **`metrics.py`**: Standardizes mAP calculation across all architectures for fair benchmarking.
* **`model_info.md`**: Technical specifications and parameter counts for each model variant.

---

## 📈 Summary of Deployment Weights
| Model | Recommended Path | Status |
| :--- | :--- | :--- |
| **YOLOv8s** | `YOLO/runs/bdd100k_run6/weights/best.pt` | **Stable** |
| **Standard DETR**| `DETR/checkpoints/experiments/DETR_Run_7/final_model.pt` | **Stable** |
| **RT-DETR** | `RT_DETR/rtdetr-l.pt` | ⚠️ **Disabled (MPS Issues)** |

---

### **Technical Setup for M4 Pro Acceleration**
To avoid the backtracking issues experienced with RT-DETR, the stable models utilize this safe-check logic:

```python
import torch
# Check for hardware acceleration availability
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

# Manual fallback suppression to prevent silent performance drops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"