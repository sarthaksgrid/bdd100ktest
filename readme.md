

# 🚦 BDD100K Multi-Model Detection Framework

[![Dataset](https://img.shields.io/badge/Dataset-BDD100K-green)](https://www.bdd100k.com/)
[![Deployed Model](https://img.shields.io/badge/Model-Deployed%20on%20Gradio-orange)](./app.py) 


> **Project Overview:** A modular computer vision laboratory benchmarking **Standard DETR** against **YOLOv8** architectures. Specifically optimized for **Apple Silicon M4 Pro** using Metal Performance Shaders (MPS) to accelerate both Transformer attention blocks and convolutional layers.

---

## 🔗 Project Resources

* **Official Dataset:** [BDD100K Website](https://huggingface.co/datasets/dgural/bdd100k) — *Access the full 10k images .*
* **Live Demo:** [Gradio Web Interface](https://huggingface.co/spaces/sarthaksgrid/bdd100k-detection) — *Interactive real-time inference dashboard featuring side-by-side architecture comparison.*
* **Images you can try:** [External Data](https://drive.google.com/drive/folders/1obWHOcdv2WyuuYXmlxDRwgeBcCPU1JdH?usp=drive_link) — *Download images from here to try with the model. Additionally you can try any image similar to a dashcam image, but it has to be straight ahead at the road and the horizon, not looking down at the ground..*
* **Detailed Analysis** - *Refer to [BDD100K_Object_Detection_Model.pdf](BDD100K_Object_Detection_Model.pdf) for detailed analysis along with curve analysis*

---

## 📂 Repository Architecture



### **1. 🏗️ Data Engineering (`/data`)**
* **`raw/`**: Source BDD100K samples.
* **`processed/`**: Standardized $640 \times 640$ images via letterbox resizing.
* **`splits/`**: Includes `stratified_split_indices.json` ensuring **0.1% rare class** (trains, riders) representation.

### **2. 🧠 Model Zoo (`/models`)**
(For more detailed info on the model and the model files, refer models_info.md in the models folder)
* **`DETR/`**: 
    * `checkpoints/experiments/`: Hierarchical storage for Run 1 through Run 7.
    * `weights/`: Localized storage for `pytorch_model.bin` and `config.json`.
* **`YOLO/`**: 
    * `weights/`: Localized storage for `yolov8s.pt` to prevent root-directory clutter.

### **3. ⚖️ Evaluation & Diagnostics (`/tools`)**
* **`cleanup_checkpoints.py`**: Automated storage manager using **Decadic Pruning** to keep only every 10th epoch + the final state.
* **`print_tree.py`**: Directory auditing utility.

---

## 🚀 Execution & Training

### **Environment Setup**
To resolve dependency issues and ensure the "Run" button works seamlessly with the M4 Pro environment:
```bash
python -m pip install -r requirements.txt

```

### **The Master Pipeline**

Initialize the centralized orchestrator to manage stratification, training, and evaluation:

```bash
python run_train_sequence.py

```

* **Pro Tip:** Type **`d`** at any prompt to utilize research-validated default hyperparameters.

---

## 📊 Empirical Benchmarking Results

Validated performance on the stratified BDD100K validation set.

### **DETR Performance (Run 7 - Epoch 50)**

| Metric | Score | Note |
| --- | --- | --- |
| **mAP (Overall)** | **0.0581** | Mean across IoU 0.50:0.95 |
| **mAP @ 0.50** | **0.1400** | Primary localization accuracy |
| **mAP @ 0.75** | **0.0416** | Strict bounding box precision |

### **YOLOv8 Performance (Baseline)**

| Metric | Score | Note |
| --- | --- | --- |
| **mAP @ 0.50-0.95** | **0.0423** | General precision across thresholds |
| **mAP @ 0.50** | **0.0857** | Baseline Target: 0.3500 |
| **mAP @ 0.75** | **0.0391** | High-precision overlap |

---

## 🛠️ Technical Implementation Notes

* **Hardware Acceleration:** Scripts utilize `torch.device("mps")`. The M4 Pro provides high throughput for the bipartite matching loss and attention mechanisms.
* **Storage Management:** Implemented a **Rolling Retention Policy**. By only saving every 10th epoch, local storage was reduced from **147 GB to <15 GB**.
* **Gradio Inference:** `app.py` is configured with absolute path resolution to `models/YOLO/weights/yolov8s.pt` to maintain a clean root directory.

---
