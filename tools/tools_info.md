# 🛠️ Tools & Utilities: Repository Maintenance

The `tools/` directory contains the administrative logic required to manage the project's disk footprint, maintain environment health, and audit code integrity. These scripts are vital for managing the heavy data requirements of the BDD100K dataset on the **Apple M4 Pro**.

---

## 📂 Key Utilities & Scripts

### 1. `print_tree.py` (The Intelligent Mapper)
**Logic:** A recursive directory crawler that generates a high-fidelity visual map of the repository.
* **Implementation:** Unlike standard tree commands, this script includes **Summarization Logic**. If a directory contains more than 15 files (e.g., `data/processed` with 10,000 images), it collapses the list and displays a count (e.g., `[10000 files]`) instead of printing every filename.
* **Filtering:** It automatically ignores metadata "noise" like `.git`, `__pycache__`, and `.DS_Store`.
* **Use Case:** Perfect for generating the "Repository Architecture" section of your main README and verifying that data splits are correctly populated.



### 2. `delete_checkpoints.py` (The Nuclear Cleanup)
**Logic:** Performs a high-speed, destructive cleanup to reclaim storage once a run is stabilized.
* **Implementation:** Uses `glob` matching to find `detr_epoch_*.pt` files.
* **Safety Logic:** Surgically programmed to preserve the `final_m4_pro_model.pt` and `loss_curve.png`, ensuring your results are safe while the "working memory" of the training run is cleared.

### 3. `audit_project.py` (The Pre-Flight Checklist)
**Logic:** A structural and data integrity auditor.
* **Implementation:** * **Data Audit:** Verifies counts for raw and processed images.
    * **Logic Audit:** Scans script contents for crucial performance flags (e.g., `do_resize=False` or `num_workers=0`).
    * **Progress Tracking:** Identifies the highest-numbered checkpoint to report real-time training status.

### 4. `cleanup_epochs.py`
**Logic:** Implements the **Decadic Retention Policy**. 
* **Implementation:** Preserves checkpoints at 10-epoch intervals (10, 20, 30...) to allow for historical analysis of model convergence without the 147 GB overhead.

---

## 🧹 Maintenance Workflows

### Documenting the Repo Structure
To generate a clean, summarized report of your current file structure:
```bash
python tools/print_tree.py