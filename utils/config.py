import torch
from pathlib import Path

# --- DIRECTORY MAPPING ---
# Base project root (assumes this file is in project_root/utils/config.py)
ROOT_DIR = Path(__file__).resolve().parent.parent

# Data Paths
DATA_DIR = ROOT_DIR / "data"
DATA_RAW = DATA_DIR / "raw"            # Location of original samples.json

# --- CRITICAL ALIGNMENT ---
# IMG_DIR is used by your DETR training and inference scripts.
# DATA_PROCESSED is used by parser/resize_images.py.
# Both point to the same folder to ensure data consistency.
IMG_DIR = DATA_DIR / "processed"       # Your perfectly resized 640x640 images
DATA_PROCESSED = IMG_DIR               # Alias to fix the ImportError in parser scripts

ANN_FILE = DATA_DIR / "splits" / "labels_640.json" # <-- POINTING TO NEW MATH JSON

# Model & Output Paths
CHECKPOINT_DIR = ROOT_DIR / "models" / "checkpoints"
VIS_DIR = ROOT_DIR / "outputs" / "visualizations"

# Create necessary directories automatically
for folder in [IMG_DIR, ANN_FILE.parent, CHECKPOINT_DIR, VIS_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

# --- HARDWARE ACCELERATION ---
# Optimized for M4 Pro (Metal Performance Shaders)
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

# --- HYPERPARAMETERS (Project Guidelines) ---
BATCH_SIZE = 16         # M4 Pro flex: Native 16 batch size
NUM_EPOCHS = 10         # Standard per Step 3
LR = 1e-4               # Recommended starting Learning Rate
WEIGHT_DECAY = 1e-4     # Recommended per Experiments section
GRAD_CLIP = 0.1         # Step 3: Crucial for Transformer stability

# --- MODEL SETTINGS ---
# CRITICAL FIX: Because data_utils.py shifts labels by -1, our classes are 0 through 9.
# We pass 10. DETR will automatically handle the Background class internally.
NUM_LABELS = 10         
NUM_QUERIES = 100       # Experiment recommendation for small signs/crowded scenes

# --- CLASS MAPPING ---
# CRITICAL FIX: Shifted to 0-based indexing to match the data_utils.py transformation
ID_TO_NAME = {
    0: "pedestrian", 1: "rider", 2: "car", 3: "truck", 
    4: "bus", 5: "train", 6: "motorcycle", 7: "bicycle",
    8: "traffic light", 9: "traffic sign"
}

print(f"[CONFIG] System initialized on {DEVICE}")