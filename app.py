import gradio as gr
from ultralytics import YOLO
from transformers import DetrImageProcessor, DetrForObjectDetection
import torch
import PIL.Image as Image
import pandas as pd
import numpy as np
import os
from pathlib import Path

# --- 1. PATH SETUP ---
ROOT_DIR = Path(__file__).resolve().parent

# FIX: Check multiple locations for weights to support both Local and Cloud deployment
# Locally: models/YOLO/weights/best.pt | Cloud: ./best.pt
YOLO_POSSIBLE_PATHS = [ROOT_DIR / "models" / "YOLO" / "weights" / "best.pt", ROOT_DIR / "best.pt"]
DETR_POSSIBLE_PATHS = [ROOT_DIR / "models" / "DETR" / "weights" / "detr_final.pt", ROOT_DIR / "detr_final.pt"]

YOLO_WEIGHTS_PATH = next((p for p in YOLO_POSSIBLE_PATHS if p.exists()), None)
DETR_WEIGHTS_PATH = next((p for p in DETR_POSSIBLE_PATHS if p.exists()), None)

# --- 2. HARDWARE ACCELERATION (FIX) ---
# Automatically selects the best available backend
if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

print(f"🚀 Initializing inference on device: {DEVICE}")

# --- 3. MODEL LOADING ---
# Load YOLOv8s
if YOLO_WEIGHTS_PATH:
    yolo_model = YOLO(str(YOLO_WEIGHTS_PATH))
    print(f"✅ Loaded YOLO Weights: {YOLO_WEIGHTS_PATH.name}")
else:
    print(f"⚠️ Warning: YOLO Research weights not found. Using pretrained baseline.")
    yolo_model = YOLO("yolov8s.pt") 

# Load DETR
try:
    detr_processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
    detr_model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
    
    if DETR_WEIGHTS_PATH:
        state_dict = torch.load(DETR_WEIGHTS_PATH, map_location="cpu")
        detr_model.load_state_dict(state_dict)
        print(f"✅ Loaded DETR Weights: {DETR_WEIGHTS_PATH.name}")
except Exception as e:
    print(f"ℹ️ Error initializing DETR components: {e}")

detr_model.to(DEVICE)
detr_model.eval()

# --- 4. CONFIGURATION ---
ID_TO_NAME = {
    0: "pedestrian", 1: "rider", 2: "car", 3: "truck", 
    4: "bus", 5: "train", 6: "motorcycle", 7: "bicycle",
    8: "traffic light", 9: "traffic sign"
}

COLOR_MAP = {
    "pedestrian": "#00FFFF", "rider": "#FF00FF", "car": "#FF3131",
    "truck": "#39FF14", "bus": "#CCFF00", "train": "#BC13FE",
    "motorcycle": "#0FF0FC", "bicycle": "#FAED27",
    "traffic light": "#FF5F1F", "traffic sign": "#FF10F0"
}

# --- 5. INFERENCE LOGIC ---
def predict_interactive(img, model_type, conf_slider, filter_class):
    if img is None:
        return None, None
    
    annotations = []
    counts = []

    if model_type == "YOLO":
        # Pass the dynamically selected DEVICE
        results = yolo_model.predict(source=img, conf=conf_slider, device=DEVICE)
        for r in results:
            for box in r.boxes:
                label_name = ID_TO_NAME.get(int(box.cls[0]), "unknown")
                if filter_class != "All" and label_name != filter_class:
                    continue
                coords = box.xyxy[0].cpu().numpy().astype(int).tolist()
                score = float(box.conf[0])
                annotations.append((coords, f"{label_name} ({score:.2f})")) 
                counts.append(label_name)
    else:
        # Pass the dynamically selected DEVICE
        inputs = detr_processor(images=img, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs = detr_model(**inputs)
        
        target_sizes = torch.tensor([img.size[::-1]])
        results = detr_processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=conf_slider
        )[0]

        results = {k: v.to("cpu") for k, v in results.items()}

        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            label_name = detr_model.config.id2label.get(label.item(), "unknown")
            if label_name == "person": label_name = "pedestrian"
            
            if filter_class != "All" and label_name != filter_class:
                continue

            box_coords = [int(i) for i in box.tolist()]
            annotations.append((box_coords, f"{label_name} ({score:.2f})"))
            counts.append(label_name)

    if counts:
        count_series = pd.Series(counts).value_counts()
        df = pd.DataFrame({"Object Class": count_series.index, "Count": count_series.values})
    else:
        df = pd.DataFrame(columns=["Object Class", "Count"])
            
    return (img, annotations), df

# --- 6. UI SETUP ---
with gr.Blocks(title="BDD100K Final Research Model", theme=gr.themes.Base()) as demo:
    gr.Markdown("<h1 style='text-align: center;'>🏙️ BDD100K Final Research Model</h1>")
    gr.Markdown(f"<p style='text-align: center;'>Current Device: <b>{DEVICE.upper()}</b></p>")
    
    with gr.Row():
        with gr.Column(scale=1):
            input_img = gr.Image(type="pil", label="Upload Scene")
            filter_dropdown = gr.Dropdown(
                choices=["All"] + list(ID_TO_NAME.values()), 
                value="All", 
                label="🎯 Target Class"
            )
        with gr.Column(scale=1):
            output_annotated = gr.AnnotatedImage(label="Inference Result", color_map=COLOR_MAP)
            model_type = gr.Radio(choices=["YOLO", "DETR"], value="YOLO", label="Model Architecture")
            conf_slider = gr.Slider(0.01, 1.0, value=0.25, label="Confidence Threshold")

    btn = gr.Button("🚀 EXECUTE INFERENCE", variant="primary")
    
    with gr.Accordion("📊 Statistical Analysis", open=True):
        stats_table = gr.Dataframe(label="Object Frequency", interactive=False)

    btn.click(
        fn=predict_interactive, 
        inputs=[input_img, model_type, conf_slider, filter_dropdown], 
        outputs=[output_annotated, stats_table]
    )

if __name__ == "__main__":
    # Settings for Hugging Face Deployment (Global) vs Local
    demo.launch()