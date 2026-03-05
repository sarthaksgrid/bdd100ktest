import sys
from pathlib import Path
import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import DetrConfig, DetrForObjectDetection, DetrImageProcessor

# --- Directory Math Updated for models/DETR ---
SCRIPT_DIR = Path(__file__).resolve().parent     # bdd100k/models/DETR
ROOT_DIR = SCRIPT_DIR.parent.parent              # bdd100k/
sys.path.append(str(ROOT_DIR))

# Import your configuration
from utils.config import DEVICE

# Set up the inputs/outputs/checkpoints inside DETR
INPUT_DIR = SCRIPT_DIR / "inputs"
OUTPUT_DIR = SCRIPT_DIR / "outputs"
CHECKPOINTS_BASE_DIR = SCRIPT_DIR / "checkpoints"

INPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINTS_BASE_DIR.mkdir(parents=True, exist_ok=True)

try:
    from utils.config import ID_TO_NAME
except ImportError:
    ID_TO_NAME = {
        0: "pedestrian", 1: "rider", 2: "car", 3: "truck", 
        4: "bus", 5: "train", 6: "motorcycle", 7: "bicycle", 
        8: "traffic light", 9: "traffic sign"
    }

def get_font():
    try:
        # Standard path for macOS
        return ImageFont.truetype("/Library/Fonts/Arial.ttf", 16)
    except IOError:
        return ImageFont.load_default()

def process_image_robustly(img_path):
    img = Image.open(img_path)
    if img.mode == 'RGBA':
        bg = Image.new('RGB', img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[3])
        return bg
    return img.convert("RGB")

def main():
    print(f"\n{'='*60}")
    print("      DETR STANDALONE INFERENCE (DYNAMIC SYNC MODE)")
    print(f"{'='*60}")

    image_paths = [p for p in INPUT_DIR.glob("*") if p.suffix.lower() in ['.jpg', '.jpeg', '.png']]
    if not image_paths:
        print(f"\n[WARNING] No images found in {INPUT_DIR}")
        return

    # Recursive search for checkpoints in subfolders
    checkpoint_files = list(CHECKPOINTS_BASE_DIR.rglob("detr_epoch_*.pt"))
    if not checkpoint_files:
        print(f"\n[ERROR] No model checkpoints found in {CHECKPOINTS_BASE_DIR}!")
        return
        
    def get_epoch_num(p):
        try:
            return int(p.stem.split('_')[-1])
        except (ValueError, IndexError):
            return -1

    # Identify and load the latest checkpoint
    latest_ckpt = max(checkpoint_files, key=get_epoch_num)
    print(f"[LOAD] Best candidate found: {latest_ckpt.relative_to(SCRIPT_DIR)}")
    
    # --- DYNAMIC CONFIGURATION SYNC ---
    # We load the weights first to inspect the saved architecture shapes
    checkpoint = torch.load(latest_ckpt, map_location=DEVICE)
    state_dict = checkpoint['model_state_dict']

    # Determine num_labels from class_labels_classifier bias (includes background)
    # If bias is shape [12], num_labels for config should be 11.
    ckpt_num_labels_with_bg = state_dict['class_labels_classifier.bias'].shape[0]
    ckpt_num_labels = ckpt_num_labels_with_bg - 1

    # Determine num_queries from query_position_embeddings weight
    ckpt_num_queries = state_dict['model.query_position_embeddings.weight'].shape[0]

    print(f"[SYNC] Detected Checkpoint Specs: {ckpt_num_labels} Classes | {ckpt_num_queries} Queries")

    # Initialize config and model with detected shapes to prevent RuntimeError
    config = DetrConfig.from_pretrained(
        "facebook/detr-resnet-50", 
        num_labels=ckpt_num_labels, 
        num_queries=ckpt_num_queries
    )
    model = DetrForObjectDetection(config).to(DEVICE)
    
    # Load state dict now that shapes match
    model.load_state_dict(state_dict)
    model.eval()

    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", do_resize=False)
    font = get_font()

    print(f"\n[INFO] Found {len(image_paths)} image(s). Starting inference on {DEVICE}...")

    with torch.no_grad():
        for img_path in image_paths:
            print(f" -> Processing: {img_path.name}")
            
            original_image = process_image_robustly(img_path)
            # Standardizing to 640x640 for the transformer backbone
            image = original_image.resize((640, 640), Image.Resampling.LANCZOS)
            
            inputs = processor(images=image, return_tensors="pt").to(DEVICE)
            outputs = model(**inputs)

            # Post-process with matching target sizes
            target_sizes = torch.tensor([[640, 640]]).to(DEVICE)
            results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.3)[0]

            draw = ImageDraw.Draw(image)
            detections_found = 0
            
            for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                box = [round(i, 2) for i in box.tolist()]
                class_id = label.item()
                class_name = ID_TO_NAME.get(class_id, f"Class_{class_id}")
                conf = score.item()
                
                # Visual output: Bounding Box and Label
                draw.rectangle(box, outline="#00FF00", width=3)
                label_text = f"{class_name} {conf:.2f}"
                
                try:
                    bbox = font.getbbox(label_text)
                    text_width, text_height = bbox[2] - bbox[0], bbox[3] - bbox[1]
                except AttributeError:
                    text_width, text_height = font.getsize(label_text)
                    
                draw.rectangle([box[0], box[1] - text_height - 4, box[0] + text_width + 4, box[1]], fill="#00FF00")
                draw.text((box[0] + 2, box[1] - text_height - 2), label_text, fill="black", font=font)
                
                detections_found += 1

            out_file = OUTPUT_DIR / f"det_{img_path.name}"
            image.save(out_file)
            print(f"    Found {detections_found} objects. Result saved.")

    print(f"\n[SUCCESS] Inference complete! Results are in: {OUTPUT_DIR.relative_to(ROOT_DIR)}")

if __name__ == "__main__":
    main()