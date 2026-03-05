import json
import os
from pathlib import Path
from tqdm import tqdm

# Import paths from your central config
try:
    from utils.config import DATA_RAW, ANN_FILE
except ImportError:
    # Fallback paths if you run the script in isolation
    DATA_RAW = Path("data/raw")
    ANN_FILE = Path("data/splits/labels.json")

def convert_to_coco():
    """
    Parses the raw FiftyOne/BDD JSON and converts it to COCO format.
    Automatically scales bounding boxes to 640x640 for optimized training.
    """
    raw_json_path = DATA_RAW / "samples.json"
    
    if not raw_json_path.exists():
        print(f"[ERROR] Could not find raw labels at: {raw_json_path}")
        return

    print(f"Loading raw annotations from {raw_json_path}...")
    with open(raw_json_path, 'r') as f:
        data_content = json.load(f)

    # Handle FiftyOne exported JSON structure
    if isinstance(data_content, dict) and "samples" in data_content:
        raw_data = data_content["samples"]
    elif isinstance(data_content, list):
        raw_data = data_content
    else:
        raw_data = data_content.get("detections", []) if isinstance(data_content, dict) else []

    # BDD100K 10 Categories required for object detection
    coco_output = {
        "images": [],
        "annotations": [],
        "categories": [
            {"id": 1, "name": "pedestrian"},
            {"id": 2, "name": "rider"},
            {"id": 3, "name": "car"},
            {"id": 4, "name": "truck"},
            {"id": 5, "name": "bus"},
            {"id": 6, "name": "train"},
            {"id": 7, "name": "motorcycle"},
            {"id": 8, "name": "bicycle"},
            {"id": 9, "name": "traffic light"},
            {"id": 10, "name": "traffic sign"}
        ]
    }

    category_map = {cat["name"]: cat["id"] for cat in coco_output["categories"]}
    ann_id_count = 1
    
    print(f"Converting {len(raw_data)} entries to COCO format (Target: 640x640)...")
    
    for i, entry in enumerate(tqdm(raw_data, desc="Generating JSON")):
        img_id = i
        
        file_path = entry.get("filepath", "")
        file_name = os.path.basename(file_path) if file_path else entry.get("name", "")
        
        if not file_name:
            continue
            
        # Image entry - Hardcoded to 640x640 to match our pre-processed images
        coco_output["images"].append({
            "id": img_id, 
            "file_name": file_name, 
            "width": 640, 
            "height": 640
        })

        # Extract detections array
        detections_field = entry.get("detections", {})
        raw_labels = []
        
        if isinstance(detections_field, dict):
            raw_labels = detections_field.get("detections", [])
        elif isinstance(detections_field, list):
            raw_labels = detections_field
        elif "labels" in entry: 
            raw_labels = entry["labels"]

        for label in raw_labels:
            # Normalize class name to match our category_map (e.g., "traffic sign")
            cat_name = (label.get("label") or label.get("category", "")).lower().replace("_", " ")
            
            if cat_name in category_map:
                # Bounding Box Extraction & Scaling
                if "bounding_box" in label: 
                    # FiftyOne format: Normalized [x, y, w, h] (0 to 1) -> convert to 640px absolute
                    bb = label["bounding_box"]
                    x, y, w, h = bb[0]*640, bb[1]*640, bb[2]*640, bb[3]*640
                elif "box2d" in label: 
                    # Raw BDD format: Pixel [x1, y1, x2, y2] out of 1280x720 -> scale to 640x640
                    b = label["box2d"]
                    x = (b["x1"] / 1280) * 640
                    y = (b["y1"] / 720) * 640
                    w = ((b["x2"] - b["x1"]) / 1280) * 640
                    h = ((b["y2"] - b["y1"]) / 720) * 640
                else:
                    continue

                # Discard invalid tiny/negative boxes
                if w <= 0 or h <= 0: continue

                # COCO requires absolute [x_min, y_min, width, height]
                coco_output["annotations"].append({
                    "id": ann_id_count,
                    "image_id": img_id,
                    "category_id": category_map[cat_name],
                    "bbox": [round(x, 2), round(y, 2), round(w, 2), round(h, 2)],
                    "area": round(w * h, 2),
                    "iscrowd": 0
                })
                ann_id_count += 1

    # Ensure output directory exists
    ANN_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    # Save with indent=4 for a human-readable, multi-line JSON file
    print(f"Saving structured JSON to: {ANN_FILE}")
    with open(ANN_FILE, 'w') as f:
        json.dump(coco_output, f, indent=4)
    
    print(f"[SUCCESS] Created {len(coco_output['annotations'])} annotations for {len(coco_output['images'])} images.")

if __name__ == "__main__":
    convert_to_coco()