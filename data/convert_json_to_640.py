
import json
from pathlib import Path

def convert_labels():
    print("[INFO] Loading original COCO labels.json...")
    
    # Since this script is inside the 'data' folder, we set our base path here
    data_dir = Path(__file__).resolve().parent
    
    # Paths based on your tree
    original_json_path = data_dir / "splits" / "labels.json"
    new_json_path = data_dir / "splits" / "labels_640.json"
    
    if not original_json_path.exists():
        print(f"[ERROR] Could not find {original_json_path}")
        print("Run your parser/bdd_to_coco.py first if you haven't!")
        return

    with open(original_json_path, 'r') as f:
        coco_data = json.load(f)

    # BDD100K original images: 1280x720 -> Processed folder images: 640x640
    scale_x = 640.0 / 1280.0
    scale_y = 640.0 / 720.0

    print("[INFO] Updating Image Metadata to 640x640...")
    for img in coco_data['images']:
        img['width'] = 640
        img['height'] = 640

    print("[INFO] Mathematically Scaling Bounding Boxes...")
    for ann in coco_data['annotations']:
        # COCO Format: [x_min, y_min, width, height]
        x, y, w, h = ann['bbox']
        
        # Apply scaling math
        new_x = x * scale_x
        new_y = y * scale_y
        new_w = w * scale_x
        new_h = h * scale_y
        
        ann['bbox'] = [new_x, new_y, new_w, new_h]
        ann['area'] = new_w * new_h  # Area must also be scaled

    with open(new_json_path, 'w') as f:
        json.dump(coco_data, f)

    print(f"\n[SUCCESS] Saved perfectly scaled annotations to:")
    print(f" -> {new_json_path}")

if __name__ == "__main__":
    convert_labels()