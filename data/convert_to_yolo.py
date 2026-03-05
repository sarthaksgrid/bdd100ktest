import json
import shutil
import sys
from pathlib import Path
from tqdm import tqdm

def main():
    print(f"\n{'='*50}")
    print(" YOLOv8 DATASET CONVERTER")
    print(f"{'='*50}")

    # Root directory math
    root = Path(__file__).resolve().parent.parent
    sys.path.append(str(root))
    
    # --- FIX: Import the exact same config DETR uses ---
    from utils.config import ANN_FILE, IMG_DIR
    
    ann_file = Path(ANN_FILE)
    img_dir = Path(IMG_DIR)
    split_file = root / "data" / "splits" / "stratified_split_indices.json"
    
    # New YOLO Directory Structure
    yolo_dir = root / "data" / "yolo_dataset"
    dirs_to_make = [
        yolo_dir / "images" / "train", yolo_dir / "images" / "val",
        yolo_dir / "labels" / "train", yolo_dir / "labels" / "val"
    ]
    for d in dirs_to_make:
        d.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Loading COCO annotations from {ann_file.name}...")
    with open(ann_file, 'r') as f:
        coco = json.load(f)
    with open(split_file, 'r') as f:
        splits = json.load(f)

    # Map image IDs to file names and annotations
    img_dict = {img['id']: img for img in coco['images']}
    ann_dict = {img['id']: [] for img in coco['images']}
    for ann in coco['annotations']:
        ann_dict[ann['image_id']].append(ann)

    def process_split(split_name, img_ids):
        print(f"\n[PROCESS] Converting {split_name} split ({len(img_ids)} images)...")
        for img_id in tqdm(img_ids):
            img_info = img_dict[img_id]
            file_name = img_info['file_name']
            
            # Source and Destination paths
            src_img = img_dir / file_name
            dst_img = yolo_dir / "images" / split_name / file_name
            label_file = yolo_dir / "labels" / split_name / f"{Path(file_name).stem}.txt"
            
            # 1. Copy Image (Ignore if it doesn't exist)
            if not src_img.exists():
                continue
            if not dst_img.exists():
                shutil.copy2(src_img, dst_img)
            
            # 2. Convert Bounding Boxes to YOLO format (Normalized x_center, y_center, w, h)
            IMG_W, IMG_H = 640.0, 640.0
            
            yolo_labels = []
            for ann in ann_dict[img_id]:
                cat_id = ann['category_id'] - 1 # YOLO requires 0-indexed classes
                x_min, y_min, w, h = ann['bbox']
                
                # Math: Center coords and normalize
                x_center = (x_min + (w / 2)) / IMG_W
                y_center = (y_min + (h / 2)) / IMG_H
                w_norm = w / IMG_W
                h_norm = h / IMG_H
                
                # Keep within bounds to prevent YOLO errors
                x_center, y_center = max(0.0, min(1.0, x_center)), max(0.0, min(1.0, y_center))
                w_norm, h_norm = max(0.0, min(1.0, w_norm)), max(0.0, min(1.0, h_norm))
                
                yolo_labels.append(f"{cat_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")
                
            with open(label_file, 'w') as f:
                f.write("\n".join(yolo_labels))

    # We map COCO integer IDs to the actual dataset IDs based on our split
    train_ids = [coco['images'][i]['id'] for i in splits['train']]
    val_ids = [coco['images'][i]['id'] for i in splits['val']]

    process_split('train', train_ids)
    process_split('val', val_ids)

    # 3. Create the dataset.yaml file
    yaml_path = yolo_dir / "bdd100k.yaml"
    yaml_content = f"""path: {yolo_dir.absolute()}
train: images/train
val: images/val

names:
  0: pedestrian
  1: rider
  2: car
  3: truck
  4: bus
  5: train
  6: motorcycle
  7: bicycle
  8: traffic light
  9: traffic sign
"""
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
        
    print(f"\n[SUCCESS] YOLO formatting complete! Config saved to: {yaml_path.name}")

if __name__ == "__main__":
    main()