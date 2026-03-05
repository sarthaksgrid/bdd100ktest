import json
import random
from pathlib import Path
from collections import defaultdict

def create_stratified_split():
    print("[INFO] Starting Stratified Splitter...")
    random.seed(42) # Lock the seed for reproducibility
    
    # This script sits inside the 'data' folder
    data_dir = Path(__file__).resolve().parent
    
    # Explicitly targeting the mathematically scaled 640x640 JSON
    json_path = data_dir / "splits" / "labels_640.json"
    out_path = data_dir / "splits" / "stratified_split_indices.json"
    
    if not json_path.exists():
        print(f"[ERROR] Could not find {json_path}")
        print("Please ensure you have generated labels_640.json first!")
        return
        
    print(f"[INFO] Reading perfectly scaled boxes from: {json_path.name}")
    with open(json_path, 'r') as f:
        coco = json.load(f)
        
    # 1. Map images to the classes they contain, and count class frequencies
    img_to_classes = defaultdict(set)
    class_counts = defaultdict(int)
    
    for ann in coco['annotations']:
        img_id = ann['image_id']
        cls_id = ann['category_id']
        img_to_classes[img_id].add(cls_id)
        class_counts[cls_id] += 1
        
    # Sort classes from Rarest to Most Common
    sorted_classes = sorted(class_counts.keys(), key=lambda c: class_counts[c])
    
    allocated_imgs = set()
    train_ids, val_ids, test_ids = [], [], []
    
    print("[INFO] Distributing images (Rarest classes first)...")
    
    # 2. Distribute images perfectly 70/20/10
    for cls in sorted_classes:
        # Find all unassigned images containing this class
        imgs_with_cls = [img for img, classes in img_to_classes.items() 
                         if cls in classes and img not in allocated_imgs]
        
        random.shuffle(imgs_with_cls)
        
        # Calculate split indices
        total = len(imgs_with_cls)
        train_end = int(total * 0.7)
        val_end = int(total * 0.9) # 70% + 20%
        
        # Slice and allocate
        train_ids.extend(imgs_with_cls[:train_end])
        val_ids.extend(imgs_with_cls[train_end:val_end])
        test_ids.extend(imgs_with_cls[val_end:])
        
        allocated_imgs.update(imgs_with_cls)
        
    # Catch any images that had ZERO annotations (empty streets)
    all_imgs = set(img['id'] for img in coco['images'])
    leftovers = list(all_imgs - allocated_imgs)
    if leftovers:
        random.shuffle(leftovers)
        t_end = int(len(leftovers) * 0.7)
        v_end = int(len(leftovers) * 0.9)
        train_ids.extend(leftovers[:t_end])
        val_ids.extend(leftovers[t_end:v_end])
        test_ids.extend(leftovers[v_end:])

    # 3. Save the specific Image IDs to a nicely formatted JSON file
    split_dict = {
        "train": train_ids,
        "val": val_ids,
        "test": test_ids
    }
    
    with open(out_path, 'w') as f:
        # indent=4 is the magic parameter that formats it cleanly across multiple lines!
        json.dump(split_dict, f, indent=4)
        
    print(f"\n[SUCCESS] Stratified split completed!")
    print(f" -> Train: {len(train_ids)} | Val: {len(val_ids)} | Test: {len(test_ids)}")
    print(f" -> Saved formatted JSON to: {out_path}")

if __name__ == "__main__":
    create_stratified_split()