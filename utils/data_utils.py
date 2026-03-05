import torch
from torch.utils.data import Dataset
from pycocotools.coco import COCO
from PIL import Image
from pathlib import Path
from torchvision import transforms

class BDD100KGPUDataset(Dataset):
    """
    Optimized Dataset loader for BDD100K on Apple Silicon.
    Includes Data Augmentation (ColorJitter) and the critical Background Label Fix.
    """
    def __init__(self, ann_file, img_dir, processor, is_training=False):
        self.coco = COCO(ann_file)
        self.ids = list(self.coco.imgs.keys())
        self.img_dir = Path(img_dir)
        self.processor = processor
        self.is_training = is_training
        
        # --- DATA AUGMENTATION ---
        # Slightly changes lighting/colors so the model learns shapes, not pixels.
        self.color_jitter = transforms.ColorJitter(
            brightness=0.2, 
            contrast=0.2, 
            saturation=0.2, 
            hue=0.05
        )

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        # 1. Get Image Metadata from COCO JSON
        img_id = self.ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        
        # 2. Load the pre-resized 640x640 image
        img_path = self.img_dir / img_info['file_name']
        
        try:
            image = Image.open(img_path).convert("RGB")
        except FileNotFoundError:
            # Fallback if an image is missing to prevent crashing
            print(f"[WARNING] Image missing: {img_path}. Falling back to first image.")
            img_info = self.coco.loadImgs(self.ids[0])[0]
            img_path = self.img_dir / img_info['file_name']
            image = Image.open(img_path).convert("RGB")

        # --- APPLY AUGMENTATION ---
        # We only want to distort images during training, NOT during validation/testing
        if self.is_training:
            image = self.color_jitter(image)

        # 3. Get Annotations
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        annotations = self.coco.loadAnns(ann_ids)
        
        target = {
            'image_id': img_id, 
            'annotations': annotations
        }
        
        # 4. Processor Transformation
        encoding = self.processor(images=image, annotations=target, return_tensors="pt")
        
        # Squeeze out the batch dimension since DataLoader will batch them later
        pixel_values = encoding["pixel_values"].squeeze(0) 
        labels = encoding["labels"][0]
        
        # --- CRITICAL BUG FIX: SHIFT LABELS ---
        # Maps COCO 1-10 to DETR 0-9. Leaves Slot 10 safely empty for Background.
        labels['class_labels'] = labels['class_labels'] - 1
        
        return pixel_values, labels

def collate_fn(batch):
    """
    Custom collate function required because bounding box lists 
    are different lengths for every image.
    """
    pixel_values = torch.stack([item[0] for item in batch])
    labels = [item[1] for item in batch]
    
    return {
        'pixel_values': pixel_values,
        'labels': labels
    }