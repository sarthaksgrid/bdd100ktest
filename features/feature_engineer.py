import torch
import cv2
import os
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
from pycocotools.coco import COCO

class BDD100KDETRDataset(Dataset):
    def __init__(self, img_dir, ann_file, transform=None):
        """
        Custom Dataset for BDD100K DETR Training
        """
        self.img_dir = img_dir
        self.coco = COCO(ann_file)
        self.ids = list(self.coco.imgs.keys())
        self.transform = transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        img_id = self.ids[index]
        file_name = self.coco.loadImgs(img_id)[0]['file_name']
        
        # Load image
        image_path = os.path.join(self.img_dir, file_name)
        image = cv2.imread(image_path)
        if image is None:
            # Fallback if image missing: return empty dummy tensor
            return torch.zeros((3, 640, 640)), {"boxes": torch.zeros((0, 4)), "labels": torch.zeros(0, dtype=torch.long)}
            
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load annotations
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        
        bboxes = []
        category_ids = []

        for ann in anns:
            # COCO format: [x_min, y_min, width, height]
            x, y, w, h = ann['bbox']
            # Convert to [x_min, y_min, x_max, y_max] and normalize
            x_min = x / 640.0
            y_min = y / 640.0
            x_max = (x + w) / 640.0
            y_max = (y + h) / 640.0
            # Strict clipping to [0, 1] range
            x_min, y_min, x_max, y_max = [max(0.0, min(float(v), 1.0)) for v in [x_min, y_min, x_max, y_max]]
            if (x_max - x_min) > 0 and (y_max - y_min) > 0:
                bboxes.append([x_min, y_min, x_max, y_max])
                category_ids.append(ann['category_id'])

        # Apply Transforms
        if self.transform and len(bboxes) > 0:
            transformed = self.transform(image=image, bboxes=bboxes, category_ids=category_ids)
            image = transformed['image']
            bboxes = transformed['bboxes']
            category_ids = transformed['category_ids']
        else:
            # MANUAL TENSOR CONVERSION FIX
            # If no transform or no bboxes, manually convert numpy array to Tensor
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        target = {
    "boxes": torch.tensor(bboxes, dtype=torch.float32) if bboxes else torch.zeros((0, 4)),
    "labels": torch.tensor(category_ids, dtype=torch.long),        # <--- NEW KEY
    "image_id": torch.tensor([img_id])
}
        return image, target

def get_train_transform():
    """
    Fixed Albumentations pipeline for M4 Pro
    Removed 'clip=True' from BboxParams to fix library incompatibility
    """
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format='coco', label_fields=['category_ids']))

def collate_fn(batch):
    """
    Handles batching of images and targets
    """
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    # Migration and safety: ensure each target uses 'labels' and has expected keys
    for t in targets:
        # migrate legacy key
        if 'class_labels' in t and 'labels' not in t:
            t['labels'] = t.pop('class_labels')
        # ensure labels tensor exists
        if 'labels' not in t:
            t['labels'] = torch.zeros(0, dtype=torch.long)
        # ensure boxes tensor exists
        if 'boxes' not in t:
            t['boxes'] = torch.zeros((0, 4), dtype=torch.float32)
    return images, targets