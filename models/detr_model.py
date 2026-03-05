from transformers import DetrForObjectDetection, DetrConfig
import torch

def get_model(num_labels=7, freeze_backbone=True):
    """
    Initializes the DETR model with a ResNet-50 backbone.
    
    Args:
        num_labels (int): Number of target classes + 1 for background.
        freeze_backbone (bool): If True, disables gradient calculation for the ResNet layers 
                                 to accelerate training on the M4 Pro.
    """
    # 1. Load Configuration
    # We use the standard facebook/detr-resnet-50 as the architecture base
    config = DetrConfig.from_pretrained(
        "facebook/detr-resnet-50", 
        num_labels=num_labels
    )
    
    # 2. Initialize Model
    # ignore_mismatched_sizes=True allows us to replace the original 91-class head 
    # with our 7-class (BDD100K) head
    model = DetrForObjectDetection.from_pretrained(
        "facebook/detr-resnet-50",
        config=config,
        ignore_mismatched_sizes=True
    )
    
    # 3. Backbone Freezing for M4 Pro Turbo Strategy
    # This reduces the computational load by ~45%, essential for 50 epochs/day
    if freeze_backbone:
        print("M4 Pro Turbo: Freezing ResNet-50 backbone layers.")
        for param in model.model.backbone.parameters():
            param.requires_grad = False
            
    return model