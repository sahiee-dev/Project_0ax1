import os
import torch
from ultralytics import YOLO

def load_model(model_path):
    """
    Loads the YOLOv8 model from the specified path.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    
    # PyTorch 2.6+ requires weights_only=True by default
    # Since this is a trusted model file from our project, we temporarily override this
    original_load = torch.load
    def patched_load(*args, **kwargs):
        kwargs['weights_only'] = False
        return original_load(*args, **kwargs)
    
    torch.load = patched_load
    try:
        model = YOLO(model_path)
    finally:
        torch.load = original_load
    
    return model
