import os
import torch
from ultralytics import YOLO

def load_models(model_paths):
    """
    Loads YOLOv8 model(s) from the specified path(s).
    Returns a dictionary of {name: model}.
    """
    if isinstance(model_paths, str):
        model_paths = [model_paths]
        
    models = {}
    
    # PyTorch 2.6+ requires weights_only=True by default
    original_load = torch.load
    def patched_load(*args, **kwargs):
        kwargs['weights_only'] = False
        return original_load(*args, **kwargs)
    
    torch.load = patched_load
    try:
        for path in model_paths:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Model file not found at {path}")
            
            basename = os.path.basename(os.path.dirname(os.path.dirname(path))) # e.g. Db, Normal_Compressed
            if "Normal" in basename:
                name = "v8n"
            elif "Db" in basename:
                name = "v8s"
            else:
                name = "default"
                
            models[name] = YOLO(path)
    finally:
        torch.load = original_load
    
    return models
