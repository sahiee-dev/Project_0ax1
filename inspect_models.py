"""
Utility script to inspect YOLOv8 model classes and verify loading.
Checks multiple model variants (Normal, Db, Haar) and prints their class names.
"""

from ultralytics import YOLO
import os

# Define model paths relative to the project root for better portability
model_paths = {
    'Normal': 'runs/detect/Normal_Compressed/weights/best.pt',
    'Db': 'runs/detect/Db/weights/best.pt',
    'Haar': 'runs/detect/Haar/weights/best.pt'
}

def inspect_all_models():
    """Iterates through predefined model paths and prints metadata."""
    for name, path in model_paths.items():
        print(f"\n--- Model: {name} ---")
        if not os.path.exists(path):
            print(f"Error: Path does not exist: {path}")
            continue
        try:
            model = YOLO(path)
            print(f"Names: {model.names}")
        except Exception as e:
            print(f"Error loading {name}: {e}")

if __name__ == "__main__":
    inspect_all_models()
