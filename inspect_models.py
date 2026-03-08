from ultralytics import YOLO
import os

model_paths = {
    'Normal': '/Users/lulu/Desktop/Project_0ax1/runs/detect/Normal_Compressed/weights/best.pt',
    'Db': '/Users/lulu/Desktop/Project_0ax1/runs/detect/Db/weights/best.pt',
    'Haar': '/Users/lulu/Desktop/Project_0ax1/runs/detect/Haar/weights/best.pt'
}

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
