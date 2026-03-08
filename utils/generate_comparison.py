import os
import cv2
import numpy as np
import sys

# Add parent dir to sys.path to allow importing
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model.loader import load_model
from model.inference import run_inference
from utils.processing import parse_results

def generate_comparisons(data_dir, output_dir, limit=5):
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    model_path = os.path.join(base_dir, "runs", "detect", "Normal_Compressed", "weights", "best.pt")
    
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        return
        
    print(f"Loading v8n from {model_path}...")
    model = load_model(model_path)
    
    os.makedirs(output_dir, exist_ok=True)
    
    image_files = [f for f in os.listdir(data_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if limit > 0:
        image_files = image_files[:limit]
        
    for img_file in image_files:
        img_path = os.path.join(data_dir, img_file)
        img = cv2.imread(img_path)
        if img is None:
            continue
            
        print(f"Processing {img_file}...")
        
        # Baseline: conf=0.18, standard
        results_baseline = run_inference(
            model, img.copy(), conf=0.18, iou=0.45, augment=False, track=False, 
            imgsz=640, enable_clahe=False, enable_tiling=False
        )
        parsed_baseline = parse_results(results_baseline)
        baseline_annotated = parsed_baseline['annotated_image']
        
        # Add label
        cv2.putText(baseline_annotated, "BASELINE (No Tiling)", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # New Pipeline: conf=0.18, tiling, clahe
        results_new = run_inference(
            model, img.copy(), conf=0.18, iou=0.45, augment=False, track=False, 
            imgsz=640, enable_clahe=True, enable_tiling=True
        )
        parsed_new = parse_results(results_new)
        new_annotated = parsed_new['annotated_image']
        
        # Add label
        cv2.putText(new_annotated, "NEW (Tiling + Geo-Filtered)", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Resize if necessary to match heights (though they should be the same)
        h1, w1 = baseline_annotated.shape[:2]
        h2, w2 = new_annotated.shape[:2]
        
        # Concatenate horizontally
        comparison = np.hstack((baseline_annotated, new_annotated))
        
        out_path = os.path.join(output_dir, f"comparison_{img_file}")
        cv2.imwrite(out_path, comparison)
        print(f"Saved to {out_path}")

if __name__ == "__main__":
    generate_comparisons("data/unified_yolo/val/images", "client_comparisons", limit=10)
