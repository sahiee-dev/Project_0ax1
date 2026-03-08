import os
import argparse
import time
import cv2
import sys

# Add parent dir to sys.path to allow importing
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model.loader import load_models
from model.inference import run_inference
from utils.processing import parse_results

def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0:
        return 0.0
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea)

def read_yolo_labels(label_path, img_w, img_h):
    boxes = []
    if not os.path.exists(label_path):
        return boxes
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                c = int(parts[0])
                xc, yc, w, h = map(float, parts[1:5])
                xmin = (xc - w/2) * img_w
                ymin = (yc - h/2) * img_h
                xmax = (xc + w/2) * img_w
                ymax = (yc + h/2) * img_h
                boxes.append([xmin, ymin, xmax, ymax])
    return boxes

def evaluate(data_dir, labels_dir=None, limit=0):
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    model_paths = {
        "v8n": os.path.join(base_dir, "runs", "detect", "Normal_Compressed", "weights", "best.pt"),
        "v8s": os.path.join(base_dir, "runs", "detect", "Db", "weights", "best.pt")
    }
    
    # Load available models
    available_paths = [p for n, p in model_paths.items() if os.path.exists(p)]
    models = load_models(available_paths) if len(available_paths) > 0 else {}
    
    for name, path in model_paths.items():
        if name not in models:
            print(f"Warning: {path} not found. Skipping {name}.")
            
    if not models:
        print("No models available for evaluation.")
        return
        
    image_files = [f for f in os.listdir(data_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if limit > 0:
        image_files = image_files[:limit]
    if not image_files:
        print(f"No images found in {data_dir}")
        return
        
    print(f"Found {len(image_files)} images for evaluation.")
    
    use_labels = labels_dir is not None and os.path.exists(labels_dir)
    if use_labels:
        print(f"Ground truth labels provided. TP/FP/FN analysis Enabled.")
    else:
        print(f"No labels provided. Only analyzing detection counts and speed.")
    
    configs = [
        # New "Baseline" at 0.18 conf
        {"model": "v8n", "size": 640, "clahe": False, "tiling": False},
        # New Inference Engineering Pivot
        {"model": "v8n", "size": 640, "clahe": True, "tiling": True}
    ]
    
    print("\nStarting Evaluation...")
    if use_labels:
        print(f"{'Model':<5} | {'Size':<4} | {'CLAHE':<5} | {'Tiling':<6} | {'TP':<4} | {'FP':<4} | {'FN':<4} | {'Precision':<9} | {'Recall':<6} | {'Avg Time(ms)':<12}")
        print("-" * 92)
    else:
        print(f"{'Model':<5} | {'Size':<4} | {'CLAHE':<5} | {'Tiling':<6} | {'Total Dets':<10} | {'Avg Conf':<8} | {'Avg Time (ms)':<13}")
        print("-" * 75)
    
    for cfg in configs:
        model_name = cfg["model"]
        if model_name not in models:
            continue
        model = models[model_name]
        imgsz = cfg["size"]
        clahe = cfg["clahe"]
        tiling = cfg["tiling"]
        
        total_time = 0.0
        total_dets = 0
        sum_conf = 0.0
        valid_images = 0
        
        # For validation metrics
        total_tp = 0
        total_fp = 0
        total_fn = 0
        
        for img_file in image_files:
            img_path = os.path.join(data_dir, img_file)
            img = cv2.imread(img_path)
            if img is None:
                continue
                
            if valid_images == 0:
                _ = run_inference(model, img, conf=0.18, iou=0.45, augment=False, track=False, imgsz=imgsz, enable_clahe=clahe, enable_tiling=tiling)
                
            start_time = time.time()
            results = run_inference(
                model, img, conf=0.18, iou=0.45, augment=False, track=False, 
                imgsz=imgsz, enable_clahe=clahe, enable_tiling=tiling
            )
            inference_time = (time.time() - start_time) * 1000
            
            parsed = parse_results(results)
            dets = parsed['detections']
            pred_boxes = [d['bbox'] for d in dets]
            
            total_dets += len(dets)
            sum_conf += sum(d['confidence'] for d in dets)
            total_time += inference_time
            
            if use_labels:
                label_file = os.path.splitext(img_file)[0] + ".txt"
                label_path = os.path.join(labels_dir, label_file)
                gt_boxes = read_yolo_labels(label_path, img.shape[1], img.shape[0])
                
                matched_gt = set()
                matched_pred = set()
                
                for i, pbox in enumerate(pred_boxes):
                    best_iou = 0
                    best_gt_idx = -1
                    for j, gbox in enumerate(gt_boxes):
                        if j in matched_gt:
                            continue
                        iou = compute_iou(pbox, gbox)
                        if iou > 0.45 and iou > best_iou:
                            best_iou = iou
                            best_gt_idx = j
                            
                    if best_gt_idx != -1:
                        matched_gt.add(best_gt_idx)
                        matched_pred.add(i)
                        
                tp = len(matched_gt)
                fp = len(pred_boxes) - len(matched_pred)
                fn = len(gt_boxes) - len(matched_gt)
                
                total_tp += tp
                total_fp += fp
                total_fn += fn

            valid_images += 1
            
        if valid_images > 0:
            avg_time = total_time / valid_images
            if use_labels:
                precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
                recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
                print(f"{model_name:<5} | {imgsz:<4} | {str(clahe):<5} | {str(tiling):<6} | {total_tp:<4} | {total_fp:<4} | {total_fn:<4} | {precision:.3f}     | {recall:.3f}  | {avg_time:.2f}")
            else:
                avg_conf = sum_conf / total_dets if total_dets > 0 else 0.0
                print(f"{model_name:<5} | {imgsz:<4} | {str(clahe):<5} | {str(tiling):<6} | {total_dets:<10} | {avg_conf:.3f}    | {avg_time:.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate YOLO models with rigorous validation metrics.")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to folder containing test images.")
    parser.add_argument("--labels_dir", type=str, default=None, help="Path to folder containing YOLO format labels (*.txt) for TP/FP/FN analysis.")
    parser.add_argument("--limit", type=int, default=0, help="Limit the number of images to evaluate.")
    args = parser.parse_args()
    evaluate(args.data_dir, args.labels_dir, args.limit)
