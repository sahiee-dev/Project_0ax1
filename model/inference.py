import os
import cv2
import numpy as np
import torch
import torchvision
from PIL import Image
from ultralytics import YOLO
from ultralytics.engine.results import Boxes

def apply_clahe(image_np, is_bgr=True, brightness_threshold=90):
    if is_bgr:
        lab = cv2.cvtColor(image_np, cv2.COLOR_BGR2LAB)
    else:
        lab = cv2.cvtColor(image_np, cv2.COLOR_RGB2LAB)
        
    l_channel, a, b = cv2.split(lab)
    
    # Conditional CLAHE: only apply if the image is relatively dark
    if np.mean(l_channel) < brightness_threshold:
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        cl = clahe.apply(l_channel)
        limg = cv2.merge((cl, a, b))
        
        if is_bgr:
            return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        else:
            return cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
    
    # Return original if bright enough
    return image_np

def run_inference(models, source, conf=0.18, iou=0.45, augment=False, track=False, imgsz=640, enable_clahe=False, enable_tiling=False):
    """
    Runs inference using the loaded model(s) on the source (image/frame).
    Supports a single model or a multi-model ensemble dictionary dict {name: model}.
    Returns the raw results.
    """
    if not isinstance(models, dict):
        # Wrap single model
        models = {"default": models}
        
    is_pil = isinstance(source, Image.Image)
    if is_pil:
        if source.mode != 'RGB':
            source = source.convert('RGB')
        img_np = np.array(source) # RGB
        is_bgr = False
    else:
        img_np = source # Assumed BGR from cv2
        if len(img_np.shape) == 3 and img_np.shape[2] == 4:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_BGRA2BGR)
        is_bgr = True

    if enable_clahe:
        # Pass a copy to avoid mutating the original source inadvertently if reused
        img_np = apply_clahe(img_np.copy(), is_bgr=is_bgr)

    h, w = img_np.shape[:2]
    
    # Pre-calculate tiles if enabled
    tiles = []
    if enable_tiling:
        overlap_w = int(w * 0.1)
        overlap_h = int(h * 0.1)
        
        w_split = w // 2
        h_split = h // 2
        
        tiles = [
            (img_np[0:h_split + overlap_h, 0:w_split + overlap_w], 0, 0),
            (img_np[0:h_split + overlap_h, w_split - overlap_w:w], w_split - overlap_w, 0),
            (img_np[h_split - overlap_h:h, 0:w_split + overlap_w], 0, h_split - overlap_h),
            (img_np[h_split - overlap_h:h, w_split - overlap_w:w], w_split - overlap_w, h_split - overlap_h)
        ]

    all_boxes = []
    base_res_shell = None
    
    for model_name, model in models.items():
        # Determine internal resolution (v8s trained on 1024, v8n trained on 640)
        current_imgsz = 1024 if "v8s" in model_name.lower() else imgsz
        
        tracker_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'utils', 'botsort_reid.yaml'))
        
        # 1. Base Full-Frame Inference
        if track:
            base_res = model.track(img_np, conf=conf, iou=iou, augment=augment, persist=True, tracker=tracker_path, imgsz=current_imgsz, verbose=False)[0]
        else:
            base_res = model(img_np, conf=conf, iou=iou, augment=augment, imgsz=current_imgsz, verbose=False)[0]
            
        if base_res_shell is None:
            base_res_shell = base_res # Steal the first result object to act as the master shell
            
        if len(base_res.boxes) > 0:
            all_boxes.append(base_res.boxes.data.clone())
            
        # 2. Tiled Overlap Inference
        for tile_img, x_off, y_off in tiles:
            # Tiled inference avoids tracking state corruption on partial frames
            tile_res = model(tile_img, conf=conf, iou=iou, augment=augment, imgsz=current_imgsz, verbose=False)[0]
            if len(tile_res.boxes) > 0:
                b = tile_res.boxes.data.clone() # [N, 6] or [N, 7]
                b[:, 0] += x_off
                b[:, 2] += x_off
                b[:, 1] += y_off
                b[:, 3] += y_off
                all_boxes.append(b)

    # 3. Aggressive Master NMS Merge
    if len(all_boxes) > 0:
        # Check tracking consistency across all collected tensors
        has_track = any(b.shape[1] == 7 for b in all_boxes) or track
        
        normalized_boxes = []
        for b in all_boxes:
            if has_track and b.shape[1] == 6:
                # Inject dummy track IDs (-1) into [x1, y1, x2, y2, id, conf, cls]
                dummy_ids = torch.full((b.shape[0], 1), -1.0, device=b.device)
                b = torch.cat([b[:, :4], dummy_ids, b[:, 4:]], dim=1)
            normalized_boxes.append(b)
            
        merged = torch.cat(normalized_boxes, dim=0)
        
        # Perform Batched NMS
        # merged shape: [Total, 6 or 7]
        boxes_coords = merged[:, :4]
        scores = merged[:, 5] if has_track else merged[:, 4]
        classes = merged[:, 6] if has_track else merged[:, 5]
        
        keep_idx = torchvision.ops.batched_nms(boxes_coords, scores, classes, iou)
        final_boxes_data = merged[keep_idx]
        
        base_res_shell.boxes = Boxes(final_boxes_data, base_res_shell.orig_shape)
    else:
        # No boxes anywhere
        base_res_shell.boxes = Boxes(torch.empty((0, 7 if track else 6), device=base_res_shell.boxes.data.device), base_res_shell.orig_shape)

    return [base_res_shell]
