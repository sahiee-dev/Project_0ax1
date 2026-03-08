def parse_results(results):
    """
    Parses YOLOv8 results to extract detections, counts, and annotated images.
    
    Args:
        results: List of YOLO results objects.
        
    Returns:
        dict: processed data containing 'image', 'counts', 'detections'
    """
    # Assuming single image inference for now, so we take the first result
    result = results[0]
    
    names = result.names
    boxes = result.boxes
    
    counts = {}
    detections = []
    
    # Store valid indices to potentially draw later or just draw manually
    # We will manually draw bounding boxes so we only show the filtered ones
    import copy
    import cv2
    img_draw = copy.deepcopy(result.orig_img)
    
    for box in boxes:
        # Geometric filtering
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        w = x2 - x1
        h = y2 - y1
        
        # Filter 1: Too small (absolute garbage)
        if w < 10 or h < 10:
            continue
            
        # Filter 2: Extreme aspect ratio (long thin poles etc)
        aspect_ratio = w / h if h > 0 else 0
        if aspect_ratio > 6.0 or aspect_ratio < 0.16:
            continue
            
        cls_id = int(box.cls[0])
        label = names[cls_id]
        conf = float(box.conf[0])
        track_id = int(box.id[0]) if box.id is not None else None
        
        # Draw on image
        color = (0, 0, 255) if cls_id == 0 else (255, 0, 0)
        cv2.rectangle(img_draw, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        txt = f"{label} {conf:.2f}"
        if track_id is not None:
             txt += f" id:{track_id}"
        cv2.putText(img_draw, txt, (int(x1), int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        detections.append({
            "label": label,
            "confidence": conf,
            "bbox": [x1, y1, x2, y2],
            "track_id": track_id
        })
        
        if label in counts:
            counts[label] += 1
        else:
            counts[label] = 1
            
    return {
        "annotated_image": img_draw,
        "counts": counts,
        "detections": detections,
        "total_weapons": sum(counts.values())
    }
