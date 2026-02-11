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
    
    # Plot the results on the image (returns numpy array)
    annotated_image = result.plot()
    
    # Extract class names and boxes
    names = result.names
    boxes = result.boxes
    
    counts = {}
    detections = []
    
    for box in boxes:
        cls_id = int(box.cls[0])
        label = names[cls_id]
        conf = float(box.conf[0])
        
        detections.append({
            "label": label,
            "confidence": conf,
            "bbox": box.xyxy[0].tolist()
        })
        
        if label in counts:
            counts[label] += 1
        else:
            counts[label] = 1
            
    return {
        "annotated_image": annotated_image,
        "counts": counts,
        "detections": detections,
        "total_weapons": sum(counts.values())
    }
