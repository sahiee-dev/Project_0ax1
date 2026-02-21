from ultralytics import YOLO

def run_inference(model, source, conf=0.25, iou=0.7, augment=False, track=False):
    """
    Runs inference using the loaded model on the source (image/frame).
    Returns the raw results.
    
    Args:
        model: Loaded YOLO model
        source: Image or frame
        conf (float): Confidence threshold (0-1)
        iou (float): NMS IoU threshold (0-1)
        augment (bool): Whether to use Test Time Augmentation (TTA)
        track (bool): Whether to use YOLO object tracking
    """
    if track:
        results = model.track(source, conf=conf, iou=iou, augment=augment, persist=True)
    else:
        results = model(source, conf=conf, iou=iou, augment=augment)
    return results
