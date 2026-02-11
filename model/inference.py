from ultralytics import YOLO

def run_inference(model, source):
    """
    Runs inference using the loaded model on the source (image/frame).
    Returns the raw results.
    """
    results = model(source)
    return results
