"""
train_all.py — Train all three weapon detection models.

Uses the Python API to avoid ultralytics CLI version constraints.
Run from project root:
    source venv/bin/activate && python data/train_all.py 2>&1 | tee data/train_all.log

Prerequisite: data/unified_yolo/ must exist (run merge_datasets.py first).
"""
import sys
import os
import torch

# Resolve project root from this script's location (data/train_all.py → project root)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)

from ultralytics import YOLO

# --- M1 Mac Optimization ---
def get_device():
    if torch.backends.mps.is_available():
        print("✓ MPS (Metal Performance Shaders) detected. Using M1 GPU acceleration.")
        return "mps"
    return "cpu"

DEVICE = get_device()
DATA_YAML = os.path.join(PROJECT_ROOT, "data", "unified_weapon_data.yaml")
COMMON = dict(
    data=DATA_YAML,
    epochs=100,
    patience=20,
    optimizer="AdamW",
    lr0=0.001,
    lrf=0.01,
    cls=2.0,            # knife class loss multiplier — counteracts gun overrepresentation
    mosaic=1.0,
    copy_paste=0.3,
    erasing=0.4,         # synthetic occlusion for partial-visibility robustness
    degrees=10,
    scale=0.5,
    fliplr=0.5,
    hsv_h=0.015,
    hsv_s=0.7,
    save_period=10,
    project=os.path.join(PROJECT_ROOT, "runs", "detect"),
    save=True,
    exist_ok=True,
    device=DEVICE,
)

MODELS = [
    # (name, weights_url, imgsz, batch, extra augmentation overrides)
    # Use model names (not paths) so ultralytics downloads fresh copies if needed,
    # avoiding PermissionError from quarantined local files.
    (
        "Normal_Compressed_v2", "yolov8n.pt", 640, 16,
        dict(copy_paste=0.3, degrees=10, mixup=0.0),
    ),
    (
        "Db_v2", "yolov8s.pt", 1024, 8,
        dict(copy_paste=0.4, degrees=15, scale=0.6, mixup=0.1),
    ),
    (
        "Haar_v2", "yolov8m.pt", 1024, 6,
        dict(copy_paste=0.5, degrees=20, scale=0.7, mixup=0.15, hsv_s=0.8),
    ),
]

for name, weights, imgsz, batch, extra in MODELS:
    print()
    print("=" * 68)
    print(f"  Training {name} — {weights}  imgsz={imgsz}  batch={batch}")
    print("=" * 68)

    cfg = {**COMMON, **extra, "imgsz": imgsz, "batch": batch, "name": name}

    model = YOLO(weights)
    model.train(**cfg)

print()
print("=" * 68)
print("  All training complete.")
print("  Weights:")
print("    runs/detect/Normal_Compressed_v2/weights/best.pt")
print("    runs/detect/Db_v2/weights/best.pt")
print("    runs/detect/Haar_v2/weights/best.pt")
print()
print("  Next: python3.10 data/evaluate_cctv.py")
print("=" * 68)
