#!/usr/bin/env bash
# ============================================================
# train_all.sh — Train all three weapon detection models
#
# Run from project root:
#   ./data/train_all.sh 2>&1 | tee data/train_all.log
#
# Prerequisite: data/unified_yolo/ must exist (run merge_datasets.py first)
# ============================================================

set -euo pipefail

DATA_YAML="data/unified_weapon_data.yaml"
EPOCHS=100
PATIENCE=20  # early stop if no improvement

# Use the Homebrew-installed yolo CLI (confirmed present at this path)
YOLO="/opt/homebrew/bin/yolo"

# ────────────────────────────────────────────────────────────
# Model 1: Standard Nano (yolov8n) — fast, lightweight
# imgsz=640 (VRAM-aware for nano)
# Diversity: lower augmentation, moderate imgsz
# ────────────────────────────────────────────────────────────
echo ""
echo "================================================================"
echo " Training Standard Nano (yolov8n) — imgsz=640"
echo "================================================================"
$YOLO detect train \
    data="$DATA_YAML" \
    model=yolov8n.pt \
    name=Normal_Compressed_v2 \
    epochs=$EPOCHS \
    patience=$PATIENCE \
    imgsz=640 \
    batch=16 \
    optimizer=AdamW \
    lr0=0.001 \
    lrf=0.01 \
    mosaic=1.0 \
    copy_paste=0.3 \
    erasing=0.4 \
    degrees=10 \
    scale=0.5 \
    fliplr=0.5 \
    hsv_h=0.015 \
    hsv_s=0.7 \
    save_period=10 \
    project=runs/detect

# ────────────────────────────────────────────────────────────
# Model 2: High Accuracy DB (yolov8s) — balanced accuracy/speed
# imgsz=1024 for small/occluded weapon detection
# Diversity: stronger augmentation vs. nano
# ────────────────────────────────────────────────────────────
echo ""
echo "================================================================"
echo " Training High Accuracy DB (yolov8s) — imgsz=1024"
echo "================================================================"
$YOLO detect train \
    data="$DATA_YAML" \
    model=yolov8s.pt \
    name=Db_v2 \
    epochs=$EPOCHS \
    patience=$PATIENCE \
    imgsz=1024 \
    batch=8 \
    optimizer=AdamW \
    lr0=0.001 \
    lrf=0.01 \
    mosaic=1.0 \
    copy_paste=0.4 \
    erasing=0.4 \
    degrees=15 \
    scale=0.6 \
    fliplr=0.5 \
    hsv_h=0.015 \
    hsv_s=0.7 \
    mixup=0.1 \
    save_period=10 \
    project=runs/detect

# ────────────────────────────────────────────────────────────
# Model 3: High Accuracy Haar (yolov8m) — max accuracy
# imgsz=1024, strongest augmentation for ensemble diversity
# ────────────────────────────────────────────────────────────
echo ""
echo "================================================================"
echo " Training High Accuracy Haar (yolov8m) — imgsz=1024"
echo "================================================================"
$YOLO detect train \
    data="$DATA_YAML" \
    model=yolov8m.pt \
    name=Haar_v2 \
    epochs=$EPOCHS \
    patience=$PATIENCE \
    imgsz=1024 \
    batch=6 \
    optimizer=AdamW \
    lr0=0.001 \
    lrf=0.01 \
    mosaic=1.0 \
    copy_paste=0.5 \
    erasing=0.5 \
    degrees=20 \
    scale=0.7 \
    fliplr=0.5 \
    hsv_h=0.02 \
    hsv_s=0.8 \
    mixup=0.15 \
    save_period=10 \
    project=runs/detect

echo ""
echo "================================================================"
echo " All training complete."
echo " Weights:"
echo "   runs/detect/Normal_Compressed_v2/weights/best.pt"
echo "   runs/detect/Db_v2/weights/best.pt"
echo "   runs/detect/Haar_v2/weights/best.pt"
echo ""
echo " Next: python data/evaluate_cctv.py"
echo "================================================================"
