"""
CCTV Holdout Evaluation Script

Evaluates all trained models on the strictly held-out CCTV test set.
Primary metric: Recall@0.5 per class (guns ≥ 0.80, knife ≥ 0.70)

Usage:
    python data/evaluate_cctv.py
"""
import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.abspath("."))

from ultralytics import YOLO

PROJECT_ROOT = Path(".")
CCTV_DATA    = PROJECT_ROOT / "data/unified_yolo/cctv_test"
YAML_PATH    = PROJECT_ROOT / "data/unified_weapon_data.yaml"
CONF         = 0.25   # match inference default
IOU          = 0.55

MODELS = {
    "Standard Nano (old)": "runs/detect/Normal_Compressed/weights/best.pt",
    "DB (old)":            "runs/detect/Db/weights/best.pt",
    "Haar (old)":          "runs/detect/Haar/weights/best.pt",
    "Standard Nano (v2)":  "runs/detect/Normal_Compressed_v2/weights/best.pt",
    "DB (v2)":             "runs/detect/Db_v2/weights/best.pt",
    "Haar (v2)":           "runs/detect/Haar_v2/weights/best.pt",
}

# Success thresholds
RECALL_GUNS_TARGET  = 0.80
RECALL_KNIFE_TARGET = 0.70

def evaluate(name, model_path):
    path = PROJECT_ROOT / model_path
    if not path.exists():
        print(f"  [SKIP] {name} — model not found: {path}")
        return None

    model = YOLO(str(path))
    results = model.val(
        data=str(YAML_PATH),
        split="test",
        conf=CONF,
        iou=IOU,
        verbose=False,
    )

    # Extract per-class metrics
    # results.box.class_result(i) → (p, r, ap50, ap)
    nc = results.box.nc if hasattr(results.box, "nc") else 2
    classes = ["guns", "knife"]

    metrics = {}
    for i in range(min(nc, len(classes))):
        row = results.box.class_result(i)
        metrics[classes[i]] = {
            "precision": row[0],
            "recall":    row[1],
            "ap50":      row[2],
        }

    metrics["overall"] = {
        "map50":    results.box.map50,
        "map50_95": results.box.map,
    }
    return metrics


def main():
    print("\n" + "=" * 70)
    print("  CCTV HOLDOUT EVALUATION")
    print(f"  Conf={CONF}  IoU={IOU}")
    print(f"  Targets: guns recall ≥ {RECALL_GUNS_TARGET}  |  knife recall ≥ {RECALL_KNIFE_TARGET}")
    print("=" * 70 + "\n")

    rows = []
    for name, path in MODELS.items():
        print(f"Evaluating: {name}")
        m = evaluate(name, path)
        if m is None:
            continue
        rows.append((name, m))

    if not rows:
        print("No results — run training first.")
        return

    # Header
    print(f"\n{'Model':<30} {'mAP50':>6} {'Guns R':>7} {'Knife R':>8}  {'Guns':>6}  {'Knife':>6}")
    print("-" * 72)

    for name, m in rows:
        map50      = m["overall"]["map50"]
        guns_r     = m.get("guns", {}).get("recall", float("nan"))
        knife_r    = m.get("knife", {}).get("recall", float("nan"))
        guns_pass  = "✓" if guns_r  >= RECALL_GUNS_TARGET  else "✗"
        knife_pass = "✓" if knife_r >= RECALL_KNIFE_TARGET else "✗"
        print(f"{name:<30} {map50:>6.3f}  {guns_r:>6.3f}  {knife_r:>7.3f}  [{guns_pass}]  [{knife_pass}]")

    print("\n✓ = meets target   ✗ = below target")
    print(f"Target: guns Recall@0.5 ≥ {RECALL_GUNS_TARGET}  |  knife Recall@0.5 ≥ {RECALL_KNIFE_TARGET}\n")


if __name__ == "__main__":
    main()
