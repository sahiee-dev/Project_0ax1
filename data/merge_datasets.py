"""
Dataset Merge Script

Merges three sources into a unified YOLO dataset:
  1. DatasetNinja (guns only, class 0)
  2. Existing Pistol Detection images (class 0: guns)
  3. Existing Knife Detection images (class 1: knife) — oversampled with augmentation

Output:
    data/unified_yolo/
        train/images/   train/labels/
        val/images/     val/labels/
        cctv_test       → copied from data/dataninja_yolo/cctv_test (NOT split)

Target distribution: ~65% guns / ~35% knife

Usage:
    python data/merge_datasets.py
"""
import os
import shutil
import random
import cv2
import numpy as np
from pathlib import Path

# ── Source paths ────────────────────────────────────────────────────────────
DATA_ROOT = Path(".")

DATANINJA_TRAIN_IMGS = DATA_ROOT / "data/dataninja_yolo/train/images"
DATANINJA_TRAIN_LBLS = DATA_ROOT / "data/dataninja_yolo/train/labels"
CCTV_TEST_SRC        = DATA_ROOT / "data/dataninja_yolo/cctv_test"

PISTOL_IMGS = DATA_ROOT / "OD-WeaponDetection/Pistol detection/Images"
PISTOL_ANNS = DATA_ROOT / "OD-WeaponDetection/Pistol detection/xmls"

KNIFE_IMGS  = DATA_ROOT / "OD-WeaponDetection/Knife_detection"

# ── Output paths ────────────────────────────────────────────────────────────
OUT = DATA_ROOT / "data/unified_yolo"

# ── Config ──────────────────────────────────────────────────────────────────
VAL_FRACTION = 0.20
KNIFE_OVERSAMPLE_FACTOR = 1.5  # duplicate knife images with augmentation
SEED = 42
random.seed(SEED)

# ── Augmentation for oversampled knife duplicates ───────────────────────────
def augment_image(img: np.ndarray) -> np.ndarray:
    """Applies random color jitter, gaussian blur, and noise to create augmented duplicate."""
    out = img.copy().astype(np.float32)

    # Color jitter: ±20 brightness, ±20 contrast (multiplicative)
    alpha = random.uniform(0.8, 1.2)   # contrast
    beta  = random.uniform(-20, 20)    # brightness
    out = out * alpha + beta
    out = np.clip(out, 0, 255)

    # Gaussian blur with 50% probability
    if random.random() < 0.5:
        ksize = random.choice([3, 5])
        out = cv2.GaussianBlur(out, (ksize, ksize), 0)

    # Gaussian noise
    noise = np.random.randn(*out.shape) * random.uniform(2, 8)
    out = np.clip(out + noise, 0, 255).astype(np.uint8)

    # Random horizontal flip (label unchanged — guns/knives are symmetric)
    if random.random() < 0.5:
        out = cv2.flip(out, 1)

    return out


def copy_pair(img_src: Path, lbl_src: Path, img_dst_dir: Path, lbl_dst_dir: Path, stem_override: str = None):
    stem = stem_override or img_src.stem
    shutil.copy2(img_src, img_dst_dir / img_src.name)
    shutil.copy2(lbl_src, lbl_dst_dir / f"{stem}.txt")


def write_pair(img: np.ndarray, lbl_content: str, stem: str, img_dst_dir: Path, lbl_dst_dir: Path):
    cv2.imwrite(str(img_dst_dir / f"{stem}.jpg"), img)
    with open(lbl_dst_dir / f"{stem}.txt", "w") as f:
        f.write(lbl_content)


# ── Helper: convert Pistol Detection .xml (Pascal VOC) → YOLO ───────────────
def parse_voc_xml_to_yolo(xml_path: Path, class_id: int) -> list[str]:
    try:
        import xml.etree.ElementTree as ET
        tree = ET.parse(xml_path)
        root = tree.getroot()
        size = root.find("size")
        w = int(size.find("width").text)
        h = int(size.find("height").text)
        lines = []
        for obj in root.findall("object"):
            bndbox = obj.find("bndbox")
            x1 = float(bndbox.find("xmin").text)
            y1 = float(bndbox.find("ymin").text)
            x2 = float(bndbox.find("xmax").text)
            y2 = float(bndbox.find("ymax").text)
            cx = ((x1 + x2) / 2) / w
            cy = ((y1 + y2) / 2) / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h
            lines.append(f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        return lines
    except Exception:
        return []


# ── Collect all samples ──────────────────────────────────────────────────────
def main():
    all_samples = []  # list of (img_path, label_lines, class_id)

    # 1. DatasetNinja (guns)
    dn_imgs = sorted(DATANINJA_TRAIN_IMGS.glob("*"))
    print(f"DatasetNinja train images: {len(dn_imgs)}")
    for img_path in dn_imgs:
        lbl_path = DATANINJA_TRAIN_LBLS / (img_path.stem + ".txt")
        if lbl_path.exists():
            all_samples.append(("dataninja", img_path, lbl_path, 0))

    # 2. Pistol Detection images (XML annotations → class 0)
    pistol_img_exts = {".jpg", ".jpeg", ".png"}
    pistol_imgs = [p for p in PISTOL_IMGS.glob("*") if p.suffix.lower() in pistol_img_exts]
    print(f"Pistol Detection images found: {len(pistol_imgs)}")
    for img_path in pistol_imgs:
        xml_path = PISTOL_ANNS / (img_path.stem + ".xml")
        if xml_path.exists():
            all_samples.append(("pistol", img_path, xml_path, 0))

    # 3. Knife Detection images
    knife_exts = {".jpg", ".jpeg", ".png"}
    knife_imgs = []
    for ext in knife_exts:
        knife_imgs.extend(KNIFE_IMGS.rglob(f"*{ext}"))
    print(f"Knife Detection images found: {len(knife_imgs)}")
    knife_base = [(p, None, 1) for p in knife_imgs]

    # Count guns vs knife before oversampling
    gun_count = len([s for s in all_samples if s[3] == 0])
    knife_count = len(knife_base)
    print(f"\nBefore balance — guns: {gun_count}, knife: {knife_count}")
    print(f"Ratio: {gun_count/(gun_count+knife_count)*100:.1f}% guns")

    # Oversample knives with augmentation to reach ~35% of total
    # Target: knife_count * 1.5
    oversample_target = int(knife_count * KNIFE_OVERSAMPLE_FACTOR)
    oversample_extra  = oversample_target - knife_count
    knife_augmented   = random.choices(knife_imgs, k=oversample_extra)
    print(f"Oversampling {oversample_extra} knife duplicates (with augmentation)")

    total_guns = gun_count
    total_knife = knife_count + oversample_extra
    total = total_guns + total_knife
    print(f"After balance — guns: {total_guns} ({100*total_guns/total:.1f}%), knife: {total_knife} ({100*total_knife/total:.1f}%)")

    # ── Build train/val split ─────────────────────────────────────────────────
    # Separate by class, stratified split
    gun_samples  = [s for s in all_samples if s[3] == 0]
    knife_samples= knife_base  # (path, None, 1)

    random.shuffle(gun_samples)
    random.shuffle(knife_samples)

    val_gun   = gun_samples[:int(len(gun_samples) * VAL_FRACTION)]
    train_gun = gun_samples[int(len(gun_samples) * VAL_FRACTION):]
    val_knife  = knife_samples[:int(len(knife_samples) * VAL_FRACTION)]
    train_knife= knife_samples[int(len(knife_samples) * VAL_FRACTION):]

    # Augmented knife extra goes to train only
    train_knife_aug = knife_augmented

    print(f"\nSplit:")
    print(f"  Train guns:  {len(train_gun)}")
    print(f"  Val guns:    {len(val_gun)}")
    print(f"  Train knife (original): {len(train_knife)}")
    print(f"  Train knife (augmented extra): {len(train_knife_aug)}")
    print(f"  Val knife:   {len(val_knife)}")

    # ── Create output directories ──────────────────────────────────────────────
    for split in ["train", "val"]:
        (OUT / split / "images").mkdir(parents=True, exist_ok=True)
        (OUT / split / "labels").mkdir(parents=True, exist_ok=True)

    # ── Copy CCTV test (hardlink) ──────────────────────────────────────────────
    cctv_out = OUT / "cctv_test"
    if cctv_out.exists():
        shutil.rmtree(cctv_out)
    shutil.copytree(CCTV_TEST_SRC, cctv_out)
    print(f"\nCopied CCTV holdout → {cctv_out}")

    # ── Write output ───────────────────────────────────────────────────────────
    def write_gun_split(samples, split):
        out_i = OUT / split / "images"
        out_l = OUT / split / "labels"
        for src, img_path, ann_path, _ in samples:
            if src == "dataninja":
                shutil.copy2(img_path, out_i / img_path.name)
                shutil.copy2(ann_path, out_l / (img_path.stem + ".txt"))
            elif src == "pistol":
                lines = parse_voc_xml_to_yolo(ann_path, class_id=0)
                if not lines:
                    continue
                shutil.copy2(img_path, out_i / img_path.name)
                with open(out_l / (img_path.stem + ".txt"), "w") as f:
                    f.write("\n".join(lines))

    def write_knife_split(paths, split, augmented=False, aug_start_idx=0):
        out_i = OUT / split / "images"
        out_l = OUT / split / "labels"
        for idx, entry in enumerate(paths):
            img_path = entry[0] if isinstance(entry, tuple) else entry
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            # Single label line: class 1, full image (no annotations — create centre bbox)
            # Knife detection dataset has no XML — use full image annotation
            lbl = f"1 0.500000 0.500000 1.000000 1.000000"

            stem = img_path.stem
            if augmented:
                img = augment_image(img)
                stem = f"{stem}_aug{aug_start_idx + idx}"

            ext = ".jpg"
            cv2.imwrite(str(out_i / f"{stem}{ext}"), img)
            with open(out_l / f"{stem}.txt", "w") as f:
                f.write(lbl)

    print("\nWriting dataset...")
    write_gun_split(train_gun, "train")
    write_gun_split(val_gun, "val")
    write_knife_split(train_knife, "train")
    write_knife_split(val_knife, "val")
    write_knife_split(train_knife_aug, "train", augmented=True, aug_start_idx=len(train_knife))

    # Stats
    print("\n✓ Dataset written to:", OUT.resolve())
    for split in ["train", "val"]:
        n = len(list((OUT / split / "images").iterdir()))
        print(f"  {split}: {n} images")


if __name__ == "__main__":
    main()
