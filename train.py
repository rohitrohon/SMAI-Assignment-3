"""
train.py — Indian Traffic Sign Classifier (T5.1)
-------------------------------------------------
Uses YOLOv8n-cls (classification head) to train on the Kaggle dataset:
  sarangdilipjodh/indian-traffic-signs-prediction85-classes

Steps:
  1. Download dataset via kaggle API (or accept a local zip path)
  2. Re-organise folder structure into ImageNet-style (train/ val/ test/)
  3. Train YOLOv8n-cls for N epochs
  4. Save best weights + export to ONNX for portability

Usage:
  python train.py --data_zip path/to/archive.zip --epochs 30 --imgsz 224
  OR (if you have kaggle CLI set up):
  python train.py --kaggle --epochs 30

Author: <your name>
"""

import argparse
import os
import shutil
import zipfile
import random
from pathlib import Path

from ultralytics import YOLO


# ──────────────────────────────────────────────
# 1.  Helpers
# ──────────────────────────────────────────────

def download_via_kaggle(dest: Path):
    """Pull the dataset with the kaggle CLI."""
    print("[info] Downloading dataset from Kaggle …")
    os.system(
        f"kaggle datasets download -d sarangdilipjodh/indian-traffic-signs-prediction85-classes "
        f"-p {dest} --unzip"
    )


def unzip(zip_path: Path, dest: Path):
    print(f"[info] Unzipping {zip_path} → {dest}")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(dest)


def find_class_dirs(root: Path):
    """
    Walk root and return a list of directories that look like class folders
    (i.e. they contain image files directly).
    """
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    class_dirs = []
    for p in sorted(root.rglob("*")):
        if p.is_dir():
            images = [f for f in p.iterdir() if f.suffix.lower() in image_exts]
            if images:
                class_dirs.append(p)
    return class_dirs


def build_imagenet_split(class_dirs, out_root: Path, val_frac=0.15, test_frac=0.10, seed=42):
    """
    Re-organise raw class folders into:
        out_root/train/<class>/…
        out_root/val/<class>/…
        out_root/test/<class>/…
    """
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    random.seed(seed)

    if out_root.exists():
        print(f"[info] {out_root} already exists – skipping split.")
        return

    print(f"[info] Building train/val/test split in {out_root} …")
    for cls_dir in class_dirs:
        cls_name = cls_dir.name
        images = sorted([f for f in cls_dir.iterdir() if f.suffix.lower() in image_exts])
        random.shuffle(images)

        n = len(images)
        n_val  = max(1, int(n * val_frac))
        n_test = max(1, int(n * test_frac))
        n_train = n - n_val - n_test

        splits = {
            "train": images[:n_train],
            "val":   images[n_train: n_train + n_val],
            "test":  images[n_train + n_val:],
        }
        for split, files in splits.items():
            dest_dir = out_root / split / cls_name
            dest_dir.mkdir(parents=True, exist_ok=True)
            for f in files:
                shutil.copy(f, dest_dir / f.name)

    print(f"[info] Split complete. Classes: {len(class_dirs)}")


# ──────────────────────────────────────────────
# 2.  Training
# ──────────────────────────────────────────────

def train(data_dir: Path, epochs: int, imgsz: int, batch: int, project: str):
    """Fine-tune YOLOv8n-cls on our dataset."""

    print("\n[info] Starting YOLOv8n-cls training …")
    model = YOLO("yolov8n-cls.pt")   # downloads ~6 MB pretrained weights automatically

    results = model.train(
        data=str(data_dir),           # folder with train/ val/ sub-dirs
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project=project,
        name="traffic_sign_cls",
        patience=10,                  # early stopping
        optimizer="AdamW",
        lr0=1e-3,
        weight_decay=5e-4,
        augment=True,
        verbose=True,
        exist_ok=True,
    )

    best_weights = Path(project) / "traffic_sign_cls" / "weights" / "best.pt"
    print(f"\n[info] Training done. Best weights saved to: {best_weights}")
    return best_weights


def export_onnx(weights_path: Path, imgsz: int):
    print("[info] Exporting to ONNX …")
    model = YOLO(str(weights_path))
    model.export(format="onnx", imgsz=imgsz, dynamic=True)
    onnx_path = weights_path.with_suffix(".onnx")
    print(f"[info] ONNX model: {onnx_path}")


# ──────────────────────────────────────────────
# 3.  Entry point
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train Indian traffic sign classifier")
    parser.add_argument("--kaggle",    action="store_true",  help="Download dataset via kaggle CLI")
    parser.add_argument("--data_zip",  type=str, default=None, help="Path to downloaded Kaggle zip")
    parser.add_argument("--raw_dir",   type=str, default="data/raw",   help="Where raw data lands")
    parser.add_argument("--split_dir", type=str, default="data/split", help="ImageNet-style split output")
    parser.add_argument("--epochs",    type=int, default=30)
    parser.add_argument("--imgsz",     type=int, default=224)
    parser.add_argument("--batch",     type=int, default=32)
    parser.add_argument("--project",   type=str, default="runs/classify")
    parser.add_argument("--export_onnx", action="store_true", help="Export best model to ONNX after training")
    args = parser.parse_args()

    raw_dir   = Path(args.raw_dir)
    split_dir = Path(args.split_dir)

    # ── Step 1: get the data ──────────────────
    if args.kaggle:
        raw_dir.mkdir(parents=True, exist_ok=True)
        download_via_kaggle(raw_dir)
    elif args.data_zip:
        raw_dir.mkdir(parents=True, exist_ok=True)
        unzip(Path(args.data_zip), raw_dir)
    else:
        if not raw_dir.exists():
            print("[error] Provide --kaggle or --data_zip <path>. Exiting.")
            return

    # ── Step 2: find class folders & build split ──
    class_dirs = find_class_dirs(raw_dir)
    if not class_dirs:
        print(f"[error] No image-containing sub-folders found under {raw_dir}. Check extraction path.")
        return
    print(f"[info] Found {len(class_dirs)} class folders.")

    build_imagenet_split(class_dirs, split_dir, val_frac=0.15, test_frac=0.10)

    # ── Step 3: train ─────────────────────────
    best_weights = train(split_dir, args.epochs, args.imgsz, args.batch, args.project)

    # ── Step 4 (optional): export ─────────────
    if args.export_onnx and best_weights.exists():
        export_onnx(best_weights, args.imgsz)

    print("\n✅  All done! Use app.py for inference.")


if __name__ == "__main__":
    main()
