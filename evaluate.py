"""
evaluate.py — Generate metrics + confusion matrix for the report
----------------------------------------------------------------
Run AFTER training is complete.

Usage:
  python evaluate.py --weights runs/classify/traffic_sign_cls/weights/best.pt
                     --test_dir data/split/test
                     --out_dir reports/

Produces:
  reports/confusion_matrix.png
  reports/top1_top5_accuracy.txt
  reports/per_class_accuracy.csv
"""

import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from ultralytics import YOLO


def run_eval(weights: str, test_dir: str, out_dir: str, imgsz: int = 224, top_k: int = 5):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    model  = YOLO(weights)
    test_p = Path(test_dir)

    class_dirs = sorted([d for d in test_p.iterdir() if d.is_dir()])
    image_exts = {".jpg", ".jpeg", ".png", ".bmp"}

    y_true, y_top1, y_top5_correct = [], [], []
    per_class_correct = defaultdict(int)
    per_class_total   = defaultdict(int)

    print(f"[eval] Running on {len(class_dirs)} classes …")
    for cls_dir in class_dirs:
        cls_name = cls_dir.name
        images   = [f for f in cls_dir.iterdir() if f.suffix.lower() in image_exts]
        for img_path in images:
            img   = np.array(Image.open(img_path).convert("RGB"))[:, :, ::-1]
            res   = model.predict(source=img, verbose=False)
            probs = res[0].probs
            names = res[0].names

            pred_top1 = names[int(probs.top1)]
            pred_top5 = [names[int(i)] for i in probs.top5]

            y_true.append(cls_name)
            y_top1.append(pred_top1)
            y_top5_correct.append(cls_name in pred_top5)

            per_class_total[cls_name]   += 1
            if pred_top1 == cls_name:
                per_class_correct[cls_name] += 1

    # ── Overall metrics ──────────────────────
    top1_acc = np.mean(np.array(y_true) == np.array(y_top1))
    top5_acc = np.mean(y_top5_correct)

    print(f"\n  Top-1 Accuracy : {top1_acc*100:.2f}%")
    print(f"  Top-5 Accuracy : {top5_acc*100:.2f}%")

    with open(out / "top1_top5_accuracy.txt", "w") as f:
        f.write(f"Top-1 Accuracy: {top1_acc*100:.2f}%\n")
        f.write(f"Top-5 Accuracy: {top5_acc*100:.2f}%\n")

    # ── Per-class accuracy ────────────────────
    rows = []
    for cls in sorted(per_class_total.keys()):
        acc = per_class_correct[cls] / per_class_total[cls] * 100
        rows.append({"Class": cls, "Correct": per_class_correct[cls],
                     "Total": per_class_total[cls], "Accuracy (%)": round(acc, 1)})
    df = pd.DataFrame(rows)
    df.to_csv(out / "per_class_accuracy.csv", index=False)
    print(f"[eval] Per-class CSV saved to {out/'per_class_accuracy.csv'}")

    # ── Confusion matrix (top 20 classes by frequency) ──
    classes_sorted = sorted(per_class_total, key=lambda c: -per_class_total[c])
    top_classes = classes_sorted[:20]

    mask_true = np.isin(y_true, top_classes)
    yt_filtered = np.array(y_true)[mask_true]
    yp_filtered = np.array(y_top1)[mask_true]

    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(yt_filtered, yp_filtered, labels=top_classes)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

    fig, ax = plt.subplots(figsize=(16, 13))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", xticklabels=top_classes,
                yticklabels=top_classes, cmap="Blues", ax=ax, linewidths=0.3)
    ax.set_title("Normalised Confusion Matrix (Top-20 Classes)", fontsize=14, pad=14)
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("True", fontsize=11)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    fig.savefig(out / "confusion_matrix.png", dpi=150)
    plt.close()
    print(f"[eval] Confusion matrix saved to {out/'confusion_matrix.png'}")

    # ── Accuracy bar chart ────────────────────
    df_sorted = df.sort_values("Accuracy (%)", ascending=True).tail(30)
    fig2, ax2 = plt.subplots(figsize=(10, 8))
    ax2.barh(df_sorted["Class"], df_sorted["Accuracy (%)"], color="steelblue")
    ax2.set_xlabel("Accuracy (%)")
    ax2.set_title("Per-class Accuracy (Top 30 classes)")
    ax2.axvline(x=top1_acc * 100, color="red", linestyle="--", label=f"Mean = {top1_acc*100:.1f}%")
    ax2.legend()
    plt.tight_layout()
    fig2.savefig(out / "per_class_accuracy.png", dpi=150)
    plt.close()
    print(f"[eval] Per-class accuracy chart saved to {out/'per_class_accuracy.png'}")

    print("\n✅  Evaluation complete. Check the reports/ directory.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights",  type=str, default="runs/classify/traffic_sign_cls/weights/best.pt")
    parser.add_argument("--test_dir", type=str, default="data/split/test")
    parser.add_argument("--out_dir",  type=str, default="reports")
    parser.add_argument("--imgsz",    type=int, default=224)
    args = parser.parse_args()
    run_eval(args.weights, args.test_dir, args.out_dir, args.imgsz)


if __name__ == "__main__":
    main()
