# Indian Traffic Sign Classifier (T5.1)

A YOLOv8n-cls model fine-tuned to recognise **84 Indian traffic sign classes** from photos or short video clips.  
Built as part of SMAI Assignment 3, Topic T5 (Road Safety & Traffic Vision).

## Project Structure

```
├── train.py             # Dataset download, split, and YOLOv8 training
├── evaluate.py          # Test-set evaluation, confusion matrix, accuracy charts
├── app.py               # Streamlit inference app (image + video)
├── generate_report.py   # Builds the final PDF report with figures
├── requirements.txt     # Python dependencies
├── data/
│   ├── raw/             # Extracted Kaggle dataset
│   └── split/           # ImageNet-style train/val/test folders
├── runs/
│   └── traffic_sign_cls/  # Training outputs, weights, curves
├── reports/             # Evaluation outputs (charts, CSV, accuracy)
└── Screenshots/         # Streamlit app screenshots for the report
```

## Dataset

**Indian Traffic Signs — 85 Classes** (84 retained after filtering)  
[Kaggle link](https://www.kaggle.com/datasets/sarangdilipjodh/indian-traffic-signs-prediction85-classes)

| Split      | Classes | Images |
|------------|---------|--------|
| Train      | 84      | 4,322  |
| Validation | 84      | 793    |
| Test       | 84      | 532    |
| **Total**  | **84**  | **5,647** |

Classes with fewer than 5 images are filtered out during preparation, bringing the count from 85 down to 84.

## Results

| Metric             | Value   |
|--------------------|---------|
| Top-1 Accuracy     | 93.98%  |
| Top-5 Accuracy     | 99.62%  |

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Download & train

**Option A — Kaggle CLI** (needs `~/.kaggle/kaggle.json`):
```bash
python train.py --kaggle --epochs 30 --imgsz 320
```

**Option B — local zip** (download the zip from Kaggle manually):
```bash
python train.py --data_zip path/to/archive.zip --epochs 30 --imgsz 320
```

Trained weights are saved to:
```
runs/traffic_sign_cls/weights/best.pt
```

### 3. Evaluate on the test set

```bash
python evaluate.py --weights runs/traffic_sign_cls/weights/best.pt \
                   --test_dir data/split/test \
                   --out_dir reports/
```

This produces confusion matrix, per-class accuracy CSV, and accuracy charts inside `reports/`.

### 4. Launch the Streamlit app

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser. The app supports:

- **Image mode** — upload a photo of a traffic sign, get top-K predictions with confidence scores.
- **Video mode** — upload a short clip, the app samples frames and shows annotated results with a frequency table.

Use the sidebar to adjust the confidence threshold, number of predictions, and model weights path.

### 5. Generate the PDF report

```bash
python generate_report.py
```

Produces `report_final.pdf` with training curves, evaluation charts, and Streamlit screenshots.

## How It Works

| Step | Detail |
|------|--------|
| Base model | `yolov8n-cls.pt` — ~3M params, pretrained on ImageNet |
| Fine-tuning | AdamW, lr=1e-3, 30 epochs, patience=15, augmentation enabled |
| Input | 320×320 RGB image |
| Output | Top-K class predictions with confidence scores |
| App | Streamlit — single image + short video (uniform frame sampling) |

## References

- Jocher, G. et al. (2023). *Ultralytics YOLOv8*. https://github.com/ultralytics/ultralytics
- Jodh, S. (2022). *Indian Traffic Signs Prediction (85 Classes)*. Kaggle.
