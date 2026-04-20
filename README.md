# 🚦 Indian Traffic Sign Classifier — T5.1

A YOLOv8n-cls model fine-tuned to recognise **85 Indian traffic sign classes** from photos or short video clips. Built as part of SMAI Assignment 3, Topic T5 (Road Safety & Traffic Vision).

---

## Demo

> Deployed on Streamlit Community Cloud / HuggingFace Spaces — _link here_

---

## Project Structure

```
indian_traffic_sign_classifier/
├── train.py           # Dataset prep + YOLOv8 training
├── app.py             # Streamlit inference app
├── requirements.txt
└── README.md
```

---

## Dataset

**Indian Traffic Signs — 85 Classes**  
[kaggle.com/datasets/sarangdilipjodh/indian-traffic-signs-prediction85-classes](https://www.kaggle.com/datasets/sarangdilipjodh/indian-traffic-signs-prediction85-classes)

- ~85 sign categories (stop, no-entry, speed limits, directional signs, etc.)
- Images scraped from Indian roads

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Download & train

**Option A — via Kaggle CLI** (recommended, needs `~/.kaggle/kaggle.json`):
```bash
python train.py --kaggle --epochs 30 --imgsz 224
```

**Option B — local zip**:
```bash
# Download the zip from Kaggle manually, then:
python train.py --data_zip /path/to/archive.zip --epochs 30
```

Trained weights will be saved to:
```
runs/classify/traffic_sign_cls/weights/best.pt
```

### 3. Run the Streamlit app
```bash
streamlit run app.py
```

Then open [http://localhost:8501](http://localhost:8501) in your browser.

---

## How it works

| Step | Detail |
|------|--------|
| **Base model** | `yolov8n-cls.pt` — 3 M params, pretrained on ImageNet |
| **Fine-tuning** | AdamW, 30 epochs, early stopping (patience=10), augmentation enabled |
| **Input** | 224×224 RGB image |
| **Output** | Top-K class predictions with confidence scores |
| **App** | Streamlit — supports single image + short video (frame sampling) |

---

## Training tips

- Start with `--epochs 30` and check for early stopping.
- If val accuracy plateaus early, try `--lr0 5e-4` or increase `--imgsz 320`.
- The dataset can be class-imbalanced; monitor per-class accuracy in `runs/classify/*/results.csv`.

---

## Results (example — fill in after training)

| Metric | Value |
|--------|-------|
| Top-1 Accuracy (val) | — |
| Top-5 Accuracy (val) | — |
| Inference speed | — ms/image (CPU) |

---

## Deployment

**HuggingFace Spaces (free)**:
1. Push this repo + trained weights to a HF Space with `SDK: streamlit`.
2. Add `best.pt` to the repo (≤ 100 MB) or load from HF Hub.

**Streamlit Community Cloud**:
1. Push to GitHub.
2. Connect at [share.streamlit.io](https://share.streamlit.io).
3. Set `KAGGLE_USERNAME` / `KAGGLE_KEY` as secrets if you want on-the-fly download.

---

## References

- Jocher, G. et al. (2023). *Ultralytics YOLOv8*. https://github.com/ultralytics/ultralytics  
- Jodh, S. (2022). *Indian Traffic Signs Prediction (85 Classes)*. Kaggle.
