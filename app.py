"""
app.py — Indian Traffic Sign Classifier  (T5.1 — Road Safety & Traffic Vision)
--------------------------------------------------------------------------------
Streamlit app:
  • Upload an image  → YOLOv8-cls predicts the traffic sign class
  • Upload a short video → frame-by-frame inference, annotated frames
  • Outputs annotated image/frames + count summary table

Usage:
  streamlit run app.py

Make sure you have trained weights at runs/classify/traffic_sign_cls/weights/best.pt
OR pass a custom path via the sidebar.

Author: <your name>
"""

import tempfile
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

# ──────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Indian Traffic Sign Classifier",
    page_icon="🚦",
    layout="wide",
)

# ──────────────────────────────────────────────
# Sidebar — model selection
# ──────────────────────────────────────────────
st.sidebar.title("⚙️ Settings")

DEFAULT_WEIGHTS = "runs/traffic_sign_cls/weights/best.pt"
weights_path = st.sidebar.text_input("Model weights path", value=DEFAULT_WEIGHTS)
conf_threshold = st.sidebar.slider("Confidence threshold", 0.1, 1.0, 0.5, 0.05)
top_k = st.sidebar.slider("Show top-K predictions", 1, 5, 3)
max_video_frames = st.sidebar.number_input("Max frames to process (video)", min_value=10, max_value=500, value=60)

st.sidebar.markdown("---")
st.sidebar.info(
    "**Dataset:** [Indian Traffic Signs (85 classes)](https://www.kaggle.com/datasets/"
    "sarangdilipjodh/indian-traffic-signs-prediction85-classes)\n\n"
    "**Model:** YOLOv8n-cls fine-tuned"
)

# ──────────────────────────────────────────────
# Load model (cached so it only loads once)
# ──────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model …")
def load_model(path: str) -> YOLO:
    return YOLO(path)


# ──────────────────────────────────────────────
# Inference helpers
# ──────────────────────────────────────────────

def predict_image(model: YOLO, img: np.ndarray, conf: float, k: int):
    """
    Run classification on a single BGR numpy image.
    Returns list of (class_name, confidence) sorted by confidence desc.
    """
    results = model.predict(source=img, verbose=False, conf=conf)
    result = results[0]

    # probs is a Probs object
    top_indices = result.probs.top5[:k]
    top_confs   = result.probs.top5conf.tolist()[:k]
    names       = result.names  # dict {int: str}

    preds = [(names[idx], float(conf_val)) for idx, conf_val in zip(top_indices, top_confs)]
    return preds


def annotate_image(img_pil: Image.Image, preds: list) -> Image.Image:
    """Draws top prediction label + confidence bar onto a PIL image."""
    draw = ImageDraw.Draw(img_pil)
    w, h = img_pil.size

    # semi-transparent overlay at the bottom
    overlay_h = min(50 + 28 * len(preds), h // 2)
    overlay = Image.new("RGBA", (w, overlay_h), (0, 0, 0, 170))
    img_pil.paste(overlay, (0, h - overlay_h), overlay)

    # try to get a decent font; fall back to default
    try:
        font_big  = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
    except Exception:
        font_big   = ImageFont.load_default()
        font_small = ImageFont.load_default()

    y_start = h - overlay_h + 6
    for i, (cls_name, conf_val) in enumerate(preds):
        color = (255, 220, 0) if i == 0 else (200, 200, 200)
        font  = font_big if i == 0 else font_small
        text  = f"{'★ ' if i==0 else f'{i+1}. '}{cls_name}  {conf_val*100:.1f}%"
        draw.text((10, y_start + i * 28), text, fill=color, font=font)

    return img_pil


# ──────────────────────────────────────────────
# Main UI
# ──────────────────────────────────────────────
st.title("🚦 Indian Traffic Sign Classifier")
st.markdown(
    "Upload an **image** or a **short video clip** and the model will identify "
    "Indian traffic signs (85 classes) using a fine-tuned **YOLOv8n-cls** model."
)

# Load model
model = None
if Path(weights_path).exists():
    try:
        model = load_model(weights_path)
        st.sidebar.success("✅ Model loaded")
    except Exception as e:
        st.sidebar.error(f"Failed to load model: {e}")
else:
    st.sidebar.warning("⚠️ Weights not found. Train the model first (`python train.py`).")

# ──────────────────────────────────────────────
# File uploader
# ──────────────────────────────────────────────
uploaded = st.file_uploader(
    "Upload image or video",
    type=["jpg", "jpeg", "png", "bmp", "mp4", "avi", "mov", "mkv"],
)

if uploaded is None:
    st.info("👆 Upload a file to get started.")
    st.stop()

if model is None:
    st.error("Model not loaded. Check the weights path in the sidebar.")
    st.stop()

file_type = uploaded.type  # e.g. "image/jpeg" or "video/mp4"

# ══════════════════════════════════════════════
# IMAGE mode
# ══════════════════════════════════════════════
if file_type.startswith("image"):
    img_pil = Image.open(uploaded).convert("RGB")
    img_np  = np.array(img_pil)[:, :, ::-1]   # RGB → BGR for ultralytics

    with st.spinner("Running inference …"):
        preds = predict_image(model, img_np, conf_threshold, top_k)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📷 Annotated Output")
        annotated = annotate_image(img_pil.copy().convert("RGBA"), preds).convert("RGB")
        st.image(annotated, use_container_width=True)

    with col2:
        st.subheader("📊 Prediction Summary")
        if preds:
            top_cls, top_conf = preds[0]
            st.metric("Top Prediction", top_cls, f"{top_conf*100:.1f}% confidence")

            st.markdown("**All top predictions:**")
            rows = [{"Rank": i + 1, "Class": cls, "Confidence": f"{c*100:.1f}%"}
                    for i, (cls, c) in enumerate(preds)]
            st.table(rows)
        else:
            st.warning("No prediction above the confidence threshold.")

# ══════════════════════════════════════════════
# VIDEO mode
# ══════════════════════════════════════════════
elif file_type.startswith("video"):
    st.info(f"Processing up to **{max_video_frames}** frames …")

    # write to temp file so OpenCV can read it
    with tempfile.NamedTemporaryFile(suffix=Path(uploaded.name).suffix, delete=False) as tmp:
        tmp.write(uploaded.read())
        tmp_path = tmp.name

    cap = cv2.VideoCapture(tmp_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_orig     = cap.get(cv2.CAP_PROP_FPS) or 25

    # evenly sample frames up to max_video_frames
    frame_indices = np.linspace(0, total_frames - 1, min(max_video_frames, total_frames), dtype=int)
    frame_set = set(frame_indices.tolist())

    annotated_frames = []
    all_top_preds    = []

    progress = st.progress(0, text="Processing frames …")
    frame_idx = 0
    processed = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx in frame_set:
            preds = predict_image(model, frame, conf_threshold, top_k)
            if preds:
                all_top_preds.append(preds[0][0])   # top class name
                frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert("RGBA")
                annotated  = annotate_image(frame_pil, preds).convert("RGB")
                annotated_frames.append(np.array(annotated))
            processed += 1
            progress.progress(processed / len(frame_set), text=f"Frame {processed}/{len(frame_set)}")
        frame_idx += 1

    cap.release()
    progress.empty()

    # ── Results ────────────────────────────────
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("🎞️ Sample Annotated Frames")
        # show up to 6 evenly-spaced annotated frames
        show_n = min(6, len(annotated_frames))
        display_indices = np.linspace(0, len(annotated_frames) - 1, show_n, dtype=int)
        cols = st.columns(3)
        for i, di in enumerate(display_indices):
            cols[i % 3].image(annotated_frames[di], use_container_width=True)

    with col2:
        st.subheader("📊 Detection Summary")
        if all_top_preds:
            counts = Counter(all_top_preds)
            st.metric("Unique sign classes detected", len(counts))
            st.metric("Total frames analysed", processed)

            st.markdown("**Sign frequency table:**")
            rows = [{"Traffic Sign": cls, "Frames": cnt, "% Frames": f"{cnt/processed*100:.1f}%"}
                    for cls, cnt in counts.most_common()]
            st.table(rows)
        else:
            st.warning("No confident detections found in the video.")

else:
    st.error("Unsupported file type.")
