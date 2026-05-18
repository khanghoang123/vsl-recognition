"""VSL Recognition — Nhận diện Ngôn ngữ Ký hiệu Tiếng Việt

Streamlit app using VideoMAEv2-Small for real-time inference.
"""

import os
import json
import time
from pathlib import Path
from collections import deque

import streamlit as st
import numpy as np
import cv2
import torch
from torchvision.transforms import Resize, CenterCrop, Normalize
from transformers import VideoMAEForVideoClassification

st.set_page_config(
    page_title="VSL Recognition",
    page_icon="🤟",
    layout="wide",
)


@st.cache_resource
def load_model():
    """Load fine-tuned VideoMAE model."""
    model_path = os.path.join(os.path.dirname(__file__), "models", "videomae_vsl_best")

    if not os.path.exists(model_path):
        return None, None

    # Load class names
    class_names_path = os.path.join(model_path, "class_names.json")
    if os.path.exists(class_names_path):
        with open(class_names_path) as f:
            class_names = json.load(f)
    else:
        class_names = [f"class_{i}" for i in range(50)]

    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = VideoMAEForVideoClassification.from_pretrained(
        model_path, num_labels=len(class_names)
    )
    model = model.to(device)
    model.eval()

    # Use fp16 on GPU for faster inference
    if device == "cuda":
        model = model.half()

    return model, class_names


def preprocess_frames(frames: list, image_size: int = 224, num_frames: int = 16):
    """Preprocess webcam frames for VideoMAE input."""
    # Uniform sample to num_frames
    total = len(frames)
    if total >= num_frames:
        indices = np.linspace(0, total - 1, num_frames, dtype=int)
    else:
        indices = np.arange(total)
        indices = np.concatenate([indices, np.full(num_frames - total, total - 1, dtype=int)])

    sampled = [frames[i] for i in indices]

    # Convert to tensor and normalize
    normalize = Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    transformed = []
    for frame in sampled:
        # frame is BGR from OpenCV, convert to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(frame_rgb).float() / 255.0
        tensor = tensor.permute(2, 0, 1)  # (3, H, W)
        tensor = Resize(image_size + 32, antialias=True)(tensor)
        tensor = CenterCrop(image_size)(tensor)
        tensor = normalize(tensor)
        transformed.append(tensor)

    return torch.stack(transformed).unsqueeze(0)  # (1, T, 3, H, W)


@torch.no_grad()
def predict(model, video_tensor, class_names, device):
    """Run inference."""
    video_tensor = video_tensor.to(device)
    if device == "cuda":
        video_tensor = video_tensor.half()

    outputs = model(pixel_values=video_tensor)
    probs = torch.softmax(outputs.logits[0].float(), dim=0)
    top5_probs, top5_idx = torch.topk(probs, min(5, len(class_names)))

    return {
        "label": class_names[top5_idx[0].item()],
        "confidence": top5_probs[0].item(),
        "top5": [(class_names[i.item()], p.item()) for i, p in zip(top5_idx, top5_probs)],
    }


# === Main UI ===
st.title("🤟 VSL Recognition")
st.subheader("Nhận diện Ngôn ngữ Ký hiệu Tiếng Việt — VideoMAEv2")

model, class_names = load_model()

if model is None:
    st.error(
        "⚠️ Model chưa được tải! \n\n"
        "Vui lòng:\n"
        "1. Train model bằng notebook `02_train_videomae.ipynb`\n"
        "2. Copy thư mục `models/videomae_vsl_best/` vào project root\n"
        "3. Reload app"
    )
    st.markdown("""
    ### Hướng dẫn nhanh

    ```bash
    # 1. Train trên Colab/Kaggle (notebook 02)
    # 2. Download models/videomae_vsl_best/ về máy
    # 3. Chạy app
    streamlit run app.py
    ```
    """)
else:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    st.success(f"Model loaded ({len(class_names)} classes) | Device: {device}")

    st.markdown("""
    ### Pipeline
    ```
    Webcam (30fps) → Buffer 16 frames → VideoMAEv2-Small → Prediction
    ```
    """)

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("### 📹 Webcam")
        run = st.checkbox("Start Webcam", value=False)

        if run:
            FRAME_WINDOW = st.image([])
            cap = cv2.VideoCapture(0)
            frame_buffer = deque(maxlen=16)
            prediction_text = st.empty()
            prediction_bar = st.empty()

            while run:
                ret, frame = cap.read()
                if not ret:
                    st.warning("Cannot access webcam")
                    break

                frame_buffer.append(frame)
                FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

                # Predict every 16 frames
                if len(frame_buffer) == 16:
                    start = time.time()
                    video_tensor = preprocess_frames(list(frame_buffer))
                    result = predict(model, video_tensor, class_names, device)
                    latency = (time.time() - start) * 1000

                    prediction_text.markdown(
                        f"### Prediction: **{result['label']}** "
                        f"({result['confidence']:.1%}) — {latency:.0f}ms"
                    )

                    # Show top-5
                    top5_str = " | ".join(
                        [f"{name}: {prob:.1%}" for name, prob in result["top5"]]
                    )
                    prediction_bar.caption(f"Top-5: {top5_str}")

                    frame_buffer.clear()

            cap.release()

    with col2:
        st.markdown("### ℹ️ Thông tin Model")
        st.markdown(f"""
        | Thông số | Giá trị |
        |----------|---------|
        | Model | VideoMAEv2-Small |
        | Parameters | ~22M |
        | Input | 16 frames × 224×224 |
        | Classes | {len(class_names)} |
        | Device | {device} |
        """)

        st.markdown("### 📚 Classes")
        st.dataframe(
            {"Index": range(len(class_names)), "Class": class_names},
            height=300,
        )

# Footer
st.markdown("---")
st.caption(
    "VSL Recognition | VideoMAEv2-Small | Multi-VSL Dataset (WACV 2025) | "
    "End-to-end video model (no MediaPipe)"
)
