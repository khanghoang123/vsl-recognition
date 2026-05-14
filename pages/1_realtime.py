"""Realtime VSL Recognition page using webcam."""

import os
import sys
import json
import time

import av
import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.keypoint_extractor import KeypointExtractor
from src.gesture_router import GestureRouter
from src.preprocessing import normalize_landmarks, compute_motion_features
from src.sentence_builder import SentenceBuilder
from src.utils import get_model_path

st.set_page_config(page_title="Nhận diện Realtime", page_icon="📹", layout="wide")
st.title("📹 Nhận diện Realtime")

# Sidebar settings
with st.sidebar:
    st.header("Cài đặt")
    confidence_threshold = st.slider(
        "Ngưỡng tin cậy (%)", 30, 99, 70, step=5
    ) / 100.0
    mode = st.radio(
        "Chế độ nhận diện",
        ["Static (Chữ cái)", "Dynamic (Từ/Cụm từ)", "Tự động (Gesture Router)"],
        index=2,
    )
    show_landmarks = st.checkbox("Hiển thị landmarks", value=True)
    st.markdown("---")
    if st.button("Xóa câu"):
        if "sentence_builder" in st.session_state:
            st.session_state.sentence_builder.clear_sentence()

# Initialize components
if "extractor" not in st.session_state:
    st.session_state.extractor = KeypointExtractor()
if "router" not in st.session_state:
    st.session_state.router = GestureRouter()
if "sentence_builder" not in st.session_state:
    st.session_state.sentence_builder = SentenceBuilder()
if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = {"label": None, "confidence": 0.0, "type": ""}
if "fps_counter" not in st.session_state:
    st.session_state.fps_counter = {"frames": 0, "start_time": time.time(), "fps": 0}

# Load models if available
static_model = None
dynamic_model = None
static_labels = {}
dynamic_labels = {}

static_model_path = get_model_path("static_cnn1d.keras")
dynamic_model_path = get_model_path("dynamic_bilstm_att.keras")

if os.path.exists(static_model_path):
    from src.static_classifier import StaticClassifier
    static_labels_path = get_model_path("static_labels.json")
    if os.path.exists(static_labels_path):
        with open(static_labels_path, "r", encoding="utf-8") as f:
            static_labels = json.load(f)
        static_model = StaticClassifier(static_model_path, static_labels)
        st.sidebar.success("Static model loaded")

if os.path.exists(dynamic_model_path):
    from src.dynamic_classifier import DynamicClassifier
    dynamic_labels_path = get_model_path("dynamic_labels.json")
    if os.path.exists(dynamic_labels_path):
        with open(dynamic_labels_path, "r", encoding="utf-8") as f:
            dynamic_labels = json.load(f)
        dynamic_model = DynamicClassifier(dynamic_model_path, dynamic_labels)
        st.sidebar.success("Dynamic model loaded")

if static_model is None and dynamic_model is None:
    st.warning(
        "Chưa có model nào được train. Vui lòng train model trước:\n"
        "```\n"
        "python training/train_static.py --data_dir data/processed/static\n"
        "python training/train_dynamic.py --data_dir data/processed/dynamic\n"
        "```\n\n"
        "Hoặc sử dụng trang **Thu thập dữ liệu** để tạo dataset."
    )
    st.info("Demo mode: Hiển thị MediaPipe hand tracking (không có nhận diện)")


def video_frame_callback(frame):
    """Process each video frame from webcam."""
    img = frame.to_ndarray(format="bgr24")
    extractor = st.session_state.extractor
    router = st.session_state.router

    # Extract landmarks
    if show_landmarks:
        landmarks, annotated = extractor.extract_with_drawing(img)
    else:
        landmarks = extractor.extract(img)
        annotated = img.copy()

    # Normalize
    norm_landmarks = normalize_landmarks(landmarks)

    # Add to router buffer
    router.add_frame(norm_landmarks)

    # FPS calculation
    fps_data = st.session_state.fps_counter
    fps_data["frames"] += 1
    elapsed = time.time() - fps_data["start_time"]
    if elapsed >= 1.0:
        fps_data["fps"] = fps_data["frames"] / elapsed
        fps_data["frames"] = 0
        fps_data["start_time"] = time.time()

    # Prediction
    label = None
    confidence = 0.0
    gesture_type = ""

    has_hand = np.any(landmarks != 0)
    if has_hand:
        if mode == "Static (Chữ cái)" or (mode == "Tự động (Gesture Router)" and router.get_gesture_type() == "static"):
            gesture_type = "Static"
            if static_model is not None:
                label, confidence = static_model.predict(
                    norm_landmarks.reshape(-1, 1).flatten(),
                    threshold=confidence_threshold,
                )
        elif mode == "Dynamic (Từ/Cụm từ)" or (mode == "Tự động (Gesture Router)" and router.get_gesture_type() == "dynamic"):
            gesture_type = "Dynamic"
            if dynamic_model is not None:
                sequence = router.get_sequence()
                norm_seq = normalize_landmarks(sequence)
                motion_seq = compute_motion_features(norm_seq)
                label, confidence = dynamic_model.predict(
                    motion_seq, threshold=confidence_threshold,
                )

    # Update state
    if label is not None:
        st.session_state.last_prediction = {
            "label": label, "confidence": confidence, "type": gesture_type
        }
        st.session_state.sentence_builder.add_sign(label, confidence)

    # Draw prediction on frame
    pred = st.session_state.last_prediction
    if pred["label"]:
        text = f"{pred['label']} ({pred['confidence']:.0%})"
        cv2.putText(annotated, text, (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        cv2.putText(annotated, f"Type: {pred['type']}", (10, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    # FPS
    cv2.putText(annotated, f"FPS: {fps_data['fps']:.0f}", (annotated.shape[1] - 120, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)

    # Hand detection indicator
    indicator_color = (0, 255, 0) if has_hand else (0, 0, 255)
    cv2.circle(annotated, (annotated.shape[1] - 20, 20), 10, indicator_color, -1)

    return av.VideoFrame.from_ndarray(annotated, format="bgr24")


# WebRTC streamer
col1, col2 = st.columns([3, 1])

with col1:
    webrtc_ctx = webrtc_streamer(
        key="vsl-recognition",
        mode=WebRtcMode.SENDRECV,
        video_frame_callback=video_frame_callback,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    )

with col2:
    st.markdown("### Kết quả")
    pred = st.session_state.get("last_prediction", {})
    if pred.get("label"):
        st.metric("Ký hiệu", pred["label"])
        st.progress(pred.get("confidence", 0))
        st.caption(f"Loại: {pred.get('type', 'N/A')}")
    else:
        st.info("Đưa tay vào camera để nhận diện")

    st.markdown("### Câu")
    sentence = st.session_state.sentence_builder.get_sentence()
    st.text_area("Nội dung", value=sentence, height=100, disabled=True)

# History
st.markdown("### Lịch sử nhận diện")
history = st.session_state.sentence_builder.get_history()
if history:
    for item in reversed(history[-10:]):
        st.text(f"  {item['sign']} ({item['confidence']:.0%})")
else:
    st.caption("Chưa có lịch sử")
