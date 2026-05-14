"""Data Collection page for VSL Recognition."""

import os
import sys
import time

import av
import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.keypoint_extractor import KeypointExtractor
from src.utils import VSL_ALPHABET, VSL_COMMON_WORDS

st.set_page_config(page_title="Thu thập dữ liệu", page_icon="📝", layout="wide")
st.title("📝 Thu thập dữ liệu")

# Initialize
if "collector_extractor" not in st.session_state:
    st.session_state.collector_extractor = KeypointExtractor()
if "collected_frames" not in st.session_state:
    st.session_state.collected_frames = []
if "is_recording" not in st.session_state:
    st.session_state.is_recording = False
if "collection_count" not in st.session_state:
    st.session_state.collection_count = 0

# Sidebar
with st.sidebar:
    st.header("Cài đặt thu thập")

    data_type = st.radio("Loại ký hiệu", ["Static (Chữ cái)", "Dynamic (Từ/Cụm từ)"])

    if data_type == "Static (Chữ cái)":
        label = st.selectbox("Chọn ký hiệu", VSL_ALPHABET)
        target_samples = st.number_input("Số mẫu cần thu", 10, 500, 50)
    else:
        label = st.selectbox("Chọn từ/cụm từ", VSL_COMMON_WORDS + ["Khác..."])
        if label == "Khác...":
            label = st.text_input("Nhập tên ký hiệu")
        sequence_length = st.number_input("Số frames/sequence", 15, 60, 30)
        target_samples = st.number_input("Số sequence cần thu", 5, 100, 20)

    output_base = st.text_input("Thư mục lưu", "data/processed")

    st.markdown("---")
    st.markdown("""
    ### Hướng dẫn
    **Static**: Mỗi lần capture lưu 1 frame landmarks.
    
    **Dynamic**: Nhấn bắt đầu ghi → thực hiện ký hiệu → tự dừng khi đủ frames.
    """)


# Determine save directory
save_type = "static" if "Static" in data_type else "dynamic"
save_dir = os.path.join(output_base, save_type, label)
os.makedirs(save_dir, exist_ok=True)

# Count existing samples
existing = len([f for f in os.listdir(save_dir) if f.endswith(".npy")]) if os.path.exists(save_dir) else 0

st.info(f"**Ký hiệu**: {label} | **Loại**: {save_type} | **Đã có**: {existing} mẫu | **Mục tiêu**: {target_samples}")


def collection_callback(frame):
    """Process frame for data collection."""
    img = frame.to_ndarray(format="bgr24")
    extractor = st.session_state.collector_extractor

    landmarks, annotated = extractor.extract_with_drawing(img)
    has_hand = np.any(landmarks != 0)

    # Recording indicator
    if st.session_state.is_recording:
        st.session_state.collected_frames.append(landmarks)
        frame_count = len(st.session_state.collected_frames)

        cv2.putText(annotated, f"RECORDING: {frame_count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # Auto-stop for dynamic when enough frames
        if save_type == "dynamic" and frame_count >= sequence_length:
            seq = np.array(st.session_state.collected_frames)
            count = st.session_state.collection_count
            fname = f"sequence_{count:04d}.npy"
            np.save(os.path.join(save_dir, fname), seq)
            st.session_state.collection_count += 1
            st.session_state.collected_frames = []
            st.session_state.is_recording = False
    else:
        status = f"Ready | Collected: {st.session_state.collection_count}"
        cv2.putText(annotated, status, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Label display
    cv2.putText(annotated, f"Label: {label}", (10, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    # Hand indicator
    color = (0, 255, 0) if has_hand else (0, 0, 255)
    cv2.circle(annotated, (annotated.shape[1] - 20, 20), 10, color, -1)

    return av.VideoFrame.from_ndarray(annotated, format="bgr24")


# Layout
col1, col2 = st.columns([3, 1])

with col1:
    webrtc_ctx = webrtc_streamer(
        key="vsl-collection",
        mode=WebRtcMode.SENDRECV,
        video_frame_callback=collection_callback,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    )

with col2:
    st.markdown("### Điều khiển")

    if save_type == "static":
        if st.button("📸 Capture (Static)", use_container_width=True):
            if st.session_state.collected_frames:
                lm = st.session_state.collected_frames[-1]
                if np.any(lm != 0):
                    count = st.session_state.collection_count
                    fname = f"sample_{count:04d}.npy"
                    np.save(os.path.join(save_dir, fname), lm)
                    st.session_state.collection_count += 1
                    st.success(f"Saved: {fname}")
                else:
                    st.warning("Không phát hiện tay!")
            # Enable continuous collection
            st.session_state.is_recording = True
            time.sleep(0.1)
            if st.session_state.collected_frames:
                lm = st.session_state.collected_frames[-1]
                if np.any(lm != 0):
                    count = st.session_state.collection_count
                    fname = f"sample_{count:04d}.npy"
                    np.save(os.path.join(save_dir, fname), lm)
                    st.session_state.collection_count += 1
            st.session_state.is_recording = False
    else:
        if st.button("🔴 Bắt đầu ghi", use_container_width=True):
            st.session_state.is_recording = True
            st.session_state.collected_frames = []

        if st.button("⏹ Dừng ghi", use_container_width=True):
            st.session_state.is_recording = False
            st.session_state.collected_frames = []

    st.markdown("---")
    st.metric("Đã thu thập", st.session_state.collection_count)
    st.metric("Tổng (bao gồm cũ)", existing + st.session_state.collection_count)
    st.progress(min(1.0, (existing + st.session_state.collection_count) / max(target_samples, 1)))
