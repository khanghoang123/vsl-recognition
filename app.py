"""Streamlit app for Vietnamese Sign Language recognition.

Local deploy expects the trained model to be downloaded from Google Drive to:
    models/videomae_olympic_best/
"""

import json
import os
import tempfile
import threading
import time
import unicodedata
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import torch
from av import VideoFrame
from PIL import Image, ImageDraw, ImageFont
from streamlit_webrtc import WebRtcMode, webrtc_streamer
from torchvision.transforms import CenterCrop, Normalize, Resize
from transformers import VideoMAEForVideoClassification


APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "models" / "videomae_olympic_best"
NUM_FRAMES = 16
IMAGE_SIZE = 224
PREDICT_EVERY_SECONDS = 0.5
UPLOAD_EXTENSIONS = [".mp4", ".avi", ".mov", ".mkv", ".webm"]
BROWSER_STREAM_WIDTH = 960
BROWSER_STREAM_HEIGHT = 720
BROWSER_STREAM_FPS = 24
START_MOTION_THRESHOLD = 3.5
END_MOTION_THRESHOLD = 1.8
CONFIDENCE_THRESHOLD = 0.35
RESULT_HOLD_SECONDS = 3.0
SMOOTHING_WINDOW = 2
PREROLL_FRAMES = 6
MIN_SEGMENT_FRAMES = 12
MAX_SEGMENT_FRAMES = 48
END_SILENCE_FRAMES = 5
COOLDOWN_SECONDS = 0.8
DEBUG_DIR = APP_DIR / "tmp_analysis" / "realtime_debug"
FONT_CANDIDATES = [
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    Path("/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf"),
    Path("C:/Windows/Fonts/arial.ttf"),
    Path("C:/Windows/Fonts/tahoma.ttf"),
]


st.set_page_config(
    page_title="VSL Recognition",
    layout="wide",
)


@st.cache_resource
def load_model():
    """Load the locally downloaded fine-tuned VideoMAE model."""
    if not MODEL_PATH.exists():
        return None, None, None

    class_names_path = MODEL_PATH / "class_names.json"
    if not class_names_path.exists():
        st.error(f"Missing class_names.json in {MODEL_PATH}")
        return None, None, None

    with open(class_names_path, encoding="utf-8") as f:
        class_names = [normalize_label(name) for name in json.load(f)]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = VideoMAEForVideoClassification.from_pretrained(
        str(MODEL_PATH),
        num_labels=len(class_names),
    )
    model.to(device)
    if device == "cuda":
        model.half()
    model.eval()

    return model, class_names, device


def normalize_label(label: str) -> str:
    """Normalize decomposed Vietnamese labels for consistent display."""
    return unicodedata.normalize("NFC", label)


@st.cache_resource
def load_overlay_font(size: int):
    for font_path in FONT_CANDIDATES:
        if font_path.exists():
            return ImageFont.truetype(str(font_path), size=size)
    return ImageFont.load_default()


def draw_text_overlay(frame: np.ndarray, lines: list[tuple[str, tuple[int, int, int], int]]) -> np.ndarray:
    """Draw Unicode text on a BGR OpenCV frame using PIL."""
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image)
    draw.rectangle((8, 8, min(frame.shape[1] - 8, 500), 100), fill=(0, 0, 0))

    for text, color, y in lines:
        font = load_overlay_font(24 if y < 45 else 18)
        draw.text((16, y), text, fill=color, font=font)

    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


def preprocess_frames(frames: list[np.ndarray]):
    """Convert BGR frames to VideoMAE tensor shape (1, T, C, H, W)."""
    total = len(frames)
    if total == 0:
        raise ValueError("No frames available for inference.")

    if total >= NUM_FRAMES:
        indices = np.linspace(0, total - 1, NUM_FRAMES, dtype=int)
    else:
        indices = np.concatenate([np.arange(total), np.full(NUM_FRAMES - total, total - 1, dtype=int)])

    normalize = Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    transformed = []
    for idx in indices:
        frame_rgb = cv2.cvtColor(center_square_crop(frames[idx]), cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(frame_rgb).float() / 255.0
        tensor = tensor.permute(2, 0, 1)
        tensor = Resize(IMAGE_SIZE + 32, antialias=True)(tensor)
        tensor = CenterCrop(IMAGE_SIZE)(tensor)
        transformed.append(normalize(tensor))

    return torch.stack(transformed).unsqueeze(0)


def read_video_frames(video_path: str | Path, max_frames: int | None = None) -> tuple[list[np.ndarray], dict]:
    """Read frames from a video file using OpenCV."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    frames: list[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
        if max_frames is not None and len(frames) >= max_frames:
            break

    cap.release()

    return frames, {
        "fps": fps,
        "frame_count": frame_count if frame_count > 0 else len(frames),
        "width": width,
        "height": height,
        "duration_seconds": (frame_count / fps) if fps > 0 and frame_count > 0 else None,
    }


def center_square_crop(frame: np.ndarray) -> np.ndarray:
    """Remove black borders first, then crop to a square around the active content."""
    height, width = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    col_energy = gray.mean(axis=0)
    row_energy = gray.mean(axis=1)
    active_cols = np.where(col_energy > 8)[0]
    active_rows = np.where(row_energy > 8)[0]

    if active_cols.size > 0 and active_rows.size > 0:
        left = int(active_cols[0])
        right = int(active_cols[-1]) + 1
        top = int(active_rows[0])
        bottom = int(active_rows[-1]) + 1
        frame = frame[top:bottom, left:right]
        height, width = frame.shape[:2]

    if height == 0 or width == 0:
        return frame

    size = min(height, width)
    top = max((height - size) // 2, 0)
    left = max((width - size) // 2, 0)
    return frame[top : top + size, left : left + size]


@torch.no_grad()
def predict(model, video_tensor, class_names, device):
    """Run one VideoMAE prediction."""
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


def predict_from_frames(model, class_names, device, frames: list[np.ndarray]):
    """Run prediction from a BGR frame list and return result + latency."""
    start = time.time()
    video_tensor = preprocess_frames(frames)
    result = predict(model, video_tensor, class_names, device)
    latency_ms = (time.time() - start) * 1000
    return result, latency_ms


def save_debug_segment(frames: list[np.ndarray], fps: int = 12) -> Path | None:
    """Persist the latest detected segment for offline inspection."""
    if not frames:
        return None

    DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    output_path = DEBUG_DIR / f"segment_{int(time.time() * 1000)}.mp4"
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        return None

    for frame in frames:
        writer.write(frame)
    writer.release()
    return output_path


class BrowserVideoProcessor:
    """WebRTC processor with gesture spotting before classification."""

    def __init__(self):
        self.raw_buffer = deque(maxlen=max(NUM_FRAMES, PREROLL_FRAMES))
        self.last_result = None
        self.last_latency_ms = None
        self.last_result_timestamp = 0.0
        self.last_motion_score = 0.0
        self.recent_results = deque(maxlen=SMOOTHING_WINDOW)
        self.prev_motion_frame = None
        self.is_inferencing = False
        self.state = "idle"
        self.segment_frames: list[np.ndarray] = []
        self.quiet_frames = 0
        self.cooldown_until = 0.0
        self.last_debug_path = None
        self.lock = threading.Lock()

    def _estimate_motion(self, image: np.ndarray) -> float:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (96, 96), interpolation=cv2.INTER_AREA)
        if self.prev_motion_frame is None:
            self.prev_motion_frame = small
            return 0.0

        score = float(np.mean(cv2.absdiff(small, self.prev_motion_frame)))
        self.prev_motion_frame = small
        return score

    def _aggregate_recent_result(self):
        if not self.recent_results:
            return None

        grouped: dict[str, list[float]] = {}
        top5_lookup = {}
        for item in self.recent_results:
            grouped.setdefault(item["label"], []).append(item["confidence"])
            top5_lookup[item["label"]] = item["top5"]

        label, scores = max(grouped.items(), key=lambda kv: (len(kv[1]), float(np.mean(kv[1]))))
        return {
            "label": label,
            "confidence": float(np.mean(scores)),
            "top5": top5_lookup[label],
        }

    def _run_inference(self, frames: list[np.ndarray], timestamp: float):
        try:
            result, latency_ms = predict_from_frames(model, class_names, device, frames)
            with self.lock:
                if result["confidence"] >= CONFIDENCE_THRESHOLD:
                    self.recent_results.append(result)
                    self.last_result = result if len(self.recent_results) < SMOOTHING_WINDOW else self._aggregate_recent_result()
                    self.last_latency_ms = latency_ms
                    self.last_result_timestamp = timestamp
                self.state = "cooldown"
                self.cooldown_until = timestamp + COOLDOWN_SECONDS
        except Exception:
            with self.lock:
                self.state = "cooldown"
                self.cooldown_until = timestamp + COOLDOWN_SECONDS
        finally:
            with self.lock:
                self.is_inferencing = False

    def recv(self, frame: VideoFrame) -> VideoFrame:
        image = frame.to_ndarray(format="bgr24")
        self.raw_buffer.append(image)
        motion_score = self._estimate_motion(image)
        now = time.time()

        frames_for_inference = None
        with self.lock:
            self.last_motion_score = motion_score
            if self.state == "cooldown" and now >= self.cooldown_until:
                self.state = "idle"

            if self.state == "idle" and motion_score >= START_MOTION_THRESHOLD:
                self.state = "collecting"
                self.segment_frames = list(self.raw_buffer)[-PREROLL_FRAMES:]
                self.quiet_frames = 0

            elif self.state == "collecting":
                self.segment_frames.append(image)
                if motion_score < END_MOTION_THRESHOLD:
                    self.quiet_frames += 1
                else:
                    self.quiet_frames = 0

                segment_finished = (
                    len(self.segment_frames) >= MAX_SEGMENT_FRAMES
                    or (len(self.segment_frames) >= MIN_SEGMENT_FRAMES and self.quiet_frames >= END_SILENCE_FRAMES)
                )
                if segment_finished and not self.is_inferencing:
                    self.is_inferencing = True
                    self.state = "predicting"
                    frames_for_inference = self.segment_frames[:]
                    self.segment_frames = []
                    self.quiet_frames = 0

        if frames_for_inference is not None:
            self.last_debug_path = save_debug_segment(frames_for_inference)
            threading.Thread(
                target=self._run_inference,
                args=(frames_for_inference, now),
                daemon=True,
            ).start()

        annotated = image.copy()
        with self.lock:
            result = self.last_result
            latency_ms = self.last_latency_ms
            motion_snapshot = self.last_motion_score
            should_show_result = (
                result is not None and (now - self.last_result_timestamp) <= RESULT_HOLD_SECONDS
            )
            state_snapshot = self.state
            segment_len = len(self.segment_frames)

        cv2.rectangle(annotated, (8, 8), (min(annotated.shape[1] - 8, 420), 92), (0, 0, 0), -1)

        if should_show_result:
            label_text = f"{result['label']} ({result['confidence']:.1%})"
            latency_text = f"{latency_ms:.0f} ms" if latency_ms is not None else ""
            annotated = draw_text_overlay(
                annotated,
                [
                    (label_text, (0, 255, 0), 18),
                    (latency_text, (255, 255, 255), 52),
                    (f"{state_snapshot} | motion {motion_snapshot:.1f} | seg {segment_len}", (180, 180, 180), 76),
                ],
            )
        else:
            cv2.putText(
                annotated,
                "No active gesture",
                (16, 34),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                annotated,
                f"{state_snapshot} | motion {motion_snapshot:.1f} | seg {segment_len}",
                (16, 62),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (180, 180, 180),
                2,
                cv2.LINE_AA,
            )

        return VideoFrame.from_ndarray(annotated, format="bgr24")

    def get_snapshot(self):
        with self.lock:
            return self.last_result, self.last_latency_ms


st.title("Vietnamese Sign Language Recognition")
st.caption("VideoMAE-Small fine-tuned on Olympic AI2025. Local app only loads ./models/videomae_olympic_best.")

model, class_names, device = load_model()

if model is None:
    st.error("Model folder not found or incomplete.")
    st.markdown(
        f"""
        Train in Colab first, then download this Google Drive folder to local:

        `models/videomae_olympic_best/`

        Expected local path:

        `{MODEL_PATH}`

        Then run:

        ```bash
        streamlit run app.py
        ```
        """
    )
    st.stop()

st.success(f"Loaded {len(class_names)} classes on {device}")

info_col, classes_col = st.columns([1, 1])

with info_col:
    st.subheader("Model")
    st.table(
        {
            "Field": ["Model", "Input", "Classes", "Device", "Local path"],
            "Value": [
                "VideoMAE-Small",
                f"{NUM_FRAMES} frames x {IMAGE_SIZE}x{IMAGE_SIZE}",
                str(len(class_names)),
                device,
                str(MODEL_PATH),
            ],
        }
    )

with classes_col:
    st.subheader("Classes")
    st.dataframe({"Index": range(len(class_names)), "Class": class_names}, height=260, width="stretch")

tab_browser, tab_upload = st.tabs(["Browser webcam (WSL-friendly)", "Upload video"])

with tab_browser:
    st.markdown(
        """
        This mode is recommended when Streamlit runs inside WSL. The browser asks Windows for webcam access,
        then streams frames into the app for prediction.
        """
    )

    browser_ctx = webrtc_streamer(
        key="browser-webcam",
        mode=WebRtcMode.SENDRECV,
        media_stream_constraints={
            "video": {
                "width": {"ideal": BROWSER_STREAM_WIDTH},
                "height": {"ideal": BROWSER_STREAM_HEIGHT},
                "frameRate": {"ideal": BROWSER_STREAM_FPS},
            },
            "audio": False,
        },
        video_processor_factory=BrowserVideoProcessor,
        async_processing=True,
    )

    if browser_ctx.state.playing:
        st.info(
            "Prediction is drawn directly on top of the webcam stream. The app waits for a gesture segment, then classifies it once, which is closer to the training setup."
        )
        if DEBUG_DIR.exists():
            debug_files = sorted(DEBUG_DIR.glob("segment_*.mp4"), key=os.path.getmtime, reverse=True)
            if debug_files:
                st.caption(f"Latest debug segment: {debug_files[0]}")
    else:
        st.info("Click Start to open your browser webcam.")

with tab_upload:
    st.markdown("Upload a short gesture video to test inference without any webcam dependency.")
    uploaded_file = st.file_uploader(
        "Choose a video",
        type=[ext.lstrip(".") for ext in UPLOAD_EXTENSIONS],
        accept_multiple_files=False,
    )

    if uploaded_file is not None:
        file_suffix = Path(uploaded_file.name).suffix.lower()
        if file_suffix not in UPLOAD_EXTENSIONS:
            st.error(f"Unsupported format: {file_suffix}")
        else:
            st.video(uploaded_file)
            temp_path = None
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=file_suffix) as tmp:
                    tmp.write(uploaded_file.getbuffer())
                    temp_path = Path(tmp.name)

                frames, video_info = read_video_frames(temp_path)
                if not frames:
                    st.error("Could not decode any frames from the uploaded video.")
                else:
                    result, latency_ms = predict_from_frames(model, class_names, device, frames)
                    meta_cols = st.columns(4)
                    meta_cols[0].metric("Frames", f"{len(frames)}")
                    meta_cols[1].metric("FPS", f"{video_info['fps']:.2f}" if video_info["fps"] else "Unknown")
                    meta_cols[2].metric("Resolution", f"{video_info['width']}x{video_info['height']}")
                    meta_cols[3].metric("Latency", f"{latency_ms:.0f} ms")

                    st.markdown(f"### Prediction: **{result['label']}** ({result['confidence']:.1%})")
                    st.caption(" | ".join(f"{name}: {prob:.1%}" for name, prob in result["top5"]))
            finally:
                if temp_path is not None and temp_path.exists():
                    temp_path.unlink(missing_ok=True)
