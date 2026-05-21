"""Streamlit app for Vietnamese Sign Language recognition.

Local deploy expects the trained model to be downloaded from Google Drive to:
    models/videomae_olympic_best/
"""

import json
import tempfile
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import torch
from av import VideoFrame
from PIL import Image, ImageDraw, ImageFont
from streamlit_webrtc import WebRtcMode, webrtc_streamer
from transformers import VideoMAEForVideoClassification

from src.inference import normalize_label, prepare_video_tensor, read_video_frames, validate_class_mapping


APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "models" / "videomae_olympic_best"
NUM_FRAMES = 16
IMAGE_SIZE = 224
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
FIXED_COUNTDOWN_SECONDS = 3.0
FIXED_DURATION_SECONDS = 1.5
FONT_CANDIDATES = [
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    Path("/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf"),
    Path("C:/Windows/Fonts/arial.ttf"),
    Path("C:/Windows/Fonts/tahoma.ttf"),
]
INFERENCE_LOCK = threading.Lock()


@dataclass(frozen=True)
class RuntimeConfig:
    """Realtime settings controlled from the Streamlit sidebar."""

    mode: str = "fixed"
    start_motion_threshold: float = START_MOTION_THRESHOLD
    end_motion_threshold: float = END_MOTION_THRESHOLD
    confidence_threshold: float = CONFIDENCE_THRESHOLD
    min_segment_frames: int = MIN_SEGMENT_FRAMES
    max_segment_frames: int = MAX_SEGMENT_FRAMES
    end_silence_frames: int = END_SILENCE_FRAMES
    result_hold_seconds: float = RESULT_HOLD_SECONDS
    smoothing_window: int = SMOOTHING_WINDOW
    preroll_frames: int = PREROLL_FRAMES
    cooldown_seconds: float = COOLDOWN_SECONDS
    fixed_countdown_seconds: float = FIXED_COUNTDOWN_SECONDS
    fixed_duration_seconds: float = FIXED_DURATION_SECONDS


st.set_page_config(
    page_title="VSL Recognition",
    layout="wide",
)


@st.cache_resource
def load_model():
    """Tải mô hình VideoMAE đã fine-tune từ máy local."""
    if not MODEL_PATH.exists():
        return None, None, None, None

    class_names_path = MODEL_PATH / "class_names.json"
    if not class_names_path.exists():
        st.error(f"Thiếu `class_names.json` trong `{MODEL_PATH}`")
        return None, None, None, None

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

    mapping_validation = validate_class_mapping(model, class_names)
    return model, class_names, device, mapping_validation


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
    """Run prediction from a BGR frame list and return result, latency, and prepared frames."""
    start = time.time()
    prepared = prepare_video_tensor(frames, num_frames=NUM_FRAMES, image_size=IMAGE_SIZE)
    with INFERENCE_LOCK:
        result = predict(model, prepared.tensor, class_names, device)
    latency_ms = (time.time() - start) * 1000
    return result, latency_ms, prepared


def render_prediction(result: dict, latency_ms: float | None = None):
    """Render a single prediction result consistently across upload and webcam."""
    st.markdown(f"### Dự đoán: **{result['label']}** ({result['confidence']:.1%})")
    if latency_ms is not None:
        st.caption(f"Thời gian suy luận: {latency_ms:.0f} ms")
    st.caption(" | ".join(f"{name}: {prob:.1%}" for name, prob in result["top5"]))


class BrowserVideoProcessor:
    """WebRTC processor for fixed-duration capture and optional motion spotting."""

    def __init__(self):
        self.config = RuntimeConfig()
        self.raw_buffer = deque(maxlen=240)
        self.last_result = None
        self.last_latency_ms = None
        self.last_prepared = None
        self.last_result_timestamp = 0.0
        self.last_motion_score = 0.0
        self.recent_results = deque(maxlen=SMOOTHING_WINDOW)
        self.prev_motion_frame = None
        self.is_inferencing = False
        self.state = "idle"
        self.segment_frames: list[np.ndarray] = []
        self.quiet_frames = 0
        self.cooldown_until = 0.0
        self.fixed_status = "idle"
        self.fixed_token = 0
        self.fixed_start_at = 0.0
        self.fixed_end_at = 0.0
        self.fixed_frames: list[np.ndarray] = []
        self.fixed_result = None
        self.fixed_latency_ms = None
        self.fixed_prepared = None
        self.fixed_error = None
        self.is_fixed_inferencing = False
        self.lock = threading.Lock()

    def update_config(self, config: RuntimeConfig):
        with self.lock:
            old_smoothing = self.config.smoothing_window
            self.config = config
            if old_smoothing != config.smoothing_window:
                recent = list(self.recent_results)[-config.smoothing_window :]
                self.recent_results = deque(recent, maxlen=config.smoothing_window)

    def request_fixed_capture(self, duration_seconds: float, countdown_seconds: float) -> int:
        """Start a countdown, then record one isolated gesture segment."""
        now = time.time()
        with self.lock:
            self.fixed_token += 1
            self.fixed_status = "countdown"
            self.fixed_start_at = now + countdown_seconds
            self.fixed_end_at = self.fixed_start_at + duration_seconds
            self.fixed_frames = []
            self.fixed_result = None
            self.fixed_latency_ms = None
            self.fixed_prepared = None
            self.fixed_error = None
            return self.fixed_token

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

    def _run_auto_inference(self, frames: list[np.ndarray], timestamp: float):
        try:
            result, latency_ms, prepared = predict_from_frames(model, class_names, device, frames)
            with self.lock:
                config = self.config
                if result["confidence"] >= config.confidence_threshold:
                    self.recent_results.append(result)
                    self.last_result = (
                        result if len(self.recent_results) < config.smoothing_window else self._aggregate_recent_result()
                    )
                    self.last_latency_ms = latency_ms
                    self.last_prepared = prepared
                    self.last_result_timestamp = timestamp
                self.state = "cooldown"
                self.cooldown_until = timestamp + config.cooldown_seconds
        except Exception:
            with self.lock:
                self.state = "cooldown"
                self.cooldown_until = timestamp + self.config.cooldown_seconds
        finally:
            with self.lock:
                self.is_inferencing = False

    def _run_fixed_inference(self, frames: list[np.ndarray], token: int):
        try:
            result, latency_ms, prepared = predict_from_frames(model, class_names, device, frames)
            with self.lock:
                if token != self.fixed_token:
                    return
                self.fixed_result = result
                self.fixed_latency_ms = latency_ms
                self.fixed_prepared = prepared
                self.fixed_status = "done"
                self.fixed_error = None
                self.last_result = result
                self.last_latency_ms = latency_ms
                self.last_prepared = prepared
                self.last_result_timestamp = time.time()
        except Exception as exc:
            with self.lock:
                if token == self.fixed_token:
                    self.fixed_status = "error"
                    self.fixed_error = str(exc)
        finally:
            with self.lock:
                self.is_fixed_inferencing = False

    def recv(self, frame: VideoFrame) -> VideoFrame:
        image = frame.to_ndarray(format="bgr24")
        self.raw_buffer.append(image)
        motion_score = self._estimate_motion(image)
        now = time.time()

        auto_frames_for_inference = None
        fixed_frames_for_inference = None
        fixed_token = None
        with self.lock:
            config = self.config
            self.last_motion_score = motion_score

            if self.fixed_status == "countdown" and now >= self.fixed_start_at:
                self.fixed_status = "recording"
                self.fixed_frames = []

            if self.fixed_status == "recording":
                self.fixed_frames.append(image)
                if now >= self.fixed_end_at and not self.is_fixed_inferencing:
                    fixed_frames_for_inference = self.fixed_frames[:] or list(self.raw_buffer)[-NUM_FRAMES:]
                    fixed_token = self.fixed_token
                    self.fixed_status = "predicting"
                    self.is_fixed_inferencing = True

            if self.state == "cooldown" and now >= self.cooldown_until:
                self.state = "idle"

            if config.mode == "auto":
                if self.state == "idle" and motion_score >= config.start_motion_threshold:
                    self.state = "collecting"
                    self.segment_frames = list(self.raw_buffer)[-config.preroll_frames :]
                    self.quiet_frames = 0

                elif self.state == "collecting":
                    self.segment_frames.append(image)
                    if motion_score < config.end_motion_threshold:
                        self.quiet_frames += 1
                    else:
                        self.quiet_frames = 0

                    segment_finished = (
                        len(self.segment_frames) >= config.max_segment_frames
                        or (
                            len(self.segment_frames) >= config.min_segment_frames
                            and self.quiet_frames >= config.end_silence_frames
                        )
                    )
                    if segment_finished and not self.is_inferencing:
                        self.is_inferencing = True
                        self.state = "predicting"
                        auto_frames_for_inference = self.segment_frames[:]
                        self.segment_frames = []
                        self.quiet_frames = 0
            elif self.state != "idle":
                self.state = "idle"
                self.segment_frames = []
                self.quiet_frames = 0

        if fixed_frames_for_inference is not None and fixed_token is not None:
            threading.Thread(
                target=self._run_fixed_inference,
                args=(fixed_frames_for_inference, fixed_token),
                daemon=True,
            ).start()

        if auto_frames_for_inference is not None:
            threading.Thread(
                target=self._run_auto_inference,
                args=(auto_frames_for_inference, now),
                daemon=True,
            ).start()

        annotated = image.copy()
        with self.lock:
            result = self.last_result
            latency_ms = self.last_latency_ms
            motion_snapshot = self.last_motion_score
            config = self.config
            should_show_result = (
                result is not None and (now - self.last_result_timestamp) <= config.result_hold_seconds
            )
            state_snapshot = self.state
            segment_len = len(self.segment_frames)
            fixed_status = self.fixed_status
            fixed_countdown_left = max(self.fixed_start_at - now, 0.0)
            fixed_recording_left = max(self.fixed_end_at - now, 0.0)

        cv2.rectangle(annotated, (8, 8), (min(annotated.shape[1] - 8, 420), 92), (0, 0, 0), -1)

        if fixed_status == "countdown":
            cv2.putText(
                annotated,
                f"Get ready: {fixed_countdown_left:.1f}s",
                (16, 38),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
        elif fixed_status == "recording":
            cv2.putText(
                annotated,
                f"Recording: {fixed_recording_left:.1f}s",
                (16, 38),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 180, 255),
                2,
                cv2.LINE_AA,
            )
        elif fixed_status == "predicting":
            cv2.putText(
                annotated,
                "Predicting...",
                (16, 38),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
        elif should_show_result:
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
            return {
                "last_result": self.last_result,
                "last_latency_ms": self.last_latency_ms,
                "last_prepared": self.last_prepared,
                "last_motion_score": self.last_motion_score,
                "state": self.state,
                "segment_len": len(self.segment_frames),
                "fixed_status": self.fixed_status,
                "fixed_token": self.fixed_token,
                "fixed_result": self.fixed_result,
                "fixed_latency_ms": self.fixed_latency_ms,
                "fixed_prepared": self.fixed_prepared,
                "fixed_error": self.fixed_error,
                "fixed_recorded_frames": len(self.fixed_frames),
            }


st.title("Nhận diện ngôn ngữ ký hiệu tiếng Việt")
st.caption("VideoMAE-Small fine-tuned trên Olympic AI2025. Ứng dụng local chỉ tải `./models/videomae_olympic_best`.")

model, class_names, device, mapping_validation = load_model()

if model is None:
    st.error("Không tìm thấy thư mục mô hình hoặc thư mục chưa đầy đủ.")
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

if not mapping_validation.ok:
    st.error("Mapping giữa class_names và model bị lệch. Dừng trước khi dự đoán để tránh hiện sai nhãn.")
    for error in mapping_validation.errors:
        st.write(f"- {error}")
    st.stop()

with st.sidebar:
    st.header("Thiết lập realtime")
    capture_mode = st.radio(
        "Chế độ webcam",
        ["Đếm ngược + fixed duration", "Tự bắt theo chuyển động"],
        index=0,
    )
    fixed_duration_seconds = st.select_slider(
        "Thời lượng ghi cố định",
        options=[1.0, 1.5, 2.0],
        value=FIXED_DURATION_SECONDS,
        help="Record one isolated sign for this many seconds after the countdown.",
    )
    fixed_countdown_seconds = st.slider("Đếm ngược", 1.0, 5.0, FIXED_COUNTDOWN_SECONDS, 0.5)

    st.divider()
    st.caption("Ngưỡng tự bắt theo chuyển động")
    start_motion_threshold = st.slider("Ngưỡng bắt đầu chuyển động", 0.5, 12.0, START_MOTION_THRESHOLD, 0.1)
    end_motion_threshold = st.slider("Ngưỡng kết thúc chuyển động", 0.2, 8.0, END_MOTION_THRESHOLD, 0.1)
    confidence_threshold = st.slider("Ngưỡng tin cậy", 0.0, 0.95, CONFIDENCE_THRESHOLD, 0.01)
    min_segment_frames = st.slider("Số frame tối thiểu của đoạn", 4, 64, MIN_SEGMENT_FRAMES, 1)
    max_segment_frames = st.slider(
        "Số frame tối đa của đoạn",
        min_segment_frames,
        96,
        max(MAX_SEGMENT_FRAMES, min_segment_frames),
        1,
    )
    end_silence_frames = st.slider("Số frame yên lặng để kết thúc", 1, 20, END_SILENCE_FRAMES, 1)
    result_hold_seconds = st.slider("Thời gian giữ kết quả", 0.5, 8.0, RESULT_HOLD_SECONDS, 0.5)

runtime_config = RuntimeConfig(
    mode="fixed" if capture_mode == "Đếm ngược + fixed duration" else "auto",
    start_motion_threshold=start_motion_threshold,
    end_motion_threshold=end_motion_threshold,
    confidence_threshold=confidence_threshold,
    min_segment_frames=min_segment_frames,
    max_segment_frames=max_segment_frames,
    end_silence_frames=end_silence_frames,
    result_hold_seconds=result_hold_seconds,
    fixed_countdown_seconds=fixed_countdown_seconds,
    fixed_duration_seconds=fixed_duration_seconds,
)

st.success(f"Đã tải {len(class_names)} lớp trên thiết bị {device}")

info_col, classes_col = st.columns([1, 1])

with info_col:
    st.subheader("Mô hình")
    st.table(
        {
            "Trường": ["Mô hình", "Đầu vào", "Số lớp", "Thiết bị", "Mapping", "Đường dẫn"],
            "Giá trị": [
                "VideoMAE-Small",
                f"{NUM_FRAMES} frames x {IMAGE_SIZE}x{IMAGE_SIZE}",
                str(len(class_names)),
                device,
                "Hợp lệ",
                str(MODEL_PATH),
            ],
        }
    )
    st.caption("Kiểm tra mapping: " + " | ".join(mapping_validation.details))

with classes_col:
    st.subheader("Các lớp")
    st.dataframe({"Index": range(len(class_names)), "Lớp": class_names}, height=260, width="stretch")

tab_browser, tab_upload = st.tabs(["Webcam trình duyệt", "Tải video lên"])

with tab_browser:
    st.markdown(
        """
        Chế độ này hợp khi chạy Streamlit trong WSL. Trình duyệt sẽ xin quyền webcam từ Windows rồi truyền khung hình vào app để dự đoán.
        """
    )
    st.caption(f"Chế độ hiện tại: {capture_mode}")

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

    processor = getattr(browser_ctx, "video_processor", None)
    if processor is not None:
        processor.update_config(runtime_config)

    if browser_ctx.state.playing:
        if runtime_config.mode == "fixed":
            st.info(
                "Chế độ demo: bấm Chụp, chờ đếm ngược, thực hiện một ký hiệu trọn vẹn rồi xem kết quả dự đoán."
            )
            capture_col, refresh_col = st.columns([1, 1])
            with capture_col:
                capture_clicked = st.button("Chụp theo thời lượng cố định", type="primary")
            with refresh_col:
                refresh_clicked = st.button("Làm mới kết quả gần nhất")

            if processor is None:
                st.warning("Bộ xử lý webcam vẫn đang khởi tạo. Chờ một chút rồi thử lại.")
            elif capture_clicked:
                token = processor.request_fixed_capture(
                    duration_seconds=runtime_config.fixed_duration_seconds,
                    countdown_seconds=runtime_config.fixed_countdown_seconds,
                )
                status_slot = st.empty()
                deadline = time.time() + runtime_config.fixed_countdown_seconds + runtime_config.fixed_duration_seconds + 12.0
                snapshot = processor.get_snapshot()
                while time.time() < deadline:
                    snapshot = processor.get_snapshot()
                    status = snapshot["fixed_status"]
                    if snapshot["fixed_token"] == token and status == "done":
                        status_slot.success("Đã chụp xong.")
                        break
                    if snapshot["fixed_token"] == token and status == "error":
                        status_slot.error(snapshot["fixed_error"] or "Chụp thất bại.")
                        break
                    status_slot.info(
                        f"Trạng thái: {status} | số frame đã ghi: {snapshot['fixed_recorded_frames']}"
                    )
                    time.sleep(0.2)
                else:
                    status_slot.warning("Hết thời gian chụp. Kiểm tra xem webcam còn đang chạy không.")

                if snapshot.get("fixed_result") is not None:
                    render_prediction(snapshot["fixed_result"], snapshot["fixed_latency_ms"])

            elif refresh_clicked and processor is not None:
                snapshot = processor.get_snapshot()
                if snapshot["fixed_result"] is None:
                    st.info("Chưa có kết quả từ chế độ thời lượng cố định.")
                else:
                    render_prediction(snapshot["fixed_result"], snapshot["fixed_latency_ms"])
        else:
            st.info(
                "Chế độ tự động sẽ chờ chuyển động, cắt một đoạn rồi dự đoán một lần. Hãy chỉnh ngưỡng ở thanh bên khi theo dõi trạng thái và mức chuyển động."
            )
            if processor is not None:
                snapshot = processor.get_snapshot()
                metric_cols = st.columns(3)
                metric_cols[0].metric("Trạng thái", snapshot["state"])
                metric_cols[1].metric("Chuyển động", f"{snapshot['last_motion_score']:.2f}")
                metric_cols[2].metric("Số frame đoạn", str(snapshot["segment_len"]))

                if st.button("Làm mới kết quả gần nhất"):
                    snapshot = processor.get_snapshot()
                    if snapshot["last_result"] is None:
                        st.info("Chưa có dự đoán nào được chấp nhận. Hãy thử giảm ngưỡng confidence hoặc ngưỡng chuyển động.")
                    else:
                        render_prediction(snapshot["last_result"], snapshot["last_latency_ms"])
    else:
        st.info("Bấm Bắt đầu để mở webcam trong trình duyệt.")

with tab_upload:
    st.markdown("Tải một video ngắn lên để kiểm tra suy luận mà không cần webcam.")
    uploaded_file = st.file_uploader(
        "Chọn video",
        type=[ext.lstrip(".") for ext in UPLOAD_EXTENSIONS],
        accept_multiple_files=False,
    )

    if uploaded_file is not None:
        file_suffix = Path(uploaded_file.name).suffix.lower()
        if file_suffix not in UPLOAD_EXTENSIONS:
            st.error(f"Định dạng không hỗ trợ: {file_suffix}")
        else:
            st.video(uploaded_file)
            temp_path = None
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=file_suffix) as tmp:
                    tmp.write(uploaded_file.getbuffer())
                    temp_path = Path(tmp.name)

                frames, video_info = read_video_frames(temp_path)
                if not frames:
                    st.error("Không giải mã được frame nào từ video đã tải lên.")
                else:
                    result, latency_ms, _prepared = predict_from_frames(model, class_names, device, frames)
                    meta_cols = st.columns(4)
                    meta_cols[0].metric("Số frame", f"{len(frames)}")
                    meta_cols[1].metric("FPS", f"{video_info['fps']:.2f}" if video_info["fps"] else "Không rõ")
                    meta_cols[2].metric("Độ phân giải", f"{video_info['width']}x{video_info['height']}")
                    meta_cols[3].metric("Độ trễ", f"{latency_ms:.0f} ms")

                    render_prediction(result, latency_ms)
            finally:
                if temp_path is not None and temp_path.exists():
                    temp_path.unlink(missing_ok=True)

