"""Inference pipeline for Vietnamese Sign Language recognition."""

import json
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from torchvision.transforms import CenterCrop, Normalize, Resize
from transformers import VideoMAEForVideoClassification

from .dataset import VideoTransform, load_video


@dataclass
class PreparedVideo:
    """Batched VideoMAE tensor plus the exact sampled frames used to build it."""

    tensor: torch.Tensor
    sampled_frames_rgb: list[np.ndarray]
    indices: list[int]


@dataclass
class MappingValidation:
    """Result of checking class_names.json against model config mappings."""

    ok: bool
    errors: list[str]
    details: list[str]


def normalize_label(label: str) -> str:
    """Normalize decomposed Vietnamese labels for stable display/comparison."""
    return unicodedata.normalize("NFC", label)


def sample_frame_indices(total_frames: int, num_frames: int = 16) -> np.ndarray:
    """Return the uniform frame indices used by training and local inference."""
    if total_frames <= 0:
        raise ValueError("No frames available for inference.")
    if total_frames >= num_frames:
        return np.linspace(0, total_frames - 1, num_frames, dtype=int)
    return np.concatenate([np.arange(total_frames), np.full(num_frames - total_frames, total_frames - 1, dtype=int)])


def center_square_crop(frame: np.ndarray, active_threshold: float = 8.0) -> np.ndarray:
    """Remove black borders first, then crop to a square around active content."""
    height, width = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    col_energy = gray.mean(axis=0)
    row_energy = gray.mean(axis=1)
    active_cols = np.where(col_energy > active_threshold)[0]
    active_rows = np.where(row_energy > active_threshold)[0]

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


def prepare_video_tensor(
    frames_bgr: list[np.ndarray],
    num_frames: int = 16,
    image_size: int = 224,
) -> PreparedVideo:
    """Convert BGR frames to a batched VideoMAE tensor and keep sampled debug frames."""
    indices = sample_frame_indices(len(frames_bgr), num_frames)
    normalize = Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    transformed = []
    sampled_frames_rgb = []
    resize = Resize(image_size + 32, antialias=True)
    crop = CenterCrop(image_size)

    for idx in indices:
        frame_rgb = cv2.cvtColor(center_square_crop(frames_bgr[int(idx)]), cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(frame_rgb).float() / 255.0
        tensor = tensor.permute(2, 0, 1)
        tensor = crop(resize(tensor))
        sampled = (tensor.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
        sampled_frames_rgb.append(sampled)
        transformed.append(normalize(tensor))

    return PreparedVideo(
        tensor=torch.stack(transformed).unsqueeze(0),
        sampled_frames_rgb=sampled_frames_rgb,
        indices=[int(i) for i in indices],
    )


def read_video_frames(video_path: str | Path, max_frames: int | None = None) -> tuple[list[np.ndarray], dict]:
    """Read BGR frames from a video file using OpenCV."""
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


def validate_class_mapping(model, class_names: list[str]) -> MappingValidation:
    """Check class_names.json order against model config id2label/label2id."""
    normalized_names = [normalize_label(name) for name in class_names]
    errors: list[str] = []
    details: list[str] = []

    config_num_labels = getattr(model.config, "num_labels", None)
    classifier = getattr(model, "classifier", None)
    classifier_out = getattr(classifier, "out_features", None)

    if config_num_labels is not None and int(config_num_labels) != len(normalized_names):
        errors.append(f"model.config.num_labels={config_num_labels} but class_names has {len(normalized_names)} labels.")
    if classifier_out is not None and int(classifier_out) != len(normalized_names):
        errors.append(f"classifier.out_features={classifier_out} but class_names has {len(normalized_names)} labels.")

    id2label = getattr(model.config, "id2label", None) or {}
    if id2label:
        missing_ids = [i for i in range(len(normalized_names)) if i not in id2label and str(i) not in id2label]
        if missing_ids:
            errors.append(f"id2label is missing ids: {missing_ids[:10]}")
        for i, expected in enumerate(normalized_names):
            raw_label = id2label.get(i, id2label.get(str(i)))
            if raw_label is None:
                continue
            actual = normalize_label(str(raw_label))
            if actual != expected:
                errors.append(f"id2label[{i}]={actual!r} but class_names[{i}]={expected!r}.")
                break
        details.append(f"id2label entries: {len(id2label)}")

    label2id = getattr(model.config, "label2id", None) or {}
    if label2id:
        normalized_label2id = {normalize_label(str(label)): int(idx) for label, idx in label2id.items()}
        for i, label in enumerate(normalized_names):
            if normalized_label2id.get(label) != i:
                errors.append(f"label2id[{label!r}]={normalized_label2id.get(label)} but expected {i}.")
                break
        details.append(f"label2id entries: {len(label2id)}")

    details.append(f"class_names entries: {len(normalized_names)}")
    if classifier_out is not None:
        details.append(f"classifier outputs: {classifier_out}")

    return MappingValidation(ok=not errors, errors=errors, details=details)


class VSLRecognizer:
    """VideoMAE recognizer for file-based or frame-buffer inference."""

    def __init__(
        self,
        model_path: str,
        class_names: Optional[list[str]] = None,
        num_frames: int = 16,
        device: Optional[str] = None,
        use_fp16: bool = True,
    ):
        self.model_path = Path(model_path)
        self.num_frames = num_frames
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_fp16 = use_fp16 and self.device == "cuda"
        self.transform = VideoTransform(mode="eval")

        if class_names is None:
            class_names_path = self.model_path / "class_names.json"
            with open(class_names_path, encoding="utf-8") as f:
                class_names = [normalize_label(label) for label in json.load(f)]
        self.class_names = class_names

        self.model = VideoMAEForVideoClassification.from_pretrained(
            str(self.model_path),
            num_labels=len(self.class_names),
        )
        self.model.to(self.device)
        if self.use_fp16:
            self.model.half()
        self.model.eval()

    @torch.no_grad()
    def _predict_tensor(self, video_tensor: torch.Tensor) -> dict:
        start = time.time()
        video_tensor = video_tensor.unsqueeze(0).to(self.device)
        if self.use_fp16:
            video_tensor = video_tensor.half()

        outputs = self.model(pixel_values=video_tensor)
        probs = torch.softmax(outputs.logits[0].float(), dim=0)
        top5_probs, top5_indices = torch.topk(probs, min(5, len(self.class_names)))

        return {
            "label": self.class_names[top5_indices[0].item()],
            "confidence": top5_probs[0].item(),
            "top5": [
                {"label": self.class_names[idx.item()], "prob": prob.item()}
                for idx, prob in zip(top5_indices, top5_probs)
            ],
            "latency_ms": (time.time() - start) * 1000,
        }

    def predict_video(self, video_path: str) -> dict:
        """Predict a sign class from a video file."""
        frames = load_video(video_path, self.num_frames)
        return self._predict_tensor(self.transform(frames))

    def predict_frames(self, frames: np.ndarray) -> dict:
        """Predict a sign class from RGB frames with shape (T, H, W, 3)."""
        total = len(frames)
        if total >= self.num_frames:
            indices = np.linspace(0, total - 1, self.num_frames, dtype=int)
        else:
            indices = np.concatenate([np.arange(total), np.full(self.num_frames - total, total - 1, dtype=int)])
        return self._predict_tensor(self.transform(frames[indices]))
