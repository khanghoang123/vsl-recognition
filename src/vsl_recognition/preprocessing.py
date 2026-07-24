"""Shared video loading and preprocessing for training, evaluation, and inference."""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision.transforms import (
    CenterCrop,
    ColorJitter,
    Normalize,
    RandomResizedCrop,
    Resize,
)
from torchvision.transforms.functional import (
    adjust_brightness,
    adjust_contrast,
    adjust_saturation,
)

VIDEO_EXTENSIONS = {".avi", ".mkv", ".mov", ".mp4", ".webm"}
IMAGE_MEAN = [0.485, 0.456, 0.406]
IMAGE_STD = [0.229, 0.224, 0.225]


@dataclass(frozen=True)
class PreparedVideo:
    """Model input and the sampled RGB frames used to create it."""

    tensor: torch.Tensor
    sampled_frames_rgb: list[np.ndarray]
    indices: list[int]


def normalize_label(label: str) -> str:
    """Normalize Vietnamese text for display and mapping checks."""
    return unicodedata.normalize("NFC", label)


def uniform_indices(total_frames: int, num_frames: int = 16) -> np.ndarray:
    """Uniformly sample a fixed number of indices, padding short clips."""
    if total_frames <= 0:
        raise ValueError("A video must contain at least one frame.")
    if num_frames <= 0:
        raise ValueError("num_frames must be positive.")
    if total_frames >= num_frames:
        return np.linspace(0, total_frames - 1, num_frames, dtype=int)
    padding = np.full(num_frames - total_frames, total_frames - 1, dtype=int)
    return np.concatenate([np.arange(total_frames), padding])


def is_bad_frame(frame_rgb: np.ndarray) -> bool:
    """Detect blank or near-solid technical frames."""
    if frame_rgb is None or frame_rgb.size == 0:
        return True
    gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
    mean = float(gray.mean())
    std = float(gray.std())
    return std < 3.0 or mean < 3.0 or mean > 252.0


def filter_bad_frames(frames_rgb: np.ndarray, min_valid_frames: int = 8) -> np.ndarray:
    """Remove technical frames, falling back when too little content remains."""
    valid = [frame for frame in frames_rgb if not is_bad_frame(frame)]
    if len(valid) < min_valid_frames:
        return frames_rgb
    return np.asarray(valid)


def read_video_rgb(video_path: str | Path) -> tuple[np.ndarray, dict[str, float | int | None]]:
    """Decode all video frames with OpenCV."""
    path = Path(video_path)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    frames: list[np.ndarray] = []

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()

    if not frames:
        raise RuntimeError(f"No frames decoded from: {path}")

    count = len(frames)
    return np.asarray(frames), {
        "frame_count": count,
        "fps": fps,
        "width": width,
        "height": height,
        "duration_seconds": count / fps if fps > 0 else None,
    }


class VideoTransform:
    """Apply spatial transforms consistently across all frames in a clip."""

    def __init__(self, mode: str = "eval", image_size: int = 224) -> None:
        if mode not in {"train", "eval"}:
            raise ValueError("mode must be 'train' or 'eval'.")
        self.mode = mode
        self.image_size = image_size
        self.normalize = Normalize(IMAGE_MEAN, IMAGE_STD)

    def __call__(self, frames_rgb: np.ndarray) -> torch.Tensor:
        video = torch.from_numpy(frames_rgb).float().div(255.0).permute(0, 3, 1, 2)
        transformed: list[torch.Tensor] = []

        if self.mode == "train":
            i, j, height, width = RandomResizedCrop.get_params(
                video[0],
                scale=(0.85, 1.0),
                ratio=(0.9, 1.1),
            )
            _, brightness, contrast, saturation, _ = ColorJitter.get_params(
                brightness=(0.9, 1.1),
                contrast=(0.9, 1.1),
                saturation=(0.9, 1.1),
                hue=None,
            )
            resize = Resize((self.image_size, self.image_size), antialias=True)
            for frame in video:
                frame = resize(frame[:, i : i + height, j : j + width])
                frame = adjust_brightness(frame, brightness)
                frame = adjust_contrast(frame, contrast)
                frame = adjust_saturation(frame, saturation)
                transformed.append(self.normalize(frame))
        else:
            resize = Resize(self.image_size + 32, antialias=True)
            crop = CenterCrop(self.image_size)
            for frame in video:
                transformed.append(self.normalize(crop(resize(frame))))

        return torch.stack(transformed)


def prepare_video(
    frames_rgb: np.ndarray,
    *,
    num_frames: int = 16,
    image_size: int = 224,
    remove_bad_frames: bool = True,
) -> PreparedVideo:
    """Prepare a decoded RGB clip using the evaluation-time training pipeline."""
    working_frames = (
        filter_bad_frames(frames_rgb) if remove_bad_frames else np.asarray(frames_rgb)
    )
    indices = uniform_indices(len(working_frames), num_frames)
    sampled = working_frames[indices]
    tensor = VideoTransform(mode="eval", image_size=image_size)(sampled)
    return PreparedVideo(
        tensor=tensor,
        sampled_frames_rgb=[frame for frame in sampled],
        indices=[int(index) for index in indices],
    )


def build_video_index(data_root: str | Path) -> dict[str, Path]:
    """Index video files by globally unique filename."""
    root = Path(data_root)
    files = [path for path in root.rglob("*") if path.suffix.lower() in VIDEO_EXTENSIONS]
    index: dict[str, Path] = {}
    duplicates: set[str] = set()
    for path in files:
        if path.name in index:
            duplicates.add(path.name)
        index[path.name] = path
    if duplicates:
        names = ", ".join(sorted(duplicates)[:5])
        raise ValueError(f"Video filenames are not unique: {names}")
    return index


def resolve_video_path(item: dict, index: dict[str, Path]) -> Path:
    """Resolve a metadata row without depending on its original Drive path."""
    video_name = item.get("video_name") or Path(item["path"]).name
    try:
        return index[video_name]
    except KeyError as exc:
        raise FileNotFoundError(f"Video not found: {video_name}") from exc
