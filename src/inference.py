"""Inference pipeline for Vietnamese Sign Language recognition."""

import json
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from transformers import VideoMAEForVideoClassification

from .dataset import VideoTransform, load_video


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
                class_names = json.load(f)
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
