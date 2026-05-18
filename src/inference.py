"""Inference pipeline for VSL recognition."""

import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from transformers import VideoMAEForVideoClassification

from .dataset import load_video, VideoTransform


class VSLRecognizer:
    """Real-time Vietnamese Sign Language recognizer using VideoMAEv2."""

    def __init__(
        self,
        model_path: str,
        class_names: list,
        num_frames: int = 16,
        device: Optional[str] = None,
    ):
        self.num_frames = num_frames
        self.class_names = class_names
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.transform = VideoTransform(mode="eval")

        # Load model
        self.model = VideoMAEForVideoClassification.from_pretrained(
            model_path,
            num_labels=len(class_names),
            ignore_mismatched_sizes=True,
        )
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def predict_video(self, video_path: str) -> dict:
        """Predict sign from video file.
        
        Returns:
            dict with 'label', 'confidence', 'top5', 'latency_ms'.
        """
        start = time.time()

        frames = load_video(video_path, self.num_frames)
        video_tensor = self.transform(frames)
        video_tensor = video_tensor.unsqueeze(0).to(self.device)
        # (1, T, C, H, W) -> (1, T, C, H, W) for VideoMAE
        # VideoMAE expects pixel_values of shape (batch, num_frames, channels, height, width)

        outputs = self.model(pixel_values=video_tensor)
        logits = outputs.logits[0]
        probs = torch.softmax(logits, dim=0)

        top5_probs, top5_indices = torch.topk(probs, min(5, len(self.class_names)))

        latency = (time.time() - start) * 1000

        return {
            "label": self.class_names[top5_indices[0].item()],
            "confidence": top5_probs[0].item(),
            "top5": [
                {"label": self.class_names[idx.item()], "prob": prob.item()}
                for idx, prob in zip(top5_indices, top5_probs)
            ],
            "latency_ms": latency,
        }

    @torch.no_grad()
    def predict_frames(self, frames: np.ndarray) -> dict:
        """Predict sign from numpy frames array.
        
        Args:
            frames: np.ndarray of shape (T, H, W, 3) uint8.
            
        Returns:
            dict with prediction results.
        """
        start = time.time()

        # Uniform sample to num_frames
        T = len(frames)
        if T >= self.num_frames:
            indices = np.linspace(0, T - 1, self.num_frames, dtype=int)
        else:
            indices = np.arange(T)
            pad = np.full(self.num_frames - T, T - 1, dtype=int)
            indices = np.concatenate([indices, pad])
        
        sampled_frames = frames[indices]
        video_tensor = self.transform(sampled_frames)
        video_tensor = video_tensor.unsqueeze(0).to(self.device)

        outputs = self.model(pixel_values=video_tensor)
        logits = outputs.logits[0]
        probs = torch.softmax(logits, dim=0)

        top5_probs, top5_indices = torch.topk(probs, min(5, len(self.class_names)))

        latency = (time.time() - start) * 1000

        return {
            "label": self.class_names[top5_indices[0].item()],
            "confidence": top5_probs[0].item(),
            "top5": [
                {"label": self.class_names[idx.item()], "prob": prob.item()}
                for idx, prob in zip(top5_indices, top5_probs)
            ],
            "latency_ms": latency,
        }
