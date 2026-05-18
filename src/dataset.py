"""Video dataset for Multi-VSL sign language recognition."""

import os
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Normalize, Resize, CenterCrop, RandomResizedCrop, ColorJitter
from torchvision.transforms.functional import adjust_brightness, adjust_contrast, adjust_saturation


def load_video_decord(video_path: str, num_frames: int = 16) -> np.ndarray:
    """Load video and uniformly sample frames using decord.
    
    Args:
        video_path: Path to video file.
        num_frames: Number of frames to sample.
        
    Returns:
        np.ndarray of shape (num_frames, H, W, 3) in uint8.
    """
    from decord import VideoReader, cpu

    vr = VideoReader(video_path, ctx=cpu(0))
    total_frames = len(vr)

    if total_frames >= num_frames:
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    else:
        indices = np.arange(total_frames)
        pad_indices = np.full(num_frames - total_frames, total_frames - 1, dtype=int)
        indices = np.concatenate([indices, pad_indices])

    frames = vr.get_batch(indices).asnumpy()
    return frames


def load_video_opencv(video_path: str, num_frames: int = 16) -> np.ndarray:
    """Fallback video loader using OpenCV."""
    import cv2

    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()

    if len(frames) == 0:
        raise ValueError(f"Could not read video: {video_path}")

    frames = np.array(frames)
    total_frames = len(frames)

    if total_frames >= num_frames:
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    else:
        indices = np.arange(total_frames)
        pad_indices = np.full(num_frames - total_frames, total_frames - 1, dtype=int)
        indices = np.concatenate([indices, pad_indices])

    return frames[indices]


def load_video(video_path: str, num_frames: int = 16) -> np.ndarray:
    """Load video with decord, fallback to opencv."""
    try:
        return load_video_decord(video_path, num_frames)
    except Exception:
        return load_video_opencv(video_path, num_frames)


class VideoTransform:
    """Video transformation for training/evaluation."""

    def __init__(self, mode: str = "train", image_size: int = 224):
        self.image_size = image_size
        self.mode = mode
        self.normalize = Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

    def __call__(self, frames: np.ndarray) -> torch.Tensor:
        """Transform video frames.
        
        Args:
            frames: np.ndarray of shape (T, H, W, 3) uint8.
            
        Returns:
            torch.Tensor of shape (T, 3, H, W) normalized.
        """
        video = torch.from_numpy(frames).float() / 255.0
        video = video.permute(0, 3, 1, 2)  # (T, H, W, 3) -> (T, 3, H, W)

        T = video.shape[0]
        transformed = []

        if self.mode == "train":
            # Random resized crop parameters (same for all frames)
            i, j, h, w = RandomResizedCrop.get_params(
                video[0], scale=(0.8, 1.0), ratio=(0.9, 1.1)
            )
            # Sample color jitter params once for temporal consistency
            _, brightness_factor, contrast_factor, saturation_factor, _ = \
                ColorJitter.get_params(
                    brightness=(0.9, 1.1),
                    contrast=(0.9, 1.1),
                    saturation=(0.9, 1.1),
                    hue=None,
                )

            for t in range(T):
                frame = video[t]
                frame = frame[:, i:i+h, j:j+w]
                frame = Resize((self.image_size, self.image_size), antialias=True)(frame)
                frame = adjust_brightness(frame, brightness_factor)
                frame = adjust_contrast(frame, contrast_factor)
                frame = adjust_saturation(frame, saturation_factor)
                frame = self.normalize(frame)
                transformed.append(frame)
        else:
            for t in range(T):
                frame = video[t]
                frame = Resize(self.image_size + 32, antialias=True)(frame)
                frame = CenterCrop(self.image_size)(frame)
                frame = self.normalize(frame)
                transformed.append(frame)

        return torch.stack(transformed)  # (T, 3, H, W)


class MultiVSLDataset(Dataset):
    """Dataset for Multi-VSL videos."""

    def __init__(
        self,
        video_paths: list,
        labels: list,
        num_frames: int = 16,
        transform: Optional[VideoTransform] = None,
    ):
        self.video_paths = video_paths
        self.labels = labels
        self.num_frames = num_frames
        self.transform = transform or VideoTransform(mode="eval")

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]

        frames = load_video(video_path, self.num_frames)
        video_tensor = self.transform(frames)

        return video_tensor, label


def create_datasets(
    data_dir: str,
    num_classes: int = 50,
    num_frames: int = 16,
    val_ratio: float = 0.2,
    seed: int = 42,
):
    """Create train/val datasets from Multi-VSL directory structure.
    
    Expected structure:
        data_dir/
            class_001/
                video1.avi
                video2.avi
            class_002/
                ...
    
    Split by signer if possible, otherwise random split.
    """
    data_path = Path(data_dir)
    classes = sorted([d.name for d in data_path.iterdir() if d.is_dir()])[:num_classes]
    class_to_idx = {c: i for i, c in enumerate(classes)}

    all_videos = []
    all_labels = []

    for cls_name in classes:
        cls_dir = data_path / cls_name
        videos = sorted([
            str(f) for f in cls_dir.iterdir()
            if f.suffix.lower() in ('.avi', '.mp4', '.mov', '.mkv')
        ])
        for v in videos:
            all_videos.append(v)
            all_labels.append(class_to_idx[cls_name])

    # Split train/val
    random.seed(seed)
    indices = list(range(len(all_videos)))
    random.shuffle(indices)

    split_idx = int(len(indices) * (1 - val_ratio))
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]

    train_videos = [all_videos[i] for i in train_indices]
    train_labels = [all_labels[i] for i in train_indices]
    val_videos = [all_videos[i] for i in val_indices]
    val_labels = [all_labels[i] for i in val_indices]

    train_dataset = MultiVSLDataset(
        train_videos, train_labels, num_frames,
        transform=VideoTransform(mode="train")
    )
    val_dataset = MultiVSLDataset(
        val_videos, val_labels, num_frames,
        transform=VideoTransform(mode="eval")
    )

    return train_dataset, val_dataset, classes
