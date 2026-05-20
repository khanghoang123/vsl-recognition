"""Model utilities for Vietnamese Sign Language recognition."""

import torch.nn as nn
from transformers import VideoMAEForVideoClassification


DEFAULT_VIDEOMAE_MODEL = "MCG-NJU/videomae-small-finetuned-kinetics"


def create_videomae_model(
    num_classes: int = 100,
    pretrained: str = DEFAULT_VIDEOMAE_MODEL,
    freeze_backbone: bool = False,
) -> nn.Module:
    """Create a VideoMAE-Small classifier for fine-tuning."""
    model = VideoMAEForVideoClassification.from_pretrained(
        pretrained,
        num_labels=num_classes,
        ignore_mismatched_sizes=True,
    )

    if freeze_backbone:
        for param in model.videomae.parameters():
            param.requires_grad = False

    return model


def get_model_info(model: nn.Module) -> dict:
    """Return parameter-count information for reporting."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_size_mb = total_params * 4 / (1024 * 1024)

    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "model_size_mb": model_size_mb,
        "total_params_str": f"{total_params / 1e6:.1f}M",
        "trainable_params_str": f"{trainable_params / 1e6:.1f}M",
        "model_size_str": f"{model_size_mb:.1f} MB",
    }
