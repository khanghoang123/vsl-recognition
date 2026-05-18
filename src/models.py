"""VideoMAEv2 model wrapper for sign language recognition."""

import torch
import torch.nn as nn
from transformers import VideoMAEForVideoClassification, VideoMAEConfig


def create_videomae_model(
    num_classes: int = 50,
    pretrained: str = "MCG-NJU/videomae-small-finetuned-kinetics",
    freeze_backbone: bool = False,
) -> nn.Module:
    """Create VideoMAEv2-Small model for fine-tuning.
    
    Args:
        num_classes: Number of sign language classes.
        pretrained: HuggingFace model name or path.
        freeze_backbone: Whether to freeze encoder layers.
        
    Returns:
        VideoMAE model with classification head.
    """
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
    """Get model parameter count and size info."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_size_mb = total_params * 4 / (1024 * 1024)  # fp32

    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "model_size_mb": model_size_mb,
        "total_params_str": f"{total_params / 1e6:.1f}M",
        "trainable_params_str": f"{trainable_params / 1e6:.1f}M",
        "model_size_str": f"{model_size_mb:.1f} MB",
    }
