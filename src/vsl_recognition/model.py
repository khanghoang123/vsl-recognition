"""Model loading, mapping validation, and prediction helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import VideoMAEForVideoClassification

from .preprocessing import normalize_label


@dataclass
class ModelBundle:
    """A loaded classifier and the metadata required for inference."""

    model: VideoMAEForVideoClassification
    class_names: list[str]
    device: torch.device
    use_fp16: bool


def validate_class_mapping(model, class_names: list[str]) -> list[str]:
    """Return mapping inconsistencies between the checkpoint and class list."""
    errors: list[str] = []
    normalized = [normalize_label(name) for name in class_names]
    classifier_out = getattr(getattr(model, "classifier", None), "out_features", None)
    if classifier_out != len(normalized):
        errors.append(f"classifier outputs {classifier_out}; expected {len(normalized)}")

    id2label = getattr(model.config, "id2label", {}) or {}
    for index, expected in enumerate(normalized):
        actual = id2label.get(index, id2label.get(str(index)))
        if actual is None:
            errors.append(f"id2label is missing class {index}")
            break
        if normalize_label(str(actual)) != expected:
            errors.append(
                f"id2label[{index}]={normalize_label(str(actual))!r}; expected {expected!r}"
            )
            break
    return errors


def load_model_bundle(
    model_dir: str | Path,
    *,
    device: str | None = None,
    fp16: bool = True,
) -> ModelBundle:
    """Load a local fine-tuned VideoMAE bundle."""
    path = Path(model_dir)
    required = ["config.json", "model.safetensors", "class_names.json"]
    missing = [name for name in required if not (path / name).exists()]
    if missing:
        raise FileNotFoundError(f"Model bundle is missing: {', '.join(missing)}")

    class_names = [
        normalize_label(name)
        for name in json.loads((path / "class_names.json").read_text(encoding="utf-8"))
    ]
    target = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    use_fp16 = fp16 and target.type == "cuda"
    model = VideoMAEForVideoClassification.from_pretrained(
        str(path),
        num_labels=len(class_names),
    )
    model.to(target)
    if use_fp16:
        model.half()
    model.eval()

    errors = validate_class_mapping(model, class_names)
    if errors:
        raise ValueError("Invalid class mapping: " + "; ".join(errors))
    return ModelBundle(model=model, class_names=class_names, device=target, use_fp16=use_fp16)


@torch.inference_mode()
def predict_batch(bundle: ModelBundle, batch: torch.Tensor) -> torch.Tensor:
    """Return class probabilities for a batch shaped (B, T, C, H, W)."""
    inputs = batch.to(bundle.device)
    if bundle.use_fp16:
        inputs = inputs.half()
    logits = bundle.model(pixel_values=inputs).logits
    return torch.softmax(logits.float(), dim=-1).cpu()
