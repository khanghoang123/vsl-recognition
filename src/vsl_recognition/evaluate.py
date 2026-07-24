"""Evaluate a local VideoMAE checkpoint and generate portfolio-ready reports."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import torch
import transformers
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    top_k_accuracy_score,
)

from .model import load_model_bundle, predict_batch
from .preprocessing import (
    build_video_index,
    normalize_label,
    prepare_video,
    read_video_rgb,
    resolve_video_path,
)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def metric_summary(
    labels: list[int],
    probabilities: np.ndarray,
    *,
    num_classes: int,
) -> dict[str, float | int]:
    predictions = probabilities.argmax(axis=1)
    return {
        "samples": len(labels),
        "top1_accuracy": float(accuracy_score(labels, predictions)),
        "top5_accuracy": float(
            top_k_accuracy_score(
                labels,
                probabilities,
                k=min(5, num_classes),
                labels=list(range(num_classes)),
            )
        ),
        "macro_f1": float(
            f1_score(labels, predictions, average="macro", zero_division=0)
        ),
    }


def support_group_summary(
    labels: list[int],
    predictions: list[int],
    total_support: dict[str, int],
) -> dict:
    """Summarize per-class F1 for head, body, and tail support bands."""
    supports = np.bincount(labels)
    class_f1 = f1_score(
        labels,
        predictions,
        labels=list(range(len(supports))),
        average=None,
        zero_division=0,
    )
    groups = {
        "tail_6_to_15": [
            index for index, count in total_support.items() if count <= 15
        ],
        "body_16_to_58": [
            index for index, count in total_support.items() if 16 <= count <= 58
        ],
        "head_59_plus": [
            index for index, count in total_support.items() if count >= 59
        ],
    }
    return {
        name: {
            "classes": len(indices),
            "mean_class_f1": float(np.mean(class_f1[indices])) if indices else None,
        }
        for name, indices in groups.items()
    }


def save_confusion_figure(
    labels: list[int],
    predictions: list[int],
    output_path: Path,
) -> None:
    matrix = confusion_matrix(labels, predictions, normalize="true")
    figure, axis = plt.subplots(figsize=(12, 10))
    image = axis.imshow(matrix, cmap="Blues", vmin=0, vmax=1)
    axis.set_title("Normalized validation confusion matrix")
    axis.set_xlabel("Predicted class index")
    axis.set_ylabel("True class index")
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def save_training_curves(history_path: Path, output_path: Path) -> None:
    history = read_json(history_path)
    epochs = [row["epoch"] for row in history]
    figure, axes = plt.subplots(1, 3, figsize=(15, 4))
    for axis, train_key, val_key, title in [
        (axes[0], "train_loss", "val_loss", "Loss"),
        (axes[1], "train_acc", "val_acc", "Accuracy"),
        (axes[2], "train_f1_macro", "val_f1_macro", "Macro-F1"),
    ]:
        axis.plot(epochs, [row[train_key] for row in history], label="Train")
        axis.plot(epochs, [row[val_key] for row in history], label="Validation")
        axis.set_title(title)
        axis.set_xlabel("Epoch")
        axis.grid(alpha=0.25)
        axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def benchmark_model(bundle, sample: torch.Tensor, runs: int = 50) -> dict:
    batch = sample.unsqueeze(0)
    for _ in range(5):
        predict_batch(bundle, batch)
    durations: list[float] = []
    for _ in range(runs):
        if bundle.device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        predict_batch(bundle, batch)
        if bundle.device.type == "cuda":
            torch.cuda.synchronize()
        durations.append((time.perf_counter() - start) * 1000)
    return {
        "runs": runs,
        "mean_ms": statistics.mean(durations),
        "median_ms": statistics.median(durations),
        "p95_ms": float(np.percentile(durations, 95)),
    }


def evaluate(
    *,
    model_dir: Path,
    metadata_path: Path,
    data_root: Path,
    output_dir: Path,
    batch_size: int,
    device: str | None,
    audit_path: Path | None,
    benchmark_runs: int,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    items = read_json(metadata_path)
    index = build_video_index(data_root)
    bundle = load_model_bundle(model_dir, device=device)
    audit = read_json(audit_path) if audit_path and audit_path.exists() else None
    total_support = (
        {
            int(index): int(count)
            for index, count in audit["dataset"]["class_support_total"].items()
        }
        if audit and "class_support_total" in audit["dataset"]
        else {index: int(count) for index, count in enumerate(np.bincount(
            [int(item["label"]) for item in items]
        ))}
    )

    all_labels: list[int] = []
    all_probabilities: list[np.ndarray] = []
    predictions: list[dict] = []
    prepared_batch: list[torch.Tensor] = []
    batch_items: list[dict] = []
    preprocess_times: list[float] = []
    model_times: list[float] = []
    first_tensor: torch.Tensor | None = None

    def run_batch() -> None:
        nonlocal first_tensor
        if not prepared_batch:
            return
        tensor_batch = torch.stack(prepared_batch)
        if first_tensor is None:
            first_tensor = prepared_batch[0].clone()
        start = time.perf_counter()
        probabilities = predict_batch(bundle, tensor_batch).numpy()
        model_times.append((time.perf_counter() - start) * 1000)
        for item, probs in zip(batch_items, probabilities, strict=True):
            label = int(item["label"])
            top_indices = probs.argsort()[-5:][::-1]
            all_labels.append(label)
            all_probabilities.append(probs)
            predictions.append(
                {
                    "video_name": item.get("video_name") or Path(item["path"]).name,
                    "target_id": label,
                    "target": normalize_label(item["class_name"]),
                    "prediction_id": int(top_indices[0]),
                    "prediction": bundle.class_names[int(top_indices[0])],
                    "confidence": float(probs[top_indices[0]]),
                    "top5": [
                        {
                            "class_id": int(index_),
                            "label": bundle.class_names[int(index_)],
                            "probability": float(probs[index_]),
                        }
                        for index_ in top_indices
                    ],
                }
            )
        prepared_batch.clear()
        batch_items.clear()

    total_start = time.perf_counter()
    for position, item in enumerate(items, start=1):
        path = resolve_video_path(item, index)
        preprocess_start = time.perf_counter()
        frames, _ = read_video_rgb(path)
        prepared = prepare_video(frames)
        preprocess_times.append((time.perf_counter() - preprocess_start) * 1000)
        prepared_batch.append(prepared.tensor)
        batch_items.append(item)
        if len(prepared_batch) == batch_size:
            run_batch()
        if position % 100 == 0:
            print(f"Evaluated {position}/{len(items)}")
    run_batch()
    total_seconds = time.perf_counter() - total_start

    probability_array = np.asarray(all_probabilities)
    predicted_ids = probability_array.argmax(axis=1).tolist()
    full_metrics = metric_summary(
        all_labels,
        probability_array,
        num_classes=len(bundle.class_names),
    )
    report = classification_report(
        all_labels,
        predicted_ids,
        labels=list(range(len(bundle.class_names))),
        target_names=bundle.class_names,
        output_dict=True,
        zero_division=0,
    )

    clean_metrics = None
    excluded_names: list[str] = []
    if audit:
        excluded_names = audit["duplicate_signature_audit"].get(
            "validation_issue_videos", []
        )
        keep = [
            index
            for index, prediction in enumerate(predictions)
            if prediction["video_name"] not in excluded_names
        ]
        if keep:
            clean_metrics = metric_summary(
                [all_labels[index] for index in keep],
                probability_array[keep],
                num_classes=len(bundle.class_names),
            )

    benchmark = (
        benchmark_model(bundle, first_tensor, benchmark_runs)
        if first_tensor is not None and benchmark_runs > 0
        else None
    )
    metrics = {
        "schema_version": 1,
        "split": "held-out validation",
        "model": {
            "architecture": bundle.model.config.model_type,
            "parameters": sum(
                parameter.numel() for parameter in bundle.model.parameters()
            ),
            "checkpoint_bytes": (model_dir / "model.safetensors").stat().st_size,
            "input_frames": int(bundle.model.config.num_frames),
            "image_size": int(bundle.model.config.image_size),
            "classes": len(bundle.class_names),
        },
        "full_validation": full_metrics,
        "support_groups": support_group_summary(
            all_labels,
            predicted_ids,
            total_support,
        ),
        "audited_clean_subset": clean_metrics,
        "excluded_validation_videos": excluded_names,
        "timing": {
            "total_evaluation_seconds": total_seconds,
            "mean_preprocessing_ms": statistics.mean(preprocess_times),
            "mean_model_batch_ms": statistics.mean(model_times),
            "batch_size": batch_size,
            "model_only_batch1": benchmark,
        },
        "environment": {
            "device": str(bundle.device),
            "fp16": bundle.use_fp16,
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "scikit_learn": sklearn.__version__,
        },
    }

    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "predictions.json").write_text(
        json.dumps(predictions, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "classification_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    save_confusion_figure(
        all_labels,
        predicted_ids,
        output_dir / "confusion_matrix.png",
    )
    history_path = model_dir / "training_history.json"
    if history_path.exists():
        save_training_curves(history_path, output_dir / "training_curves.png")
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("reports"))
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default=None)
    parser.add_argument("--audit", type=Path, default=None)
    parser.add_argument("--benchmark-runs", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = evaluate(
        model_dir=args.model_dir,
        metadata_path=args.metadata,
        data_root=args.data_root,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        device=args.device,
        audit_path=args.audit,
        benchmark_runs=args.benchmark_runs,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
