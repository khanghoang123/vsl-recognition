"""Fine-tune VideoMAE on a fixed metadata split."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from transformers import AutoImageProcessor, VideoMAEForVideoClassification

from .preprocessing import (
    VideoTransform,
    build_video_index,
    filter_bad_frames,
    normalize_label,
    read_video_rgb,
    resolve_video_path,
    uniform_indices,
)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class MetadataDataset(Dataset):
    """Dataset backed by a fixed JSON manifest."""

    def __init__(
        self,
        items: list[dict],
        video_index: dict[str, Path],
        *,
        num_frames: int,
        image_size: int,
        mode: str,
        remove_bad_frames: bool,
        min_valid_frames: int,
    ) -> None:
        self.items = items
        self.video_index = video_index
        self.num_frames = num_frames
        self.transform = VideoTransform(mode=mode, image_size=image_size)
        self.remove_bad_frames = remove_bad_frames
        self.min_valid_frames = min_valid_frames

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        item = self.items[index]
        path = resolve_video_path(item, self.video_index)
        frames, _ = read_video_rgb(path)
        if self.remove_bad_frames:
            frames = filter_bad_frames(frames, self.min_valid_frames)
        indices = uniform_indices(len(frames), self.num_frames)
        return self.transform(frames[indices]), int(item["label"])


def epoch_metrics(labels: list[int], predictions: list[int]) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(
            f1_score(labels, predictions, average="macro", zero_division=0)
        ),
    }


def run_epoch(
    model,
    loader,
    criterion,
    device,
    *,
    optimizer=None,
    scheduler=None,
    scaler=None,
    amp_enabled: bool,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_samples = 0
    labels_all: list[int] = []
    predictions_all: list[int] = []

    context = torch.enable_grad if training else torch.no_grad
    with context():
        for videos, labels in loader:
            videos = videos.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(
                device_type=device.type,
                enabled=amp_enabled,
            ):
                logits = model(pixel_values=videos).logits
                loss = criterion(logits, labels)
            if training:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
            total_loss += float(loss.item()) * len(labels)
            total_samples += len(labels)
            labels_all.extend(labels.cpu().tolist())
            predictions_all.extend(logits.argmax(dim=1).cpu().tolist())

    metrics = epoch_metrics(labels_all, predictions_all)
    metrics["loss"] = total_loss / max(total_samples, 1)
    return metrics


def save_bundle(
    output_dir: Path,
    model,
    processor,
    class_names: list[str],
    history: list[dict],
    metrics: dict,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir, safe_serialization=True)
    processor.save_pretrained(output_dir)
    (output_dir / "class_names.json").write_text(
        json.dumps(class_names, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "training_history.json").write_text(
        json.dumps(history, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def train(args: argparse.Namespace) -> None:
    config = read_json(args.config)
    metadata_dir = args.metadata_dir
    train_items = read_json(metadata_dir / "train.json")
    validation_items = read_json(metadata_dir / "val.json")
    class_names = [
        normalize_label(name)
        for name in read_json(metadata_dir / "class_names.json")
    ]
    set_seed(int(config["seed"]))

    video_index = build_video_index(args.data_root)
    common = {
        "video_index": video_index,
        "num_frames": int(config["num_frames"]),
        "image_size": int(config["image_size"]),
        "remove_bad_frames": bool(config["filter_bad_frames"]),
        "min_valid_frames": int(config["min_valid_frames"]),
    }
    train_dataset = MetadataDataset(train_items, mode="train", **common)
    validation_dataset = MetadataDataset(validation_items, mode="eval", **common)

    sampler = None
    if config["weighted_sampler"]:
        counts = Counter(int(item["label"]) for item in train_items)
        weights = [1.0 / counts[int(item["label"])] for item in train_items]
        sampler = WeightedRandomSampler(weights, len(weights), replacement=True)

    batch_size = int(config["batch_size"])
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=sampler is None,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
        persistent_workers=args.workers > 0,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.workers > 0,
    )

    id2label = {index: name for index, name in enumerate(class_names)}
    label2id = {name: index for index, name in id2label.items()}
    processor = AutoImageProcessor.from_pretrained(config["model_name"])
    model = VideoMAEForVideoClassification.from_pretrained(
        config["model_name"],
        num_labels=len(class_names),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    total_steps = int(config["epochs"]) * len(train_loader)
    warmup_steps = int(config["warmup_epochs"]) * len(train_loader)

    def schedule(step: int) -> float:
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    criterion = nn.CrossEntropyLoss(
        label_smoothing=float(config["label_smoothing"])
    )
    amp_enabled = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    history: list[dict] = []
    best_macro_f1 = -1.0

    for epoch in range(1, int(config["epochs"]) + 1):
        started = time.perf_counter()
        train_result = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            amp_enabled=amp_enabled,
        )
        validation_result = run_epoch(
            model,
            validation_loader,
            criterion,
            device,
            amp_enabled=amp_enabled,
        )
        record = {
            "epoch": epoch,
            "train_loss": train_result["loss"],
            "train_acc": train_result["accuracy"],
            "train_f1_macro": train_result["macro_f1"],
            "val_loss": validation_result["loss"],
            "val_acc": validation_result["accuracy"],
            "val_f1_macro": validation_result["macro_f1"],
            "seconds": time.perf_counter() - started,
        }
        history.append(record)
        print(json.dumps(record))

        if record["val_f1_macro"] > best_macro_f1:
            best_macro_f1 = record["val_f1_macro"]
            save_bundle(
                args.output_dir,
                model,
                processor,
                class_names,
                history,
                record,
            )

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "history": history,
                "config": config,
            },
            args.output_dir.parent / "last_checkpoint.pt",
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--metadata-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
