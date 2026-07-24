"""Audit a recovered training run for split integrity and reproducibility."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

from .preprocessing import build_video_index, is_bad_frame, resolve_video_path


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized_video_name(item: dict) -> str:
    return item.get("video_name") or Path(item["path"]).name


def source_group(item: dict) -> str:
    """Group filename variants such as 123.mp4 and 123_1.mp4."""
    stem = Path(normalized_video_name(item)).stem
    return re.sub(r"_\d+$", "", stem)


def sampled_frames(path: Path, count: int = 5) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return []
    indices = set(np.linspace(0, total - 1, min(count, total), dtype=int).tolist())
    frames: list[np.ndarray] = []
    index = 0
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        if index in indices:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        index += 1
        if len(frames) == len(indices):
            break
    cap.release()
    return frames


def video_signature(path: Path) -> tuple[str | None, float]:
    """Match the lightweight signature used by the original EDA notebook."""
    frames = sampled_frames(path)
    if not frames:
        return None, 1.0
    signatures: list[str] = []
    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        small = cv2.resize(gray, (16, 16), interpolation=cv2.INTER_AREA)
        bits = (small > small.mean()).astype(np.uint8).flatten()
        signatures.append("".join(map(str, bits.tolist())))
    payload = "|".join(signatures).encode("utf-8")
    signature = hashlib.md5(payload, usedforsecurity=False).hexdigest()
    return signature, sum(is_bad_frame(frame) for frame in frames) / len(frames)


def audit_run(
    metadata_dir: Path,
    data_root: Path,
    model_dir: Path,
    *,
    scan_signatures: bool = True,
) -> dict:
    train = read_json(metadata_dir / "train.json")
    validation = read_json(metadata_dir / "val.json")
    class_names = read_json(metadata_dir / "class_names.json")
    index = build_video_index(data_root)

    train_names = {normalized_video_name(item) for item in train}
    validation_names = {normalized_video_name(item) for item in validation}
    train_groups = {source_group(item) for item in train}
    validation_groups = {source_group(item) for item in validation}
    missing = [
        normalized_video_name(item)
        for item in [*train, *validation]
        if normalized_video_name(item) not in index
    ]
    class_support = Counter(
        int(item["label"]) for item in [*train, *validation]
    )

    audit = {
        "schema_version": 1,
        "dataset": {
            "train_videos": len(train),
            "validation_videos": len(validation),
            "total_videos": len(train) + len(validation),
            "classes": len(class_names),
            "train_classes": len({int(item["label"]) for item in train}),
            "validation_classes": len({int(item["label"]) for item in validation}),
            "class_support_total": {
                str(label): class_support[label] for label in sorted(class_support)
            },
            "exact_path_overlap": sorted(train_names & validation_names),
            "source_group_overlap": sorted(train_groups & validation_groups),
            "missing_videos": missing,
        },
        "artifacts": {},
        "duplicate_signature_audit": {
            "enabled": scan_signatures,
            "duplicate_groups": None,
            "duplicate_videos": None,
            "cross_split_groups": [],
            "validation_issue_videos": [],
        },
    }

    artifact_paths = {
        "train_manifest": metadata_dir / "train.json",
        "validation_manifest": metadata_dir / "val.json",
        "class_names": metadata_dir / "class_names.json",
        "model_config": model_dir / "config.json",
        "model_weights": model_dir / "model.safetensors",
        "training_history": model_dir / "training_history.json",
        "legacy_metrics": model_dir / "metrics.json",
    }
    for name, path in artifact_paths.items():
        if path.exists():
            audit["artifacts"][name] = {
                "file": path.name,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }

    if scan_signatures:
        groups: dict[str, list[dict]] = defaultdict(list)
        for split, items in (("train", train), ("validation", validation)):
            for item in items:
                path = resolve_video_path(item, index)
                signature, bad_ratio = video_signature(path)
                if signature:
                    groups[signature].append(
                        {
                            "split": split,
                            "video_name": normalized_video_name(item),
                            "class_name": item["class_name"],
                            "label": int(item["label"]),
                            "sampled_bad_frame_ratio": bad_ratio,
                        }
                    )

        duplicate_groups = [items for items in groups.values() if len(items) > 1]
        cross_split = [
            items
            for items in duplicate_groups
            if len({item["split"] for item in items}) > 1
        ]
        issue_names = sorted(
            {
                item["video_name"]
                for group in cross_split
                for item in group
                if item["split"] == "validation"
            }
        )
        audit["duplicate_signature_audit"] = {
            "enabled": True,
            "duplicate_groups": len(duplicate_groups),
            "duplicate_videos": sum(len(group) for group in duplicate_groups),
            "cross_split_groups": cross_split,
            "validation_issue_videos": issue_names,
        }

    dataset = audit["dataset"]
    audit["checks"] = {
        "all_videos_resolved": not dataset["missing_videos"],
        "all_classes_present": (
            dataset["classes"]
            == dataset["train_classes"]
            == dataset["validation_classes"]
        ),
        "no_exact_overlap": not dataset["exact_path_overlap"],
        "no_source_group_overlap": not dataset["source_group_overlap"],
        "no_signature_overlap": not audit["duplicate_signature_audit"][
            "cross_split_groups"
        ],
    }
    return audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("reports/data_audit.json"))
    parser.add_argument("--skip-signatures", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = audit_run(
        args.metadata_dir,
        args.data_root,
        args.model_dir,
        scan_signatures=not args.skip_signatures,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report["checks"], indent=2))
    print(f"Audit written to {args.output}")


if __name__ == "__main__":
    main()
