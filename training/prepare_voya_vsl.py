"""Download and prepare VOYA_VSL dataset for training.

VOYA_VSL (HuggingFace): ~161 classes, pre-extracted MediaPipe keypoints.
Each sample: (60 frames, 1605 features) - includes pose, face, hand landmarks.

Usage:
    python training/prepare_voya_vsl.py --num_classes 20 --output_dir data/processed
    python training/prepare_voya_vsl.py --all --output_dir data/processed
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# Feature layout in the 1605-dim vector (MediaPipe Holistic):
# Pose: 33 landmarks × 3 coords (x,y,z) = 99 features [0:99]
# Left hand: 21 landmarks × 3 coords = 63 features [99:162]
# Right hand: 21 landmarks × 3 coords = 63 features [162:225]
# Face: 460 landmarks × 3 coords = 1380 features [225:1605]
# Total: 99 + 63 + 63 + 1380 = 1605
POSE_START, POSE_END = 0, 99
LEFT_HAND_START, LEFT_HAND_END = 99, 162
RIGHT_HAND_START, RIGHT_HAND_END = 162, 225
FACE_START, FACE_END = 225, 1605


def extract_hand_features(sequence: np.ndarray) -> np.ndarray:
    """Extract only hand landmarks from full 1605-dim feature vector.

    Returns: (T, 126) - left hand (63) + right hand (63)
    """
    left = sequence[:, LEFT_HAND_START:LEFT_HAND_END]
    right = sequence[:, RIGHT_HAND_START:RIGHT_HAND_END]
    return np.concatenate([left, right], axis=-1)


def extract_hand_and_pose_features(sequence: np.ndarray) -> np.ndarray:
    """Extract hand + pose landmarks (for richer context).

    Returns: (T, 225) - pose (99) + left hand (63) + right hand (63)
    """
    pose = sequence[:, POSE_START:POSE_END]
    left = sequence[:, LEFT_HAND_START:LEFT_HAND_END]
    right = sequence[:, RIGHT_HAND_START:RIGHT_HAND_END]
    return np.concatenate([pose, left, right], axis=-1)


def download_and_prepare(
    num_classes: int = 20,
    download_all: bool = False,
    output_dir: str = "data/processed",
    feature_mode: str = "hands_only",
):
    """Download VOYA_VSL and prepare for training."""
    from huggingface_hub import hf_hub_download, list_repo_files

    print("Listing VOYA_VSL files...")
    all_files = list_repo_files("Kateht/VOYA_VSL", repo_type="dataset")
    npz_files = sorted([f for f in all_files if f.endswith(".npz")])
    total_classes = len(npz_files)

    if download_all:
        num_classes = total_classes
    num_classes = min(num_classes, total_classes)
    print(f"Preparing {num_classes} / {total_classes} classes")

    # Download labels
    labels_path = hf_hub_download("Kateht/VOYA_VSL", "labels.json", repo_type="dataset")
    with open(labels_path, "r", encoding="utf-8") as f:
        all_labels = json.load(f)

    # Process classes
    static_dir = os.path.join(output_dir, "static")
    dynamic_dir = os.path.join(output_dir, "dynamic")
    os.makedirs(static_dir, exist_ok=True)
    os.makedirs(dynamic_dir, exist_ok=True)

    label_mapping = {}
    total_samples = 0

    for i, npz_file in enumerate(npz_files[:num_classes]):
        class_key = npz_file.replace("Merged/", "").replace(".npz", "")
        class_name = all_labels.get(class_key, class_key)

        # Clean class name for filesystem
        safe_name = class_name.replace("/", "_").replace("\\", "_").replace(" ", "_")
        safe_name = f"{i:04d}_{safe_name}"

        print(f"  [{i+1}/{num_classes}] Downloading {class_key}: {class_name}")

        try:
            path = hf_hub_download("Kateht/VOYA_VSL", npz_file, repo_type="dataset")
            data = np.load(path)
            sequences = data["sequences"]  # (N, 60, 1605)
            labels = data["labels"]        # (N,)

            # Extract features based on mode
            if feature_mode == "hands_only":
                processed = np.array([extract_hand_features(s) for s in sequences])
            elif feature_mode == "hands_and_pose":
                processed = np.array([extract_hand_and_pose_features(s) for s in sequences])
            else:
                processed = sequences

            # Save as dynamic sequences
            dyn_dir = os.path.join(dynamic_dir, safe_name)
            os.makedirs(dyn_dir, exist_ok=True)

            for j in range(len(processed)):
                np.save(os.path.join(dyn_dir, f"seq_{j:04d}.npy"), processed[j])

            # Also save middle frame as static sample
            static_cls_dir = os.path.join(static_dir, safe_name)
            os.makedirs(static_cls_dir, exist_ok=True)

            for j in range(len(processed)):
                mid_frame = processed[j][30]  # Middle frame of 60
                np.save(os.path.join(static_cls_dir, f"sample_{j:04d}.npy"), mid_frame)

            label_mapping[str(i)] = class_name
            total_samples += len(processed)

            print(f"    → {len(processed)} samples saved ({processed[0].shape})")

        except Exception as e:
            print(f"    ERROR: {e}")
            continue

    # Save label mappings
    meta = {
        "labels": label_mapping,
        "feature_mode": feature_mode,
        "num_classes": len(label_mapping),
        "total_samples": total_samples,
        "sequence_length": 60,
    }

    meta_path = os.path.join(output_dir, "meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"\nDone! {len(label_mapping)} classes, {total_samples} total samples")
    print(f"  Static data: {static_dir}")
    print(f"  Dynamic data: {dynamic_dir}")
    print(f"  Metadata: {meta_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare VOYA_VSL dataset")
    parser.add_argument("--num_classes", type=int, default=20)
    parser.add_argument("--all", action="store_true", dest="download_all")
    parser.add_argument("--output_dir", type=str, default="data/processed")
    parser.add_argument("--feature_mode", type=str, default="hands_only",
                        choices=["hands_only", "hands_and_pose", "full"])
    args = parser.parse_args()

    download_and_prepare(
        num_classes=args.num_classes,
        download_all=args.download_all,
        output_dir=args.output_dir,
        feature_mode=args.feature_mode,
    )
