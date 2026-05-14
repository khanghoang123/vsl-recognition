"""Evaluation script for trained VSL models.

Usage:
    python training/evaluate.py --model_type static --data_dir data/processed/static
    python training/evaluate.py --model_type dynamic --data_dir data/processed/dynamic
"""

import argparse
import json
import os
import sys

import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tensorflow as tf
from tensorflow import keras

from src.preprocessing import normalize_landmarks, compute_motion_features


def evaluate_static(model_path: str, data_dir: str, labels_path: str):
    """Evaluate static CNN-1D model."""
    model = keras.models.load_model(model_path)
    with open(labels_path, "r", encoding="utf-8") as f:
        labels = json.load(f)

    X, y = [], []
    class_dirs = sorted([
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d))
    ])

    for idx, class_name in enumerate(class_dirs):
        class_path = os.path.join(data_dir, class_name)
        for fname in os.listdir(class_path):
            if fname.endswith(".npy"):
                data = np.load(os.path.join(class_path, fname))
                if data.ndim == 1 and len(data) == 126:
                    X.append(normalize_landmarks(data))
                    y.append(idx)

    X = np.array(X)[..., np.newaxis]
    y = np.array(y)

    print(f"Evaluating on {len(X)} samples, {len(set(y))} classes")

    y_pred_probs = model.predict(X, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)

    acc = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred, average="macro")

    print(f"\nAccuracy: {acc:.4f}")
    print(f"F1-Score (macro): {f1:.4f}")
    print(f"\nClassification Report:")
    target_names = [labels.get(str(i), f"class_{i}") for i in range(len(set(y)))]
    print(classification_report(y, y_pred, target_names=target_names))

    cm = confusion_matrix(y, y_pred)
    print(f"\nConfusion Matrix:\n{cm}")

    return acc, f1


def evaluate_dynamic(model_path: str, data_dir: str, labels_path: str,
                     sequence_length: int = 60):
    """Evaluate dynamic Bi-LSTM + Attention model."""
    from src.dynamic_classifier import MultiHeadAttentionBlock

    model = keras.models.load_model(
        model_path,
        custom_objects={"MultiHeadAttentionBlock": MultiHeadAttentionBlock},
    )
    with open(labels_path, "r", encoding="utf-8") as f:
        labels = json.load(f)

    X, y = [], []
    class_dirs = sorted([
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d))
    ])

    for idx, class_name in enumerate(class_dirs):
        class_path = os.path.join(data_dir, class_name)
        for fname in os.listdir(class_path):
            if fname.endswith(".npy"):
                seq = np.load(os.path.join(class_path, fname))
                if seq.ndim == 2:
                    if seq.shape[1] > 126:
                        seq = seq[:, :126]
                    elif seq.shape[1] < 126:
                        pad = np.zeros((seq.shape[0], 126 - seq.shape[1]))
                        seq = np.concatenate([seq, pad], axis=1)

                    if len(seq) < sequence_length:
                        pad_count = sequence_length - len(seq)
                        padding = np.tile(seq[-1:], (pad_count, 1))
                        seq = np.concatenate([seq, padding], axis=0)
                    else:
                        indices = np.linspace(0, len(seq) - 1, sequence_length).astype(int)
                        seq = seq[indices]

                    seq = normalize_landmarks(seq)
                    seq = compute_motion_features(seq)
                    X.append(seq)
                    y.append(idx)

    X = np.array(X)
    y = np.array(y)

    print(f"Evaluating on {len(X)} sequences, {len(set(y))} classes")
    print(f"Input shape: {X.shape}")

    y_pred_probs = model.predict(X, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)

    acc = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred, average="macro")

    print(f"\nAccuracy: {acc:.4f}")
    print(f"F1-Score (macro): {f1:.4f}")
    print(f"\nClassification Report:")
    target_names = [labels.get(str(i), f"class_{i}") for i in range(len(set(y)))]
    print(classification_report(y, y_pred, target_names=target_names))

    return acc, f1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate VSL models")
    parser.add_argument("--model_type", type=str, choices=["static", "dynamic"], required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--sequence_length", type=int, default=60)
    args = parser.parse_args()

    if args.model_type == "static":
        evaluate_static(
            model_path="models/static_cnn1d.keras",
            data_dir=args.data_dir,
            labels_path="models/static_labels.json",
        )
    else:
        evaluate_dynamic(
            model_path="models/dynamic_bilstm_att.keras",
            data_dir=args.data_dir,
            labels_path="models/dynamic_labels.json",
            sequence_length=args.sequence_length,
        )
