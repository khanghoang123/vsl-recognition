"""Training script for dynamic sign Bi-LSTM + Attention model.

Usage:
    python training/train_dynamic.py --data_dir data/processed/dynamic --epochs 80
"""

import argparse
import json
import os
import sys

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tensorflow as tf
from tensorflow import keras

from src.dynamic_classifier import build_dynamic_model
from src.preprocessing import normalize_landmarks, compute_motion_features, uniform_sample_frames


def load_data(data_dir: str, sequence_length: int = 30, use_motion: bool = True):
    """Load sequence keypoints data from .npy/.npz files."""
    X, y = [], []
    labels = {}

    class_dirs = sorted([
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d))
    ])

    for idx, class_name in enumerate(class_dirs):
        labels[idx] = class_name
        class_path = os.path.join(data_dir, class_name)

        for fname in os.listdir(class_path):
            fpath = os.path.join(class_path, fname)
            if fname.endswith(".npy"):
                seq = np.load(fpath)
            elif fname.endswith(".npz"):
                data = np.load(fpath)
                seq = data["sequences"] if "sequences" in data else data[list(data.keys())[0]]
            else:
                continue

            if seq.ndim == 2:
                seq = process_sequence(seq, sequence_length, use_motion)
                if seq is not None:
                    X.append(seq)
                    y.append(idx)
            elif seq.ndim == 3:
                for s in seq:
                    s = process_sequence(s, sequence_length, use_motion)
                    if s is not None:
                        X.append(s)
                        y.append(idx)

    return np.array(X), np.array(y), labels


def process_sequence(
    seq: np.ndarray,
    sequence_length: int,
    use_motion: bool,
) -> np.ndarray:
    """Process a raw sequence: sample, normalize, add motion features."""
    total_frames = len(seq)
    if total_frames < 3:
        return None

    feature_dim = seq.shape[1]

    # If features > 126, take only first 126 (hand landmarks)
    if feature_dim > 126:
        seq = seq[:, :126]
    elif feature_dim < 126:
        pad = np.zeros((total_frames, 126 - feature_dim))
        seq = np.concatenate([seq, pad], axis=1)

    # Uniform sampling
    indices = uniform_sample_frames(total_frames, sequence_length)
    seq = seq[indices]

    # Normalize
    seq = normalize_landmarks(seq)

    # Add motion features
    if use_motion:
        seq = compute_motion_features(seq)

    return seq


def train(
    data_dir: str,
    output_dir: str = "models",
    epochs: int = 80,
    batch_size: int = 32,
    sequence_length: int = 30,
    use_motion: bool = True,
    val_split: float = 0.15,
    test_split: float = 0.1,
):
    """Train dynamic Bi-LSTM + Attention model."""
    print("Loading data...")
    X, y, labels = load_data(data_dir, sequence_length, use_motion)
    print(f"Loaded {len(X)} sequences, {len(labels)} classes")
    print(f"Input shape: {X.shape}")
    print(f"Classes: {labels}")

    # Split data
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_split, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=val_split / (1 - test_split),
        random_state=42, stratify=y_trainval
    )
    print(f"Training: {len(X_train)}, Validation: {len(X_val)}, Test: {len(X_test)}")

    # Class weights
    class_weights_arr = compute_class_weight(
        "balanced", classes=np.unique(y_train), y=y_train
    )
    class_weights = {i: w for i, w in enumerate(class_weights_arr)}

    # Build model
    num_classes = len(labels)
    input_dim = 126
    model = build_dynamic_model(
        num_classes=num_classes,
        sequence_length=sequence_length,
        input_dim=input_dim,
        use_motion=use_motion,
    )
    model.summary()

    # Compile
    optimizer = keras.optimizers.AdamW(
        learning_rate=0.0005, weight_decay=0.01
    )
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=10, restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6
        ),
    ]

    # Train
    print("\nTraining...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1,
    )

    # Evaluate
    print("\nEvaluation on test set:")
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"  Test accuracy: {test_acc:.4f}")
    print(f"  Test loss: {test_loss:.4f}")

    # Save
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, "dynamic_bilstm_att.keras")
    model.save(model_path)
    print(f"Model saved to {model_path}")

    labels_path = os.path.join(output_dir, "dynamic_labels.json")
    with open(labels_path, "w", encoding="utf-8") as f:
        json.dump(labels, f, ensure_ascii=False, indent=2)

    hist_path = os.path.join(output_dir, "dynamic_history.json")
    hist_data = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    with open(hist_path, "w") as f:
        json.dump(hist_data, f)

    return model, history, labels


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train dynamic VSL Bi-LSTM model")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="models")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--sequence_length", type=int, default=30)
    parser.add_argument("--no_motion", action="store_true")
    args = parser.parse_args()

    train(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        use_motion=not args.no_motion,
    )
