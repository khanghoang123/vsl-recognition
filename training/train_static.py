"""Training script for static sign (alphabet) CNN-1D model.

Usage:
    python training/train_static.py --data_dir data/processed/static --epochs 100
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

from src.static_classifier import build_static_model
from src.preprocessing import normalize_landmarks, augment_landmarks


def load_data(data_dir: str):
    """Load keypoints data from .npy files organized by class folders."""
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
            if fname.endswith(".npy"):
                data = np.load(os.path.join(class_path, fname))
                if data.ndim == 1 and len(data) == 126:
                    X.append(data)
                    y.append(idx)
                elif data.ndim == 2:
                    for row in data:
                        if len(row) == 126:
                            X.append(row)
                            y.append(idx)

    return np.array(X), np.array(y), labels


def augment_dataset(X: np.ndarray, y: np.ndarray, factor: int = 3):
    """Augment dataset by generating variations."""
    X_aug, y_aug = [X], [y]
    for i in range(factor):
        augmented = np.array([augment_landmarks(x, seed=i * len(X) + j)
                              for j, x in enumerate(X)])
        X_aug.append(augmented)
        y_aug.append(y)
    return np.concatenate(X_aug), np.concatenate(y_aug)


def train(
    data_dir: str,
    output_dir: str = "models",
    epochs: int = 100,
    batch_size: int = 64,
    augment_factor: int = 3,
    val_split: float = 0.15,
    test_split: float = 0.1,
):
    """Train static CNN-1D model."""
    print("Loading data...")
    X, y, labels = load_data(data_dir)
    print(f"Loaded {len(X)} samples, {len(labels)} classes")
    print(f"Classes: {labels}")

    # Normalize
    print("Normalizing landmarks...")
    X = normalize_landmarks(X)

    # Split: grouped by video would be ideal, but for static signs we split by sample
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_split, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=val_split / (1 - test_split),
        random_state=42, stratify=y_trainval
    )

    # Augment training data only
    print(f"Augmenting training data (factor={augment_factor})...")
    X_train, y_train = augment_dataset(X_train, y_train, factor=augment_factor)
    print(f"Training: {len(X_train)}, Validation: {len(X_val)}, Test: {len(X_test)}")

    # Reshape for Conv1D: (N, 126) -> (N, 126, 1)
    X_train = X_train[..., np.newaxis]
    X_val = X_val[..., np.newaxis]
    X_test = X_test[..., np.newaxis]

    # Compute class weights for imbalanced data
    class_weights_arr = compute_class_weight(
        "balanced", classes=np.unique(y_train), y=y_train
    )
    class_weights = {i: w for i, w in enumerate(class_weights_arr)}

    # Build model
    num_classes = len(labels)
    model = build_static_model(num_classes=num_classes)
    model.summary()

    # Compile with AdamW
    optimizer = keras.optimizers.AdamW(
        learning_rate=0.001, weight_decay=0.01
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
    model_path = os.path.join(output_dir, "static_cnn1d.keras")
    model.save(model_path)
    print(f"Model saved to {model_path}")

    labels_path = os.path.join(output_dir, "static_labels.json")
    with open(labels_path, "w", encoding="utf-8") as f:
        json.dump(labels, f, ensure_ascii=False, indent=2)
    print(f"Labels saved to {labels_path}")

    # Save training history
    hist_path = os.path.join(output_dir, "static_history.json")
    hist_data = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    with open(hist_path, "w") as f:
        json.dump(hist_data, f)

    return model, history, labels


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train static VSL CNN-1D model")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="models")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--augment_factor", type=int, default=3)
    args = parser.parse_args()

    train(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        augment_factor=args.augment_factor,
    )
