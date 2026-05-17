"""CNN-1D model for static sign (alphabet) classification.

Reference: Paper comparison MLP vs CNN on ASL (Jurnal RESTI),
           Paper HUST 2025 (VSL Alphabet Recognition).
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def build_static_model(num_classes: int, input_dim: int = 126) -> keras.Model:
    """Build CNN-1D model for static sign classification.

    Input: (126,) hand landmarks from one frame.
    Architecture: 3x Conv1D blocks + GlobalAvgPool + Dense head.
    """
    inputs = keras.Input(shape=(input_dim, 1), name="landmarks")

    # Conv block 1
    x = layers.Conv1D(64, kernel_size=3, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Conv block 2
    x = layers.Conv1D(128, kernel_size=3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)

    # Conv block 3
    x = layers.Conv1D(256, kernel_size=3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)

    # Head
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(num_classes, activation="softmax", name="output")(x)

    model = keras.Model(inputs=inputs, outputs=x, name="static_cnn1d")
    return model


class StaticClassifier:
    """Wrapper for static sign classification inference."""

    def __init__(self, model_path: str, labels: dict):
        self.model = keras.models.load_model(model_path)
        self.labels = labels  # {index: label_name}
        self.idx_to_label = {int(k): v for k, v in labels.items()}

    def predict(self, landmarks: np.ndarray, threshold: float = 0.5):
        """Predict static sign from landmarks.

        Args:
            landmarks: (126,) array of hand landmarks.
            threshold: minimum confidence to return a prediction.

        Returns:
            (label, confidence) or (None, 0.0) if below threshold.
        """
        if np.all(landmarks == 0):
            return None, 0.0

        x = landmarks.reshape(1, -1, 1).astype(np.float32)
        probs = self.model.predict(x, verbose=0)[0]
        idx = int(np.argmax(probs))
        confidence = float(probs[idx])

        if confidence < threshold:
            return None, confidence

        label = self.idx_to_label.get(idx, f"class_{idx}")
        return label, confidence
