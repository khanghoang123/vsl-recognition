"""Bi-LSTM + Multi-Head Attention model for dynamic sign classification.

Reference: "Recognizing VSL Using Deep Neural Networks"
           Nguyen Quang Duy, Luong Thai Le - UT Communications, 2025.
           Accuracy: 99.51%
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class MultiHeadAttentionBlock(layers.Layer):
    """Multi-Head Attention layer for temporal focusing."""

    def __init__(self, embed_dim: int, num_heads: int = 4, **kwargs):
        super().__init__(**kwargs)
        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim // num_heads,
        )
        self.layernorm = layers.LayerNormalization()

    def call(self, x):
        attn_output = self.attention(query=x, value=x, key=x)
        return self.layernorm(x + attn_output)


def build_dynamic_model(
    num_classes: int,
    sequence_length: int = 30,
    input_dim: int = 126,
    use_motion: bool = True,
) -> keras.Model:
    """Build Bi-LSTM + Multi-Head Attention model for dynamic signs.

    Input: (sequence_length, input_dim) or (sequence_length, input_dim*2) with motion.
    """
    feature_dim = input_dim * 2 if use_motion else input_dim
    inputs = keras.Input(shape=(sequence_length, feature_dim), name="sequence")

    # Bi-LSTM layers
    x = layers.Bidirectional(
        layers.LSTM(128, return_sequences=True)
    )(inputs)
    x = layers.Bidirectional(
        layers.LSTM(64, return_sequences=True)
    )(x)

    # Multi-Head Attention
    x = MultiHeadAttentionBlock(embed_dim=128, num_heads=4)(x)

    # Temporal pooling: mean over time steps
    x = layers.GlobalAveragePooling1D()(x)

    # Classification head
    x = layers.BatchNormalization()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(num_classes, activation="softmax", name="output")(x)

    model = keras.Model(inputs=inputs, outputs=x, name="dynamic_bilstm_attention")
    return model


class DynamicClassifier:
    """Wrapper for dynamic sign classification inference."""

    def __init__(self, model_path: str, labels: dict):
        self.model = keras.models.load_model(
            model_path,
            custom_objects={"MultiHeadAttentionBlock": MultiHeadAttentionBlock},
        )
        self.labels = labels
        self.idx_to_label = {int(k): v for k, v in labels.items()}

    def predict(self, sequence: np.ndarray, threshold: float = 0.5):
        """Predict dynamic sign from landmark sequence.

        Args:
            sequence: (T, features) array of landmark sequences.
            threshold: minimum confidence to return a prediction.

        Returns:
            (label, confidence) or (None, 0.0) if below threshold.
        """
        has_data = np.any(sequence != 0, axis=1)
        if np.sum(has_data) < 5:
            return None, 0.0

        x = np.expand_dims(sequence, axis=0).astype(np.float32)
        probs = self.model.predict(x, verbose=0)[0]
        idx = int(np.argmax(probs))
        confidence = float(probs[idx])

        if confidence < threshold:
            return None, confidence

        label = self.idx_to_label.get(idx, f"class_{idx}")
        return label, confidence
