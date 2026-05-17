"""Data preprocessing utilities for VSL recognition.

Includes: uniform sampling, temporal padding, normalization, augmentation.
References: AIO2025 (sampling/padding), Paper HUST 2025 (augmentation).
"""

import numpy as np


def uniform_sample_frames(total_frames: int, target_frames: int, stride: int = 1) -> np.ndarray:
    """Select frame indices using uniform sampling with temporal padding.

    Reference: AIO2025 - HMDB51Dataset._select_indices
    """
    if total_frames <= 0:
        raise ValueError("Video has no frames")
    if total_frames == 1:
        return np.zeros(target_frames, dtype=np.int64)

    steps = max(target_frames * stride, target_frames)
    grid = np.linspace(0, total_frames - 1, num=steps)
    idxs = grid[::stride].astype(np.int64)

    # Temporal padding: repeat last frame if not enough
    if len(idxs) < target_frames:
        pad = np.full(target_frames - len(idxs), idxs[-1], dtype=np.int64)
        idxs = np.concatenate([idxs, pad])

    return idxs[:target_frames]


def normalize_landmarks(landmarks: np.ndarray) -> np.ndarray:
    """Normalize landmarks relative to wrist position.

    Translates so wrist is at origin, scales by max distance from wrist.
    Works for (126,) single frame or (T, 126) sequence.
    """
    if landmarks.ndim == 1:
        return _normalize_single(landmarks)
    return np.array([_normalize_single(frame) for frame in landmarks])


def _normalize_single(landmarks: np.ndarray) -> np.ndarray:
    """Normalize a single frame's landmarks (126,)."""
    result = landmarks.copy()

    for hand_offset in [0, 63]:  # left hand, right hand
        hand = result[hand_offset:hand_offset + 63]
        if np.all(hand == 0):
            continue

        # Reshape to (21, 3)
        points = hand.reshape(21, 3)

        # Translate: wrist (landmark 0) to origin
        wrist = points[0].copy()
        points -= wrist

        # Scale by max distance from wrist
        distances = np.linalg.norm(points, axis=1)
        max_dist = np.max(distances)
        if max_dist > 0:
            points /= max_dist

        result[hand_offset:hand_offset + 63] = points.flatten()

    return result


def compute_motion_features(sequence: np.ndarray) -> np.ndarray:
    """Compute frame differencing (motion) features.

    Reference: SMIF concept from AIO2025 LS-ViT.
    Input: (T, 126) landmark sequence.
    Output: (T, 252) concatenation of position + motion features.
    """
    motion = np.zeros_like(sequence)
    motion[1:] = sequence[1:] - sequence[:-1]

    return np.concatenate([sequence, motion], axis=-1)


def augment_landmarks(landmarks: np.ndarray, seed: int = None) -> np.ndarray:
    """Apply data augmentation to landmarks.

    Augmentations (NO horizontal flip - VSL is hand-specific):
    - Random scaling (±10%)
    - Random rotation (±15 degrees, on x-y plane)
    - Gaussian noise
    - Random translation
    """
    rng = np.random.RandomState(seed)
    result = landmarks.copy()

    # Random scaling
    scale = rng.uniform(0.9, 1.1)
    result *= scale

    # Random rotation (x-y plane only)
    angle = rng.uniform(-15, 15) * np.pi / 180
    cos_a, sin_a = np.cos(angle), np.sin(angle)

    if result.ndim == 1:
        result = _rotate_single(result, cos_a, sin_a)
    else:
        for i in range(len(result)):
            result[i] = _rotate_single(result[i], cos_a, sin_a)

    # Gaussian noise
    noise = rng.normal(0, 0.002, result.shape)
    result += noise

    return result


def _rotate_single(landmarks: np.ndarray, cos_a: float, sin_a: float) -> np.ndarray:
    """Rotate x-y coordinates of a single frame."""
    result = landmarks.copy()
    for hand_offset in [0, 63]:
        hand = result[hand_offset:hand_offset + 63]
        if np.all(hand == 0):
            continue
        points = hand.reshape(21, 3)
        x_new = points[:, 0] * cos_a - points[:, 1] * sin_a
        y_new = points[:, 0] * sin_a + points[:, 1] * cos_a
        points[:, 0] = x_new
        points[:, 1] = y_new
        result[hand_offset:hand_offset + 63] = points.flatten()
    return result
