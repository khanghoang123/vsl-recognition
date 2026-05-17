"""Gesture Router: classify input as static or dynamic gesture.

Based on temporal motion analysis of landmark sequences.
Reference: "A Unified Hand-Landmark-Based DL Framework" (EAI 2026, Binh Duong Univ.)
"""

import numpy as np
from collections import deque


class GestureRouter:
    """Routes input to static or dynamic classifier based on hand motion."""

    def __init__(
        self,
        buffer_size: int = 30,
        motion_threshold: float = 0.015,
        min_frames_for_dynamic: int = 15,
    ):
        self.buffer_size = buffer_size
        self.motion_threshold = motion_threshold
        self.min_frames_for_dynamic = min_frames_for_dynamic
        self.landmark_buffer = deque(maxlen=buffer_size)

    def add_frame(self, landmarks: np.ndarray):
        """Add a frame's landmarks to the buffer."""
        self.landmark_buffer.append(landmarks.copy())

    def get_gesture_type(self) -> str:
        """Determine if current gesture is static or dynamic.

        Returns "static", "dynamic", or "insufficient" if not enough frames.
        """
        if len(self.landmark_buffer) < self.min_frames_for_dynamic:
            return "static"

        sequence = np.array(list(self.landmark_buffer))

        # Only consider frames where at least one hand is detected
        hand_present = np.any(sequence != 0, axis=1)
        if np.sum(hand_present) < self.min_frames_for_dynamic:
            return "static"

        valid_frames = sequence[hand_present]
        if len(valid_frames) < 2:
            return "static"

        diffs = np.diff(valid_frames, axis=0)
        mean_displacement = np.mean(np.abs(diffs))

        if mean_displacement > self.motion_threshold:
            return "dynamic"
        return "static"

    def get_sequence(self) -> np.ndarray:
        """Get the current landmark sequence for dynamic classification."""
        if len(self.landmark_buffer) == 0:
            return np.zeros((self.buffer_size, 126))

        sequence = np.array(list(self.landmark_buffer))

        # Pad if not enough frames (temporal padding: repeat last frame)
        if len(sequence) < self.buffer_size:
            pad_count = self.buffer_size - len(sequence)
            padding = np.tile(sequence[-1:], (pad_count, 1))
            sequence = np.concatenate([sequence, padding], axis=0)

        return sequence

    def reset(self):
        """Clear the landmark buffer."""
        self.landmark_buffer.clear()
