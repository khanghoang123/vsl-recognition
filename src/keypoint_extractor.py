"""MediaPipe hand landmark extraction for VSL recognition."""

import numpy as np
import mediapipe as mp
import cv2


class KeypointExtractor:
    """Extract hand landmarks using MediaPipe Hands.

    Extracts 21 landmarks per hand (left + right) with x, y, z coordinates,
    producing a 126-dimensional feature vector per frame.
    """

    NUM_LANDMARKS_PER_HAND = 21
    COORDS_PER_LANDMARK = 3
    FEATURES_PER_HAND = NUM_LANDMARKS_PER_HAND * COORDS_PER_LANDMARK  # 63
    TOTAL_FEATURES = FEATURES_PER_HAND * 2  # 126

    def __init__(
        self,
        static_image_mode: bool = False,
        max_num_hands: int = 2,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils

    def extract(self, frame: np.ndarray) -> np.ndarray:
        """Extract hand landmarks from a BGR frame.

        Returns a (126,) array: [left_hand(63), right_hand(63)].
        If a hand is not detected, its portion is filled with zeros.
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)

        left_hand = np.zeros(self.FEATURES_PER_HAND)
        right_hand = np.zeros(self.FEATURES_PER_HAND)

        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmarks, handedness in zip(
                results.multi_hand_landmarks, results.multi_handedness
            ):
                label = handedness.classification[0].label
                coords = []
                for lm in hand_landmarks.landmark:
                    coords.extend([lm.x, lm.y, lm.z])
                coords = np.array(coords)

                # MediaPipe mirrors: "Left" in results = user's right hand
                if label == "Left":
                    right_hand = coords
                else:
                    left_hand = coords

        return np.concatenate([left_hand, right_hand])

    def extract_with_drawing(self, frame: np.ndarray):
        """Extract landmarks and draw them on the frame.

        Returns (landmarks_array, annotated_frame).
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)
        annotated = frame.copy()

        left_hand = np.zeros(self.FEATURES_PER_HAND)
        right_hand = np.zeros(self.FEATURES_PER_HAND)

        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmarks, handedness in zip(
                results.multi_hand_landmarks, results.multi_handedness
            ):
                label = handedness.classification[0].label
                coords = []
                for lm in hand_landmarks.landmark:
                    coords.extend([lm.x, lm.y, lm.z])
                coords = np.array(coords)

                if label == "Left":
                    right_hand = coords
                else:
                    left_hand = coords

                self.mp_drawing.draw_landmarks(
                    annotated,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing.DrawingSpec(
                        color=(0, 255, 0), thickness=2, circle_radius=2
                    ),
                    self.mp_drawing.DrawingSpec(
                        color=(255, 255, 255), thickness=1
                    ),
                )

        landmarks = np.concatenate([left_hand, right_hand])
        return landmarks, annotated

    def close(self):
        self.hands.close()

    def __del__(self):
        self.close()
