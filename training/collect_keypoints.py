"""Collect keypoints from webcam for training data.

Usage:
    python training/collect_keypoints.py --label "A" --type static --num_samples 100
    python training/collect_keypoints.py --label "Xin chào" --type dynamic --num_sequences 30
"""

import argparse
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.keypoint_extractor import KeypointExtractor


def collect_static(label: str, output_dir: str, num_samples: int = 100):
    """Collect static sign keypoints from webcam."""
    extractor = KeypointExtractor(static_image_mode=False)
    save_dir = os.path.join(output_dir, label)
    os.makedirs(save_dir, exist_ok=True)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open webcam")
        return

    collected = 0
    print(f"Collecting '{label}' - press SPACE to capture, Q to quit")
    print(f"Target: {num_samples} samples")

    while collected < num_samples:
        ret, frame = cap.read()
        if not ret:
            break

        landmarks, annotated = extractor.extract_with_drawing(frame)
        has_hand = np.any(landmarks != 0)

        status = f"Collected: {collected}/{num_samples}"
        color = (0, 255, 0) if has_hand else (0, 0, 255)
        cv2.putText(annotated, status, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(annotated, f"Label: {label}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.imshow("Collect Static Signs", annotated)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord(" ") and has_hand:
            fname = f"sample_{collected:04d}.npy"
            np.save(os.path.join(save_dir, fname), landmarks)
            collected += 1
            print(f"  Captured {collected}/{num_samples}")

    cap.release()
    cv2.destroyAllWindows()
    extractor.close()
    print(f"Done! Collected {collected} samples for '{label}'")


def collect_dynamic(label: str, output_dir: str, num_sequences: int = 30,
                    sequence_length: int = 30):
    """Collect dynamic sign keypoint sequences from webcam."""
    extractor = KeypointExtractor(static_image_mode=False)
    save_dir = os.path.join(output_dir, label)
    os.makedirs(save_dir, exist_ok=True)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open webcam")
        return

    collected = 0
    recording = False
    buffer = []

    print(f"Collecting '{label}' - press SPACE to start/stop recording, Q to quit")
    print(f"Target: {num_sequences} sequences of {sequence_length} frames")

    while collected < num_sequences:
        ret, frame = cap.read()
        if not ret:
            break

        landmarks, annotated = extractor.extract_with_drawing(frame)

        if recording:
            buffer.append(landmarks)
            status = f"RECORDING [{len(buffer)}/{sequence_length}]"
            cv2.putText(annotated, status, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            if len(buffer) >= sequence_length:
                seq = np.array(buffer)
                fname = f"sequence_{collected:04d}.npy"
                np.save(os.path.join(save_dir, fname), seq)
                collected += 1
                buffer = []
                recording = False
                print(f"  Captured {collected}/{num_sequences}")
        else:
            status = f"Ready - Collected: {collected}/{num_sequences}"
            cv2.putText(annotated, status, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.putText(annotated, f"Label: {label}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.imshow("Collect Dynamic Signs", annotated)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord(" "):
            if not recording:
                recording = True
                buffer = []
                print("  Recording started...")
            else:
                recording = False
                buffer = []
                print("  Recording cancelled")

    cap.release()
    cv2.destroyAllWindows()
    extractor.close()
    print(f"Done! Collected {collected} sequences for '{label}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect VSL keypoints from webcam")
    parser.add_argument("--label", type=str, required=True, help="Sign label")
    parser.add_argument("--type", type=str, choices=["static", "dynamic"], required=True)
    parser.add_argument("--output_dir", type=str, default="data/processed")
    parser.add_argument("--num_samples", type=int, default=100, help="For static")
    parser.add_argument("--num_sequences", type=int, default=30, help="For dynamic")
    parser.add_argument("--sequence_length", type=int, default=30, help="For dynamic")
    args = parser.parse_args()

    output = os.path.join(args.output_dir, args.type)
    if args.type == "static":
        collect_static(args.label, output, args.num_samples)
    else:
        collect_dynamic(args.label, output, args.num_sequences, args.sequence_length)
