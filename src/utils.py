"""Utility functions for VSL recognition."""

import json
import os
import numpy as np


def load_labels(path: str) -> dict:
    """Load label mapping from JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_labels(labels: dict, path: str):
    """Save label mapping to JSON file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(labels, f, ensure_ascii=False, indent=2)


def get_model_path(model_name: str) -> str:
    """Get the path to a model file."""
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, "models", model_name)


def get_labels_path() -> str:
    """Get the path to labels.json."""
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, "models", "labels.json")


VSL_ALPHABET = [
    "A", "B", "C", "D", "Đ", "E", "G", "H", "I", "K",
    "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U",
    "V", "X", "Y", "Sắc", "Huyền",
]

VSL_COMMON_WORDS = [
    "Xin chào", "Cảm ơn", "Xin lỗi", "Tôi", "Bạn",
    "Tên", "Tuổi", "Nhà", "Trường", "Ăn",
    "Uống", "Đi", "Đến", "Yêu", "Thích",
    "Giúp", "Hiểu", "Không", "Có", "Vui",
]
