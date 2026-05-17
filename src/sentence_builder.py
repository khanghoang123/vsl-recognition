"""Sentence builder: accumulates recognized signs into sentences."""

import time
from collections import deque


class SentenceBuilder:
    """Builds sentences from a stream of recognized signs."""

    def __init__(
        self,
        cooldown_seconds: float = 1.5,
        max_history: int = 50,
    ):
        self.cooldown = cooldown_seconds
        self.history = deque(maxlen=max_history)
        self.sentence_parts = []
        self.last_sign = None
        self.last_sign_time = 0.0

    def add_sign(self, sign: str, confidence: float):
        """Add a recognized sign. Applies cooldown to avoid duplicates."""
        now = time.time()

        if sign == self.last_sign and (now - self.last_sign_time) < self.cooldown:
            return

        self.last_sign = sign
        self.last_sign_time = now
        self.sentence_parts.append(sign)
        self.history.append({
            "sign": sign,
            "confidence": confidence,
            "time": now,
        })

    def get_sentence(self) -> str:
        """Get the current accumulated sentence."""
        return " ".join(self.sentence_parts)

    def get_history(self) -> list:
        """Get recognition history."""
        return list(self.history)

    def clear_sentence(self):
        """Clear the current sentence."""
        self.sentence_parts.clear()

    def backspace(self):
        """Remove the last sign from the sentence."""
        if self.sentence_parts:
            self.sentence_parts.pop()

    def add_space(self):
        """Add a space/separator."""
        self.sentence_parts.append(" ")
