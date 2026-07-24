import numpy as np
import pytest

from vsl_recognition.preprocessing import (
    filter_bad_frames,
    is_bad_frame,
    uniform_indices,
)


def test_uniform_indices_cover_full_clip():
    assert uniform_indices(30, 4).tolist() == [0, 9, 19, 29]


def test_uniform_indices_pad_short_clip():
    assert uniform_indices(2, 4).tolist() == [0, 1, 1, 1]


def test_uniform_indices_reject_empty_video():
    with pytest.raises(ValueError):
        uniform_indices(0, 16)


def test_bad_frame_filter_falls_back_when_too_few_valid_frames():
    blank = np.zeros((8, 8, 3), dtype=np.uint8)
    textured = np.indices((8, 8)).sum(axis=0).astype(np.uint8)
    textured = np.repeat(textured[..., None] * 20, 3, axis=2)
    frames = np.asarray([blank, textured])
    assert is_bad_frame(blank)
    assert not is_bad_frame(textured)
    assert np.array_equal(filter_bad_frames(frames, min_valid_frames=2), frames)
