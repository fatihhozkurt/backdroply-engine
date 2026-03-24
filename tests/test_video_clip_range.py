from __future__ import annotations

import pytest
from fastapi import HTTPException

from app.video_clip import normalize_clip_range


def test_normalize_clip_range_defaults_to_full_video():
    start, end = normalize_clip_range(8.0, None, None, 18.0)
    assert start == pytest.approx(0.0)
    assert end == pytest.approx(8.0)


def test_normalize_clip_range_clamps_end_to_total():
    start, end = normalize_clip_range(8.0, 2.5, 19.0, 18.0)
    assert start == pytest.approx(2.5)
    assert end == pytest.approx(8.0)


def test_normalize_clip_range_rejects_negative_start():
    with pytest.raises(HTTPException) as exc:
        normalize_clip_range(8.0, -0.1, 3.0, 18.0)
    assert exc.value.status_code == 400


def test_normalize_clip_range_rejects_invalid_order():
    with pytest.raises(HTTPException) as exc:
        normalize_clip_range(8.0, 3.0, 3.0, 18.0)
    assert exc.value.status_code == 400


def test_normalize_clip_range_rejects_clip_longer_than_processing_limit():
    with pytest.raises(HTTPException) as exc:
        normalize_clip_range(20.0, 1.0, 8.5, 5.0)
    assert exc.value.status_code == 413
