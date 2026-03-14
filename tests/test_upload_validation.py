from __future__ import annotations

from pathlib import Path

import cv2
import pytest
from fastapi import HTTPException

from app.security import validate_image_file, validate_video_file


def test_validate_image_file_rejects_script_signature(tmp_path: Path):
    payload = b"\x89PNG\r\n\x1a\n<script>alert('x')</script>"
    image_path = tmp_path / "evil.png"
    image_path.write_bytes(payload)

    with pytest.raises(HTTPException) as exc:
        validate_image_file(image_path, max_mb=1)

    assert exc.value.status_code == 400
    assert "script-like payload" in str(exc.value.detail).lower()


def test_validate_image_file_rejects_non_image_payload(tmp_path: Path):
    image_path = tmp_path / "fake.png"
    image_path.write_bytes(b"not-an-image")

    with pytest.raises(HTTPException) as exc:
        validate_image_file(image_path, max_mb=1)

    assert exc.value.status_code == 400


def test_validate_video_file_rejects_over_duration(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    video_path = tmp_path / "sample.mp4"
    # "ftyp" in first bytes is enough for magic sniff.
    video_path.write_bytes(b"\x00\x00\x00\x18ftypisom")

    class FakeCapture:
        def isOpened(self):
            return True

        def get(self, prop):
            if prop == cv2.CAP_PROP_FPS:
                return 25.0
            if prop == cv2.CAP_PROP_FRAME_COUNT:
                return 250.0  # 10 seconds
            if prop == cv2.CAP_PROP_FRAME_WIDTH:
                return 1280.0
            if prop == cv2.CAP_PROP_FRAME_HEIGHT:
                return 720.0
            return 0.0

        def release(self):
            return None

    monkeypatch.setattr(cv2, "VideoCapture", lambda _: FakeCapture())

    with pytest.raises(HTTPException) as exc:
        validate_video_file(video_path, max_mb=10, max_seconds=3)

    assert exc.value.status_code == 413
    assert "duration exceeds" in str(exc.value.detail).lower()


def test_validate_video_file_rejects_invalid_resolution(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    video_path = tmp_path / "sample.webm"
    video_path.write_bytes(b"\x1a\x45\xdf\xa3dummy")

    class FakeCapture:
        def isOpened(self):
            return True

        def get(self, prop):
            if prop == cv2.CAP_PROP_FPS:
                return 30.0
            if prop == cv2.CAP_PROP_FRAME_COUNT:
                return 60.0
            if prop == cv2.CAP_PROP_FRAME_WIDTH:
                return 8.0
            if prop == cv2.CAP_PROP_FRAME_HEIGHT:
                return 8.0
            return 0.0

        def release(self):
            return None

    monkeypatch.setattr(cv2, "VideoCapture", lambda _: FakeCapture())

    with pytest.raises(HTTPException) as exc:
        validate_video_file(video_path, max_mb=10, max_seconds=10)

    assert exc.value.status_code == 400
    assert "resolution is invalid" in str(exc.value.detail).lower()

