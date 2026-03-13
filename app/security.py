from __future__ import annotations

import base64
import binascii
import hmac
import io
import re
from pathlib import Path

import cv2
import numpy as np
from fastapi import HTTPException, status
from PIL import Image

from .config import settings


SCRIPT_PATTERN = re.compile(
    rb"(<script|<\?php|powershell|cmd\.exe|/bin/sh|eval\(|base64_decode|onerror=)",
    flags=re.IGNORECASE,
)
SAFE_NAME_PATTERN = re.compile(r"[^a-zA-Z0-9._-]+")

IMAGE_MAGIC = {
    b"\x89PNG\r\n\x1a\n": "png",
    b"\xff\xd8\xff": "jpg",
    b"RIFF": "webp",
}
VIDEO_MAGIC = {
    b"\x1a\x45\xdf\xa3": "webm",
    b"ftyp": "mp4",
}


def _read_head(path: Path, size: int = 8192) -> bytes:
    with path.open("rb") as f:
        return f.read(size)


def safe_filename(name: str) -> str:
    clean = Path(name).name.strip()
    clean = SAFE_NAME_PATTERN.sub("_", clean)
    return clean[:120] or "upload.bin"


def parse_hex_color(bg_color: str | None) -> tuple[int, int, int] | None:
    if not bg_color:
        return None
    c = bg_color.strip().lower()
    if c in {"transparent", "none"}:
        return None
    if not re.fullmatch(r"#[0-9a-f]{6}", c):
        raise HTTPException(status_code=400, detail="Invalid bg_color. Use #RRGGBB or transparent.")
    r = int(c[1:3], 16)
    g = int(c[3:5], 16)
    b = int(c[5:7], 16)
    return r, g, b


def _assert_non_script(data: bytes):
    if SCRIPT_PATTERN.search(data):
        raise HTTPException(status_code=400, detail="Blocked: suspicious script-like payload detected.")


def _sniff_image(head: bytes) -> str | None:
    for sig, fmt in IMAGE_MAGIC.items():
        if sig == b"RIFF" and head.startswith(sig) and b"WEBP" in head[:32]:
            return fmt
        if head.startswith(sig):
            return fmt
    return None


def _sniff_video(head: bytes) -> str | None:
    for sig, fmt in VIDEO_MAGIC.items():
        if sig == b"ftyp" and sig in head[:32]:
            return fmt
        if head.startswith(sig):
            return fmt
    return None


def validate_image_file(path: Path, max_mb: int):
    if not path.exists():
        raise HTTPException(status_code=400, detail="Image file missing.")
    size_mb = path.stat().st_size / (1024 * 1024)
    if size_mb > max_mb:
        raise HTTPException(status_code=413, detail=f"Image exceeds {max_mb} MB.")
    head = _read_head(path)
    _assert_non_script(head)
    if _sniff_image(head) is None:
        raise HTTPException(status_code=400, detail="Unsupported image format.")
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise HTTPException(status_code=400, detail="Image cannot be decoded.")
    h, w = image.shape[:2]
    if h < 2 or w < 2:
        raise HTTPException(status_code=400, detail="Image dimensions are invalid.")
    max_pixels = max(1, int(settings.engine_max_image_pixels))
    if int(h * w) > max_pixels:
        raise HTTPException(
            status_code=413,
            detail=f"Image resolution exceeds limit ({max_pixels} pixels).",
        )


def validate_video_file(path: Path, max_mb: int, max_seconds: int):
    if not path.exists():
        raise HTTPException(status_code=400, detail="Video file missing.")
    size_mb = path.stat().st_size / (1024 * 1024)
    if size_mb > max_mb:
        raise HTTPException(status_code=413, detail=f"Video exceeds {max_mb} MB.")
    head = _read_head(path)
    _assert_non_script(head)
    if _sniff_video(head) is None:
        raise HTTPException(status_code=400, detail="Unsupported video format. Use MP4/WEBM.")
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise HTTPException(status_code=400, detail="Video cannot be decoded.")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    if width < 16 or height < 16:
        raise HTTPException(status_code=400, detail="Video resolution is invalid.")
    max_pixels = max(1, int(settings.engine_max_video_pixels))
    if int(width * height) > max_pixels:
        raise HTTPException(
            status_code=413,
            detail=f"Video resolution exceeds limit ({max_pixels} pixels).",
        )
    duration = frames / max(fps, 1e-6)
    if duration > max_seconds:
        raise HTTPException(
            status_code=413,
            detail=f"Video duration exceeds {max_seconds} seconds.",
        )


def data_url_to_gray_mask(data_url: str | None) -> np.ndarray | None:
    if not data_url:
        return None
    value = data_url.strip()
    if not value:
        return None
    prefix = "base64,"
    idx = value.find(prefix)
    if idx > 0:
        value = value[idx + len(prefix) :]
    try:
        payload = base64.b64decode(value)
    except binascii.Error as exc:
        raise HTTPException(status_code=400, detail="Mask base64 decode failed.") from exc
    _assert_non_script(payload[:1024])
    try:
        img = Image.open(io.BytesIO(payload)).convert("L")
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail="Mask image decode failed.") from exc
    arr = np.array(img, dtype=np.uint8)
    return np.where(arr > 0, 255, 0).astype(np.uint8)


def assert_internal_token(token: str | None, expected: str):
    if not token or not hmac.compare_digest(str(token), str(expected)):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unauthorized engine access.",
        )
