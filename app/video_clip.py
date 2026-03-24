from __future__ import annotations

from fastapi import HTTPException


def normalize_clip_range(
    total_duration_sec: float,
    clip_start_sec: float | None,
    clip_end_sec: float | None,
    max_process_sec: float,
) -> tuple[float, float]:
    total = max(0.0, float(total_duration_sec))
    start = 0.0 if clip_start_sec is None else float(clip_start_sec)
    end = total if clip_end_sec is None else float(clip_end_sec)
    if start < 0.0:
        raise HTTPException(status_code=400, detail="clip_start_sec must be >= 0.")
    if end <= 0.0:
        raise HTTPException(status_code=400, detail="clip_end_sec must be > 0.")
    if start >= total:
        raise HTTPException(status_code=400, detail="clip_start_sec exceeds video duration.")
    end = min(end, total)
    if end <= start:
        raise HTTPException(status_code=400, detail="clip_end_sec must be greater than clip_start_sec.")
    clip_len = end - start
    max_len = float(max(1.0, max_process_sec))
    if clip_len > max_len + 1e-6:
        raise HTTPException(
            status_code=413,
            detail=f"Selected clip duration exceeds {int(max_len)} seconds.",
        )
    return start, end

