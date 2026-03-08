from __future__ import annotations

import json
import shutil
import time
import uuid
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from .config import settings
from .processing import clear_model_cache, extract_video_frame, process_image, process_video
from .schemas import FrameResponse, HealthResponse, ProcessResponse
from .security import (
    assert_internal_token,
    data_url_to_gray_mask,
    parse_hex_color,
    safe_filename,
    validate_image_file,
    validate_video_file,
)

app = FastAPI(title=settings.app_name, version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

JOBS: dict[str, dict] = {}
JOB_INDEX_DIR = settings.workdir / "job-index"
JOB_INDEX_DIR.mkdir(parents=True, exist_ok=True)


def _manifest_path(job_id: str) -> Path:
    return JOB_INDEX_DIR / f"{job_id}.json"


def _write_manifest(job_id: str, info: dict):
    manifest = _manifest_path(job_id)
    manifest.write_text(json.dumps(info), encoding="utf-8")


def _read_manifest(job_id: str) -> dict | None:
    manifest = _manifest_path(job_id)
    if not manifest.exists():
        return None
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            return payload
    except Exception:  # noqa: BLE001
        return None
    return None


def _delete_manifest(job_id: str):
    _manifest_path(job_id).unlink(missing_ok=True)


def _job_root_from_output(path: Path) -> Path:
    # Expected layout: .../jobs/<job-folder>/output/output.ext
    if len(path.parents) >= 2:
        return path.parents[1]
    return path.parent


def _cleanup_job_files(path: Path):
    if not path.is_absolute():
        return
    root = _job_root_from_output(path)
    root_parts = {part.lower() for part in root.parts}
    if "jobs" not in root_parts:
        return
    if root.exists() and root.is_dir():
        shutil.rmtree(root, ignore_errors=True)
        return
    if path.exists():
        path.unlink(missing_ok=True)


def _auth(x_engine_token: str | None = Header(default=None)):
    assert_internal_token(x_engine_token, settings.engine_shared_token)


def _clean_old_jobs():
    cutoff = time.time() - (settings.engine_keep_jobs_hours * 3600)
    expired = [k for k, v in JOBS.items() if v["created_at"] < cutoff]
    for job_id in expired:
        info = JOBS.pop(job_id, None)
        if not info:
            continue
        path = Path(info["path"])
        _cleanup_job_files(path)
        _delete_manifest(job_id)

    for manifest in JOB_INDEX_DIR.glob("*.json"):
        job_id = manifest.stem
        payload = _read_manifest(job_id)
        if not payload:
            manifest.unlink(missing_ok=True)
            continue
        created_at = float(payload.get("created_at", 0.0))
        if created_at >= cutoff:
            continue
        path_str = str(payload.get("path", ""))
        if path_str:
            _cleanup_job_files(Path(path_str))
        manifest.unlink(missing_ok=True)


def _write_upload(file: UploadFile, target_dir: Path) -> Path:
    target_dir.mkdir(parents=True, exist_ok=True)
    name = safe_filename(file.filename or "upload.bin")
    out = target_dir / name
    with out.open("wb") as f:
        while True:
            chunk = file.file.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
    return out


def _register_job(path: Path, output_name: str, output_mime: str, media_type: str) -> str:
    _clean_old_jobs()
    job_id = uuid.uuid4().hex
    info = {
        "path": str(path),
        "output_name": output_name,
        "output_mime": output_mime,
        "media_type": media_type,
        "created_at": time.time(),
    }
    JOBS[job_id] = info
    _write_manifest(job_id, info)
    return job_id


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="ok", app=settings.app_name)


@app.post("/v1/process/image", response_model=ProcessResponse, dependencies=[Depends(_auth)])
def process_image_endpoint(
    file: UploadFile = File(...),
    quality: str = Form(default="ultra"),
    bg_color: str = Form(default="transparent"),
    keep_mask_data_url: str | None = Form(default=None),
    erase_mask_data_url: str | None = Form(default=None),
):
    quality = "ultra" if quality not in {"ultra", "balanced"} else quality
    bg_rgb = parse_hex_color(bg_color)
    if bg_rgb is not None and not settings.engine_allow_color_bg:
        raise HTTPException(status_code=400, detail="Color background is disabled by policy.")

    job_root = settings.workdir / "jobs" / uuid.uuid4().hex
    src = _write_upload(file, job_root / "input")
    validate_image_file(src, settings.engine_max_image_mb)
    keep_mask = data_url_to_gray_mask(keep_mask_data_url)
    erase_mask = data_url_to_gray_mask(erase_mask_data_url)

    output_name = "output.png"
    output = job_root / "output" / output_name
    result = process_image(
        image_path=src,
        output_path=output,
        quality=quality,
        erase_mask=erase_mask,
        keep_mask=keep_mask,
        bg_rgb=bg_rgb,
    )
    job_id = _register_job(output, output_name=output_name, output_mime=result["output_mime"], media_type="image")
    return ProcessResponse(
        job_id=job_id,
        media_type="image",
        quality=quality,
        output_name=output_name,
        output_mime=result["output_mime"],
        download_url=f"/v1/jobs/{job_id}/download",
        status="completed",
        qc_suspect_frames=result["qc_suspect_frames"],
        model_used=result["main_model"],
    )


@app.post("/v1/process/video", response_model=ProcessResponse, dependencies=[Depends(_auth)])
def process_video_endpoint(
    file: UploadFile = File(...),
    quality: str = Form(default="ultra"),
    bg_color: str = Form(default="transparent"),
    keep_mask_data_url: str | None = Form(default=None),
    erase_mask_data_url: str | None = Form(default=None),
):
    quality = "ultra" if quality not in {"ultra", "balanced"} else quality
    bg_rgb = parse_hex_color(bg_color)
    if bg_rgb is not None and not settings.engine_allow_color_bg:
        raise HTTPException(status_code=400, detail="Color background is disabled by policy.")

    job_root = settings.workdir / "jobs" / uuid.uuid4().hex
    src = _write_upload(file, job_root / "input")
    validate_video_file(src, settings.engine_max_video_mb, settings.engine_max_video_seconds)
    keep_mask = data_url_to_gray_mask(keep_mask_data_url)
    erase_mask = data_url_to_gray_mask(erase_mask_data_url)

    ext = "webm" if bg_rgb is None else "mp4"
    mime = "video/webm" if bg_rgb is None else "video/mp4"
    output_name = f"output.{ext}"
    output = job_root / "output" / output_name
    result = process_video(
        video_path=src,
        output_path=output,
        quality=quality,
        erase_mask=erase_mask,
        keep_mask=keep_mask,
        bg_rgb=bg_rgb,
    )
    job_id = _register_job(output, output_name=output_name, output_mime=mime, media_type="video")
    return ProcessResponse(
        job_id=job_id,
        media_type="video",
        quality=quality,
        output_name=output_name,
        output_mime=mime,
        download_url=f"/v1/jobs/{job_id}/download",
        status="completed",
        qc_suspect_frames=result["qc_suspect_frames"],
        model_used=result["main_model"],
    )


@app.post("/v1/frame/extract", response_model=FrameResponse, dependencies=[Depends(_auth)])
def frame_extract(
    file: UploadFile = File(...),
    time_sec: float = Form(default=0.0),
):
    job_root = settings.workdir / "jobs" / uuid.uuid4().hex
    src = _write_upload(file, job_root / "input")
    validate_video_file(src, settings.engine_max_video_mb, settings.engine_max_video_seconds)
    frame = extract_video_frame(src, time_sec=time_sec)
    return FrameResponse(**frame)


@app.get("/v1/jobs/{job_id}/download", dependencies=[Depends(_auth)])
def download_job(job_id: str):
    info = JOBS.get(job_id)
    if not info:
        info = _read_manifest(job_id)
        if info:
            JOBS[job_id] = info
    if not info:
        raise HTTPException(status_code=404, detail="Job not found or expired.")
    path = Path(info["path"])
    if not path.exists():
        raise HTTPException(status_code=404, detail="Output file is unavailable.")
    return FileResponse(path, filename=info["output_name"], media_type=info["output_mime"])


@app.post("/v1/cache/clear", dependencies=[Depends(_auth)])
def clear_cache():
    clear_model_cache()
    return {"status": "ok"}
