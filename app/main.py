from __future__ import annotations

import json
import logging
import multiprocessing as mp
import subprocess
import shutil
import threading
import time
import uuid
from pathlib import Path

import imageio_ffmpeg
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from .config import settings
from .processing import (
    clear_model_cache,
    ensure_runtime_provider_ready,
    extract_video_frame,
    force_cpu_only_mode,
    ort_runtime_info,
    prewarm_models,
    process_image,
    process_video,
)
from .schemas import FrameResponse, HealthResponse, ProcessResponse
from .security import (
    assert_internal_token,
    data_url_to_gray_mask,
    parse_hex_color,
    safe_filename,
    validate_image_file,
    validate_video_file,
)
from .video_clip import normalize_clip_range

app = FastAPI(title=settings.app_name, version="1.0.0")
logger = logging.getLogger("backdroply.engine")
processing_logger = logging.getLogger("backdroply.engine.processing")
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
PROCESS_SEMAPHORE = threading.BoundedSemaphore(max(1, int(settings.engine_max_concurrent_jobs)))
ACTIVE_PROCESS_COUNT = 0
PROCESS_COUNT_LOCK = threading.Lock()


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


def _processing_slot():
    global ACTIVE_PROCESS_COUNT
    wait_sec = max(0.0, float(settings.engine_queue_wait_seconds))
    acquired = PROCESS_SEMAPHORE.acquire(timeout=wait_sec)
    if not acquired:
        raise HTTPException(
            status_code=429,
            detail="Engine is busy. Retry shortly.",
        )
    with PROCESS_COUNT_LOCK:
        ACTIVE_PROCESS_COUNT += 1
    try:
        yield
    finally:
        with PROCESS_COUNT_LOCK:
            ACTIVE_PROCESS_COUNT = max(0, ACTIVE_PROCESS_COUNT - 1)
        PROCESS_SEMAPHORE.release()


@app.on_event("startup")
def _startup_prewarm():
    level_name = (settings.engine_log_level or "INFO").strip().upper()
    log_level = getattr(logging, level_name, logging.INFO)
    logging.getLogger().setLevel(log_level)
    for target in (logger, processing_logger):
        target.setLevel(log_level)
        if not target.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
            target.addHandler(handler)
        target.propagate = False
    runtime_probe = ensure_runtime_provider_ready()
    logger.warning(
        "Runtime probe mode=%s reason=%s",
        runtime_probe.get("mode", "unknown"),
        runtime_probe.get("reason", ""),
    )
    if not settings.engine_prewarm_on_startup:
        return
    try:
        prewarm_models(settings.engine_prewarm_quality)
        return
    except Exception as exc:  # noqa: BLE001
        logger.warning("Startup prewarm failed on configured providers: %s", exc)

    if not settings.engine_ort_allow_cpu_fallback:
        logger.warning("CPU fallback is disabled. Continuing without prewarm.")
        return

    try:
        force_cpu_only_mode()
        prewarm_models(settings.engine_prewarm_quality)
        logger.warning("Engine switched to CPU-only mode after prewarm failure.")
    except Exception as exc:  # noqa: BLE001
        logger.warning("CPU fallback prewarm failed too. Continuing without prewarm: %s", exc)


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


def _normalize_clip_range(
    total_duration_sec: float,
    clip_start_sec: float | None,
    clip_end_sec: float | None,
) -> tuple[float, float]:
    return normalize_clip_range(
        total_duration_sec=total_duration_sec,
        clip_start_sec=clip_start_sec,
        clip_end_sec=clip_end_sec,
        max_process_sec=float(max(1, int(settings.engine_max_video_seconds))),
    )


def _trim_video_for_clip(src: Path, dst: Path, start_sec: float, end_sec: float):
    ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()
    if not Path(ffmpeg_bin).exists():
        raise HTTPException(status_code=500, detail="ffmpeg binary not found.")
    dst.parent.mkdir(parents=True, exist_ok=True)

    cmd_reencode = [
        ffmpeg_bin,
        "-y",
        "-ss",
        f"{start_sec:.3f}",
        "-to",
        f"{end_sec:.3f}",
        "-i",
        str(src),
        "-map",
        "0:v:0",
        "-map",
        "0:a:0?",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "160k",
        "-movflags",
        "+faststart",
        str(dst),
    ]
    completed = subprocess.run(cmd_reencode, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=False)
    if completed.returncode != 0 or not dst.exists() or dst.stat().st_size <= 0:
        tail = (completed.stderr or b"").decode("utf-8", errors="ignore")[-500:]
        raise HTTPException(status_code=500, detail=f"Video clipping failed: {tail or 'unknown ffmpeg error'}")


def _video_worker(conn, kwargs: dict):
    try:
        result = process_video(**kwargs)
        conn.send({"ok": True, "result": result})
    except Exception as exc:  # noqa: BLE001
        conn.send({"ok": False, "error": str(exc)})
    finally:
        conn.close()


def _resolve_video_worker_timeout_sec(request_quality: str) -> int:
    configured = int(getattr(settings, "engine_video_worker_timeout_sec", 0) or 0)
    if configured > 0:
        return configured
    factor = float(getattr(settings, "engine_video_worker_timeout_factor", 20.0) or 20.0)
    factor = max(8.0, factor)
    derived = int(max(120, int(settings.engine_max_video_seconds) * factor))
    if (request_quality or "").strip().lower() == "ultra":
        derived = max(derived, 180)
    return derived


def _run_video_isolated(request_quality: str, worker_timeout_sec: int | None = None, **kwargs) -> dict:
    # Guard engine against native crashes in cv2/onnx stacks during long video jobs.
    timeout_sec = (
        int(worker_timeout_sec) if worker_timeout_sec is not None else _resolve_video_worker_timeout_sec(request_quality)
    )
    timeout_sec = max(120, timeout_sec)
    logger.info("Video worker start: quality=%s timeout=%ss", request_quality, timeout_sec)
    ctx = mp.get_context("spawn")
    parent_conn, child_conn = ctx.Pipe(duplex=False)
    proc = ctx.Process(target=_video_worker, args=(child_conn, kwargs), daemon=True)
    proc.start()
    child_conn.close()

    payload: dict | None = None
    deadline = time.time() + float(timeout_sec)
    while time.time() < deadline:
        if parent_conn.poll(0.2):
            try:
                payload = parent_conn.recv()
            except EOFError:
                payload = None
            break
        if not proc.is_alive():
            break

    if payload is None:
        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=3)
            logger.warning("Video worker timeout: quality=%s timeout=%ss", request_quality, timeout_sec)
            raise HTTPException(status_code=504, detail="Video processing timed out in engine.")
        exit_code = proc.exitcode
        logger.warning("Video worker exit: quality=%s code=%s", request_quality, exit_code)
        raise HTTPException(status_code=502, detail=f"Video processing worker exited unexpectedly (code={exit_code}).")

    proc.join(timeout=3)
    if not bool(payload.get("ok")):
        error_text = str(payload.get("error", "unknown engine error"))
        raise HTTPException(status_code=502, detail=f"Video processing failed: {error_text}")
    result = payload.get("result")
    if not isinstance(result, dict):
        raise HTTPException(status_code=502, detail="Video processing returned invalid result.")
    return result


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="ok", app=settings.app_name)


@app.get("/v1/stats", dependencies=[Depends(_auth)])
def engine_stats():
    with PROCESS_COUNT_LOCK:
        active = int(ACTIVE_PROCESS_COUNT)
    max_jobs = max(1, int(settings.engine_max_concurrent_jobs))
    pending = max(0, len(JOBS))
    return {
        "status": "ok",
        "active_processing_jobs": active,
        "max_concurrent_jobs": max_jobs,
        "queue_wait_seconds": float(settings.engine_queue_wait_seconds),
        "registered_jobs": pending,
        "runtime": ort_runtime_info(),
    }


@app.post("/v1/process/image", response_model=ProcessResponse, dependencies=[Depends(_auth), Depends(_processing_slot)])
def process_image_endpoint(
    file: UploadFile = File(...),
    quality: str = Form(default="ultra"),
    bg_color: str = Form(default="transparent"),
    keep_mask_data_url: str | None = Form(default=None),
    erase_mask_data_url: str | None = Form(default=None),
    watermark_enabled: bool | None = Form(default=None),
):
    started_at = time.perf_counter()
    quality = "ultra" if quality not in {"ultra", "balanced"} else quality
    bg_rgb = parse_hex_color(bg_color)
    if bg_rgb is not None and not settings.engine_allow_color_bg:
        raise HTTPException(status_code=400, detail="Color background is disabled by policy.")

    job_root = settings.workdir / "jobs" / uuid.uuid4().hex
    src = _write_upload(file, job_root / "input")
    logger.info(
        "image.request.start filename=%s size_bytes=%s quality=%s bg_color=%s watermark=%s",
        src.name,
        src.stat().st_size if src.exists() else -1,
        quality,
        bg_color,
        watermark_enabled,
    )
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
        watermark_enabled=watermark_enabled,
    )
    job_id = _register_job(output, output_name=output_name, output_mime=result["output_mime"], media_type="image")
    logger.info(
        "image.request.done job_id=%s elapsed_sec=%.3f model=%s qc_suspect=%s",
        job_id,
        time.perf_counter() - started_at,
        result["main_model"],
        result["qc_suspect_frames"],
    )
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


@app.post("/v1/process/video", response_model=ProcessResponse, dependencies=[Depends(_auth), Depends(_processing_slot)])
def process_video_endpoint(
    file: UploadFile = File(...),
    quality: str = Form(default="ultra"),
    bg_color: str = Form(default="transparent"),
    keep_mask_data_url: str | None = Form(default=None),
    erase_mask_data_url: str | None = Form(default=None),
    watermark_enabled: bool | None = Form(default=None),
    clip_start_sec: float | None = Form(default=None),
    clip_end_sec: float | None = Form(default=None),
):
    started_at = time.perf_counter()
    quality = "ultra" if quality not in {"ultra", "balanced"} else quality
    bg_rgb = parse_hex_color(bg_color)
    if bg_rgb is not None and not settings.engine_allow_color_bg:
        raise HTTPException(status_code=400, detail="Color background is disabled by policy.")

    job_root = settings.workdir / "jobs" / uuid.uuid4().hex
    src = _write_upload(file, job_root / "input")
    logger.info(
        "video.request.start filename=%s size_bytes=%s quality=%s bg_color=%s watermark=%s clip_start=%s clip_end=%s isolated_worker=%s",
        src.name,
        src.stat().st_size if src.exists() else -1,
        quality,
        bg_color,
        watermark_enabled,
        clip_start_sec,
        clip_end_sec,
        settings.engine_video_use_isolated_worker,
    )
    raw_meta = validate_video_file(src, settings.engine_max_video_mb, settings.engine_max_video_upload_seconds)
    keep_mask = data_url_to_gray_mask(keep_mask_data_url)
    erase_mask = data_url_to_gray_mask(erase_mask_data_url)

    clip_start, clip_end = _normalize_clip_range(
        total_duration_sec=float(raw_meta.get("duration", 0.0)),
        clip_start_sec=clip_start_sec,
        clip_end_sec=clip_end_sec,
    )
    logger.info(
        "video.request.clip_normalized duration_sec=%.3f clip_start=%.3f clip_end=%.3f clip_len=%.3f",
        float(raw_meta.get("duration", 0.0)),
        clip_start,
        clip_end,
        max(0.0, clip_end - clip_start),
    )
    source_for_process = src
    if clip_start > 0.0 or clip_end < float(raw_meta.get("duration", 0.0)) - 1e-6:
        clipped = job_root / "input" / "clip.mp4"
        _trim_video_for_clip(src, clipped, clip_start, clip_end)
        validate_video_file(clipped, settings.engine_max_video_mb, settings.engine_max_video_seconds)
        source_for_process = clipped
        logger.info(
            "video.request.clip_rendered clip_path=%s clip_size_bytes=%s",
            str(clipped),
            clipped.stat().st_size if clipped.exists() else -1,
        )
    elif float(raw_meta.get("duration", 0.0)) > float(max(1, int(settings.engine_max_video_seconds))):
        raise HTTPException(
            status_code=413,
            detail=f"Video exceeds {int(settings.engine_max_video_seconds)}s processing limit. Select a shorter clip range.",
        )

    ext = "webm" if bg_rgb is None else "mp4"
    mime = "video/webm" if bg_rgb is None else "video/mp4"
    output_name = f"output.{ext}"
    output = job_root / "output" / output_name

    def run_video_job(request_quality: str) -> dict:
        common_kwargs = dict(
            video_path=source_for_process,
            output_path=output,
            quality=request_quality,
            erase_mask=erase_mask,
            keep_mask=keep_mask,
            bg_rgb=bg_rgb,
            watermark_enabled=watermark_enabled,
        )
        if settings.engine_video_use_isolated_worker:
            return _run_video_isolated(request_quality=request_quality, **common_kwargs)
        try:
            return process_video(**common_kwargs)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=502, detail=f"Video processing failed: {exc}") from exc

    actual_quality = quality
    try:
        result = run_video_job(quality)
    except HTTPException as exc:
        detail_text = str(exc.detail or "")
        lower_detail = detail_text.lower()
        if quality == "ultra" and ("worker exited unexpectedly" in lower_detail or "timed out" in lower_detail):
            actual_quality = "balanced"
            logger.warning("Ultra video pass failed (%s). Retrying with balanced profile.", detail_text)
            result = run_video_job("balanced")
        else:
            logger.warning("video.request.failed quality=%s detail=%s", quality, detail_text)
            raise
    job_id = _register_job(output, output_name=output_name, output_mime=mime, media_type="video")
    logger.info(
        "video.request.done job_id=%s elapsed_sec=%.3f quality_requested=%s quality_actual=%s model=%s qc_suspect=%s",
        job_id,
        time.perf_counter() - started_at,
        quality,
        actual_quality,
        result["main_model"],
        result["qc_suspect_frames"],
    )
    return ProcessResponse(
        job_id=job_id,
        media_type="video",
        quality=actual_quality,
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
    validate_video_file(src, settings.engine_max_video_mb, settings.engine_max_video_upload_seconds)
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


@app.post("/v1/prewarm", dependencies=[Depends(_auth)])
def prewarm_endpoint(quality: str | None = None):
    normalized = None if quality is None or not quality.strip() else quality.strip().lower()
    if normalized not in {None, "ultra", "balanced"}:
        raise HTTPException(status_code=400, detail="quality must be ultra, balanced or empty.")
    loaded = prewarm_models(normalized)
    return {"status": "ok", "loaded_models": loaded}
