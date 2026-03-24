from __future__ import annotations

import base64
import logging
import os
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np
import onnxruntime as ort
from rembg import new_session, remove

from .config import settings

DEFAULT_MODEL_CANDIDATES = ["u2netp"]
_SESSION_CACHE: dict[str, object] = {}
_SESSION_PROVIDER_CACHE: dict[str, list[str]] = {}
_SESSION_ERRORS: dict[str, list[str]] = {}
_SESSION_LOCK = threading.Lock()
_RUNTIME_BOOT_MODE = "uninitialized"
_RUNTIME_BOOT_REASON = ""
logger = logging.getLogger("backdroply.engine.processing")


@dataclass
class VideoMeta:
    width: int
    height: int
    fps: float
    frame_count: int


@dataclass
class BorderModel:
    centers_lab: np.ndarray
    radii: np.ndarray
    border_px: int


@dataclass
class AutoParams:
    main_model: str
    aux_model: str | None
    rescue_model: str | None
    feather_strength: float
    alpha_cutoff: int
    min_blob_percent: float
    temporal_smooth: float
    temporal_flow_strength: float
    edge_refine_strength: float
    grabcut_iterations: int
    keep_largest_component: bool
    border_distance_multiplier: float
    border_alpha_guard: int
    protect_dilate_px: int
    frame_recheck_enabled: bool
    frame_recheck_edge_threshold: float
    frame_recheck_disagreement_threshold: float


@dataclass
class RenderStats:
    frame_count: int
    edge_leak_mean: float
    edge_leak_max: float
    area_mean: float
    area_min: float
    comp_max: int
    suspect_frames: list[int]


def _available_ort_providers() -> list[str]:
    try:
        providers = list(ort.get_available_providers())
    except Exception:  # noqa: BLE001
        providers = []
    if "CPUExecutionProvider" not in providers:
        providers.append("CPUExecutionProvider")
    return providers


def _set_runtime_boot_status(mode: str, reason: str = ""):
    global _RUNTIME_BOOT_MODE, _RUNTIME_BOOT_REASON
    _RUNTIME_BOOT_MODE = mode
    _RUNTIME_BOOT_REASON = reason[:400]


def _probe_cuda_in_subprocess(timeout_sec: int) -> tuple[bool, str]:
    script = """
import sys
from rembg import new_session

try:
    session = new_session("u2netp", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    inner = getattr(session, "inner_session", None)
    providers = []
    if inner is not None and hasattr(inner, "get_providers"):
        providers = list(inner.get_providers())
    if "CUDAExecutionProvider" in providers:
        sys.exit(0)
    sys.exit(3)
except Exception:
    sys.exit(2)
"""
    try:
        completed = subprocess.run(
            [sys.executable, "-c", script],
            env=os.environ.copy(),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            timeout=max(3, timeout_sec),
            check=False,
        )
    except subprocess.TimeoutExpired:
        return False, "cuda probe timeout"
    if completed.returncode == 0:
        return True, "cuda provider usable"
    stderr_tail = (completed.stderr or b"").decode("utf-8", errors="ignore")[-300:].strip()
    reason = stderr_tail or f"cuda probe exit code {completed.returncode}"
    return False, reason


def ensure_runtime_provider_ready() -> dict[str, str]:
    available = _available_ort_providers()
    preferred = settings.ort_provider_order

    if "CUDAExecutionProvider" not in preferred:
        _set_runtime_boot_status("cpu-configured", "cuda not preferred by config")
        return {"mode": _RUNTIME_BOOT_MODE, "reason": _RUNTIME_BOOT_REASON}

    if "CUDAExecutionProvider" not in available:
        if settings.engine_ort_allow_cpu_fallback:
            force_cpu_only_mode()
            _set_runtime_boot_status("cpu-fallback", "cuda provider not available")
            return {"mode": _RUNTIME_BOOT_MODE, "reason": _RUNTIME_BOOT_REASON}
        _set_runtime_boot_status("gpu-unavailable", "cuda provider not available and fallback disabled")
        return {"mode": _RUNTIME_BOOT_MODE, "reason": _RUNTIME_BOOT_REASON}

    ok, reason = _probe_cuda_in_subprocess(timeout_sec=int(settings.engine_ort_probe_timeout_sec))
    if ok:
        _set_runtime_boot_status("gpu", reason)
        return {"mode": _RUNTIME_BOOT_MODE, "reason": _RUNTIME_BOOT_REASON}

    if settings.engine_ort_allow_cpu_fallback:
        force_cpu_only_mode()
        _set_runtime_boot_status("cpu-fallback", f"cuda probe failed: {reason}")
        return {"mode": _RUNTIME_BOOT_MODE, "reason": _RUNTIME_BOOT_REASON}

    _set_runtime_boot_status("gpu-probe-failed", reason)
    return {"mode": _RUNTIME_BOOT_MODE, "reason": _RUNTIME_BOOT_REASON}


def _provider_attempts() -> list[list[str]]:
    available = _available_ort_providers()
    preferred = [provider for provider in settings.ort_provider_order if provider in available]
    attempts: list[list[str]] = []
    if preferred:
        attempts.append(preferred)
    elif "CPUExecutionProvider" in available:
        attempts.append(["CPUExecutionProvider"])

    if settings.engine_ort_allow_cpu_fallback and "CPUExecutionProvider" in available:
        if not attempts or attempts[-1] != ["CPUExecutionProvider"]:
            attempts.append(["CPUExecutionProvider"])

    if not attempts:
        attempts.append(["CPUExecutionProvider"])

    deduped: list[list[str]] = []
    seen: set[tuple[str, ...]] = set()
    for attempt in attempts:
        key = tuple(attempt)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(attempt)
    return deduped


def _session_for(model_name: str):
    with _SESSION_LOCK:
        cached = _SESSION_CACHE.get(model_name)
        if cached is not None:
            return cached

    attempts = _provider_attempts()
    errors: list[str] = []
    for providers in attempts:
        try:
            session = new_session(model_name, providers=providers)
            used_providers = providers
            inner_session = getattr(session, "inner_session", None)
            if inner_session is not None and hasattr(inner_session, "get_providers"):
                used_providers = list(inner_session.get_providers())
            with _SESSION_LOCK:
                _SESSION_CACHE[model_name] = session
                _SESSION_PROVIDER_CACHE[model_name] = used_providers
                _SESSION_ERRORS.pop(model_name, None)
            return session
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{providers}: {type(exc).__name__}: {exc}")

    with _SESSION_LOCK:
        _SESSION_ERRORS[model_name] = errors[-4:]
    raise RuntimeError(
        f"Could not initialize model '{model_name}' with providers {attempts}. "
        f"Last error: {errors[-1] if errors else 'unknown'}"
    )


def clear_model_cache():
    with _SESSION_LOCK:
        _SESSION_CACHE.clear()
        _SESSION_PROVIDER_CACHE.clear()
        _SESSION_ERRORS.clear()


def force_cpu_only_mode():
    """Switch runtime provider preference to CPU and drop loaded sessions."""
    settings.engine_ort_providers = "CPUExecutionProvider"
    settings.engine_ort_runtime_flavor = "cpu"
    clear_model_cache()
    _set_runtime_boot_status("cpu-fallback", "forced to cpu-only mode")


def ort_runtime_info() -> dict:
    try:
        device = str(ort.get_device())
    except Exception:  # noqa: BLE001
        device = "unknown"
    with _SESSION_LOCK:
        session_providers = {key: list(value) for key, value in _SESSION_PROVIDER_CACHE.items()}
        session_errors = {key: list(value) for key, value in _SESSION_ERRORS.items()}
    return {
        "runtime_flavor": settings.engine_ort_runtime_flavor,
        "ort_version": getattr(ort, "__version__", "unknown"),
        "ort_device": device,
        "available_providers": _available_ort_providers(),
        "preferred_provider_order": settings.ort_provider_order,
        "provider_attempts": _provider_attempts(),
        "allow_cpu_fallback": bool(settings.engine_ort_allow_cpu_fallback),
        "boot_mode": _RUNTIME_BOOT_MODE,
        "boot_reason": _RUNTIME_BOOT_REASON,
        "loaded_models": session_providers,
        "session_errors": session_errors,
    }


def _runtime_is_cpu() -> bool:
    flavor = settings.engine_ort_runtime_flavor.strip().lower()
    if flavor == "cpu":
        return True
    with _SESSION_LOCK:
        cached = list(_SESSION_PROVIDER_CACHE.values())
    if cached:
        return not any("CUDAExecutionProvider" in providers for providers in cached)
    return "CUDAExecutionProvider" not in _available_ort_providers()


def _model_candidates_for_quality(quality: str) -> list[str]:
    configured = settings.model_candidates(quality)
    if configured:
        return configured
    return DEFAULT_MODEL_CANDIDATES


def _video_meta(video_path: Path) -> VideoMeta:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError("Video cannot be opened.")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    return VideoMeta(width=width, height=height, fps=fps, frame_count=frame_count)


def _sample_frame_indices(frame_count: int, count: int = 8) -> list[int]:
    if frame_count <= 1:
        return [0]
    count = max(2, min(count, frame_count))
    return np.linspace(0, frame_count - 1, count, dtype=int).tolist()


def _read_frames(video_path: Path, indices: list[int], max_side: int = 960) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    frames: list[np.ndarray] = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame_bgr = cap.read()
        if not ok:
            continue
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w = frame_rgb.shape[:2]
        if max(h, w) > max_side:
            scale = max_side / float(max(h, w))
            frame_rgb = cv2.resize(
                frame_rgb,
                (max(1, int(w * scale)), max(1, int(h * scale))),
                interpolation=cv2.INTER_AREA,
            )
        frames.append(frame_rgb)
    cap.release()
    return frames


def _infer_alpha(
    session,
    frame_rgb: np.ndarray,
    max_side: int | None = None,
    post_process_mask: bool = True,
) -> np.ndarray:
    h, w = frame_rgb.shape[:2]
    work = frame_rgb
    scaled = False
    if max_side is not None and max(h, w) > max_side:
        scale = max_side / float(max(h, w))
        nw = max(1, int(w * scale))
        nh = max(1, int(h * scale))
        work = cv2.resize(frame_rgb, (nw, nh), interpolation=cv2.INTER_AREA)
        scaled = True

    mask_img = remove(work, session=session, only_mask=True, post_process_mask=post_process_mask)
    alpha = np.asarray(mask_img)
    if alpha.ndim == 3:
        alpha = alpha[:, :, 0]
    alpha = alpha.astype(np.uint8)
    if scaled:
        alpha = cv2.resize(alpha, (w, h), interpolation=cv2.INTER_LINEAR)
    return alpha


def _inference_side_limit(quality: str, media_type: str) -> int:
    q = (quality or "").strip().lower()
    if media_type == "video":
        cpu_mode = _runtime_is_cpu()
        if q == "ultra":
            configured = settings.engine_video_cpu_ultra_max_side if cpu_mode else settings.engine_video_max_side_ultra
            return max(512, int(configured))
        configured = settings.engine_video_cpu_balanced_max_side if cpu_mode else settings.engine_video_max_side_balanced
        return max(448, int(configured))
    return 1400 if q == "ultra" else 1100


def _build_border_model(frames_rgb: list[np.ndarray], border_ratio: float = 0.08, k: int = 4) -> BorderModel:
    if not frames_rgb:
        raise RuntimeError("No sample frames for border model.")
    h, w = frames_rgb[0].shape[:2]
    border_px = max(8, int(min(h, w) * border_ratio))

    samples = []
    for frame_rgb in frames_rgb:
        lab = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2LAB)
        border = np.concatenate(
            [
                lab[:border_px, :, :].reshape(-1, 3),
                lab[-border_px:, :, :].reshape(-1, 3),
                lab[:, :border_px, :].reshape(-1, 3),
                lab[:, -border_px:, :].reshape(-1, 3),
            ],
            axis=0,
        )
        if len(border) > 12000:
            pick = np.random.choice(len(border), 12000, replace=False)
            border = border[pick]
        samples.append(border)
    points = np.concatenate(samples, axis=0).astype(np.float32)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 35, 0.2)
    _, labels, centers = cv2.kmeans(points, k, None, criteria, 4, flags=cv2.KMEANS_PP_CENTERS)
    labels = labels.reshape(-1)
    radii = np.zeros((k,), dtype=np.float32)
    for idx in range(k):
        cluster = points[labels == idx]
        if len(cluster) == 0:
            radii[idx] = 14.0
            continue
        d = np.linalg.norm(cluster - centers[idx], axis=1)
        radii[idx] = float(np.percentile(d, 85)) + 3.0

    return BorderModel(centers_lab=centers.astype(np.float32), radii=radii, border_px=border_px)


def _border_connected_bg_mask(
    frame_rgb: np.ndarray,
    border_model: BorderModel,
    dist_multiplier: float,
) -> np.ndarray:
    h, w = frame_rgb.shape[:2]
    bgmask_max_side = int(max(256, settings.engine_video_bgmask_max_side))
    scale = 1.0
    proc_rgb = frame_rgb
    if max(h, w) > bgmask_max_side:
        scale = bgmask_max_side / float(max(h, w))
        nw = max(64, int(round(w * scale)))
        nh = max(64, int(round(h * scale)))
        proc_rgb = cv2.resize(frame_rgb, (nw, nh), interpolation=cv2.INTER_AREA)

    lab = cv2.cvtColor(proc_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    centers = border_model.centers_lab
    radii = border_model.radii * max(0.6, dist_multiplier)

    diff = lab[:, :, None, :] - centers[None, None, :, :]
    dist2 = np.sum(diff * diff, axis=3)
    norm_score2 = np.min(dist2 / ((radii[None, None, :] ** 2) + 1e-6), axis=2)
    bg_like = (norm_score2 < 1.0).astype(np.uint8)
    bg_like = cv2.morphologyEx(
        bg_like,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    )

    num, labels, _, _ = cv2.connectedComponentsWithStats(bg_like, connectivity=8)
    if num <= 1:
        return np.zeros_like(bg_like, dtype=np.uint8)

    border_labels = set(np.unique(np.concatenate([labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1]])))
    border_labels.discard(0)
    if not border_labels:
        out_small = np.zeros_like(bg_like, dtype=np.uint8)
        if scale != 1.0:
            return cv2.resize(out_small, (w, h), interpolation=cv2.INTER_NEAREST).astype(np.uint8)
        return out_small
    lut = np.zeros((num,), dtype=np.uint8)
    for lb in border_labels:
        lut[int(lb)] = 1
    out_small = lut[labels].astype(np.uint8)
    if scale != 1.0:
        return cv2.resize(out_small, (w, h), interpolation=cv2.INTER_NEAREST).astype(np.uint8)
    return out_small


def _largest_component(mask: np.ndarray) -> np.ndarray:
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num <= 1:
        return np.zeros_like(mask, dtype=np.uint8)
    idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return np.where(labels == idx, 255, 0).astype(np.uint8)


def _grabcut_refine(frame_bgr: np.ndarray, alpha: np.ndarray, iterations: int) -> np.ndarray:
    if iterations <= 0:
        return alpha
    mask = np.full(alpha.shape, cv2.GC_PR_BGD, dtype=np.uint8)
    mask[alpha <= 10] = cv2.GC_BGD
    mask[(alpha > 10) & (alpha < 180)] = cv2.GC_PR_FGD
    mask[alpha >= 180] = cv2.GC_FGD
    bg_model = np.zeros((1, 65), np.float64)
    fg_model = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(
            frame_bgr,
            mask,
            None,
            bg_model,
            fg_model,
            iterations,
            mode=cv2.GC_INIT_WITH_MASK,
        )
    except cv2.error:
        return alpha
    fg = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    return np.where(fg > 0, alpha, 0).astype(np.uint8)


def _component_clean(alpha: np.ndarray, min_area_px: int, keep_largest: bool) -> np.ndarray:
    mask = (alpha > 8).astype(np.uint8)
    if not keep_largest and min_area_px <= 0:
        return alpha
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num <= 1:
        return alpha
    keep = np.zeros((num,), dtype=np.uint8)
    if keep_largest:
        largest_idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        keep[largest_idx] = 1
    if min_area_px > 0:
        for idx in range(1, num):
            if stats[idx, cv2.CC_STAT_AREA] >= min_area_px:
                keep[idx] = 1
    cleaned = keep[labels]
    return np.where(cleaned > 0, alpha, 0).astype(np.uint8)


def _recover_near_subject_components(
    alpha: np.ndarray,
    reference_alpha: np.ndarray,
    proximity_px: int,
    min_component_area: int,
) -> np.ndarray:
    if alpha.shape != reference_alpha.shape:
        return alpha

    subject_seed = np.where(alpha >= 120, 255, 0).astype(np.uint8)
    main_subject = _largest_component(subject_seed)
    if np.count_nonzero(main_subject) == 0:
        return alpha

    k = int(max(9, proximity_px))
    if k % 2 == 0:
        k += 1
    near_zone = cv2.dilate(main_subject, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)), iterations=1)
    candidate_mask = np.where(reference_alpha >= 104, 255, 0).astype(np.uint8)

    num, labels, stats, _ = cv2.connectedComponentsWithStats(candidate_mask, connectivity=8)
    if num <= 1:
        return alpha
    keep = np.zeros((num,), dtype=np.uint8)
    keep[0] = 0
    for idx in range(1, num):
        area = int(stats[idx, cv2.CC_STAT_AREA])
        if area < min_component_area:
            continue
        comp = labels == idx
        if np.any(near_zone[comp] > 0):
            keep[idx] = 1
    recovered = keep[labels] > 0
    if not np.any(recovered):
        return alpha

    out = alpha.copy()
    out[recovered] = np.maximum(out[recovered], reference_alpha[recovered])
    return out


def _suppress_border_components(
    alpha: np.ndarray,
    protect_zone: np.ndarray | None,
    alpha_min: int = 56,
) -> np.ndarray:
    mask = np.where(alpha >= alpha_min, 255, 0).astype(np.uint8)
    num, labels, _, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num <= 1:
        return alpha

    border_labels = set(
        np.unique(np.concatenate([labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1]])).tolist()
    )
    border_labels.discard(0)
    if not border_labels:
        return alpha

    protected_labels: set[int] = set()
    if protect_zone is not None and np.any(protect_zone > 0):
        protected_labels = set(np.unique(labels[protect_zone > 0]).tolist())
    drop_labels = border_labels.difference(protected_labels)
    if not drop_labels:
        return alpha

    drop_mask = np.isin(labels, np.array(list(drop_labels), dtype=labels.dtype))
    out = alpha.copy()
    if protect_zone is not None:
        out[drop_mask & (protect_zone == 0)] = 0
    else:
        out[drop_mask] = 0
    return out


def _strict_needed(alpha: np.ndarray, leak_zone: np.ndarray, threshold: int) -> bool:
    if threshold <= 0:
        return False
    if np.any(leak_zone):
        leak_mean = float(alpha[leak_zone].mean())
        return leak_mean >= float(threshold)
    edge_band = max(4, int(min(alpha.shape[:2]) * 0.04))
    edge_mask = np.zeros(alpha.shape[:2], dtype=np.uint8)
    edge_mask[:edge_band, :] = 1
    edge_mask[-edge_band:, :] = 1
    edge_mask[:, :edge_band] = 1
    edge_mask[:, -edge_band:] = 1
    return float(alpha[edge_mask > 0].mean()) >= float(threshold + 6)


def _merge_strict_alpha(main_alpha: np.ndarray, strict_alpha: np.ndarray) -> np.ndarray:
    main = main_alpha.astype(np.uint8)
    strict = strict_alpha.astype(np.uint8)
    core = np.where(main > 205, 255, 0).astype(np.uint8)
    shell = cv2.dilate(core, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (37, 37)), iterations=1)
    bg_clean = np.minimum(main, strict).astype(np.uint8)
    fg_preserve = np.maximum(main, strict).astype(np.uint8)
    return np.where(shell > 0, fg_preserve, bg_clean).astype(np.uint8)


def _fuse_human_general(main_alpha: np.ndarray, aux_alpha: np.ndarray, main_model: str, aux_model: str) -> np.ndarray:
    human_name = "u2net_human_seg"
    if main_model == human_name and aux_model != human_name:
        alpha_h = main_alpha
        alpha_g = aux_alpha
    elif aux_model == human_name and main_model != human_name:
        alpha_h = aux_alpha
        alpha_g = main_alpha
    else:
        return np.minimum(main_alpha, aux_alpha).astype(np.uint8)

    core = np.where(alpha_h > 165, 255, 0).astype(np.uint8)
    expand = cv2.dilate(core, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (61, 61)), iterations=1)
    add = np.where((expand > 0) & (alpha_g > 135), alpha_g, 0).astype(np.uint8)
    fused = np.maximum(alpha_h, add)
    return fused.astype(np.uint8)


def _temporal_fg_lock(alpha: np.ndarray, prev_lock: np.ndarray | None) -> np.ndarray:
    bin_mask = np.where(alpha > 110, 255, 0).astype(np.uint8)
    if prev_lock is None:
        return _largest_component(bin_mask)
    gate = cv2.dilate(prev_lock, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (71, 71)), iterations=1)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(bin_mask, connectivity=8)
    if num <= 1:
        return prev_lock
    keep = np.zeros((num,), dtype=np.uint8)
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    keep[largest] = 1
    for idx in range(1, num):
        comp = labels == idx
        if np.any(gate[comp] > 0):
            keep[idx] = 1
    locked = np.where(keep[labels] > 0, 255, 0).astype(np.uint8)
    if np.count_nonzero(locked) == 0:
        return _largest_component(bin_mask)
    return locked


def _refine_alpha(
    alpha: np.ndarray,
    feather_strength: float,
    alpha_cutoff: int,
    min_area_px: int,
    keep_largest: bool,
    temporal_smooth: float,
    prev_alpha: np.ndarray | None,
) -> np.ndarray:
    out = alpha.copy()
    if feather_strength > 0:
        out = cv2.GaussianBlur(out, (0, 0), feather_strength)
    if alpha_cutoff > 0:
        scale = 255.0 / max(1, 255 - alpha_cutoff)
        out = np.clip((out.astype(np.float32) - alpha_cutoff) * scale, 0, 255).astype(np.uint8)

    out = _component_clean(out, min_area_px=min_area_px, keep_largest=keep_largest)
    if temporal_smooth > 0 and prev_alpha is not None:
        out = cv2.addWeighted(
            out.astype(np.float32),
            1.0 - temporal_smooth,
            prev_alpha.astype(np.float32),
            temporal_smooth,
            0.0,
        )
        out = np.clip(out, 0, 255).astype(np.uint8)
    return out


def _mean_on_mask(img: np.ndarray, mask: np.ndarray) -> float:
    valid = mask > 0
    if not np.any(valid):
        return 0.0
    return float(img[valid].mean())


def _choose_models_and_cache(
    sample_frames: list[np.ndarray],
    quality: str,
    meta: VideoMeta | None = None,
) -> tuple[str, str | None, str | None, dict[str, list[np.ndarray]]]:
    candidates = _model_candidates_for_quality(quality)
    if not candidates:
        candidates = DEFAULT_MODEL_CANDIDATES

    # CPU runtime on dense video can be memory-fragile with multi-model auto-selection.
    # Force a single lightweight model to prevent native OOM kills.
    if _runtime_is_cpu():
        preferred_cpu_order = ["u2netp", "silueta", "u2net_human_seg", "u2net", "isnet-general-use"]
        chosen = None
        for name in preferred_cpu_order:
            if name in candidates:
                chosen = name
                break
        if chosen is None:
            chosen = candidates[0]
        session = _session_for(chosen)
        alphas = [_infer_alpha(session, frame, max_side=960) for frame in sample_frames]
        return chosen, None, None, {chosen: alphas}

    cache: dict[str, list[np.ndarray]] = {}
    model_scores: dict[str, float] = {}

    h, w = sample_frames[0].shape[:2]
    edge = np.zeros((h, w), dtype=np.uint8)
    border = max(8, int(min(h, w) * 0.08))
    edge[:border, :] = 1
    edge[-border:, :] = 1
    edge[:, :border] = 1
    edge[:, -border:] = 1
    center = np.zeros((h, w), dtype=np.uint8)
    center[int(h * 0.25) : int(h * 0.75), int(w * 0.25) : int(w * 0.75)] = 1

    for model in candidates:
        session = _session_for(model)
        alphas = [_infer_alpha(session, frame, max_side=960) for frame in sample_frames]
        cache[model] = alphas

        center_scores = np.array([_mean_on_mask(a, center) for a in alphas], dtype=np.float32)
        edge_scores = np.array([_mean_on_mask(a, edge) for a in alphas], dtype=np.float32)
        area_scores = np.array([float((a > 140).mean()) for a in alphas], dtype=np.float32)
        stability_penalty = float(edge_scores.std() * 0.45)
        score = float(center_scores.mean() - 1.65 * edge_scores.mean() - 42.0 * area_scores.mean() - stability_penalty)
        if meta is not None and (quality or "").strip().lower() == "ultra":
            dense_video = (meta.fps >= 45.0) or (meta.frame_count >= 260)
            if dense_video:
                if model == "u2netp":
                    score += 9.0
                elif model == "silueta":
                    score += 4.0
                elif model in {"isnet-general-use", "u2net"}:
                    score -= 34.0
                elif model == "u2net_human_seg":
                    score -= 18.0
        model_scores[model] = score

    sorted_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)
    main_model = sorted_models[0][0]
    aux_model = sorted_models[1][0] if len(sorted_models) > 1 else None
    rescue_model = sorted_models[2][0] if len(sorted_models) > 2 else aux_model
    if aux_model == main_model:
        aux_model = None
    if rescue_model == main_model:
        rescue_model = None
    return main_model, aux_model, rescue_model, cache


def prewarm_models(quality: str | None = None) -> list[str]:
    qualities: list[str] = [quality] if quality else ["balanced", "ultra"]
    loaded: list[str] = []
    for q in qualities:
        for model in _model_candidates_for_quality(q):
            _session_for(model)
            if model not in loaded:
                loaded.append(model)
    return loaded


def _auto_params(
    quality: str,
    sample_frames: list[np.ndarray],
    border_model: BorderModel,
    alpha_cache: list[np.ndarray],
    main_model: str,
    aux_model: str | None,
    rescue_model: str | None,
) -> AutoParams:
    h, w = sample_frames[0].shape[:2]
    center = np.zeros((h, w), dtype=np.uint8)
    center[int(h * 0.25) : int(h * 0.75), int(w * 0.25) : int(w * 0.75)] = 1

    leak_values = []
    center_values = []
    for frame_rgb, alpha in zip(sample_frames, alpha_cache):
        bg_mask = _border_connected_bg_mask(frame_rgb, border_model, dist_multiplier=1.0)
        leak_values.append(_mean_on_mask(alpha, bg_mask) / 255.0)
        center_values.append(_mean_on_mask(alpha, center) / 255.0)

    leak = float(np.mean(leak_values))
    center_fg = float(np.mean(center_values))
    if quality == "ultra":
        base_cutoff = 182
        feather = 1.18
        temporal = 0.26
        grabcut = 1
        dist_mult = 1.12
        min_blob = 0.14
        protect = 39
    else:
        base_cutoff = 170
        feather = 1.05
        temporal = 0.22
        grabcut = 1
        dist_mult = 1.02
        min_blob = 0.09
        protect = 31

    leak_boost = int(np.clip((leak - 0.05) * 210.0, 0, 34))
    center_relief = -6 if center_fg < 0.23 else 0
    alpha_cutoff = int(np.clip(base_cutoff + leak_boost + center_relief, 145, 220))
    dist_mult = float(np.clip(dist_mult + leak * 0.55, 0.9, 1.5))
    min_blob = float(np.clip(min_blob + leak * 0.35, 0.05, 0.60))
    use_aux = ((quality == "ultra") or (leak > 0.08)) and aux_model is not None

    return AutoParams(
        main_model=main_model,
        aux_model=aux_model if use_aux else None,
        rescue_model=rescue_model if quality == "ultra" else None,
        feather_strength=feather,
        alpha_cutoff=alpha_cutoff,
        min_blob_percent=min_blob,
        temporal_smooth=temporal,
        temporal_flow_strength=float(np.clip(settings.engine_temporal_flow_strength, 0.0, 0.55)),
        edge_refine_strength=float(np.clip(settings.engine_edge_refine_strength, 0.0, 1.0)),
        grabcut_iterations=grabcut,
        keep_largest_component=True,
        border_distance_multiplier=dist_mult,
        border_alpha_guard=242,
        protect_dilate_px=protect,
        frame_recheck_enabled=bool(settings.engine_enable_frame_recheck),
        frame_recheck_edge_threshold=float(max(2.0, settings.engine_recheck_edge_threshold)),
        frame_recheck_disagreement_threshold=float(max(5.0, settings.engine_recheck_disagreement_threshold)),
    )


def _tighten_auto_params(params: AutoParams) -> AutoParams:
    return AutoParams(
        main_model=params.main_model,
        aux_model=params.aux_model,
        rescue_model=params.rescue_model,
        feather_strength=float(min(2.0, params.feather_strength + 0.2)),
        alpha_cutoff=int(min(224, params.alpha_cutoff + 12)),
        min_blob_percent=float(min(0.85, params.min_blob_percent + 0.08)),
        temporal_smooth=float(min(0.48, params.temporal_smooth + 0.08)),
        temporal_flow_strength=float(min(0.58, params.temporal_flow_strength + 0.05)),
        edge_refine_strength=float(min(1.0, params.edge_refine_strength + 0.06)),
        grabcut_iterations=int(min(3, params.grabcut_iterations + 1)),
        keep_largest_component=True,
        border_distance_multiplier=float(min(1.48, params.border_distance_multiplier + 0.12)),
        border_alpha_guard=int(max(226, params.border_alpha_guard - 8)),
        protect_dilate_px=int(min(57, params.protect_dilate_px + 8)),
        frame_recheck_enabled=params.frame_recheck_enabled,
        frame_recheck_edge_threshold=max(2.0, params.frame_recheck_edge_threshold - 0.5),
        frame_recheck_disagreement_threshold=max(5.0, params.frame_recheck_disagreement_threshold - 2.0),
    )


def _fuse_alpha_maps(
    main_alpha: np.ndarray,
    aux_alpha: np.ndarray,
    main_model: str,
    aux_model: str,
) -> np.ndarray:
    # Human-specific model complements general models better with this strategy.
    if "human_seg" in main_model or "human_seg" in aux_model:
        return _fuse_human_general(main_alpha, aux_alpha, main_model=main_model, aux_model=aux_model)

    union = np.maximum(main_alpha, aux_alpha)
    core = np.where(union > 176, 255, 0).astype(np.uint8)
    core = cv2.dilate(core, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (45, 45)), iterations=1)
    conservative = np.minimum(main_alpha, aux_alpha)
    permissive = np.maximum(main_alpha, aux_alpha)
    return np.where(core > 0, permissive, conservative).astype(np.uint8)


def _edge_aware_refine(alpha: np.ndarray, frame_bgr: np.ndarray, strength: float) -> np.ndarray:
    if strength <= 0.0:
        return alpha
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, threshold1=50, threshold2=130)
    edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
    transition = np.where((alpha > 0) & (alpha < 255), 255, 0).astype(np.uint8)
    roi = np.where((edges > 0) | (transition > 0), 255, 0).astype(np.uint8)
    if np.count_nonzero(roi) == 0:
        return alpha

    diameter = 7 if strength < 0.75 else 9
    sigma_color = float(45.0 + (35.0 * strength))
    sigma_space = float(6.0 + (4.0 * strength))
    smooth = cv2.bilateralFilter(alpha, diameter, sigma_color, sigma_space)
    out = alpha.copy()
    out[roi > 0] = smooth[roi > 0]
    return out.astype(np.uint8)


def _soft_edge_matte(alpha: np.ndarray, strength: float) -> np.ndarray:
    if strength <= 0.0:
        return alpha
    a = alpha.astype(np.uint8)
    transition = np.where((a > 0) & (a < 255), 255, 0).astype(np.uint8)
    hard_fg = np.where(a >= 128, 255, 0).astype(np.uint8)
    hard_boundary = cv2.morphologyEx(
        hard_fg,
        cv2.MORPH_GRADIENT,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    )
    seed = np.where((transition > 0) | (hard_boundary > 0), 255, 0).astype(np.uint8)
    if np.count_nonzero(seed) == 0:
        return a

    radius = int(np.clip(round(2 + 5 * strength), 2, 7))
    k = radius * 2 + 1
    band = cv2.dilate(seed, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)), iterations=1)

    sigma = float(0.75 + 1.8 * strength)
    smooth = cv2.GaussianBlur(a, (0, 0), sigmaX=sigma, sigmaY=sigma)

    weight = cv2.GaussianBlur(
        (band.astype(np.float32) / 255.0),
        (0, 0),
        sigmaX=max(0.7, sigma * 0.7),
        sigmaY=max(0.7, sigma * 0.7),
    )
    mix = float(np.clip(0.42 + 0.38 * strength, 0.30, 0.82))
    weight = np.clip(weight * mix, 0.0, 0.82)

    out = a.astype(np.float32) * (1.0 - weight) + smooth.astype(np.float32) * weight
    out[(band == 0) & (a <= 2)] = 0.0
    out[(band == 0) & (a >= 253)] = 255.0
    return np.clip(out, 0, 255).astype(np.uint8)


def _warp_previous_alpha(
    prev_alpha: np.ndarray,
    gray: np.ndarray,
    prev_gray: np.ndarray,
) -> tuple[np.ndarray, float]:
    # Flow from current -> previous allows remapping prev alpha to current frame coordinates.
    h, w = gray.shape[:2]
    flow_max_side = int(max(256, settings.engine_video_flow_max_side))
    scale = 1.0
    gray_flow = gray
    prev_gray_flow = prev_gray
    alpha_flow = prev_alpha
    if max(h, w) > flow_max_side:
        scale = flow_max_side / float(max(h, w))
        nw = max(64, int(round(w * scale)))
        nh = max(64, int(round(h * scale)))
        gray_flow = cv2.resize(gray, (nw, nh), interpolation=cv2.INTER_AREA)
        prev_gray_flow = cv2.resize(prev_gray, (nw, nh), interpolation=cv2.INTER_AREA)
        alpha_flow = cv2.resize(prev_alpha, (nw, nh), interpolation=cv2.INTER_LINEAR)

    flow = cv2.calcOpticalFlowFarneback(
        gray_flow,
        prev_gray_flow,
        None,
        0.5,
        3,
        15,
        3,
        5,
        1.2,
        0,
    )
    fh, fw = gray_flow.shape[:2]
    grid_x, grid_y = np.meshgrid(np.arange(fw, dtype=np.float32), np.arange(fh, dtype=np.float32))
    map_x = grid_x + flow[:, :, 0]
    map_y = grid_y + flow[:, :, 1]
    warped_prev_flow = cv2.remap(alpha_flow, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    if scale != 1.0:
        warped_prev = cv2.resize(warped_prev_flow, (w, h), interpolation=cv2.INTER_LINEAR)
    else:
        warped_prev = warped_prev_flow

    # Background motion is often much larger; gate refresh decision on the previous foreground area.
    fg_mask = alpha_flow > 96
    if np.any(fg_mask):
        motion_mag = float(np.mean(np.linalg.norm(flow[fg_mask], axis=1)))
    else:
        motion_mag = float(np.mean(np.linalg.norm(flow, axis=2)))

    if scale != 1.0:
        motion_mag = motion_mag / max(scale, 1e-6)
    return warped_prev, motion_mag


def _flow_guided_temporal_blend(
    alpha: np.ndarray,
    prev_alpha: np.ndarray | None,
    frame_bgr: np.ndarray,
    prev_gray: np.ndarray | None,
    strength: float,
) -> tuple[np.ndarray, np.ndarray]:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    if prev_alpha is None or prev_gray is None or strength <= 0.0:
        return alpha, gray

    warped_prev, motion_mag = _warp_previous_alpha(prev_alpha, gray, prev_gray)
    motion_gate = float(np.clip(1.0 - (motion_mag / 5.8), 0.0, 1.0))
    blend_w = float(np.clip(strength * motion_gate, 0.0, 0.65))
    if blend_w <= 0.0:
        return alpha, gray

    blended = cv2.addWeighted(
        alpha.astype(np.float32),
        1.0 - blend_w,
        warped_prev.astype(np.float32),
        blend_w,
        0.0,
    )
    return np.clip(blended, 0, 255).astype(np.uint8), gray


def _quality_stats(edge_values: np.ndarray, area_values: np.ndarray, comp_values: np.ndarray) -> RenderStats:
    if len(edge_values) == 0:
        return RenderStats(0, 0.0, 0.0, 0.0, 0.0, 0, [])
    edge_thr = max(8.0, float(edge_values.mean() + 2.2 * (edge_values.std() + 1e-6)))
    area_mean = float(area_values.mean())
    area_std = float(area_values.std())
    comp_thr = max(3, int(np.ceil(comp_values.mean() + 2.0 * (comp_values.std() + 1e-6))))
    sus_edge = np.where(edge_values > edge_thr)[0]
    if area_mean < 0.02:
        # For tiny subjects, area jitter is not a reliable quality signal.
        sus_area = np.array([], dtype=np.int32)
    else:
        area_thr = max(0.01, float(area_mean - 2.1 * (area_std + 1e-6)))
        sus_area = np.where(area_values < area_thr)[0]
    sus_comp = np.where(comp_values > comp_thr)[0]
    combined = sorted(set(sus_edge.tolist() + sus_area.tolist() + sus_comp.tolist()))
    return RenderStats(
        frame_count=int(len(edge_values)),
        edge_leak_mean=float(edge_values.mean()),
        edge_leak_max=float(edge_values.max()),
        area_mean=float(area_values.mean()),
        area_min=float(area_values.min()),
        comp_max=int(comp_values.max()),
        suspect_frames=combined,
    )


def _ffmpeg_cmd_transparent(
    ffmpeg_bin: str,
    input_video: Path,
    output_video: Path,
    width: int,
    height: int,
    fps: float,
) -> list[str]:
    vp9_crf = int(np.clip(settings.engine_video_vp9_crf, 10, 40))
    vp9_cpu_used = int(np.clip(settings.engine_video_vp9_cpu_used, 0, 8))
    deadline = (settings.engine_video_vp9_deadline or "realtime").strip().lower()
    if deadline not in {"realtime", "good", "best"}:
        deadline = "realtime"
    threads = int(max(0, settings.engine_video_vp9_threads))
    return [
        ffmpeg_bin,
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgba",
        "-s",
        f"{width}x{height}",
        "-r",
        f"{fps:.6f}",
        "-i",
        "-",
        "-i",
        str(input_video),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0?",
        "-c:v",
        "libvpx-vp9",
        "-b:v",
        "0",
        "-crf",
        str(vp9_crf),
        "-pix_fmt",
        "yuva420p",
        "-row-mt",
        "1",
        "-deadline",
        deadline,
        "-cpu-used",
        str(vp9_cpu_used),
        "-threads",
        str(threads) if threads > 0 else "0",
        "-c:a",
        "libopus",
        "-b:a",
        "160k",
        "-shortest",
        str(output_video),
    ]


def _ffmpeg_cmd_solid(
    ffmpeg_bin: str,
    input_video: Path,
    output_video: Path,
    width: int,
    height: int,
    fps: float,
) -> list[str]:
    return [
        ffmpeg_bin,
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{width}x{height}",
        "-r",
        f"{fps:.6f}",
        "-i",
        "-",
        "-i",
        str(input_video),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0?",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "16",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "160k",
        "-shortest",
        str(output_video),
    ]


def _apply_manual_masks(
    alpha: np.ndarray,
    erase_mask: np.ndarray | None,
    keep_mask: np.ndarray | None,
) -> np.ndarray:
    out = alpha.copy()
    h, w = out.shape[:2]
    if erase_mask is not None:
        em = cv2.resize(erase_mask, (w, h), interpolation=cv2.INTER_NEAREST)
        out[em > 0] = 0
    if keep_mask is not None:
        km = cv2.resize(keep_mask, (w, h), interpolation=cv2.INTER_NEAREST)
        out[km > 0] = 255
    return out


def _draw_watermark_icon(canvas: np.ndarray, cx: int, cy: int, size: int, color: tuple[int, ...], thickness: int):
    arm = max(4, size)
    cv2.line(canvas, (cx - arm, cy), (cx + arm, cy), color, thickness, cv2.LINE_AA)
    cv2.line(canvas, (cx, cy - arm), (cx, cy + arm), color, thickness, cv2.LINE_AA)
    diag = max(3, int(round(arm * 0.72)))
    thin = max(1, thickness - 1)
    cv2.line(canvas, (cx - diag, cy - diag), (cx + diag, cy + diag), color, thin, cv2.LINE_AA)
    cv2.line(canvas, (cx - diag, cy + diag), (cx + diag, cy - diag), color, thin, cv2.LINE_AA)
    cv2.circle(canvas, (cx, cy), max(1, int(round(arm * 0.18))), color, -1, cv2.LINE_AA)


def _watermark_layout(width: int, height: int, text: str) -> tuple[int, int, int, int, int, int, int, int, float, int, int]:
    font = cv2.FONT_HERSHEY_SIMPLEX
    short_side = max(1, min(width, height))
    font_scale = max(0.50, min(1.12, short_side / 880.0))
    thickness = max(2, int(round(font_scale * 2.1)))
    icon_size = max(10, int(round(15 * font_scale)))
    gap = max(6, int(round(8 * font_scale)))
    margin = max(10, int(round(22 * font_scale)))
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    total_w = icon_size * 2 + gap + text_w
    total_h = text_h + baseline
    x = max(margin, width - total_w - margin)
    y = max(text_h + margin, height - margin)
    return x, y, icon_size, gap, text_w, text_h, baseline, thickness, font_scale, total_w, total_h


def _watermark_patch_rgba(
    text: str,
    icon_size: int,
    gap: int,
    text_h: int,
    thickness: int,
    font_scale: float,
    total_w: int,
    total_h: int,
) -> tuple[np.ndarray, int]:
    # Draw watermark at higher resolution and downsample for cleaner edges.
    supersample = 4
    pad = max(5, int(round(icon_size * 0.42)))
    patch_w = max(1, total_w + pad * 2)
    patch_h = max(1, total_h + pad * 2)
    hi_w = patch_w * supersample
    hi_h = patch_h * supersample
    hi = np.zeros((hi_h, hi_w, 4), dtype=np.uint8)

    baseline_y = int(round((pad + text_h) * supersample))
    icon_cx = int(round((pad + icon_size) * supersample))
    icon_cy = int(round((pad + (text_h / 2.0)) * supersample))
    text_x = int(round((pad + icon_size * 2 + gap) * supersample))
    thickness_hi = max(2, int(round(thickness * supersample)))
    icon_hi = max(6, int(round(icon_size * supersample)))
    shadow_shift = max(1, int(round(supersample * 0.55)))

    _draw_watermark_icon(
        hi,
        icon_cx + shadow_shift,
        icon_cy + shadow_shift,
        icon_hi,
        (6, 12, 24, 132),
        max(2, thickness_hi + 1),
    )
    _draw_watermark_icon(
        hi,
        icon_cx,
        icon_cy,
        icon_hi,
        (130, 220, 255, 246),
        thickness_hi,
    )

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(
        hi,
        text,
        (text_x + shadow_shift, baseline_y + shadow_shift),
        font,
        font_scale * supersample,
        (6, 12, 24, 148),
        max(2, thickness_hi + 1),
        cv2.LINE_AA,
    )
    cv2.putText(
        hi,
        text,
        (text_x, baseline_y),
        font,
        font_scale * supersample,
        (244, 249, 255, 254),
        thickness_hi,
        cv2.LINE_AA,
    )

    patch = cv2.resize(hi, (patch_w, patch_h), interpolation=cv2.INTER_LANCZOS4)
    return patch, pad


def _blend_watermark_patch_bgr(frame_bgr: np.ndarray, patch_rgba: np.ndarray, x0: int, y0: int) -> np.ndarray:
    out = frame_bgr.copy()
    h, w = out.shape[:2]
    ph, pw = patch_rgba.shape[:2]
    x1 = x0 + pw
    y1 = y0 + ph
    cx0 = max(0, x0)
    cy0 = max(0, y0)
    cx1 = min(w, x1)
    cy1 = min(h, y1)
    if cx0 >= cx1 or cy0 >= cy1:
        return out

    sx0 = cx0 - x0
    sy0 = cy0 - y0
    sx1 = sx0 + (cx1 - cx0)
    sy1 = sy0 + (cy1 - cy0)
    src = patch_rgba[sy0:sy1, sx0:sx1]

    alpha = src[:, :, 3:4].astype(np.float32) / 255.0
    if float(alpha.max(initial=0.0)) <= 0.0:
        return out
    dst = out[cy0:cy1, cx0:cx1].astype(np.float32)
    src_rgb = src[:, :, :3].astype(np.float32)
    blended = src_rgb * alpha + dst * (1.0 - alpha)
    out[cy0:cy1, cx0:cx1] = np.clip(blended, 0, 255).astype(np.uint8)
    return out


def _blend_watermark_patch_rgba(frame_rgba: np.ndarray, patch_rgba: np.ndarray, x0: int, y0: int) -> np.ndarray:
    out = frame_rgba.copy()
    h, w = out.shape[:2]
    ph, pw = patch_rgba.shape[:2]
    x1 = x0 + pw
    y1 = y0 + ph
    cx0 = max(0, x0)
    cy0 = max(0, y0)
    cx1 = min(w, x1)
    cy1 = min(h, y1)
    if cx0 >= cx1 or cy0 >= cy1:
        return out

    sx0 = cx0 - x0
    sy0 = cy0 - y0
    sx1 = sx0 + (cx1 - cx0)
    sy1 = sy0 + (cy1 - cy0)
    src = patch_rgba[sy0:sy1, sx0:sx1]
    dst = out[cy0:cy1, cx0:cx1]
    out[cy0:cy1, cx0:cx1] = _alpha_compose_rgba(dst, src)
    return out


def _apply_watermark_bgr(frame_bgr: np.ndarray, watermark_enabled: bool | None = None) -> np.ndarray:
    enabled = settings.engine_watermark_enabled if watermark_enabled is None else bool(watermark_enabled)
    if not enabled:
        return frame_bgr
    text = (settings.engine_watermark_text or "").strip()
    if not text:
        return frame_bgr
    h, w = frame_bgr.shape[:2]
    if h < 48 or w < 96:
        return frame_bgr

    x, y, icon_size, gap, _, text_h, _, thickness, font_scale, total_w, total_h = _watermark_layout(w, h, text)
    patch, pad = _watermark_patch_rgba(
        text=text,
        icon_size=icon_size,
        gap=gap,
        text_h=text_h,
        thickness=thickness,
        font_scale=font_scale,
        total_w=total_w,
        total_h=total_h,
    )
    return _blend_watermark_patch_bgr(frame_bgr, patch, x - pad, y - text_h - pad)


def _alpha_compose_rgba(base: np.ndarray, overlay: np.ndarray) -> np.ndarray:
    base_f = base.astype(np.float32)
    ov_f = overlay.astype(np.float32)
    ov_a = ov_f[:, :, 3:4] / 255.0
    base_a = base_f[:, :, 3:4] / 255.0
    out_rgb = ov_f[:, :, :3] * ov_a + base_f[:, :, :3] * (1.0 - ov_a)
    out_a = (ov_a + base_a * (1.0 - ov_a)) * 255.0
    out = np.concatenate([out_rgb, out_a], axis=2)
    return np.clip(out, 0, 255).astype(np.uint8)


def _apply_watermark_rgba(frame_rgba: np.ndarray, watermark_enabled: bool | None = None) -> np.ndarray:
    enabled = settings.engine_watermark_enabled if watermark_enabled is None else bool(watermark_enabled)
    if not enabled:
        return frame_rgba
    text = (settings.engine_watermark_text or "").strip()
    if not text:
        return frame_rgba
    h, w = frame_rgba.shape[:2]
    if h < 48 or w < 96:
        return frame_rgba

    x, y, icon_size, gap, _, text_h, _, thickness, font_scale, total_w, total_h = _watermark_layout(w, h, text)
    patch, pad = _watermark_patch_rgba(
        text=text,
        icon_size=icon_size,
        gap=gap,
        text_h=text_h,
        thickness=thickness,
        font_scale=font_scale,
        total_w=total_w,
        total_h=total_h,
    )
    return _blend_watermark_patch_rgba(frame_rgba, patch, x - pad, y - text_h - pad)


def _render_with_params(
    video_path: Path,
    output_video: Path,
    ffmpeg_bin: str,
    meta: VideoMeta,
    border_model: BorderModel,
    params: AutoParams,
    erase_mask: np.ndarray | None,
    keep_mask: np.ndarray | None,
    bg_rgb: tuple[int, int, int] | None,
    watermark_enabled: bool | None,
    main_max_side: int,
    aux_max_side: int,
    rescue_max_side: int,
    video_infer_stride: int,
    strict_enabled: bool,
) -> RenderStats:
    cmd = (
        _ffmpeg_cmd_transparent(ffmpeg_bin, video_path, output_video, meta.width, meta.height, meta.fps)
        if bg_rgb is None
        else _ffmpeg_cmd_solid(ffmpeg_bin, video_path, output_video, meta.width, meta.height, meta.fps)
    )
    session_main = _session_for(params.main_model)
    session_aux = _session_for(params.aux_model) if params.aux_model else None
    session_rescue = _session_for(params.rescue_model) if params.rescue_model else None
    strict_session = None
    strict_threshold = max(0, int(settings.engine_ultra_strict_threshold))
    strict_ratio = float(np.clip(float(settings.engine_ultra_strict_max_ratio_pct) / 100.0, 0.0, 1.0))
    strict_max_side = int(settings.engine_strict_model_max_side)
    if strict_max_side <= 0:
        strict_max_side = int(max(512, main_max_side))
    strict_model = (settings.engine_ultra_strict_model or "").strip()
    if strict_enabled and strict_model and strict_model not in {params.main_model, params.aux_model or ""}:
        try:
            strict_session = _session_for(strict_model)
        except Exception:  # noqa: BLE001
            strict_session = None
    min_area_px = int((meta.width * meta.height) * (params.min_blob_percent / 100.0))
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError("Video read error.")
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    prev_alpha = None
    prev_lock = None
    prev_gray = None
    h, w = meta.height, meta.width
    accessory_recover_enabled = bool(settings.engine_video_accessory_recover)
    proximity_ratio = float(np.clip(settings.engine_video_accessory_proximity_ratio, 0.01, 0.16))
    min_area_ratio = float(np.clip(settings.engine_video_accessory_min_area_ratio, 0.00003, 0.002))
    accessory_proximity_px = max(13, int(min(h, w) * proximity_ratio))
    accessory_min_area = max(96, int((h * w) * min_area_ratio))
    edge_band = max(8, int(min(h, w) * 0.08))
    edge_mask = np.zeros((h, w), dtype=np.uint8)
    edge_mask[:edge_band, :] = 1
    edge_mask[-edge_band:, :] = 1
    edge_mask[:, :edge_band] = 1
    edge_mask[:, -edge_band:] = 1
    edge_values: list[float] = []
    area_values: list[float] = []
    comp_values: list[float] = []
    frame_index = 0
    total_frames = max(1, int(meta.frame_count))
    infer_stride = max(1, int(video_infer_stride))
    fps_cap = max(0, int(settings.engine_video_process_fps_cap))
    if fps_cap > 0 and meta.fps > fps_cap:
        fps_stride = max(1, int(np.ceil(meta.fps / float(fps_cap))))
        infer_stride = max(infer_stride, fps_stride)
    strict_budget = max(0, int(total_frames * strict_ratio)) if strict_enabled else 0
    strict_used = 0
    grabcut_frame_stride = max(1, int(settings.engine_grabcut_frame_stride))
    motion_refresh_threshold = float(max(2.0, settings.engine_video_motion_refresh_threshold))
    qc_stride = max(1, int(settings.engine_video_qc_sample_stride))
    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frame_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            should_infer = prev_alpha is None or prev_gray is None or (frame_index % infer_stride == 0)
            alpha = None
            reference_alpha = None
            aux_disagreement = 0.0
            bg_mask = None
            protect_zone = None
            if not should_infer and prev_alpha is not None and prev_gray is not None:
                propagated_alpha, motion_mag = _warp_previous_alpha(prev_alpha, frame_gray, prev_gray)
                if motion_mag <= motion_refresh_threshold:
                    alpha = propagated_alpha
                else:
                    should_infer = True

            if should_infer:
                alpha_main = _infer_alpha(session_main, frame_rgb, max_side=main_max_side)
                alpha = alpha_main
                reference_alpha = alpha_main
                if session_aux is not None:
                    aux_alpha = _infer_alpha(session_aux, frame_rgb, max_side=aux_max_side)
                    aux_disagreement = float(np.mean(np.abs(alpha_main.astype(np.int16) - aux_alpha.astype(np.int16))))
                    reference_alpha = np.maximum(reference_alpha, aux_alpha)
                    alpha = _fuse_alpha_maps(
                        alpha_main,
                        aux_alpha,
                        main_model=params.main_model,
                        aux_model=params.aux_model or params.main_model,
                    )

                if (
                    strict_session is not None
                    and strict_used < strict_budget
                    and strict_threshold > 0
                ):
                    leak_probe = _border_connected_bg_mask(
                        frame_rgb,
                        border_model=border_model,
                        dist_multiplier=params.border_distance_multiplier,
                    )
                    if _strict_needed(alpha, leak_probe > 0, strict_threshold):
                        try:
                            strict_alpha = _infer_alpha(strict_session, frame_rgb, max_side=strict_max_side)
                            alpha = _merge_strict_alpha(alpha, strict_alpha)
                            strict_used += 1
                        except Exception:  # noqa: BLE001
                            strict_session = None
            if alpha is None:
                alpha = np.zeros((h, w), dtype=np.uint8)

            if should_infer:
                bg_mask = _border_connected_bg_mask(frame_rgb, border_model, dist_multiplier=params.border_distance_multiplier)
                sure_fg = (alpha >= 225).astype(np.uint8) * 255
                main_fg = _largest_component(sure_fg)
                if params.protect_dilate_px > 0:
                    k = params.protect_dilate_px
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
                    protect_zone = cv2.dilate(main_fg, kernel, iterations=1)
                else:
                    protect_zone = main_fg

                suppress = (bg_mask > 0) & (protect_zone == 0) & (alpha < params.border_alpha_guard)
                alpha[suppress] = 0
                edge_now = float(alpha[edge_mask > 0].mean())
                if edge_now > 6.0:
                    hard_suppress = (bg_mask > 0) & (protect_zone == 0) & (alpha < 252)
                    alpha[hard_suppress] = 0
                fg_lock = _temporal_fg_lock(alpha, prev_lock)
            else:
                fg_lock = prev_lock if prev_lock is not None else np.where(alpha > 110, 255, 0).astype(np.uint8)
            keep_zone = cv2.dilate(fg_lock, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17)), iterations=1)
            alpha[(keep_zone == 0) & (alpha < 248)] = 0
            alpha = _suppress_border_components(alpha, protect_zone=protect_zone, alpha_min=56)
            grabcut_iters = 0
            if should_infer and params.grabcut_iterations > 0 and (frame_index % grabcut_frame_stride == 0):
                grabcut_iters = params.grabcut_iterations
            if grabcut_iters > 0:
                alpha = _grabcut_refine(frame_bgr, alpha, iterations=grabcut_iters)
            alpha = _apply_manual_masks(alpha, erase_mask=erase_mask, keep_mask=keep_mask)
            if should_infer:
                alpha, frame_gray = _flow_guided_temporal_blend(
                    alpha,
                    prev_alpha=prev_alpha,
                    frame_bgr=frame_bgr,
                    prev_gray=prev_gray,
                    strength=params.temporal_flow_strength,
                )
            alpha = _refine_alpha(
                alpha,
                feather_strength=params.feather_strength if should_infer else float(max(0.55, params.feather_strength * 0.55)),
                alpha_cutoff=params.alpha_cutoff if should_infer else int(max(118, params.alpha_cutoff - 24)),
                min_area_px=min_area_px,
                keep_largest=params.keep_largest_component,
                temporal_smooth=params.temporal_smooth if should_infer else float(max(params.temporal_smooth, 0.34)),
                prev_alpha=prev_alpha,
            )
            edge_refine_due = should_infer or (frame_index % max(3, infer_stride * 2) == 0)
            if edge_refine_due:
                alpha = _edge_aware_refine(alpha, frame_bgr=frame_bgr, strength=params.edge_refine_strength)

            if prev_alpha is not None:
                cur_edge = float(alpha[edge_mask > 0].mean())
                prev_edge = float(prev_alpha[edge_mask > 0].mean())
                if cur_edge > prev_edge + 2.0:
                    stable_zone = cv2.dilate(
                        np.where(prev_alpha > 110, 255, 0).astype(np.uint8),
                        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)),
                        iterations=1,
                    )
                    alpha[(stable_zone == 0) & (alpha < 250)] = 0

            cur_edge = float(alpha[edge_mask > 0].mean())
            if (
                should_infer
                and
                session_rescue is not None
                and params.frame_recheck_enabled
                and (
                    cur_edge > params.frame_recheck_edge_threshold
                    or aux_disagreement > params.frame_recheck_disagreement_threshold
                )
            ):
                rescue_alpha = _infer_alpha(session_rescue, frame_rgb, max_side=rescue_max_side)
                rescue_fused = _fuse_alpha_maps(
                    alpha,
                    rescue_alpha,
                    main_model=params.main_model,
                    aux_model=params.rescue_model or params.main_model,
                )
                rescue_fused = _apply_manual_masks(rescue_fused, erase_mask=erase_mask, keep_mask=keep_mask)
                rescue_fused = _refine_alpha(
                    rescue_fused,
                    feather_strength=params.feather_strength,
                    alpha_cutoff=params.alpha_cutoff,
                    min_area_px=min_area_px,
                    keep_largest=params.keep_largest_component,
                    temporal_smooth=0.0,
                    prev_alpha=None,
                )
                rescue_fused = _edge_aware_refine(rescue_fused, frame_bgr=frame_bgr, strength=params.edge_refine_strength)
                rescue_edge = float(rescue_fused[edge_mask > 0].mean())
                if rescue_edge + 0.35 < cur_edge:
                    alpha = rescue_fused

            if (
                should_infer
                and accessory_recover_enabled
                and reference_alpha is not None
            ):
                alpha = _recover_near_subject_components(
                    alpha=alpha,
                    reference_alpha=reference_alpha,
                    proximity_px=accessory_proximity_px,
                    min_component_area=accessory_min_area,
                )

            if frame_index % qc_stride == 0:
                fg = (alpha > 128).astype(np.uint8)
                edge_values.append(float(alpha[edge_mask > 0].mean()))
                area_values.append(float(fg.mean()))
                qc_fg = fg if max(h, w) <= 960 else cv2.resize(fg, (max(1, w // 2), max(1, h // 2)), interpolation=cv2.INTER_NEAREST)
                num_comp, _, _, _ = cv2.connectedComponentsWithStats(qc_fg, connectivity=8)
                comp_values.append(float(num_comp - 1))

            if bg_rgb is None:
                rgba = np.dstack([frame_rgb, alpha]).astype(np.uint8)
                rgba = _apply_watermark_rgba(rgba, watermark_enabled=watermark_enabled)
                payload = rgba.tobytes()
            else:
                alpha_f = (alpha.astype(np.float32) / 255.0)[:, :, None]
                solid = np.full(frame_rgb.shape, (bg_rgb[0], bg_rgb[1], bg_rgb[2]), dtype=np.uint8)
                comp_rgb = (frame_rgb.astype(np.float32) * alpha_f + solid.astype(np.float32) * (1.0 - alpha_f)).astype(np.uint8)
                comp_bgr = cv2.cvtColor(comp_rgb, cv2.COLOR_RGB2BGR)
                comp_bgr = _apply_watermark_bgr(comp_bgr, watermark_enabled=watermark_enabled)
                payload = comp_bgr.tobytes()
            if proc.stdin:
                proc.stdin.write(payload)

            prev_alpha = alpha
            prev_lock = np.where(alpha > 120, 255, 0).astype(np.uint8)
            prev_gray = frame_gray
            frame_index += 1
    except Exception as exc:  # noqa: BLE001
        proc.kill()
        raise RuntimeError(f"Render failed: {exc}") from exc
    finally:
        cap.release()
        if proc.stdin:
            proc.stdin.close()
            # communicate() tries to flush stdin if present; avoid flushing a closed pipe.
            proc.stdin = None

    _, ffmpeg_err = proc.communicate()
    if proc.returncode != 0:
        msg = ffmpeg_err.decode("utf-8", errors="ignore")[-1200:]
        raise RuntimeError(f"ffmpeg failed: {msg}")
    if not output_video.exists():
        raise RuntimeError("Output video is missing.")

    return _quality_stats(
        edge_values=np.array(edge_values, dtype=np.float32),
        area_values=np.array(area_values, dtype=np.float32),
        comp_values=np.array(comp_values, dtype=np.float32),
    )


def _prepare_auto_pipeline(video_path: Path, quality: str):
    meta = _video_meta(video_path)
    if meta.width <= 0 or meta.height <= 0:
        raise RuntimeError("Video resolution unreadable.")
    sample_idx = _sample_frame_indices(meta.frame_count, count=8)
    sample_frames = _read_frames(video_path, sample_idx, max_side=960)
    if not sample_frames:
        raise RuntimeError("No sample frames.")
    border_model = _build_border_model(sample_frames, border_ratio=0.08, k=4)
    main_model, aux_model, rescue_model, cache = _choose_models_and_cache(sample_frames, quality=quality, meta=meta)
    params = _auto_params(
        quality=quality,
        sample_frames=sample_frames,
        border_model=border_model,
        alpha_cache=cache[main_model],
        main_model=main_model,
        aux_model=aux_model,
        rescue_model=rescue_model,
    )
    return meta, border_model, params


def _video_infer_stride(quality: str, meta: VideoMeta) -> int:
    q = (quality or "").strip().lower()
    base = settings.engine_video_infer_stride_ultra if q == "ultra" else settings.engine_video_infer_stride_balanced
    stride = max(1, min(12, int(base)))
    if meta.fps >= 45.0:
        stride = min(12, stride + 1)
    if meta.frame_count >= 360:
        stride = min(12, stride + 1)
    if _runtime_is_cpu():
        target_keyframes = settings.engine_video_cpu_max_keyframes_ultra if q == "ultra" else settings.engine_video_cpu_max_keyframes_balanced
        target_keyframes = max(6, int(target_keyframes))
        stride = max(stride, int(np.ceil(max(1, meta.frame_count) / float(target_keyframes))))
        if q == "ultra" and meta.fps >= 30.0:
            stride = max(stride, 6)
    return stride


def _video_pass_count(quality: str, meta: VideoMeta) -> int:
    if (quality or "").strip().lower() != "ultra":
        return 1
    configured = max(1, int(settings.engine_ultra_max_passes))
    if _runtime_is_cpu():
        configured = min(configured, 1)
    if meta.fps >= 45.0 or meta.frame_count >= 300:
        configured = min(configured, 1)
    return configured


def _should_qc_retry(quality: str, stats: RenderStats | None) -> bool:
    if stats is None:
        return False
    if (quality or "").strip().lower() != "ultra":
        return False
    if _runtime_is_cpu():
        return False
    threshold = max(1, int(settings.engine_video_qc_retry_threshold))
    return len(stats.suspect_frames) >= threshold


def process_video(
    video_path: Path,
    output_path: Path,
    quality: str,
    erase_mask: np.ndarray | None = None,
    keep_mask: np.ndarray | None = None,
    bg_rgb: tuple[int, int, int] | None = None,
    watermark_enabled: bool | None = None,
) -> dict:
    started_at = time.perf_counter()
    ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()
    if not Path(ffmpeg_bin).exists():
        raise RuntimeError("ffmpeg binary not found.")

    meta, border_model, params = _prepare_auto_pipeline(video_path, quality=quality)
    infer_stride = _video_infer_stride(quality, meta)
    max_passes = _video_pass_count(quality, meta)
    logger.info(
        "video.pipeline.start quality=%s fps=%.3f frames=%s size=%sx%s infer_stride=%s max_passes=%s main_model=%s aux_model=%s rescue_model=%s cpu_mode=%s",
        quality,
        meta.fps,
        meta.frame_count,
        meta.width,
        meta.height,
        infer_stride,
        max_passes,
        params.main_model,
        params.aux_model,
        params.rescue_model,
        _runtime_is_cpu(),
    )
    disable_grabcut = bool(settings.engine_video_disable_grabcut) or (
        _runtime_is_cpu() and settings.engine_video_cpu_disable_grabcut
    )
    disable_recheck = _runtime_is_cpu() and settings.engine_video_cpu_disable_recheck
    if disable_grabcut or disable_recheck:
        params = AutoParams(
            main_model=params.main_model,
            aux_model=params.aux_model,
            rescue_model=params.rescue_model,
            feather_strength=params.feather_strength,
            alpha_cutoff=params.alpha_cutoff,
            min_blob_percent=params.min_blob_percent,
            temporal_smooth=params.temporal_smooth,
            temporal_flow_strength=params.temporal_flow_strength,
            edge_refine_strength=params.edge_refine_strength,
            grabcut_iterations=0 if disable_grabcut else params.grabcut_iterations,
            keep_largest_component=params.keep_largest_component,
            border_distance_multiplier=params.border_distance_multiplier,
            border_alpha_guard=params.border_alpha_guard,
            protect_dilate_px=params.protect_dilate_px,
            frame_recheck_enabled=(False if disable_recheck else params.frame_recheck_enabled),
            frame_recheck_edge_threshold=params.frame_recheck_edge_threshold,
            frame_recheck_disagreement_threshold=params.frame_recheck_disagreement_threshold,
        )
    params_list = [params]
    if quality == "ultra":
        extra_passes = max(0, max_passes - 1)
        for _ in range(extra_passes):
            params_list.append(_tighten_auto_params(params_list[-1]))

    output_dir = output_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    best_video: Path | None = None
    best_stats: RenderStats | None = None
    best_params: AutoParams | None = None
    passes_run = 0

    suffix = ".webm" if bg_rgb is None else ".mp4"
    for idx, pass_params in enumerate(params_list, start=1):
        if (
            best_stats is not None
            and (
                len(best_stats.suspect_frames) == 0
                or (
                    len(best_stats.suspect_frames) <= 1
                    and best_stats.edge_leak_mean < 4.8
                )
            )
        ):
            break
        pass_video = output_dir / f"{output_path.stem}_pass{idx}{suffix}"
        pass_stats = _render_with_params(
            video_path=video_path,
            output_video=pass_video,
            ffmpeg_bin=ffmpeg_bin,
            meta=meta,
            border_model=border_model,
            params=pass_params,
            erase_mask=erase_mask,
            keep_mask=keep_mask,
            bg_rgb=bg_rgb,
            watermark_enabled=watermark_enabled,
            main_max_side=_inference_side_limit(quality, "video"),
            aux_max_side=960,
            rescue_max_side=960,
            video_infer_stride=infer_stride,
            strict_enabled=((quality or "").strip().lower() == "ultra"),
        )
        passes_run += 1
        logger.info(
            "video.pipeline.pass_done pass=%s quality=%s qc_suspect=%s edge_leak_mean=%.3f edge_leak_max=%.3f area_mean=%.4f comp_max=%s",
            idx,
            quality,
            len(pass_stats.suspect_frames),
            pass_stats.edge_leak_mean,
            pass_stats.edge_leak_max,
            pass_stats.area_mean,
            pass_stats.comp_max,
        )
        if best_stats is None:
            best_video = pass_video
            best_stats = pass_stats
            best_params = pass_params
        else:
            prev_key = (len(best_stats.suspect_frames), best_stats.edge_leak_mean, best_stats.comp_max)
            cur_key = (len(pass_stats.suspect_frames), pass_stats.edge_leak_mean, pass_stats.comp_max)
            if cur_key < prev_key:
                best_video = pass_video
                best_stats = pass_stats
                best_params = pass_params
        if quality != "ultra":
            break

    if best_video is not None and best_stats is not None and best_params is not None and _should_qc_retry(quality, best_stats):
        retry_params = _tighten_auto_params(best_params)
        stride_delta = max(0, int(settings.engine_video_qc_retry_infer_stride_delta))
        retry_infer_stride = max(1, infer_stride - stride_delta)
        retry_video = output_dir / f"{output_path.stem}_qcretry{suffix}"
        logger.info(
            "video.pipeline.qc_retry.start quality=%s suspect=%s edge_leak_mean=%.3f infer_stride=%s->%s",
            quality,
            len(best_stats.suspect_frames),
            best_stats.edge_leak_mean,
            infer_stride,
            retry_infer_stride,
        )
        retry_stats = _render_with_params(
            video_path=video_path,
            output_video=retry_video,
            ffmpeg_bin=ffmpeg_bin,
            meta=meta,
            border_model=border_model,
            params=retry_params,
            erase_mask=erase_mask,
            keep_mask=keep_mask,
            bg_rgb=bg_rgb,
            watermark_enabled=watermark_enabled,
            main_max_side=_inference_side_limit(quality, "video"),
            aux_max_side=960,
            rescue_max_side=960,
            video_infer_stride=retry_infer_stride,
            strict_enabled=((quality or "").strip().lower() == "ultra"),
        )
        passes_run += 1
        logger.info(
            "video.pipeline.qc_retry.done quality=%s qc_suspect=%s edge_leak_mean=%.3f",
            quality,
            len(retry_stats.suspect_frames),
            retry_stats.edge_leak_mean,
        )
        prev_key = (len(best_stats.suspect_frames), best_stats.edge_leak_mean, best_stats.comp_max)
        cur_key = (len(retry_stats.suspect_frames), retry_stats.edge_leak_mean, retry_stats.comp_max)
        if cur_key < prev_key:
            best_video = retry_video
            best_stats = retry_stats
            best_params = retry_params

    if best_video is None or best_stats is None or best_params is None:
        raise RuntimeError("Video processing terminated unexpectedly.")
    if output_path.exists():
        output_path.unlink()
    best_video.replace(output_path)
    elapsed = time.perf_counter() - started_at
    logger.info(
        "video.pipeline.completed quality=%s elapsed_sec=%.3f passes_run=%s selected_model=%s selected_aux=%s qc_suspect=%s",
        quality,
        elapsed,
        passes_run,
        best_params.main_model,
        best_params.aux_model,
        len(best_stats.suspect_frames),
    )
    return {
        "output_path": str(output_path),
        "main_model": best_params.main_model,
        "aux_model": best_params.aux_model,
        "qc_suspect_frames": len(best_stats.suspect_frames),
        "passes": passes_run,
    }


def _image_auto_params(frame_rgb: np.ndarray, quality: str):
    border_model = _build_border_model([frame_rgb], border_ratio=0.08, k=4)
    main_model, aux_model, rescue_model, cache = _choose_models_and_cache([frame_rgb], quality=quality)
    params = _auto_params(
        quality=quality,
        sample_frames=[frame_rgb],
        border_model=border_model,
        alpha_cache=cache[main_model],
        main_model=main_model,
        aux_model=aux_model,
        rescue_model=rescue_model,
    )
    return border_model, params


def process_image(
    image_path: Path,
    output_path: Path,
    quality: str,
    erase_mask: np.ndarray | None = None,
    keep_mask: np.ndarray | None = None,
    bg_rgb: tuple[int, int, int] | None = None,
    watermark_enabled: bool | None = None,
) -> dict:
    frame_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if frame_bgr is None:
        raise RuntimeError("Image read failed.")
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    h, w = frame_rgb.shape[:2]

    border_model, params = _image_auto_params(frame_rgb, quality=quality)
    session_main = _session_for(params.main_model)
    alpha_main = _infer_alpha(session_main, frame_rgb, max_side=_inference_side_limit(quality, "image"))
    alpha = alpha_main
    reference_alpha = alpha_main.copy()
    aux_disagreement = 0.0
    if params.aux_model:
        session_aux = _session_for(params.aux_model)
        aux_alpha = _infer_alpha(session_aux, frame_rgb, max_side=1280)
        aux_disagreement = float(np.mean(np.abs(alpha_main.astype(np.int16) - aux_alpha.astype(np.int16))))
        alpha = _fuse_alpha_maps(alpha_main, aux_alpha, params.main_model, params.aux_model)
        reference_alpha = np.maximum(reference_alpha, aux_alpha)

    bg_mask = _border_connected_bg_mask(frame_rgb, border_model=border_model, dist_multiplier=params.border_distance_multiplier)
    sure_fg = (alpha >= 225).astype(np.uint8) * 255
    main_fg = _largest_component(sure_fg)
    k = params.protect_dilate_px
    protect_zone = cv2.dilate(main_fg, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)), iterations=1)
    suppress = (bg_mask > 0) & (protect_zone == 0) & (alpha < params.border_alpha_guard)
    alpha[suppress] = 0
    alpha = _grabcut_refine(frame_bgr, alpha, iterations=params.grabcut_iterations)
    alpha = _apply_manual_masks(alpha, erase_mask=erase_mask, keep_mask=keep_mask)

    min_area_px = int((w * h) * (params.min_blob_percent / 100.0))
    alpha = _refine_alpha(
        alpha,
        feather_strength=params.feather_strength,
        alpha_cutoff=params.alpha_cutoff,
        min_area_px=min_area_px,
        keep_largest=False,
        temporal_smooth=0.0,
        prev_alpha=None,
    )
    alpha = _edge_aware_refine(alpha, frame_bgr=frame_bgr, strength=params.edge_refine_strength)
    alpha = _soft_edge_matte(alpha, strength=float(np.clip(0.20 + (params.edge_refine_strength * 0.62), 0.20, 0.88)))
    alpha = _apply_manual_masks(alpha, erase_mask=erase_mask, keep_mask=keep_mask)

    edge_band = max(8, int(min(h, w) * 0.08))
    edge_mask = np.zeros((h, w), dtype=np.uint8)
    edge_mask[:edge_band, :] = 1
    edge_mask[-edge_band:, :] = 1
    edge_mask[:, :edge_band] = 1
    edge_mask[:, -edge_band:] = 1
    leak_before_rescue = float(alpha[edge_mask > 0].mean())
    if (
        params.frame_recheck_enabled
        and params.rescue_model
        and (
            leak_before_rescue > params.frame_recheck_edge_threshold
            or aux_disagreement > params.frame_recheck_disagreement_threshold
        )
    ):
        rescue_alpha = _infer_alpha(_session_for(params.rescue_model), frame_rgb, max_side=1200)
        candidate = _fuse_alpha_maps(alpha, rescue_alpha, params.main_model, params.rescue_model)
        candidate = _apply_manual_masks(candidate, erase_mask=erase_mask, keep_mask=keep_mask)
        candidate = _refine_alpha(
            candidate,
            feather_strength=params.feather_strength,
            alpha_cutoff=params.alpha_cutoff,
            min_area_px=min_area_px,
            keep_largest=False,
            temporal_smooth=0.0,
            prev_alpha=None,
        )
        candidate = _edge_aware_refine(candidate, frame_bgr=frame_bgr, strength=params.edge_refine_strength)
        candidate = _soft_edge_matte(candidate, strength=float(np.clip(0.20 + (params.edge_refine_strength * 0.62), 0.20, 0.88)))
        candidate = _apply_manual_masks(candidate, erase_mask=erase_mask, keep_mask=keep_mask)
        candidate_leak = float(candidate[edge_mask > 0].mean())
        if candidate_leak + 0.35 < leak_before_rescue:
            alpha = candidate
            reference_alpha = np.maximum(reference_alpha, rescue_alpha)

    proximity_px = max(19, int(min(h, w) * 0.05))
    min_accessory_area = max(160, int((w * h) * 0.00018))
    alpha = _recover_near_subject_components(
        alpha=alpha,
        reference_alpha=reference_alpha,
        proximity_px=proximity_px,
        min_component_area=min_accessory_area,
    )
    alpha = _apply_manual_masks(alpha, erase_mask=erase_mask, keep_mask=keep_mask)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if bg_rgb is None:
        rgba = np.dstack([frame_rgb, alpha]).astype(np.uint8)
        rgba = _apply_watermark_rgba(rgba, watermark_enabled=watermark_enabled)
        out_bgra = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA)
        cv2.imwrite(str(output_path), out_bgra)
        output_kind = "image/png"
    else:
        alpha_f = (alpha.astype(np.float32) / 255.0)[:, :, None]
        solid = np.full(frame_rgb.shape, (bg_rgb[0], bg_rgb[1], bg_rgb[2]), dtype=np.uint8)
        comp_rgb = (frame_rgb.astype(np.float32) * alpha_f + solid.astype(np.float32) * (1.0 - alpha_f)).astype(np.uint8)
        comp_bgr = cv2.cvtColor(comp_rgb, cv2.COLOR_RGB2BGR)
        comp_bgr = _apply_watermark_bgr(comp_bgr, watermark_enabled=watermark_enabled)
        cv2.imwrite(str(output_path), comp_bgr)
        output_kind = "image/png"

    leak = float(alpha[edge_mask > 0].mean())
    return {
        "output_path": str(output_path),
        "output_mime": output_kind,
        "main_model": params.main_model,
        "aux_model": params.aux_model,
        "qc_suspect_frames": int(1 if leak > 8 else 0),
    }


def extract_video_frame(video_path: Path, time_sec: float = 0.0) -> dict:
    cap = cv2.VideoCapture(str(video_path))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    target = max(0, int(float(time_sec) * fps))
    if total > 0:
        target = min(target, total - 1)
    cap.set(cv2.CAP_PROP_POS_FRAMES, target)
    ok, frame_bgr = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError("Frame extraction failed.")
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    ok2, enc = cv2.imencode(".png", cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
    if not ok2:
        raise RuntimeError("Frame encoding failed.")
    b64 = base64.b64encode(enc.tobytes()).decode("ascii")
    h, w = frame_rgb.shape[:2]
    return {
        "frame_png_base64": b64,
        "width": w,
        "height": h,
    }


def temp_dir(prefix: str) -> Path:
    return Path(tempfile.mkdtemp(prefix=prefix))
