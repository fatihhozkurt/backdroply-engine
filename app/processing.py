from __future__ import annotations

import base64
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np
from rembg import new_session, remove

from .config import settings

# Keep defaults light enough to avoid OOM/SIGKILL in constrained Docker hosts.
# A single lightweight model is the safest default for local Docker memory limits.
MODEL_CANDIDATES = ["u2netp"]
_SESSION_CACHE: dict[str, object] = {}


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
    feather_strength: float
    alpha_cutoff: int
    min_blob_percent: float
    temporal_smooth: float
    grabcut_iterations: int
    keep_largest_component: bool
    border_distance_multiplier: float
    border_alpha_guard: int
    protect_dilate_px: int


@dataclass
class RenderStats:
    frame_count: int
    edge_leak_mean: float
    edge_leak_max: float
    area_mean: float
    area_min: float
    comp_max: int
    suspect_frames: list[int]


def _session_for(model_name: str):
    if model_name not in _SESSION_CACHE:
        _SESSION_CACHE[model_name] = new_session(model_name)
    return _SESSION_CACHE[model_name]


def clear_model_cache():
    _SESSION_CACHE.clear()


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
        return 1280 if q == "ultra" else 960
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
    lab = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    centers = border_model.centers_lab
    radii = border_model.radii * max(0.6, dist_multiplier)

    diff = lab[:, :, None, :] - centers[None, None, :, :]
    dist = np.linalg.norm(diff, axis=3)
    norm_score = np.min(dist / (radii[None, None, :] + 1e-6), axis=2)
    bg_like = (norm_score < 1.0).astype(np.uint8)
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
        return np.zeros_like(bg_like, dtype=np.uint8)
    lut = np.zeros((num,), dtype=np.uint8)
    for lb in border_labels:
        lut[int(lb)] = 1
    return lut[labels].astype(np.uint8)


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


def _choose_models_and_cache(sample_frames: list[np.ndarray]) -> tuple[str, str, dict[str, list[np.ndarray]]]:
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

    for model in MODEL_CANDIDATES:
        session = _session_for(model)
        alphas = [_infer_alpha(session, frame, max_side=960) for frame in sample_frames]
        cache[model] = alphas

        center_scores = np.array([_mean_on_mask(a, center) for a in alphas], dtype=np.float32)
        edge_scores = np.array([_mean_on_mask(a, edge) for a in alphas], dtype=np.float32)
        area_scores = np.array([float((a > 140).mean()) for a in alphas], dtype=np.float32)
        score = float(center_scores.mean() - 1.7 * edge_scores.mean() - 45.0 * area_scores.mean())
        model_scores[model] = score

    sorted_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)
    main_model = sorted_models[0][0]
    aux_model = sorted_models[1][0] if len(sorted_models) > 1 else sorted_models[0][0]
    return main_model, aux_model, cache


def _auto_params(
    quality: str,
    sample_frames: list[np.ndarray],
    border_model: BorderModel,
    alpha_cache: list[np.ndarray],
    main_model: str,
    aux_model: str,
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
        feather = 1.35
        temporal = 0.30
        grabcut = 2
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
    use_aux = (quality == "ultra") or (leak > 0.08)

    return AutoParams(
        main_model=main_model,
        aux_model=aux_model if use_aux else None,
        feather_strength=feather,
        alpha_cutoff=alpha_cutoff,
        min_blob_percent=min_blob,
        temporal_smooth=temporal,
        grabcut_iterations=grabcut,
        keep_largest_component=True,
        border_distance_multiplier=dist_mult,
        border_alpha_guard=242,
        protect_dilate_px=protect,
    )


def _tighten_auto_params(params: AutoParams) -> AutoParams:
    return AutoParams(
        main_model=params.main_model,
        aux_model=params.aux_model,
        feather_strength=float(min(2.0, params.feather_strength + 0.2)),
        alpha_cutoff=int(min(224, params.alpha_cutoff + 12)),
        min_blob_percent=float(min(0.85, params.min_blob_percent + 0.08)),
        temporal_smooth=float(min(0.48, params.temporal_smooth + 0.08)),
        grabcut_iterations=int(min(3, params.grabcut_iterations + 1)),
        keep_largest_component=True,
        border_distance_multiplier=float(min(1.48, params.border_distance_multiplier + 0.12)),
        border_alpha_guard=int(max(226, params.border_alpha_guard - 8)),
        protect_dilate_px=int(min(57, params.protect_dilate_px + 8)),
    )


def _quality_stats(edge_values: np.ndarray, area_values: np.ndarray, comp_values: np.ndarray) -> RenderStats:
    if len(edge_values) == 0:
        return RenderStats(0, 0.0, 0.0, 0.0, 0.0, 0, [])
    edge_thr = max(8.0, float(edge_values.mean() + 2.2 * (edge_values.std() + 1e-6)))
    area_thr = max(0.10, float(area_values.mean() - 2.1 * (area_values.std() + 1e-6)))
    comp_thr = max(3, int(np.ceil(comp_values.mean() + 2.0 * (comp_values.std() + 1e-6))))
    sus_edge = np.where(edge_values > edge_thr)[0]
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
        "-lossless",
        "1",
        "-pix_fmt",
        "yuva420p",
        "-row-mt",
        "1",
        "-deadline",
        "good",
        "-cpu-used",
        "2",
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


def _watermark_layout(width: int, height: int, text: str) -> tuple[int, int, int, int, int, int, int, int]:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.42, min(0.95, width / 1800.0))
    thickness = max(1, int(round(scale * 2.0)))
    icon_size = max(8, int(round(13 * scale)))
    gap = max(5, int(round(7 * scale)))
    margin = max(8, int(round(18 * scale)))
    (text_w, text_h), baseline = cv2.getTextSize(text, font, scale, thickness)
    total_w = icon_size * 2 + gap + text_w
    x = max(margin, width - total_w - margin)
    y = max(text_h + margin, height - margin)
    return x, y, icon_size, gap, text_w, text_h, baseline, thickness


def _apply_watermark_bgr(frame_bgr: np.ndarray) -> np.ndarray:
    if not settings.engine_watermark_enabled:
        return frame_bgr
    text = (settings.engine_watermark_text or "").strip()
    if not text:
        return frame_bgr
    h, w = frame_bgr.shape[:2]
    if h < 48 or w < 96:
        return frame_bgr

    x, y, icon_size, gap, text_w, text_h, baseline, thickness = _watermark_layout(w, h, text)
    box_pad = max(4, int(round(icon_size * 0.45)))
    box_x0 = max(0, x - box_pad)
    box_y0 = max(0, y - text_h - box_pad)
    box_x1 = min(w - 1, x + icon_size * 2 + gap + text_w + box_pad)
    box_y1 = min(h - 1, y + baseline + box_pad)

    out = frame_bgr.copy()
    overlay = out.copy()
    cv2.rectangle(overlay, (box_x0, box_y0), (box_x1, box_y1), (8, 16, 34), -1, cv2.LINE_AA)
    out = cv2.addWeighted(overlay, 0.34, out, 0.66, 0.0)

    icon_cx = x + icon_size
    icon_cy = y - (text_h // 2)
    _draw_watermark_icon(out, icon_cx + 1, icon_cy + 1, icon_size, (10, 18, 38), max(1, thickness + 1))
    _draw_watermark_icon(out, icon_cx, icon_cy, icon_size, (120, 210, 255), thickness)

    text_x = x + icon_size * 2 + gap
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(out, text, (text_x + 1, y + 1), font, max(0.42, min(0.95, w / 1800.0)), (8, 16, 34), thickness + 1, cv2.LINE_AA)
    cv2.putText(out, text, (text_x, y), font, max(0.42, min(0.95, w / 1800.0)), (232, 240, 255), thickness, cv2.LINE_AA)
    return out


def _alpha_compose_rgba(base: np.ndarray, overlay: np.ndarray) -> np.ndarray:
    base_f = base.astype(np.float32)
    ov_f = overlay.astype(np.float32)
    ov_a = ov_f[:, :, 3:4] / 255.0
    base_a = base_f[:, :, 3:4] / 255.0
    out_rgb = ov_f[:, :, :3] * ov_a + base_f[:, :, :3] * (1.0 - ov_a)
    out_a = (ov_a + base_a * (1.0 - ov_a)) * 255.0
    out = np.concatenate([out_rgb, out_a], axis=2)
    return np.clip(out, 0, 255).astype(np.uint8)


def _apply_watermark_rgba(frame_rgba: np.ndarray) -> np.ndarray:
    if not settings.engine_watermark_enabled:
        return frame_rgba
    text = (settings.engine_watermark_text or "").strip()
    if not text:
        return frame_rgba
    h, w = frame_rgba.shape[:2]
    if h < 48 or w < 96:
        return frame_rgba

    x, y, icon_size, gap, text_w, text_h, baseline, thickness = _watermark_layout(w, h, text)
    wm = np.zeros_like(frame_rgba, dtype=np.uint8)

    box_pad = max(4, int(round(icon_size * 0.45)))
    box_x0 = max(0, x - box_pad)
    box_y0 = max(0, y - text_h - box_pad)
    box_x1 = min(w - 1, x + icon_size * 2 + gap + text_w + box_pad)
    box_y1 = min(h - 1, y + baseline + box_pad)
    cv2.rectangle(wm, (box_x0, box_y0), (box_x1, box_y1), (8, 16, 34, 132), -1, cv2.LINE_AA)

    icon_cx = x + icon_size
    icon_cy = y - (text_h // 2)
    _draw_watermark_icon(wm, icon_cx + 1, icon_cy + 1, icon_size, (10, 18, 38, 220), max(1, thickness + 1))
    _draw_watermark_icon(wm, icon_cx, icon_cy, icon_size, (120, 210, 255, 220), thickness)

    text_x = x + icon_size * 2 + gap
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(wm, text, (text_x + 1, y + 1), font, max(0.42, min(0.95, w / 1800.0)), (8, 16, 34, 220), thickness + 1, cv2.LINE_AA)
    cv2.putText(wm, text, (text_x, y), font, max(0.42, min(0.95, w / 1800.0)), (232, 240, 255, 220), thickness, cv2.LINE_AA)

    return _alpha_compose_rgba(frame_rgba, wm)


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
    main_max_side: int,
    aux_max_side: int,
) -> RenderStats:
    cmd = (
        _ffmpeg_cmd_transparent(ffmpeg_bin, video_path, output_video, meta.width, meta.height, meta.fps)
        if bg_rgb is None
        else _ffmpeg_cmd_solid(ffmpeg_bin, video_path, output_video, meta.width, meta.height, meta.fps)
    )
    session_main = _session_for(params.main_model)
    session_aux = _session_for(params.aux_model) if params.aux_model else None
    min_area_px = int((meta.width * meta.height) * (params.min_blob_percent / 100.0))
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError("Video read error.")
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    prev_alpha = None
    prev_lock = None
    h, w = meta.height, meta.width
    edge_band = max(8, int(min(h, w) * 0.08))
    edge_mask = np.zeros((h, w), dtype=np.uint8)
    edge_mask[:edge_band, :] = 1
    edge_mask[-edge_band:, :] = 1
    edge_mask[:, :edge_band] = 1
    edge_mask[:, -edge_band:] = 1
    edge_values: list[float] = []
    area_values: list[float] = []
    comp_values: list[float] = []
    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            alpha = _infer_alpha(session_main, frame_rgb, max_side=main_max_side)
            if session_aux is not None:
                aux_alpha = _infer_alpha(session_aux, frame_rgb, max_side=aux_max_side)
                alpha = _fuse_human_general(
                    alpha,
                    aux_alpha,
                    main_model=params.main_model,
                    aux_model=params.aux_model or params.main_model,
                )

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
            keep_zone = cv2.dilate(fg_lock, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17)), iterations=1)
            alpha[(keep_zone == 0) & (alpha < 248)] = 0
            alpha = _grabcut_refine(frame_bgr, alpha, iterations=params.grabcut_iterations)
            alpha = _apply_manual_masks(alpha, erase_mask=erase_mask, keep_mask=keep_mask)
            alpha = _refine_alpha(
                alpha,
                feather_strength=params.feather_strength,
                alpha_cutoff=params.alpha_cutoff,
                min_area_px=min_area_px,
                keep_largest=params.keep_largest_component,
                temporal_smooth=params.temporal_smooth,
                prev_alpha=prev_alpha,
            )

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

            fg = (alpha > 128).astype(np.uint8)
            edge_values.append(float(alpha[edge_mask > 0].mean()))
            area_values.append(float(fg.mean()))
            num_comp, _, _, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
            comp_values.append(float(num_comp - 1))

            if bg_rgb is None:
                rgba = np.dstack([frame_rgb, alpha]).astype(np.uint8)
                rgba = _apply_watermark_rgba(rgba)
                payload = rgba.tobytes()
            else:
                alpha_f = (alpha.astype(np.float32) / 255.0)[:, :, None]
                solid = np.full(frame_rgb.shape, (bg_rgb[0], bg_rgb[1], bg_rgb[2]), dtype=np.uint8)
                comp_rgb = (frame_rgb.astype(np.float32) * alpha_f + solid.astype(np.float32) * (1.0 - alpha_f)).astype(np.uint8)
                comp_bgr = cv2.cvtColor(comp_rgb, cv2.COLOR_RGB2BGR)
                comp_bgr = _apply_watermark_bgr(comp_bgr)
                payload = comp_bgr.tobytes()
            if proc.stdin:
                proc.stdin.write(payload)

            prev_alpha = alpha
            prev_lock = np.where(alpha > 120, 255, 0).astype(np.uint8)
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
    main_model, aux_model, cache = _choose_models_and_cache(sample_frames)
    params = _auto_params(
        quality=quality,
        sample_frames=sample_frames,
        border_model=border_model,
        alpha_cache=cache[main_model],
        main_model=main_model,
        aux_model=aux_model,
    )
    return meta, border_model, params


def process_video(
    video_path: Path,
    output_path: Path,
    quality: str,
    erase_mask: np.ndarray | None = None,
    keep_mask: np.ndarray | None = None,
    bg_rgb: tuple[int, int, int] | None = None,
) -> dict:
    ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()
    if not Path(ffmpeg_bin).exists():
        raise RuntimeError("ffmpeg binary not found.")

    meta, border_model, params = _prepare_auto_pipeline(video_path, quality=quality)
    params_list = [params]
    if quality == "ultra":
        for _ in range(4):
            params_list.append(_tighten_auto_params(params_list[-1]))

    output_dir = output_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    best_video: Path | None = None
    best_stats: RenderStats | None = None
    best_params: AutoParams | None = None
    passes_run = 0

    suffix = ".webm" if bg_rgb is None else ".mp4"
    for idx, pass_params in enumerate(params_list, start=1):
        if best_stats is not None and len(best_stats.suspect_frames) == 0:
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
            main_max_side=_inference_side_limit(quality, "video"),
            aux_max_side=960,
        )
        passes_run += 1
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

    if best_video is None or best_stats is None or best_params is None:
        raise RuntimeError("Video processing terminated unexpectedly.")
    if output_path.exists():
        output_path.unlink()
    best_video.replace(output_path)
    return {
        "output_path": str(output_path),
        "main_model": best_params.main_model,
        "aux_model": best_params.aux_model,
        "qc_suspect_frames": len(best_stats.suspect_frames),
        "passes": passes_run,
    }


def _image_auto_params(frame_rgb: np.ndarray, quality: str):
    border_model = _build_border_model([frame_rgb], border_ratio=0.08, k=4)
    main_model, aux_model, cache = _choose_models_and_cache([frame_rgb])
    params = _auto_params(
        quality=quality,
        sample_frames=[frame_rgb],
        border_model=border_model,
        alpha_cache=cache[main_model],
        main_model=main_model,
        aux_model=aux_model,
    )
    return border_model, params


def process_image(
    image_path: Path,
    output_path: Path,
    quality: str,
    erase_mask: np.ndarray | None = None,
    keep_mask: np.ndarray | None = None,
    bg_rgb: tuple[int, int, int] | None = None,
) -> dict:
    frame_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if frame_bgr is None:
        raise RuntimeError("Image read failed.")
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    h, w = frame_rgb.shape[:2]

    border_model, params = _image_auto_params(frame_rgb, quality=quality)
    session_main = _session_for(params.main_model)
    alpha = _infer_alpha(session_main, frame_rgb, max_side=_inference_side_limit(quality, "image"))
    if params.aux_model:
        session_aux = _session_for(params.aux_model)
        aux_alpha = _infer_alpha(session_aux, frame_rgb, max_side=1280)
        alpha = _fuse_human_general(alpha, aux_alpha, params.main_model, params.aux_model)

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
        keep_largest=params.keep_largest_component,
        temporal_smooth=0.0,
        prev_alpha=None,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if bg_rgb is None:
        rgba = np.dstack([frame_rgb, alpha]).astype(np.uint8)
        rgba = _apply_watermark_rgba(rgba)
        out_bgra = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA)
        cv2.imwrite(str(output_path), out_bgra)
        output_kind = "image/png"
    else:
        alpha_f = (alpha.astype(np.float32) / 255.0)[:, :, None]
        solid = np.full(frame_rgb.shape, (bg_rgb[0], bg_rgb[1], bg_rgb[2]), dtype=np.uint8)
        comp_rgb = (frame_rgb.astype(np.float32) * alpha_f + solid.astype(np.float32) * (1.0 - alpha_f)).astype(np.uint8)
        comp_bgr = cv2.cvtColor(comp_rgb, cv2.COLOR_RGB2BGR)
        comp_bgr = _apply_watermark_bgr(comp_bgr)
        cv2.imwrite(str(output_path), comp_bgr)
        output_kind = "image/png"

    edge_band = max(8, int(min(h, w) * 0.08))
    edge_mask = np.zeros((h, w), dtype=np.uint8)
    edge_mask[:edge_band, :] = 1
    edge_mask[-edge_band:, :] = 1
    edge_mask[:, :edge_band] = 1
    edge_mask[:, -edge_band:] = 1
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
