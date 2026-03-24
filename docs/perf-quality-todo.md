# Backdroply Engine: Video Perf + Quality TODO

## 1) Root-cause findings (2026-03-15)
- Runtime was CPU-only (`onnxruntime` CPU provider), while video path was doing expensive per-frame segmentation + full-resolution optical flow + lossless VP9 encoding.
- Historical backend DB evidence showed ultra jobs taking `~425-660s` and timing out depending on backend timeout.
- ETA estimator was over-influenced by old, slow samples and was reporting significantly high values.

## 2) Completed optimizations
- Added CPU-adaptive video strategy:
  - Foreground-aware, downscaled optical-flow warp for inter-frame propagation.
  - More aggressive keyframe inference stride for CPU.
  - CPU-specific max inference side and pass count cap.
  - Optional CPU grabcut/recheck disable path (enabled by default for video CPU path).
- Reduced transparent video encode cost:
  - Switched from VP9 lossless to configurable high-quality VP9 (`CRF`) with realtime deadline and CPU-used tuning.
- Reduced non-output overhead:
  - QC sampling stride and downscaled connected-components for QC.
- Improved ETA:
  - Switched to recency-weighted, clipped estimator to react faster after engine improvements.

## 3) Current measured results (backend endpoints)
- `vidu-video-3191802828768854.mp4` (ultra, transparent): steady-state `~102-108s`.
- `15427335_1080_1920_50fps.mp4` (ultra, transparent): steady-state `~82-84s`.
- Very high-resolution `15363990_2160_3840_30fps.mp4` is rejected by configured pixel cap (`ENGINE_MAX_VIDEO_PIXELS=2500000`) by design.
- Occasional first-run after restart may be slower due warmup/cache effects.

## 4) Gaps to "best-in-market" quality claim
- No objective SOTA benchmark suite (public dataset + ground truth masks + temporal consistency metrics) exists yet in repo.
- Current pipeline still relies on image segmentation model family for video path; this is not ideal for temporal matting quality in hard scenes.
- Therefore, "better than all competitors" is not yet provable.

## 5) Next high-impact roadmap
1. Add benchmark harness and quality gate:
   - Datasets: fixed mixed set (portrait, full-body, fast motion, hair/fur, black-white dynamic backgrounds).
   - Metrics: SAD, MAD, grad/connectivity, temporal warping error, edge-F score.
   - CI gate: fail build if regression in p95 latency or quality metrics.
2. Add dedicated video matting backend (A/B selectable):
   - Recurrent video matting model path for temporal coherence.
   - Keep current pipeline as fallback path.
3. Add object-class routing:
   - Portrait/human -> video matting model.
   - Generic object -> high-quality DIS model for image/keyframes + temporal propagation.
4. GPU production path:
   - ORT GPU/TensorRT profile, pinned model resolutions, and per-node worker sizing.
   - Keep CPU fallback with explicit "best-effort quality" profile.
5. Scale architecture:
   - Worker pool autoscaling by queue depth and per-job SLA class.
   - Strong backpressure and queue shedding for overload safety.
6. Observability:
   - Per-stage timing emitted per job (infer, flow, refine, encode, upload).
   - Dashboards + SLO alerts for timeout/error and p95 processing time.

