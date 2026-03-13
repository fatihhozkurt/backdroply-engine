# Backdroply Engine

FastAPI-based background removal engine for image and video workloads.

## Scope

- Image background removal (`/v1/process/image`)
- Video background removal (`/v1/process/video`)
- Frame extraction for brush correction (`/v1/frame/extract`)
- Job output download (`/v1/jobs/{job_id}/download`)
- Cache clear endpoint (`/v1/cache/clear`)
- Runtime stats endpoint (`/v1/stats`)
- Model prewarm endpoint (`/v1/prewarm`)
- Health endpoint (`/health`)
- Multi-model cascade with automatic model scoring
- Border-aware suppression and edge-aware alpha refinement
- Optical-flow temporal smoothing for video consistency
- Per-frame QC recheck/fallback for difficult scenes (ultra mode)

All processing endpoints are protected with a shared token header.

## Local Development

Requirements:

- Python 3.12
- FFmpeg

Run:

```bash
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 9000
```

## Main Environment Variables

- `ENGINE_SHARED_TOKEN`
- `ENGINE_WORKDIR`
- `ENGINE_MAX_IMAGE_MB`
- `ENGINE_MAX_VIDEO_MB`
- `ENGINE_MAX_VIDEO_SECONDS`
- `ENGINE_DEFAULT_QUALITY`
- `ENGINE_ALLOW_COLOR_BG`
- `ENGINE_WATERMARK_ENABLED`
- `ENGINE_WATERMARK_TEXT`
- `ENGINE_MODEL_CANDIDATES_ULTRA`
- `ENGINE_MODEL_CANDIDATES_BALANCED`
- `ENGINE_ULTRA_MAX_PASSES`
- `ENGINE_TEMPORAL_FLOW_STRENGTH`
- `ENGINE_EDGE_REFINE_STRENGTH`
- `ENGINE_ENABLE_FRAME_RECHECK`
- `ENGINE_RECHECK_EDGE_THRESHOLD`
- `ENGINE_RECHECK_DISAGREEMENT_THRESHOLD`
- `ENGINE_MAX_CONCURRENT_JOBS`
- `ENGINE_QUEUE_WAIT_SECONDS`
- `ENGINE_PREWARM_ON_STARTUP`
- `ENGINE_PREWARM_QUALITY`
- `ENGINE_ORT_PROVIDERS`
- `ENGINE_ORT_ALLOW_CPU_FALLBACK`
- `ENGINE_ORT_RUNTIME_FLAVOR`

Default candidate model is memory-safe (`u2netp`).  
If your deployment has enough RAM/CPU, you can opt in heavier models by overriding candidate env values.

## GPU Primary + CPU Fallback

- Build with GPU ONNX Runtime on GPU nodes by setting Docker build arg `ENGINE_ORT_RUNTIME=gpu`.
- Keep `ENGINE_ORT_ALLOW_CPU_FALLBACK=true` to continue serving on CPU if CUDA provider is unavailable.
- Optionally force provider order with `ENGINE_ORT_PROVIDERS` (example: `CUDAExecutionProvider,CPUExecutionProvider`).
- Runtime provider state is visible under `/v1/stats` in the `runtime` field.
