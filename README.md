# Backdroply Engine

FastAPI-based background removal engine for image and video workloads.

## Scope

- Image background removal (`/v1/process/image`)
- Video background removal (`/v1/process/video`)
- Frame extraction for brush correction (`/v1/frame/extract`)
- Job output download (`/v1/jobs/{job_id}/download`)
- Cache clear endpoint (`/v1/cache/clear`)
- Health endpoint (`/health`)

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
