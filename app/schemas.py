from __future__ import annotations

from pydantic import BaseModel, Field


class ProcessResponse(BaseModel):
    job_id: str
    media_type: str
    quality: str
    output_name: str
    output_mime: str
    download_url: str
    status: str
    qc_suspect_frames: int | None = None
    model_used: str | None = None
    note: str | None = None


class FrameResponse(BaseModel):
    frame_png_base64: str
    width: int = Field(ge=1)
    height: int = Field(ge=1)


class HealthResponse(BaseModel):
    status: str
    app: str
