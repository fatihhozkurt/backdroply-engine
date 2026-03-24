from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "Backdroply Engine"
    engine_port: int = 9000
    engine_workdir: str = "/app/data"
    engine_allowed_origins: str = "http://localhost:5173,http://localhost:8080"
    engine_log_level: str = "INFO"
    engine_shared_token: str = "change_me_engine_token"
    engine_max_image_mb: int = 15
    engine_max_video_mb: int = 120
    engine_max_video_upload_seconds: int = 120
    engine_max_video_seconds: int = 18
    engine_max_image_pixels: int = 8_500_000
    engine_max_video_pixels: int = 2_500_000
    engine_default_quality: str = "ultra"
    engine_allow_color_bg: bool = True
    engine_keep_jobs_hours: int = 12
    engine_watermark_enabled: bool = True
    engine_watermark_text: str = "made by backdroply"
    # Keep defaults memory-safe; heavier models can be opt-in through env.
    engine_model_candidates_ultra: str = "u2netp"
    engine_model_candidates_balanced: str = "u2netp"
    engine_ultra_max_passes: int = 2
    engine_video_process_fps_cap: int = 0
    engine_grabcut_frame_stride: int = 2
    engine_ultra_strict_model: str = "isnet-general-use"
    engine_ultra_strict_threshold: int = 10
    engine_ultra_strict_max_ratio_pct: int = 35
    engine_strict_model_max_side: int = 960
    engine_video_infer_stride_ultra: int = 2
    engine_video_infer_stride_balanced: int = 3
    engine_video_max_side_ultra: int = 1024
    engine_video_max_side_balanced: int = 896
    engine_video_motion_refresh_threshold: float = 8.0
    engine_video_flow_max_side: int = 640
    engine_video_qc_sample_stride: int = 2
    engine_video_cpu_ultra_max_side: int = 896
    engine_video_cpu_balanced_max_side: int = 704
    engine_video_cpu_max_keyframes_ultra: int = 28
    engine_video_cpu_max_keyframes_balanced: int = 22
    engine_video_cpu_disable_grabcut: bool = True
    engine_video_cpu_disable_recheck: bool = True
    engine_video_vp9_cpu_used: int = 5
    engine_video_vp9_crf: int = 18
    engine_video_vp9_deadline: str = "realtime"
    engine_video_vp9_threads: int = 0
    engine_temporal_flow_strength: float = 0.22
    engine_edge_refine_strength: float = 0.7
    engine_enable_frame_recheck: bool = True
    engine_recheck_edge_threshold: float = 9.0
    engine_recheck_disagreement_threshold: float = 28.0
    engine_max_concurrent_jobs: int = 2
    engine_queue_wait_seconds: float = 2.0
    engine_video_use_isolated_worker: bool = False
    engine_video_disable_grabcut: bool = True
    engine_video_bgmask_max_side: int = 640
    engine_video_accessory_recover: bool = True
    engine_video_accessory_proximity_ratio: float = 0.05
    engine_video_accessory_min_area_ratio: float = 0.00015
    engine_video_worker_timeout_sec: int = 0
    engine_video_worker_timeout_factor: float = 20.0
    engine_prewarm_on_startup: bool = True
    engine_prewarm_quality: str = "ultra"
    engine_video_qc_retry_threshold: int = 220
    engine_video_qc_retry_infer_stride_delta: int = 1
    engine_ort_providers: str = ""
    engine_ort_allow_cpu_fallback: bool = True
    engine_ort_runtime_flavor: str = "cpu"
    engine_ort_probe_timeout_sec: int = 12

    @property
    def workdir(self) -> Path:
        p = Path(self.engine_workdir).expanduser().resolve()
        p.mkdir(parents=True, exist_ok=True)
        (p / "jobs").mkdir(parents=True, exist_ok=True)
        return p

    @property
    def allowed_origins(self) -> list[str]:
        return [x.strip() for x in self.engine_allowed_origins.split(",") if x.strip()]

    def model_candidates(self, quality: str) -> list[str]:
        raw = self.engine_model_candidates_ultra if (quality or "").strip().lower() == "ultra" else self.engine_model_candidates_balanced
        seen: set[str] = set()
        ordered: list[str] = []
        for token in raw.split(","):
            model = token.strip()
            if not model or model in seen:
                continue
            seen.add(model)
            ordered.append(model)
        if not ordered:
            return ["u2netp"]
        return ordered

    @property
    def ort_provider_order(self) -> list[str]:
        raw = self.engine_ort_providers.strip()
        if not raw:
            return ["CUDAExecutionProvider", "ROCMExecutionProvider", "CPUExecutionProvider"]
        seen: set[str] = set()
        ordered: list[str] = []
        for token in raw.split(","):
            provider = token.strip()
            if not provider or provider in seen:
                continue
            seen.add(provider)
            ordered.append(provider)
        if not ordered:
            return ["CPUExecutionProvider"]
        return ordered


settings = Settings()
