from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "Backdroply Engine"
    engine_port: int = 9000
    engine_workdir: str = "/app/data"
    engine_allowed_origins: str = "http://localhost:5173,http://localhost:8080"
    engine_shared_token: str = "change_me_engine_token"
    engine_max_image_mb: int = 15
    engine_max_video_mb: int = 120
    engine_max_video_seconds: int = 18
    engine_max_image_pixels: int = 8_500_000
    engine_max_video_pixels: int = 2_500_000
    engine_default_quality: str = "ultra"
    engine_allow_color_bg: bool = True
    engine_keep_jobs_hours: int = 12
    engine_watermark_enabled: bool = True
    engine_watermark_text: str = "Made by Fatih"

    @property
    def workdir(self) -> Path:
        p = Path(self.engine_workdir).expanduser().resolve()
        p.mkdir(parents=True, exist_ok=True)
        (p / "jobs").mkdir(parents=True, exist_ok=True)
        return p

    @property
    def allowed_origins(self) -> list[str]:
        return [x.strip() for x in self.engine_allowed_origins.split(",") if x.strip()]


settings = Settings()
