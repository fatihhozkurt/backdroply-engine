from __future__ import annotations

from app.config import Settings


def test_allowed_origins_parses_csv():
    settings = Settings(engine_allowed_origins="https://a.com, https://b.com ,,")
    assert settings.allowed_origins == ["https://a.com", "https://b.com"]
