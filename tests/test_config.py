from __future__ import annotations

from app.config import Settings


def test_allowed_origins_parses_csv():
    settings = Settings(engine_allowed_origins="https://a.com, https://b.com ,,")
    assert settings.allowed_origins == ["https://a.com", "https://b.com"]


def test_model_candidates_dedupe_and_fallback():
    settings = Settings(
        engine_model_candidates_ultra="u2net, u2netp, u2netp,",
        engine_model_candidates_balanced="",
    )
    assert settings.model_candidates("ultra") == ["u2net", "u2netp"]
    assert settings.model_candidates("balanced") == ["u2netp"]


def test_ort_provider_order_default_and_dedupe():
    default_settings = Settings(engine_ort_providers="")
    assert default_settings.ort_provider_order == ["CUDAExecutionProvider", "ROCMExecutionProvider", "CPUExecutionProvider"]

    explicit_settings = Settings(
        engine_ort_providers="CUDAExecutionProvider, CPUExecutionProvider, CUDAExecutionProvider,",
    )
    assert explicit_settings.ort_provider_order == ["CUDAExecutionProvider", "CPUExecutionProvider"]
