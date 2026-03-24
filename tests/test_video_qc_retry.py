from __future__ import annotations

from app.processing import RenderStats, _should_qc_retry


def _stats(suspect_count: int) -> RenderStats:
    return RenderStats(
        frame_count=100,
        edge_leak_mean=5.0,
        edge_leak_max=9.0,
        area_mean=0.24,
        area_min=0.09,
        comp_max=4,
        suspect_frames=list(range(max(0, int(suspect_count)))),
    )


def test_should_qc_retry_for_ultra_when_suspect_frames_high(monkeypatch):
    monkeypatch.setattr("app.processing._runtime_is_cpu", lambda: False)
    monkeypatch.setattr("app.processing.settings.engine_video_qc_retry_threshold", 80)
    assert _should_qc_retry("ultra", _stats(120)) is True


def test_should_qc_retry_disabled_for_non_ultra(monkeypatch):
    monkeypatch.setattr("app.processing._runtime_is_cpu", lambda: False)
    monkeypatch.setattr("app.processing.settings.engine_video_qc_retry_threshold", 20)
    assert _should_qc_retry("balanced", _stats(120)) is False


def test_should_qc_retry_disabled_in_cpu_mode(monkeypatch):
    monkeypatch.setattr("app.processing._runtime_is_cpu", lambda: True)
    monkeypatch.setattr("app.processing.settings.engine_video_qc_retry_threshold", 20)
    assert _should_qc_retry("ultra", _stats(120)) is False
