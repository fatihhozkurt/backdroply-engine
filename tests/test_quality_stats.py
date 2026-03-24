from __future__ import annotations

import numpy as np

from app.processing import _quality_stats


def test_quality_stats_tiny_foreground_not_all_flagged():
    # Mirrors tiny-subject videos where area mean is ~0.3%.
    edge = np.full((144,), 0.55, dtype=np.float32)
    area = np.full((144,), 0.003, dtype=np.float32)
    comp = np.ones((144,), dtype=np.float32)
    stats = _quality_stats(edge_values=edge, area_values=area, comp_values=comp)
    assert stats.frame_count == 144
    assert len(stats.suspect_frames) == 0


def test_quality_stats_normal_foreground_detects_area_drop():
    edge = np.full((30,), 1.0, dtype=np.float32)
    area = np.full((30,), 0.22, dtype=np.float32)
    area[10] = 0.001
    area[11] = 0.002
    comp = np.ones((30,), dtype=np.float32)
    stats = _quality_stats(edge_values=edge, area_values=area, comp_values=comp)
    assert len(stats.suspect_frames) >= 2
    assert 10 in stats.suspect_frames
    assert 11 in stats.suspect_frames
