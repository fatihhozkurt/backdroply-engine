from __future__ import annotations

import numpy as np

from app.processing import VideoMeta, _choose_models_and_cache


def _alpha_profile(h: int, w: int, center_val: int, edge_val: int, fg_ratio: float) -> np.ndarray:
    alpha = np.full((h, w), edge_val, dtype=np.uint8)
    ch0, ch1 = int(h * 0.25), int(h * 0.75)
    cw0, cw1 = int(w * 0.25), int(w * 0.75)
    alpha[ch0:ch1, cw0:cw1] = center_val
    # Add coarse foreground mass near center to control area term.
    fh = max(1, int(h * fg_ratio))
    fw = max(1, int(w * fg_ratio))
    sy = max(0, (h - fh) // 2)
    sx = max(0, (w - fw) // 2)
    alpha[sy : sy + fh, sx : sx + fw] = max(center_val, 200)
    return alpha


def test_dense_ultra_prefers_u2netp_with_speed_penalty(monkeypatch):
    h, w = 120, 120
    frame = np.zeros((h, w, 3), dtype=np.uint8)

    # isnet is slightly better on raw quality score, but dense-video speed penalty should flip selection to u2netp.
    alpha_isnet = _alpha_profile(h, w, center_val=236, edge_val=8, fg_ratio=0.33)
    alpha_u2netp = _alpha_profile(h, w, center_val=232, edge_val=9, fg_ratio=0.34)

    def fake_session_for(model_name: str):
        return model_name

    def fake_infer_alpha(session, frame_rgb, max_side=None, post_process_mask=True):
        del frame_rgb, max_side, post_process_mask
        return alpha_isnet if session == "isnet-general-use" else alpha_u2netp

    monkeypatch.setattr("app.processing._runtime_is_cpu", lambda: False)
    monkeypatch.setattr("app.processing._model_candidates_for_quality", lambda q: ["isnet-general-use", "u2netp"])
    monkeypatch.setattr("app.processing._session_for", fake_session_for)
    monkeypatch.setattr("app.processing._infer_alpha", fake_infer_alpha)

    meta = VideoMeta(width=w, height=h, fps=50.0, frame_count=320)
    main_model, aux_model, rescue_model, cache = _choose_models_and_cache([frame, frame], quality="ultra", meta=meta)
    assert main_model == "u2netp"
    assert aux_model == "isnet-general-use"
    assert rescue_model in {"u2netp", "isnet-general-use", None}
    assert set(cache.keys()) == {"isnet-general-use", "u2netp"}
