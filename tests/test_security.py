from __future__ import annotations

import base64
import io

import numpy as np
import pytest
from fastapi import HTTPException
from PIL import Image

from app.security import assert_internal_token, data_url_to_gray_mask, parse_hex_color, safe_filename


def _mask_data_url() -> str:
    arr = np.zeros((3, 3), dtype=np.uint8)
    arr[1, 1] = 255
    img = Image.fromarray(arr, mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def test_parse_hex_color_accepts_hex():
    assert parse_hex_color("#112233") == (17, 34, 51)


def test_parse_hex_color_rejects_invalid():
    with pytest.raises(HTTPException):
        parse_hex_color("112233")


def test_safe_filename_sanitizes_path():
    assert safe_filename("../../evil?.png") == "evil_.png"


def test_data_url_to_gray_mask_binary_output():
    mask = data_url_to_gray_mask(_mask_data_url())
    assert mask is not None
    assert mask.shape == (3, 3)
    assert int(mask[1, 1]) == 255
    assert int(mask[0, 0]) == 0


def test_assert_internal_token_rejects_invalid():
    with pytest.raises(HTTPException):
        assert_internal_token("wrong", "expected")
