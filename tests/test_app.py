"""app.py: pure-logic helpers (crop_center, discover_checkpoints, build_app)
-- not the full Gradio server round-trip, which is exercised manually (see
Stage 6 verification notes), but enough to catch import/wiring regressions."""

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

pytest.importorskip("gradio")

import app  # noqa: E402


def test_crop_center_snaps_to_block_multiple():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    cropped = app.crop_center(img, 50)
    assert cropped.shape[0] % app.BLOCK_SIZE == 0
    assert cropped.shape[1] % app.BLOCK_SIZE == 0
    assert cropped.shape[0] <= 50 and cropped.shape[1] <= 50


def test_crop_center_never_exceeds_image_size():
    img = np.zeros((20, 20, 3), dtype=np.uint8)
    cropped = app.crop_center(img, 128)
    assert cropped.shape[0] <= 20 and cropped.shape[1] <= 20


def test_discover_checkpoints_returns_list():
    result = app.discover_checkpoints()
    assert isinstance(result, list)


def test_load_config_baseline_and_dct():
    gbticl, coeff, fixed_basis = app.load_config(app.BASELINE_LABEL, "(none)", "(none)")
    assert gbticl is not None and coeff is not None and fixed_basis is False

    gbticl2, coeff2, fixed_basis2 = app.load_config(app.DCT_LABEL, "(none)", "(none)")
    assert gbticl2 is None and coeff2 is not None and fixed_basis2 is True


def test_build_app_constructs_without_error():
    demo = app.build_app()
    assert demo is not None
