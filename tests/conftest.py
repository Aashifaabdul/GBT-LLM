"""Shared fixtures: put the repo root on sys.path and provide a `device`
fixture using the project's own get_device() (GPU if available, else CPU)."""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gbticl_pipeline.device_utils import get_device  # noqa: E402


@pytest.fixture(scope="session")
def device():
    return get_device()
