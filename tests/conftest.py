"""Shared pytest fixtures: makes gbticl_pipeline importable regardless of
which directory pytest is invoked from, and exposes the project's own
device-selection logic so tests run on GPU when available (matching how the
codec actually runs) without hardcoding 'cuda' anywhere in the test files."""

import sys
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gbticl_pipeline.device_utils import get_device  # noqa: E402


@pytest.fixture(scope="session")
def device():
    return get_device()
