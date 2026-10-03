import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.reference.config import tiny_config  # noqa: E402


@pytest.fixture(scope="session")
def cfg():
    return tiny_config()


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)
