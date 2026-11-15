import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session")
def raw():
    from bci_common.data import make_synthetic_mi
    return make_synthetic_mi(seed=42)
