import os
from pathlib import Path

import pytest

try:                                   # the device tests need 64-bit JAX; the rest do not import JAX
    import jax
    jax.config.update("jax_enable_x64", True)
except ImportError:
    pass

DATA = os.environ.get("CRYDATAPATH")


@pytest.fixture(scope="session")
def data_dir():
    if not DATA or not (Path(DATA) / "cosmics_0.data").is_file():
        pytest.skip("CRY 1.7 release tables not installed (set CRYDATAPATH)")
    return Path(DATA)
