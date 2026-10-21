from pathlib import Path

import pytest

from thesiskb.core.config import REPO_ROOT, get_models_config


@pytest.fixture
def config_dir() -> Path:
    return REPO_ROOT / "config"


@pytest.fixture
def models(config_dir):
    get_models_config.cache_clear()
    return get_models_config(config_dir)
