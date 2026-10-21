import pytest

from thesiskb.core.config import Settings, get_people_config, get_pipeline_config


def test_models_yaml_defines_all_roles(models):
    for role in ("answer", "judge", "extraction", "vision", "embedding", "reranker", "asr"):
        assert models.role(role).model


def test_embedding_dim_matches_schema(models):
    from thesiskb.core.models import EMBEDDING_DIM

    assert models.role("embedding").dim == EMBEDDING_DIM


def test_unknown_role_raises(models):
    with pytest.raises(KeyError, match="nope"):
        models.role("nope")


def test_provider_parsing(models):
    assert models.role("answer").provider == "ollama_chat"
    assert models.role("embedding").provider == "ollama"


def test_pipeline_config(config_dir):
    get_pipeline_config.cache_clear()
    cfg = get_pipeline_config(config_dir)
    assert cfg.retrieval.hybrid_top_k == 30
    assert cfg.retrieval.rerank_top_k == 8
    assert cfg.model_extra["ocr"]["languages"] == ["deu", "eng"]


def test_people_config(config_dir):
    get_people_config.cache_clear()
    assert get_people_config(config_dir).people


def test_settings_from_env(monkeypatch):
    monkeypatch.setenv("TKB_OLLAMA_BASE_URL", "http://gpu-box:11434")
    assert Settings(_env_file=None).ollama_base_url == "http://gpu-box:11434"
