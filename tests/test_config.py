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


@pytest.mark.parametrize("name", ["models.local.yaml", "models.api.yaml", "models.cpu.yaml"])
def test_example_configs_load_and_cover_all_roles(config_dir, name):
    import yaml

    from thesiskb.core.config import ModelsConfig

    cfg = ModelsConfig.model_validate(
        yaml.safe_load((config_dir / "examples" / name).read_text(encoding="utf-8"))
    )
    for role in (
        "answer",
        "judge",
        "router",
        "extraction",
        "vision",
        "embedding",
        "reranker",
        "asr",
        "diarization",
    ):
        assert cfg.role(role).model
    # embeddings stay local in every setup (stored vectors, privacy)
    assert cfg.role("embedding").is_local
    if name == "models.api.yaml":
        cloud = [r for r, c in cfg.roles.items() if not c.is_local]
        assert set(cloud) == {"answer", "judge", "router", "extraction"}
        assert all(f"{r}_local" in cfg.roles and cfg.role(f"{r}_local").is_local for r in cloud)


def test_doctor_cloud_check(monkeypatch, config_dir):
    import yaml

    from thesiskb.core.config import ModelsConfig
    from thesiskb.doctor import Status, check_cloud

    api = ModelsConfig.model_validate(
        yaml.safe_load((config_dir / "examples" / "models.api.yaml").read_text(encoding="utf-8"))
    )
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    r = check_cloud(api)
    assert r.status == Status.FAIL and "ANTHROPIC_API_KEY" in r.detail
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    assert check_cloud(api).status == Status.OK
