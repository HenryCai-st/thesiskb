"""Settings from environment (.env) plus typed YAML config from config/."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    """Secrets and endpoints. Read from env vars prefixed TKB_ or a .env file."""

    model_config = SettingsConfigDict(env_prefix="TKB_", env_file=".env", extra="ignore")

    database_url: str = "postgresql+psycopg://thesiskb:thesiskb@localhost:5432/thesiskb"
    ollama_base_url: str = "http://localhost:11434"
    hf_token: str | None = None
    api_token: str | None = None  # if set, the API/UI/MCP-over-HTTP require this bearer token
    tesseract_cmd: str = "tesseract"
    tessdata_dir: str | None = None  # language packs folder if not the install default
    public_url: str | None = None  # base for links in chat replies; default: the request's URL
    searxng_url: str = "http://localhost:8888"  # web search (M17): docker compose --profile web
    config_dir: Path = REPO_ROOT / "config"
    # which model setup to use, relative to config_dir: models.yaml (fully local, default) or
    # e.g. examples/models.api.yaml (cloud LLM, local KB)
    models_file: str = "models.yaml"


class RoleConfig(BaseModel):
    """One model role (answer, embedding, ...). Extra keys like `dim` are allowed."""

    model_config = {"extra": "allow"}

    model: str
    backend: str = "litellm"
    api_base: str | None = None  # may contain "{ollama_base_url}"
    api_key: str | None = None
    vram_gb: float = 0.0
    params: dict[str, Any] = Field(default_factory=dict)

    @property
    def provider(self) -> str:
        return self.model.split("/", 1)[0] if "/" in self.model else ""

    @property
    def model_name(self) -> str:
        return self.model.split("/", 1)[1] if "/" in self.model else self.model

    @property
    def served_by_ollama(self) -> bool:
        return self.provider.startswith("ollama") or "{ollama_base_url}" in (self.api_base or "")

    @property
    def is_local(self) -> bool:
        """Runs on this machine: Ollama, a localhost endpoint, or an in-process backend."""
        if self.backend != "litellm" or self.served_by_ollama:
            return True
        base = (self.api_base or "").lower()
        return any(h in base for h in ("://localhost", "://127.0.0.1", "://[::1]"))


class ModelsConfig(BaseModel):
    roles: dict[str, RoleConfig]

    def role(self, name: str) -> RoleConfig:
        try:
            return self.roles[name]
        except KeyError as e:
            raise KeyError(f"Model role '{name}' not defined in models.yaml") from e


class ChunkingConfig(BaseModel):
    target_tokens: int = 600
    max_tokens: int = 800
    overlap_ratio: float = 0.12


class RetrievalConfig(BaseModel):
    hybrid_top_k: int = 30
    rerank_top_k: int = 8
    rrf_k: int = 60
    min_rerank_score: float = 0.02  # below this the answer path refuses instead of generating


class EntitiesConfig(BaseModel):
    auto_merge: bool = True  # merge near-duplicate entity names after each ingest (ADR 0014)
    keep_snapshots: int = 10  # automatic graph snapshots kept in data/backups/entities/


class PipelineConfig(BaseModel):
    model_config = {"extra": "allow"}

    data_dir: Path = Path("data")
    raw_dir: Path = Path("data/raw")
    processed_dir: Path = Path("data/processed")
    pipeline_version: str = "0.1.0"
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    entities: EntitiesConfig = Field(default_factory=EntitiesConfig)


class Person(BaseModel):
    name: str
    aliases: list[str] = Field(default_factory=list)
    role: str | None = None


class PeopleConfig(BaseModel):
    people: list[Person] = Field(default_factory=list)


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


@lru_cache
def get_settings() -> Settings:
    # also export .env to the process environment: LiteLLM reads provider keys
    # (ANTHROPIC_API_KEY, OPENAI_API_KEY, ...) from there, not from Settings
    try:
        from dotenv import load_dotenv

        load_dotenv(".env", override=False)
    except ImportError:
        pass
    return Settings()


@lru_cache
def get_models_config(config_dir: Path | None = None) -> ModelsConfig:
    settings = get_settings()
    d = config_dir or settings.config_dir
    path = Path(settings.models_file)
    return ModelsConfig.model_validate(_load_yaml(path if path.is_absolute() else d / path))


@lru_cache
def get_pipeline_config(config_dir: Path | None = None) -> PipelineConfig:
    d = config_dir or get_settings().config_dir
    return PipelineConfig.model_validate(_load_yaml(d / "pipeline.yaml"))


@lru_cache
def get_people_config(config_dir: Path | None = None) -> PeopleConfig:
    d = config_dir or get_settings().config_dir
    local = d / "people.local.yaml"
    return PeopleConfig.model_validate(_load_yaml(local if local.exists() else d / "people.yaml"))
