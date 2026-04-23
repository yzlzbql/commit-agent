from __future__ import annotations

from pathlib import Path
import tomllib
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field


class ModelCfg(BaseModel):
    name: str = "gpt-5.4"
    name_env: str | None = "OPENAI_MODEL"
    api_key_env: str = "OPENAI_API_KEY"
    base_url_env: str | None = "OPENAI_BASE_URL"
    organization_env: str | None = "OPENAI_ORGANIZATION"
    project_env: str | None = "OPENAI_PROJECT"
    reasoning_effort: Literal["low", "medium", "high"] | None = None
    verbosity: Literal["low", "medium", "high"] | None = None
    temperature: float | None = None
    max_completion_tokens: int | None = None
    timeout_seconds: float = 120.0


class LimitsCfg(BaseModel):
    max_steps: int | None = None
    max_retries: int = 3
    max_repeat_tool_calls: int = 0
    doom_loop_threshold: int = 2
    enable_compacted_summary: bool = True


class VerifyCfg(BaseModel):
    commands: list[str] = Field(default_factory=list)


class LspServerCfg(BaseModel):
    name: str
    extensions: list[str] = Field(default_factory=list)
    command: list[str] = Field(default_factory=list)
    root_markers: list[str] = Field(default_factory=list)


class LspCfg(BaseModel):
    servers: list[LspServerCfg] = Field(default_factory=list)


class AnalysisCfg(BaseModel):
    prefer_lsp: bool = True
    default_timeout_ms: int = 5000
    index_max_files: int = 10000
    ctags_command: list[str] = Field(default_factory=lambda: ["ctags"])


class AgentConfig(BaseModel):
    model: ModelCfg = Field(default_factory=ModelCfg)
    limits: LimitsCfg = Field(default_factory=LimitsCfg)
    verify: VerifyCfg = Field(default_factory=VerifyCfg)
    lsp: LspCfg = Field(default_factory=LspCfg)
    analysis: AnalysisCfg = Field(default_factory=AnalysisCfg)

    @classmethod
    def load(
        cls,
        cwd: Path,
        *,
        env_search_paths: list[Path] | None = None,
        config_search_paths: list[Path] | None = None,
    ) -> "AgentConfig":
        paths = []
        for item in env_search_paths or []:
            path = Path(item).resolve()
            if path not in paths:
                paths.append(path)
        base = cwd.resolve()
        if base not in paths:
            paths.append(base)
        for item in paths:
            env_path = item / ".env"
            if env_path.exists():
                load_dotenv(env_path, override=False)

        config_paths = []
        for item in config_search_paths or []:
            path = Path(item).resolve()
            if path not in config_paths:
                config_paths.append(path)
        if base not in config_paths:
            config_paths.append(base)

        data: dict[str, object] = {}
        for item in config_paths:
            path = item / ".agent" / "config.toml"
            if not path.exists():
                continue
            loaded = tomllib.loads(path.read_text(encoding="utf-8"))
            data = _merge_dicts(data, loaded)
        if not data:
            return cls()
        return cls.model_validate(data)


def _merge_dicts(base: dict[str, object], override: dict[str, object]) -> dict[str, object]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_dicts(merged[key], value)
            continue
        merged[key] = value
    return merged
