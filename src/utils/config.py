"""Configuration management."""

import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
import typer 
from pydantic import BaseModel, Field, model_validator
from rich.console import Console
from contextvars import ContextVar
from contextlib import contextmanager

from src.utils.config_schemes import (
    A2AScheme, LLMConfigScheme, 
    BeanstalkdScheme, CopilotConfigScheme, 
    BillingScheme, LocalesConfig, 
    ElasticsearchConfigScheme, MetricsConfig, WebSearchScheme
)

logger = logging.getLogger(__name__)


class Config(BaseModel):
    """Main configuration for step 00."""

    workspace: Path
    default_agent: str
    agents_path: Path = Field(default=Path("agents"))

    llm: LLMConfigScheme
    a2a: A2AScheme
    beanstalkd: BeanstalkdScheme
    copilot: CopilotConfigScheme
    elasticsearch: ElasticsearchConfigScheme
    billing: BillingScheme
    locales: LocalesConfig
    metrics_config: MetricsConfig
    web_search: WebSearchScheme

    @model_validator(mode="after")
    def resolve_paths(self) -> "Config":
        """Resolve relative paths to absolute using workspace."""
        for field_name in (
            "agents_path",
        ):
            path = getattr(self, field_name)
            if not path.is_absolute():
                setattr(self, field_name, self.workspace / path)
        return self

    @classmethod
    def load(cls, workspace_dir: Path) -> "Config":
        """Load configuration from workspace directory."""
        config_data = cls._load_config(workspace_dir)
        config_data["workspace"] = workspace_dir
        return cls.model_validate(config_data)

    @classmethod
    def _load_config(cls, workspace_dir: Path) -> dict[str, Any]:
        """Load config from YAML file."""
        config_file = workspace_dir / "config.user.yaml"
        if not config_file.exists():
            raise FileNotFoundError(f"Config file not found: {config_file}")

        with open(config_file) as f:
            return yaml.safe_load(f) or {}




@lru_cache(maxsize=1)
def get_config() -> Config:
    """Returns the config"""
    console = Console()

    DEFAULT_WORKSPACE = Path(__file__).resolve().parents[2] / "default_workspace"
    config_file = DEFAULT_WORKSPACE.expanduser() / "config.user.yaml"

    if not config_file.exists():
        console.print(f"[yellow]No configuration found at {config_file}[/yellow]")
        raise typer.Exit(1)

    cfg = Config.load(DEFAULT_WORKSPACE)
    return cfg


log_context: ContextVar[dict] = ContextVar("log_context", default={})


@contextmanager
def push_log_context(**fields):
    """Устанавливает значения в `log_context`
    и гарантированно сбрасывает их в finally.
    Использовать только в верхнем уровне обработки задачи"""

    token = log_context.set(fields)
    try:
        yield
    finally:
        log_context.reset(token)

inited_config = get_config()