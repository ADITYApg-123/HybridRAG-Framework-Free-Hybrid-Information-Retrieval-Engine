"""Configuration loader and logging initialization for HybridRAG."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


def load_config(config_path: str | Path | None = None) -> Dict[str, Any]:
    """Load YAML configuration and resolve project-relative paths."""
    target_path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    if not target_path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {target_path}")

    with target_path.open("r", encoding="utf-8") as f:
        config: Dict[str, Any] = yaml.safe_load(f)

    return config


def resolve_project_path(relative_or_abs: str | Path) -> Path:
    """Resolve a path relative to the repository root if not already absolute."""
    path = Path(relative_or_abs)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def setup_logging(level: str = "INFO") -> logging.Logger:
    """Configure standardized logging across HybridRAG modules."""
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger("hybrid_rag")
