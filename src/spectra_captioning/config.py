"""Configuration loading and management.

Loads configuration from config.yaml with optional CLI overrides.
"""

from pathlib import Path

import yaml


_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "config.yaml"


def load_config(config_path: Path | None = None) -> dict:
    """Load configuration from a YAML file.

    Args:
        config_path: Path to the config file. Defaults to the project-root
            ``config.yaml``.

    Returns:
        Parsed configuration dictionary.
    """
    path = config_path or _DEFAULT_CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with open(path) as f:
        return yaml.safe_load(f)


def apply_overrides(config: dict, overrides: dict) -> dict:
    """Apply CLI overrides on top of the base config.

    ``overrides`` uses dot-separated keys, e.g.
    ``{"captioning.model": "gemini-3.6-flash"}`` sets
    ``config["captioning"]["model"]``.
    """
    for dotted_key, value in overrides.items():
        keys = dotted_key.split(".")
        target = config
        for key in keys[:-1]:
            target = target.setdefault(key, {})
        target[keys[-1]] = value
    return config
