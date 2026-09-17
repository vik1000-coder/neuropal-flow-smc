"""Config loading and hashing.

Configs are plain YAML dicts wrapped in a small dot-accessible mapping so that
``cfg.model.ridge`` and ``cfg["model"]["ridge"]`` both work.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


class Config(dict):
    """A dict that also supports attribute access, recursively."""

    def __getattr__(self, key: str) -> Any:
        try:
            val = self[key]
        except KeyError as exc:  # pragma: no cover - defensive
            raise AttributeError(key) from exc
        return _wrap(val)

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    def get_path(self, dotted: str, default: Any = None) -> Any:
        """Look up a nested key by dotted path, e.g. ``cfg.get_path('model.ridge')``."""
        node: Any = self
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return _wrap(node)


def _wrap(val: Any) -> Any:
    if isinstance(val, dict) and not isinstance(val, Config):
        return Config(val)
    return val


def load_config(path: str | Path) -> Config:
    """Load a YAML config file into a :class:`Config`."""
    path = Path(path)
    with open(path, "r") as fh:
        data = yaml.safe_load(fh) or {}
    return Config(data)


def config_hash(cfg: dict) -> str:
    """Deterministic short hash of a config dict (order-independent)."""
    payload = json.dumps(cfg, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]
