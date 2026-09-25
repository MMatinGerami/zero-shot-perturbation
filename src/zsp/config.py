"""Configuration loading."""

from __future__ import annotations

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


class Config(dict):
    @property
    def seed(self) -> int:
        return int(self["seed"])

    def path(self, key: str) -> Path:
        p = REPO_ROOT / self["paths"][key]
        p.mkdir(parents=True, exist_ok=True)
        return p


def load_config(path: str | Path = REPO_ROOT / "configs" / "default.yaml") -> Config:
    with open(path) as fh:
        return Config(yaml.safe_load(fh))
