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

    def context_names(self, extra: list[str] | tuple = ()) -> list[str]:
        """Built contexts to use: the core screens plus any of `evaluation.extra_sources`
        that are asked for by name. Extras stay out unless requested, so tables made before
        they existed reproduce unchanged."""
        extras = set(self["evaluation"].get("extra_sources", []))
        if unknown := set(extra) - extras:
            raise ValueError(f"not in evaluation.extra_sources: {sorted(unknown)}")
        proc = self.path("processed")
        built = [n for n in self["contexts"] if (proc / f"{n}_pert.parquet").exists()]
        return [n for n in built if n not in extras] + [n for n in extra if n in built]


def load_config(path: str | Path = REPO_ROOT / "configs" / "default.yaml") -> Config:
    with open(path) as fh:
        return Config(yaml.safe_load(fh))
