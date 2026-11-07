"""Configuration loading and canonical file locations.

Every stage reads the same ``config.yaml``; file names of intermediate products
are defined once here (``Paths``) so stages never hard-code each other's paths.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
log = logging.getLogger("ecoregions")


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
                        datefmt="%H:%M:%S")


def load_config(path: str | os.PathLike | None = None) -> dict[str, Any]:
    path = Path(path) if path else Path(os.environ.get("ECOREGIONS_CONFIG", ROOT / "config.yaml"))
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["_root"] = str(path.resolve().parent)
    return cfg


@dataclass
class Paths:
    """Canonical locations of every raw and derived product."""

    cfg: dict[str, Any]

    @cached_property
    def root(self) -> Path:
        return Path(self.cfg["_root"])

    def _dir(self, key: str) -> Path:
        p = self.root / self.cfg["paths"][key]
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def raw(self) -> Path:
        return self._dir("raw")

    @property
    def interim(self) -> Path:
        return self._dir("interim")

    @property
    def outputs(self) -> Path:
        return self._dir("outputs")

    @property
    def qa(self) -> Path:
        return self._dir("qa")

    def raster(self, name: str) -> Path:
        """Derived raster on the 90 m analysis grid."""
        return self.interim / f"{name}.tif"

    def coarse(self, name: str) -> Path:
        """Derived raster on the coarse (segmentation) grid."""
        return self.interim / "coarse" / f"{name}.tif"

    def vector(self, name: str) -> Path:
        return self.interim / f"{name}.gpkg"

    @property
    def grid_json(self) -> Path:
        return self.interim / "grid.json"

    @property
    def gazetteer(self) -> Path:
        return self.root / self.cfg["naming"]["gazetteer"]
