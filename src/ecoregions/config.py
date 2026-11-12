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


DEFAULT_REGION = "pak-afg"


def merge(base: dict, over: dict) -> dict:
    """``over`` on top of ``base``: nested dicts merge key by key; anything else (lists included) replaces."""
    out = dict(base)
    for k, v in over.items():
        out[k] = merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_config(path: str | os.PathLike | None = None, region: str | None = None) -> dict[str, Any]:
    """The shared ``config.yaml`` with one mapped region's settings (``regions/<region>.yaml``) on top.
    The project maps the planet region by region; everything specific to one region lives in its file."""
    path = Path(path) if path else Path(os.environ.get("ECOREGIONS_CONFIG", ROOT / "config.yaml"))
    region = region or os.environ.get("ECOREGIONS_REGION", DEFAULT_REGION)
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    rfile = path.resolve().parent / "regions" / f"{region}.yaml"
    if not rfile.exists():
        raise FileNotFoundError(f"no region file {rfile}")
    with open(rfile, encoding="utf-8") as f:
        cfg = merge(cfg, yaml.safe_load(f) or {})
    cfg["_root"] = str(path.resolve().parent)
    cfg["_region"] = region
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
    def raw_region(self) -> Path:
        """Downloads clipped or tiled to this region (DEM, climate, soil, land cover, population, phenology)."""
        return self._dir("raw_region")

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
