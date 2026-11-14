"""Stage 0 — input checks that take seconds, run before a build, so a bad input stops it in minutes, not hours.

* Province seeds: each province's seeds must form one group when seeds within ``link_km`` of each other are joined
  (a cluster appended to the wrong province, as the Iranian Makran and Sistan seeds once were to the Ghazni
  Plateau, forms a second group far away).
* Elevation tiles: every Copernicus tile that exists inside the bounding box is on disk (127 were once missing).

The coastline check needs the DEM and runs in the ``dem`` stage (land the DEM adds beyond Natural Earth).
"""
from __future__ import annotations

import logging

import numpy as np
from scipy.sparse.csgraph import connected_components

from .config import Paths

log = logging.getLogger(__name__)


def km_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Great-circle distances (km) between every lon/lat row of ``a`` and of ``b``."""
    a, b = np.radians(a)[:, None, :], np.radians(b)[None, :, :]
    h = np.sin((b[..., 1] - a[..., 1]) / 2) ** 2 + np.cos(a[..., 1]) * np.cos(b[..., 1]) * np.sin((b[..., 0] - a[..., 0]) / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(h))


def seed_groups(seeds, link_km: float) -> int:
    """Number of groups among one province's seeds, joining seeds within ``link_km``."""
    s = np.asarray(seeds, float)
    return connected_components(km_between(s, s) <= link_km, directed=False)[0]


def check_seeds(units: list, link_km: float) -> list[str]:
    return [f"province {u['code']} {u['name']}: its {len(u['seeds'])} seeds form {n} groups more than {link_km:.0f} km apart"
            for u in units if (n := seed_groups(u["seeds"], link_km)) > 1]


def check_tiles(cfg, paths: Paths) -> list[str]:
    from .download import dem_tile_names

    have = {p.stem for p in (paths.raw_region / "copernicus_dem").glob("*.tif")}
    missing = sorted(set(dem_tile_names(cfg)) - have)
    return [f"{len(missing)} Copernicus tiles missing from {paths.raw_region / 'copernicus_dem'}, e.g. {missing[:3]}"] if missing else []


def run(cfg) -> None:
    paths = Paths(cfg)
    problems = check_seeds(cfg["provinces"]["units"], cfg["provinces"].get("seed_link_km", 400)) + check_tiles(cfg, paths)
    for p in problems:
        log.error("preflight: %s", p)
    if problems:
        raise SystemExit(f"preflight: {len(problems)} problem(s); fix the inputs before building")
    log.info("preflight: province seeds grouped; all elevation tiles present")


if __name__ == "__main__":  # self-check: one tight province passes, a second far cluster fails
    near = [[61.0, 31.0], [61.5, 31.2], [62.0, 31.4]]
    assert seed_groups(near, 400) == 1
    assert seed_groups(near + [[58.4, 26.4], [59.0, 27.3]], 400) == 2
    print("ok")
