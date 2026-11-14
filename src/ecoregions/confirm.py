"""Confirmation data roles (pre-registered, docs/preregistration.md): which records may be used to build and tune a map
(development) and which are sealed for a single confirmatory test.

The mask is drawn from geography only, before any record is read: an area's land is divided into ``block_deg``
squares; ``share`` of the squares with enough land in the area are drawn with a fixed seed as confirmation blocks; a
``buffer_km`` strip around them is excluded from development (records there are used for nothing). Readers of
species records call ``role`` and drop everything that is not "development" while a map is built.
"""
from __future__ import annotations

import json
import logging

import geopandas as gpd
import numpy as np
import shapely

from .config import Paths

log = logging.getLogger(__name__)


def draw_mask(country_geom, block_deg: float, share: float, min_land: float, buffer_km: float, seed: int) -> dict:
    """Confirmation blocks and buffer (lon/lat polygons) for one area; deterministic for a given seed."""
    w, s, e, n = country_geom.bounds
    blocks = []
    for lon in np.arange(np.floor(w / block_deg) * block_deg, e, block_deg):
        for lat in np.arange(np.floor(s / block_deg) * block_deg, n, block_deg):
            b = shapely.box(lon, lat, lon + block_deg, lat + block_deg)
            if b.intersection(country_geom).area / b.area >= min_land:
                blocks.append((round(float(lon), 3), round(float(lat), 3)))
    rng = np.random.default_rng(seed)
    pick = sorted(rng.choice(len(blocks), size=max(1, round(share * len(blocks))), replace=False).tolist())
    conf = [blocks[i] for i in pick]
    union = shapely.unary_union([shapely.box(a, b, a + block_deg, b + block_deg) for a, b in conf])
    # buffer in degrees: km / (111 km x cos(latitude at the area's centre)): at least buffer_km in every direction
    lat0 = np.radians((s + n) / 2)
    buf = union.buffer(buffer_km / (111.0 * np.cos(lat0)))
    # only the area's own records are sealed: records of neighbouring, already-mapped areas were used in development
    inside = country_geom.buffer(0.02)
    return dict(blocks=conf, candidates=len(blocks), block_deg=block_deg, buffer_km=buffer_km, seed=seed,
                confirmation=shapely.to_geojson(union.intersection(inside)),
                buffer=shapely.to_geojson(buf.difference(union).intersection(inside)))


def load_mask(cfg) -> dict | None:
    c = cfg.get("confirmation")
    if not c:
        return None
    f = Paths(cfg).root / c["mask"]
    m = json.loads(f.read_text(encoding="utf-8"))
    m["_conf"] = shapely.from_geojson(m["confirmation"])
    m["_buf"] = shapely.from_geojson(m["buffer"])
    return m


def role(mask: dict | None, lon, lat) -> np.ndarray:
    """'development', 'buffer' or 'confirmation' for each lon/lat (all 'development' without a mask)."""
    lon, lat = np.asarray(lon, float), np.asarray(lat, float)
    out = np.full(lon.shape, "development", dtype=object)
    if mask is None:
        return out
    out[shapely.contains_xy(mask["_buf"], lon, lat)] = "buffer"
    out[shapely.contains_xy(mask["_conf"], lon, lat)] = "confirmation"
    return out


def cell_roles(mask: dict, cells: list) -> np.ndarray:
    """Role of each (lon0, lat0, deg, ...) cell: 'confirmation' if wholly inside the confirmation squares, 'development'
    if it touches neither squares nor buffer, '' if it crosses a role boundary (left out, preregistration section 4)."""
    sealed = mask["_conf"].union(mask["_buf"])
    out = []
    for c in cells:
        b = shapely.box(c[0], c[1], c[0] + c[2], c[1] + c[2])
        out.append("confirmation" if mask["_conf"].contains(b) else "" if b.intersects(sealed) else "development")
    return np.array(out, dtype=object)


def opened(cfg, gpkg) -> bool:
    """True when the confirmation records may be read: ``confirmation.open`` set and the map's hash equal to the frozen
    ``confirmation.map_sha256`` (recorded in the ledger first). Any mismatch stops the run."""
    import hashlib

    c = cfg.get("confirmation") or {}
    if not c.get("open"):
        return False
    sha = hashlib.sha256(gpkg.read_bytes()).hexdigest()
    if sha != c.get("map_sha256"):
        raise SystemExit(f"confirmation: the map's hash {sha} differs from the frozen confirmation.map_sha256; refusing")
    return True


def run(cfg) -> None:
    """Draw and write the mask for ``confirmation.area`` (ISO3 in Natural Earth), refusing to overwrite an existing one."""
    paths = Paths(cfg)
    c = cfg["confirmation"]
    out = paths.root / c["mask"]
    if out.exists():
        raise SystemExit(f"{out} exists: the confirmation mask is drawn once and never redrawn")
    ne = gpd.read_file(paths.raw / "naturalearth" / "ne_10m_admin_0_countries.shp")
    geom = ne[ne.ADM0_A3 == c["area"]].to_crs(4326).union_all()
    m = draw_mask(geom, c["block_deg"], c["share"], c["min_land"], c["buffer_km"], c["seed"])
    m["area"] = c["area"]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(m, indent=1), encoding="utf-8")
    log.info("confirmation mask for %s: %d of %d %g-degree blocks, %g km buffer -> %s", c["area"], len(m["blocks"]),
             m["candidates"], c["block_deg"], c["buffer_km"], out)
