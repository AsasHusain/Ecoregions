"""Stage ``fitcheck``: does each vegetation type's outline follow the landscape evidence for it?

Each type in ``fitcheck.families`` is matched to an evidence layer it should stand out on (``ls_bare`` for sand
deserts, ``ls_bright`` for salt crusts and ice, ``ls_green`` for closed forests, low ``hand`` for floodplain forests).
On a 1 km grid, on natural ground only (WorldCover cropland and built-up excluded), the type is compared with a ring
``fitcheck.ring_km`` wide around it. A cell shows the evidence when it lies beyond the midpoint of the type's and the
ring's medians (the midpoint is set on the published map, so both maps are judged by one threshold):

- precision: share of the type's cells that show the evidence;
- recall: share of the evidence cells in the type and its ring that lie inside the type;
- fit: their harmonic mean.

The published map of the area (``fitcheck.published``) is scored the same way, and the stage fails when a type's fit
falls more than ``fitcheck.max_drop`` below it: a rebuild may not draw a landscape worse than the map it replaces.
Writes ``outputs/<area>/fitcheck.csv``. A quality gate, not independent validation: where a layer is also a model
input, agreement with it is expected; the species-record tests are the validation.
"""
from __future__ import annotations

import logging

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.features import rasterize
from scipy import ndimage as ndi

from .config import Paths
from .grid import Grid

log = logging.getLogger(__name__)
WORLDCOVER_CONVERTED = (40, 50)  # cropland, built-up


def layer(path, grid: Grid, how=Resampling.average) -> np.ndarray:
    with rasterio.open(path) as s:
        a = s.read(1, out_shape=grid.shape, resampling=how).astype("float32")
        if s.nodata is not None and not np.isnan(s.nodata):
            a[a == s.nodata] = np.nan
    return a


def type_raster(gpkg, grid: Grid, types: list) -> np.ndarray:
    g = gpd.read_file(gpkg, layer="ecoregions").to_crs(grid.rio_crs)
    code = {t: i + 1 for i, t in enumerate(types)}
    g = g[g["type"].isin(code)]
    return rasterize([(geom, code[t]) for geom, t in zip(g.geometry, g["type"])], out_shape=grid.shape,
                     transform=grid.transform, fill=0, dtype="uint8")


def score(lab: np.ndarray, k: int, ev: np.ndarray, natural: np.ndarray, ring_cells: int, sign: float, mid=None):
    inside = lab == k
    if not inside.any():
        return None
    ring = ndi.binary_dilation(inside, iterations=ring_cells) & ~inside
    i, r = ev[inside & natural], ev[ring & natural]
    i, r = i[np.isfinite(i)], r[np.isfinite(r)]
    if i.size < 20 or r.size < 20:
        return None
    if mid is None:
        mid = (np.median(i) + np.median(r)) / 2
    pos_i, pos_r = sign * (i - mid) > 0, sign * (r - mid) > 0
    precision = pos_i.mean()
    recall = pos_i.sum() / max(pos_i.sum() + pos_r.sum(), 1)
    fit = 2 * precision * recall / max(precision + recall, 1e-9)
    return dict(inside_median=round(float(np.median(i)), 3), ring_median=round(float(np.median(r)), 3),
                precision=round(float(precision), 3), recall=round(float(recall), 3), fit=round(float(fit), 3)), mid


def run(cfg) -> None:
    paths = Paths(cfg)
    fc = {"ring_km": 10, "max_drop": 0.05, "cell_m": 1080, **cfg.get("fitcheck", {})}
    g90 = Grid.load(paths.grid_json)
    grid = g90.coarsen(int(round(fc["cell_m"] / g90.res)))
    ring_cells = max(1, round(fc["ring_km"] * 1000 / grid.res))
    wc = layer(paths.raster("worldcover"), grid, Resampling.mode)
    natural = ~np.isin(wc, WORLDCOVER_CONVERTED)
    fams = fc.get("families", {})
    types = sorted({t for ts in fams.values() for t in ts["types"]})
    current = type_raster(paths.outputs / "ecoregions.gpkg", grid, types)
    pub_path = fc.get("published", {}).get(cfg["_region"])
    published = type_raster(f"/vsizip/{paths.root / pub_path}/ecoregions.gpkg", grid, types) if pub_path else None
    rows = []
    for name, fam in fams.items():
        ev = layer(paths.coarse(fam["layer"]), grid)
        sign = -1.0 if fam.get("lower") else 1.0
        for t in fam["types"]:
            k = types.index(t) + 1
            ref = score(published, k, ev, natural, ring_cells, sign) if published is not None else None
            mid = ref[1] if ref else None
            cur = score(current, k, ev, natural, ring_cells, sign, mid)
            if cur is None:
                continue
            row = dict(type=t, family=name, layer=fam["layer"], **cur[0])
            if ref:
                row.update(published_fit=ref[0]["fit"], change=round(cur[0]["fit"] - ref[0]["fit"], 3))
            rows.append(row)
            log.info("fitcheck %-24s %-9s fit %.2f (precision %.2f, recall %.2f)%s", t, fam["layer"], cur[0]["fit"],
                     cur[0]["precision"], cur[0]["recall"], f"; published {ref[0]['fit']:.2f}" if ref else "")
    df = pd.DataFrame(rows)
    df.to_csv(paths.outputs / "fitcheck.csv", index=False)
    lost = coverage(cfg, paths, fc)
    accept = {a["type"]: a["reason"] for a in fc.get("accept", [])}  # reviewed exceptions, as in bordercheck.accept
    dropped = df.type[df.change < -fc["max_drop"]] if "change" in df else df.type.iloc[0:0]
    for t in sorted(set(lost.type[lost.type.isin(list(accept))]) | set(dropped[dropped.isin(list(accept))])):
        log.info("fitcheck: %s accepted: %s", t, accept[t])
    lost = lost[~lost.type.isin(list(accept))]
    worse = df[(df.change < -fc["max_drop"]) & ~df.type.isin(list(accept))] if "change" in df else df.iloc[0:0]
    msgs = [f"{r.type} fit {r.fit:.2f} (published {r.published_fit:.2f})" for r in worse.itertuples()]
    msgs += [f"{r.type}: {r.mapped_km2:,.0f} km² mapped of {r.pixel_km2:,.0f} km² of its communities" for r in lost.itertuples()]
    if msgs:
        raise SystemExit("fitcheck: " + "; ".join(msgs))


def coverage(cfg, paths: Paths, fc: dict) -> pd.DataFrame:
    """Types whose communities cover at least ``min_type_km2`` of 90 m pixels but which keep less than
    ``min_kept`` of that area as mapped units: narrow belts (a north-slope forest in a meadow unit) outvoted when
    units take their majority type. Writes ``outputs/<area>/type_coverage.csv``; returns the lost types."""
    from .communities import load_rules

    rules = load_rules(paths.root / cfg["communities"]["rules"])
    eco = [c["ecoregion"] for c in rules["communities"]]
    from .boundaries import load_boundaries, mapped_area

    px = np.zeros(len(eco) + 1, np.int64)
    with rasterio.open(paths.raster("communities")) as s:
        own = mapped_area(cfg, paths, *load_boundaries(paths)).to_crs(s.crs).geometry  # a parcel's strip is not its own
        for _, w in s.block_windows(1):
            inside = rasterize([(g, 1) for g in own], out_shape=(w.height, w.width), transform=s.window_transform(w),
                               fill=0).astype(bool)
            if inside.any():
                px += np.bincount(s.read(1, window=w)[inside], minlength=px.size)[: px.size]
        cell_km2 = abs(s.transform.a * s.transform.e) / 1e6
    pix = pd.Series(px[1:] * cell_km2, index=eco).groupby(level=0).sum()
    mapped = gpd.read_file(paths.outputs / "ecoregions.gpkg", layer="ecoregions", columns=["type", "area_km2"],
                           ignore_geometry=True).groupby("type").area_km2.sum()
    cov = pd.DataFrame({"pixel_km2": pix.round(0), "mapped_km2": mapped.reindex(pix.index).fillna(0).round(0)})
    cov["kept"] = (cov.mapped_km2 / cov.pixel_km2.clip(lower=1)).round(3)
    cov.rename_axis("type").reset_index().to_csv(paths.outputs / "type_coverage.csv", index=False)
    lost = cov[(cov.pixel_km2 >= fc.get("min_type_km2", 500)) & (cov.kept < fc.get("min_kept", 0.1))]
    for t, r in lost.iterrows():
        log.warning("fitcheck: %s keeps %.0f of %.0f km² of its communities as mapped units", t, r.mapped_km2, r.pixel_km2)
    return lost.rename_axis("type").reset_index()
