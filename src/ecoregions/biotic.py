"""Ecoregions: one 10 km grid over everything mapped, built in one run (docs/plan.md; tests registered in
docs/preregistration.md, section 10).

Cells are those of the world grid of ``global_fauna`` (EASE-Grid 2.0, EPSG:6933), so animal presence, computed once
for the whole world, needs no resampling, and land added later joins the same grid. Steps, each writing to
``biotic.out``:

* ``cells``: grid cells with at least ``min_land`` of their area in the mapped countries; each cell's one-degree
  block and cross-validation fold (blocks drawn at random into ``folds`` folds, registered seed).
* ``env``: climate (CHELSA v2.1, read over HTTP for the window only), elevation and relief (GMTED2010) per cell.
* ``plants``: wild, native plant records per cell and species, development records only: records inside any area's
  sealed squares or their buffer, and records made on or after ``temporal_cut``, are never read into the counts.

    PYTHONPATH=src python -m ecoregions.biotic cells|env|plants
"""
from __future__ import annotations

import logging
import sys

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import scipy.sparse as sp
import shapely
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.warp import reproject, transform_bounds
from rasterio.windows import from_bounds

from .config import Paths, load_config
from .grid import Grid

log = logging.getLogger(__name__)
CRS = "EPSG:6933"
WORLD = (-17367530.45, -7314540.83, 17367530.45, 7314540.83)  # EPSG:6933 extent, as global_fauna


def setup(cfg):
    bc = cfg["biotic"]
    out = Paths(cfg).root / bc["out"]
    out.mkdir(parents=True, exist_ok=True)
    return bc, out, Grid.from_bounds(CRS, WORLD, bc["cell_m"])


def study_area(cfg) -> shapely.Geometry:
    """The mapped countries of every area in ``biotic.areas``, as each area maps them (lon/lat)."""
    from .boundaries import load_boundaries, mapped_area

    cfgs = [load_config(region=r) for r in cfg["biotic"]["areas"]]
    countries, disputed = load_boundaries(Paths(cfgs[0]))
    return shapely.union_all([mapped_area(c, Paths(c), countries, disputed).union_all() for c in cfgs])


def cells(cfg) -> pd.DataFrame:
    bc, out, g = setup(cfg)
    area = gpd.GeoSeries([study_area(cfg)], crs=4326).to_crs(CRS).iloc[0]
    xmin, ymin, xmax, ymax = area.bounds
    c0, c1 = int((xmin - g.x0) // g.res), int((xmax - g.x0) // g.res) + 1
    r0, r1 = int((g.y0 - ymax) // g.res), int((g.y0 - ymin) // g.res) + 1
    k = 10  # land share from a nested 1 km grid
    fine = rasterio.transform.Affine(g.res / k, 0, g.x0 + c0 * g.res, 0, -g.res / k, g.y0 - r0 * g.res)
    m = rasterize([(area, 1)], out_shape=((r1 - r0) * k, (c1 - c0) * k), transform=fine, dtype="uint8")
    share = m.reshape(r1 - r0, k, c1 - c0, k).mean((1, 3))
    rr, cc = np.nonzero(share >= bc["min_land"])
    rows, cols = rr + r0, cc + c0
    x, y = g.x0 + (cols + 0.5) * g.res, g.y0 - (rows + 0.5) * g.res
    lon, lat = gpd.points_from_xy(x, y, crs=CRS).to_crs(4326).x, gpd.points_from_xy(x, y, crs=CRS).to_crs(4326).y
    df = pd.DataFrame(dict(cell=rows * g.width + cols, row=rows, col=cols, x=x, y=y, lon=np.round(lon, 4),
                           lat=np.round(lat, 4), land=np.round(share[rr, cc], 3)))
    # cross-validation folds by one-degree block (registered: preregistration section 10)
    f = bc["folds"]
    df["block"] = (np.floor(df.lon / f["block_deg"]).astype(int).astype(str) + "_"
                   + np.floor(df.lat / f["block_deg"]).astype(int).astype(str))
    blocks = np.sort(df.block.unique())
    fold = np.random.default_rng(f["seed"]).integers(0, f["n"], len(blocks))
    df["fold"] = df.block.map(dict(zip(blocks, fold)))
    df.to_csv(out / "cells.csv", index=False)
    log.info("biotic cells: %d cells of %g km over %s (%d one-degree blocks in %d folds)", len(df), bc["cell_m"] / 1000,
             ", ".join(bc["areas"]), len(blocks), f["n"])
    return df


def _cell_mean(src_path: str, df: pd.DataFrame, g: Grid, scale=1.0, offset=0.0) -> np.ndarray:
    """Average of a raster over each cell (read only over the cells' window; nodata ignored)."""
    c0, c1, r0, r1 = df.col.min(), df.col.max() + 1, df.row.min(), df.row.max() + 1
    dst_tr = rasterio.transform.Affine(g.res, 0, g.x0 + c0 * g.res, 0, -g.res, g.y0 - r0 * g.res)
    dst = np.full((r1 - r0, c1 - c0), np.nan, "float32")
    with rasterio.open(src_path) as s:
        b = transform_bounds(CRS, s.crs, g.x0 + c0 * g.res, g.y0 - r1 * g.res, g.x0 + c1 * g.res, g.y0 - r0 * g.res)
        w = from_bounds(*b, s.transform).round_offsets().round_lengths()
        a = s.read(1, window=w, boundless=True, masked=True).astype("float32").filled(np.nan)
        reproject(a, dst, src_transform=s.window_transform(w), src_crs=s.crs, src_nodata=np.nan, dst_transform=dst_tr,
                  dst_crs=CRS, dst_nodata=np.nan, resampling=Resampling.average)
    return dst[df.row - r0, df.col - c0] * scale + offset


def env(cfg) -> pd.DataFrame:
    bc, out, g = setup(cfg)
    df = pd.read_csv(out / "cells.csv")
    ch = cfg["sources"]["chelsa"]
    res = df[["cell"]].copy()
    for var in bc["env"]["chelsa"]:
        url = "/vsicurl/" + ch["url"].format(var=var)
        v = _cell_mean(url, df, g)
        if var == "ai" and np.nanmedian(v) > 100:  # stored x 10000 in some releases
            v = v / 10000
        res[var] = v
        log.info("biotic env: %s (median %.3g)", var, np.nanmedian(v))
    months = ch["summer_months"]
    pr = {m: _cell_mean("/vsicurl/" + ch["monthly_url"].format(month=m), df, g) for m in range(1, 13)}
    total = sum(pr.values())
    res["summer_rain"] = sum(pr[m] for m in months) / np.where(total > 0, total, np.nan)
    dem = f"/vsizip/{Paths(cfg).raw / 'gmted2010' / 'mn30_grd.zip'}/mn30_grd"
    res["elev"] = _cell_mean(dem, df, g)
    # relief: spread of 1 km elevation inside each cell
    c0, c1, r0, r1 = df.col.min(), df.col.max() + 1, df.row.min(), df.row.max() + 1
    k = 10
    fine_tr = rasterio.transform.Affine(g.res / k, 0, g.x0 + c0 * g.res, 0, -g.res / k, g.y0 - r0 * g.res)
    fine = np.full(((r1 - r0) * k, (c1 - c0) * k), np.nan, "float32")
    with rasterio.open(dem) as s:
        reproject(rasterio.band(s, 1), fine, dst_transform=fine_tr, dst_crs=CRS, dst_nodata=np.nan,
                  resampling=Resampling.average)
    blk = fine.reshape(r1 - r0, k, c1 - c0, k)
    res["relief"] = (np.nanpercentile(blk, 90, axis=(1, 3)) - np.nanpercentile(blk, 10, axis=(1, 3)))[df.row - r0, df.col - c0]
    res.to_csv(out / "env.csv", index=False)
    log.info("biotic env: %d cells, %d variables, %d cells with a gap", len(res), res.shape[1] - 1,
             int(res.drop(columns="cell").isna().any(axis=1).sum()))
    return res


def plants(cfg) -> None:
    from .confirm import load_mask, role
    from .native import screen

    bc, out, g = setup(cfg)
    df = pd.read_csv(out / "cells.csv")
    keys = list(dict.fromkeys(d["key"] for r in bc["areas"] for d in load_config(region=r)["sources"]["gbif_dwca"]["downloads"]))
    cols = ["gbifID", "kingdom", "species", "basisOfRecord", "cultivated", "eventDate", "year", "decimalLatitude",
            "decimalLongitude", "coordinateUncertaintyInMeters"]
    raw = Paths(cfg).raw / "gbif_dwca"
    recs = []
    for key in keys:
        for ch in pd.read_csv(raw / f"occurrences_{key}.csv.gz", dtype=str, usecols=cols, chunksize=2_000_000):
            ch = ch[(ch.kingdom == "Plantae") & (ch.cultivated != "True") & (ch.basisOfRecord != "FOSSIL_SPECIMEN")]
            unc = pd.to_numeric(ch.coordinateUncertaintyInMeters, errors="coerce")
            recs.append(ch[~(unc > cfg["vegcheck"]["max_uncertainty_m"])].dropna(subset=["species", "decimalLatitude", "decimalLongitude"]))
        log.info("biotic plants: read %s", key)
    recs = pd.concat(recs).drop_duplicates("gbifID")
    n0 = len(recs)
    lon, lat = recs.decimalLongitude.astype(float).values, recs.decimalLatitude.astype(float).values
    dev = np.ones(len(recs), bool)
    for r in bc["areas"]:  # development only if every area's mask says so (preregistration section 10)
        dev &= role(load_mask(load_config(region=r)), lon, lat) == "development"
    late = recs.eventDate.fillna("").str[:10] >= bc["temporal_cut"]
    recs = recs[dev & ~late.values]
    log.info("biotic plants: %d wild plant records; %d kept for development (%d in sealed squares or buffers, %d dated "
             "%s or later)", n0, len(recs), int((~dev).sum()), int(late.sum()), bc["temporal_cut"])
    recs = screen(cfg, recs[["species", "decimalLatitude", "decimalLongitude"]])  # native status, as in every area
    pts =gpd.GeoSeries(gpd.points_from_xy(recs.decimalLongitude.astype(float), recs.decimalLatitude.astype(float)), crs=4326).to_crs(CRS)
    cell = ((g.y0 - pts.y.values) // g.res).astype(int) * g.width + ((pts.x.values - g.x0) // g.res).astype(int)
    idx = pd.Series(np.arange(len(df)), index=df.cell.values)
    keep = np.isin(cell, df.cell.values)
    sp_names = np.sort(recs.species[keep].unique())
    col = pd.Series(np.arange(len(sp_names)), index=sp_names)
    m = sp.coo_matrix((np.ones(int(keep.sum()), "int32"), (idx[cell[keep]].values, col[recs.species[keep]].values)),
                      shape=(len(df), len(sp_names))).tocsr()
    sp.save_npz(out / "plants.npz", m)
    pd.Series(sp_names, name="species").to_csv(out / "plant_species.csv", index=False)
    n_rec = np.asarray(m.sum(1)).ravel()
    log.info("biotic plants: %d native records in %d species on %d of %d cells (median %d records per sampled cell)",
             int(m.sum()), len(sp_names), int((n_rec > 0).sum()), len(df), int(np.median(n_rec[n_rec > 0])))


if __name__ == "__main__":
    from .config import setup_logging

    setup_logging()
    step = sys.argv[1]
    {"cells": cells, "env": env, "plants": plants}[step](load_config())
