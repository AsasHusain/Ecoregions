"""Global faunal presence: every land vertebrate range on one equal-area grid, for the global Faunal Ecoregions.

The regional ``fauna`` stage reads an area's ranges at once; the whole world (IUCN and BirdLife, ~28 GB) does not fit
in memory, so ranges are indexed once by their attributes and read one species at a time by feature id. Species,
codes and filters are those of ``fauna`` (terrestrial, not marine-dependent; ``fauna.presence/origin/seasonal``).

Grid: EASE-Grid 2.0 global (EPSG:6933, equal-area), ``global_fauna.cell_m`` cells with a nested ``fine_m`` grid for
range cover and elevation (USGS GMTED2010 mean elevation, 30 arc-seconds, averaged to ``fine_m``). Presence as in
``fauna``: cover >= ``fauna.min_share`` of a cell (a species that never reaches it keeps its best cells), for the
ranges as drawn and within the species' elevation limits.

Outputs in ``global_fauna.out``: ``presence_{range,elev}.npz`` (cells x species), ``species.csv``, ``cells.csv``,
``grid.json``.
"""
from __future__ import annotations

import logging
import os
import time

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio
import scipy.sparse as sp
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.transform import Affine

from .config import Paths
from .fauna import species_traits
from .grid import Grid, block_reduce, reproject_file

log = logging.getLogger(__name__)
CRS = "EPSG:6933"
WORLD = (-17367530.45, -7314540.83, 17367530.45, 7314540.83)  # EPSG:6933 extent


def range_index(path, layer: str, key: str, ids: set, fc: dict) -> pd.DataFrame:
    """Feature ids of the range polygons with the configured presence/origin/seasonal codes, by species id."""
    a = pyogrio.read_dataframe(path, layer=layer, columns=[key, "presence", "origin", "seasonal"], read_geometry=False,
                               fid_as_index=True)
    a = a[a.presence.isin(fc["presence"]) & a.origin.isin(fc["origin"]) & a.seasonal.isin(fc["seasonal"])]
    a = a.assign(id=a[key].astype(int), fid=a.index.astype(int))
    return a[a.id.isin(ids)][["id", "fid"]].assign(path=str(path), layer=layer)


def run(cfg) -> None:
    paths, fc, gc = Paths(cfg), cfg["fauna"], cfg["global_fauna"]
    out = paths.root / gc["out"]
    out.mkdir(parents=True, exist_ok=True)
    cell = Grid.from_bounds(CRS, WORLD, gc["cell_m"])
    k = gc["cell_m"] // gc["fine_m"]
    fine = Grid(cell.crs, cell.x0, cell.y0, gc["fine_m"], cell.width * k, cell.height * k)
    cell.save(out / "grid.json")
    dem = reproject_file(f"/vsizip/{paths.raw / 'gmted2010' / 'mn30_grd.zip'}/mn30_grd", fine, Resampling.average)
    ne = gpd.read_file(paths.raw / "naturalearth" / "ne_10m_admin_0_countries.shp").to_crs(CRS)
    on_land = rasterize(((g, 1) for g in ne.geometry), out_shape=fine.shape, transform=fine.transform, dtype="uint8") > 0
    dem[~on_land] = np.nan  # GMTED gives the oceans 0 m, not no-data
    land = block_reduce(on_land.astype("float32"), k)  # share of each cell on land (Natural Earth)
    log.info("global fauna: %d x %d cells of %g km, %d at least half land", cell.width, cell.height, gc["cell_m"] / 1000,
             int((land >= 0.5).sum()))

    traits = species_traits(paths.raw / "iucn" / "redlist_assessments_2026-11-06.zip", fc["classes"])
    ids = set(traits.index)
    srcs = [(p, "exported", "id_no") for p in sorted((paths.raw / "iucn" / "ranges").glob("*/spatial_data.gpkg"))]
    srcs.append((paths.raw / "birdlife" / "BOTW_2025.gpkg", "all_species", "sisid"))
    idx = pd.concat([range_index(p, lay, key, ids, fc) for p, lay, key in srcs])
    log.info("global fauna: %d range polygons for %d species", len(idx), idx.id.nunique())

    cols, rows, t0 = {"range": [], "elev": []}, [], time.time()
    limit = int(os.environ.get("GLOBAL_FAUNA_LIMIT", 0))  # smoke tests: the first N species only
    for n, (sid, g) in enumerate(idx.groupby("id")):
        if limit and n >= limit:
            break
        geoms = pd.concat([pyogrio.read_dataframe(p, layer=lay, fids=f.fid.tolist(), columns=[])
                           for (p, lay), f in g.groupby(["path", "layer"])])
        geoms = gpd.GeoSeries(geoms.geometry, crs=4326).to_crs(CRS).simplify(gc["fine_m"] / 2)
        geoms = geoms[~geoms.is_empty]
        if geoms.empty:
            continue
        # window on the fine grid, snapped to whole cells
        xmin, ymin, xmax, ymax = geoms.total_bounds
        c0 = max(0, int((xmin - cell.x0) // gc["cell_m"]))
        c1 = min(cell.width, int((xmax - cell.x0) // gc["cell_m"]) + 1)
        r0 = max(0, int((cell.y0 - ymax) // gc["cell_m"]))
        r1 = min(cell.height, int((cell.y0 - ymin) // gc["cell_m"]) + 1)
        tr = Affine(gc["fine_m"], 0, cell.x0 + c0 * gc["cell_m"], 0, -gc["fine_m"], cell.y0 - r0 * gc["cell_m"])
        r = rasterize(((geom, 1) for geom in geoms), out_shape=((r1 - r0) * k, (c1 - c0) * k), transform=tr,
                      dtype="uint8").astype(bool)
        if not r.any():
            continue
        t = traits.loc[sid]
        d = dem[r0 * k:r1 * k, c0 * k:c1 * k]
        e = r & ~np.isnan(d)
        if not np.isnan(t.elev_lo):
            e &= d >= t.elev_lo
        if not np.isnan(t.elev_hi):
            e &= d <= t.elev_hi
        gidx = (np.arange(r0, r1)[:, None] * cell.width + np.arange(c0, c1)[None, :]).ravel()
        for kk, m in (("range", r), ("elev", e)):
            cov = block_reduce(m.astype("float32"), k).ravel()
            pres = cov >= fc["min_share"]
            if not pres.any() and cov.max() > 0:  # small-range species: keep its best cells
                pres = cov >= cov.max() * 0.999
            cols[kk].append(gidx[pres])
        rows.append(dict(id=sid, name=t.scientificName, cls=t.className, range_cells=len(cols["range"][-1]),
                         elev_cells=len(cols["elev"][-1])))
        if n % 1000 == 0:
            log.info("global fauna: %d species, %.0f min", n, (time.time() - t0) / 60)

    n_cells = cell.width * cell.height
    for kk, cl in cols.items():
        m = sp.csc_matrix((np.ones(sum(map(len, cl)), bool),
                           (np.concatenate(cl), np.repeat(np.arange(len(cl)), [len(c) for c in cl]))),
                          shape=(n_cells, len(cl)), dtype=bool)
        sp.save_npz(out / f"presence_{kk}.npz", m)
    pd.DataFrame(rows).to_csv(out / "species.csv", index=False)
    pd.DataFrame({"cell": np.arange(n_cells), "land_share": land.ravel()}).to_csv(out / "cells.csv", index=False)
    log.info("global fauna: %d species written to %s", len(rows), out)
