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
        a = a * s.scales[0] + s.offsets[0]  # CHELSA stores scaled integers (rain x 10, temperature in K x 10)
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


def border_drawn(X: sp.csr_matrix, df: pd.DataFrame, country_of, shifts, ratio=3.0, min_pairs=6) -> np.ndarray:
    """Range maps whose edge follows a national border: for each species and each border (pair of countries), the
    number of range-edge pairs of neighbouring cells (one cell with the species, one without) that cross that border,
    against the same count for copies of the borders shifted ``shifts`` (m). Flagged when, on some border, at least
    ``min_pairs`` edge pairs cross it and at least ``ratio`` times as many as on any shifted copy. Counted border by
    border, so a short stretch of a long range edge is found (2026-11-22: wildcat, kestrel, black francolin, night
    heron and woolly hare one-sided on the Makran coast, missed by the 100 km run test)."""
    A = sp.triu(neighbours(df), 1).tocoo()
    i, j = A.row, A.col
    Xc = X.tocsc().astype(np.int8)
    edge = abs(Xc[i] - Xc[j]).tocsc()  # pairs x species: 1 where exactly one of the two cells has the species

    def per_border(dx, dy):
        c = country_of(dx, dy)
        lo, hi = np.minimum(c[i], c[j]), np.maximum(c[i], c[j])
        cross = (lo != hi) & (lo >= 0)
        key = lo * 1000 + hi
        keys = np.unique(key[cross])
        K = sp.csr_matrix((np.ones(cross.sum()), (np.searchsorted(keys, key[cross]), np.flatnonzero(cross))),
                          shape=(len(keys), len(i)))
        return pd.DataFrame((K @ edge).toarray(), index=keys)

    real = per_border(0, 0)
    sh = [per_border(dx, dy).reindex(real.index, fill_value=0) for dx, dy in shifts]
    shifted = np.maximum.reduce([x.to_numpy() for x in sh])
    hit = (real.to_numpy() >= min_pairs) & (real.to_numpy() >= ratio * np.maximum(shifted, 1))
    return hit.any(0)


def animals(cfg) -> None:
    """Range-map presence on the cells, from the world run of ``global_fauna``; range maps that stop at a border on
    level ground (``joint_faunal.clipped_species``) and the reviewed ``faunal.clipped_known`` are left out."""
    from .boundaries import load_boundaries
    from .joint_faunal import clipped_species

    bc, out, g = setup(cfg)
    df = pd.read_csv(out / "cells.csv")
    src = Paths(cfg).root / cfg["global_fauna"]["out"]
    X = sp.load_npz(src / f"presence_{cfg['faunal']['presence']}.npz").tocsr()[df.cell.to_numpy()]
    spc = pd.read_csv(src / "species.csv")
    have = np.asarray(X.getnnz(0)).ravel() > 0
    X, spc = X[:, have].tocsr(), spc[have].reset_index(drop=True)
    X.data[:] = 1
    countries = load_boundaries(Paths(cfg))[0]
    area = gpd.GeoSeries([study_area(cfg)], crs=4326).buffer(0.5).iloc[0]
    countries = countries[countries.intersects(area)]
    dem = f"/vsizip/{(Paths(cfg).raw / 'gmted2010' / 'mn30_grd.zip').as_posix()}/mn30_grd"
    clipped = clipped_species(g, df, X, spc.name.tolist(), countries, dem) | set(cfg["faunal"].get("clipped_known", {}))
    from .boundaries import display_countries
    cty = display_countries(*load_boundaries(Paths(cfg))).to_crs(CRS)
    cty = cty[cty.intersects(gpd.GeoSeries([area], crs=4326).to_crs(CRS).iloc[0])].reset_index(drop=True)
    xs, ys = df.x.to_numpy(), df.y.to_numpy()

    def country_of(dx, dy):  # country index of each cell centre with the borders shifted by (dx, dy)
        c = np.full(len(df), -1)
        for n_, geom in enumerate(cty.geometry):
            c[shapely.contains_xy(geom, xs - dx, ys - dy)] = n_
        return c

    s = bc.get("border_drawn", {})
    drawn = border_drawn(X, df, country_of, [(0, 30000), (0, -30000), (30000, 0), (-30000, 0)],
                         s.get("ratio", 3.0), s.get("min_pairs", 6))
    spc["cells"] = np.asarray(X.getnnz(0)).ravel()
    spc["border_run"] = spc.name.isin(clipped)
    spc["border_edge"] = drawn
    spc["clipped"] = spc.border_run | spc.border_edge
    spc.to_csv(out / "animal_species.csv", index=False)
    keep = ~spc.clipped.to_numpy()
    sp.save_npz(out / "animals.npz", X[:, keep].tocsr())
    log.info("biotic animals: %d species on the cells (%s); %d left out as range maps that stop at a border",
             int(keep.sum()), ", ".join(f"{c} {n}" for c, n in spc[keep].cls.value_counts().items()), int((~keep).sum()))


def fill_gaps(E: pd.DataFrame, df: pd.DataFrame, rounds: int = 20) -> pd.DataFrame:
    """Missing values (coastal cells beyond a layer's edge) from the mean of their neighbouring cells, outward."""
    A = neighbours(df)
    E = E.copy()
    for v in E.columns:
        x = E[v].to_numpy(float, copy=True)
        for _ in range(rounds):
            miss = np.isnan(x)
            if not miss.any():
                break
            have = (~miss).astype(float)
            s, c = A @ np.where(miss, 0, x), A @ have
            fill = miss & (c > 0)
            x[fill] = s[fill] / c[fill]
        E[v] = np.where(np.isnan(x), np.nanmedian(x), x)
    return E


# ---------------------------------------------------------------- plant turnover model (GDM)
def ramps(x: np.ndarray, knots: np.ndarray) -> np.ndarray:
    """Monotone piecewise-linear basis (linear I-splines): one ramp per knot interval, 0 below it, 1 above."""
    return np.stack([np.clip((x - a) / max(b - a, 1e-9), 0, 1) for a, b in zip(knots[:-1], knots[1:])], -1)


def gdm_fit(F: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Generalized dissimilarity model: d = 1 - exp(-(b0 + F @ b)), all b >= 0, least squares (Ferrier et al. 2007)."""
    from scipy.optimize import minimize

    A = np.hstack([np.ones((len(F), 1)), F])

    def loss(b):
        e = np.exp(-(A @ b))
        r = (1 - e) - d
        return float(r @ r) / len(d), (2 / len(d)) * (A.T @ (r * e))

    b0 = np.full(A.shape[1], 0.05)
    return minimize(loss, b0, jac=True, method="L-BFGS-B", bounds=[(0, None)] * A.shape[1]).x


def gdm_predict(F: np.ndarray, b: np.ndarray) -> np.ndarray:
    return 1 - np.exp(-(b[0] + F @ b[1:]))


def plantmodel(cfg) -> None:
    """Plant turnover from climate and terrain, fitted on pairs of well-recorded cells (each rarefied to the same
    number of records, so sampling effort does not pose as turnover) with geographic distance as a covariate; checked
    on held-out folds; every cell then gets its coordinates in plant-turnover space (``plant_T.npy``)."""
    bc, out, g = setup(cfg)
    pm = bc["plant_model"]
    df = pd.read_csv(out / "cells.csv")
    E = fill_gaps(pd.read_csv(out / "env.csv").set_index("cell").loc[df.cell, pm["vars"]].reset_index(drop=True), df)
    X = sp.load_npz(out / "plants.npz").tocsr()
    rng = np.random.default_rng(pm["seed"])
    n_rec = np.asarray(X.sum(1)).ravel()
    sites = np.flatnonzero(n_rec >= pm["rarefy"])
    rows = []
    for i in sites:  # rarefy each site to the same number of records
        row = X.getrow(i)
        draw = rng.multivariate_hypergeometric(row.data.astype(np.int64), pm["rarefy"])
        rows.append(row.indices[draw > 0])
    P = sp.csr_matrix((np.ones(sum(map(len, rows)), bool), np.concatenate(rows),
                       np.cumsum([0] + [len(r) for r in rows])), shape=(len(sites), X.shape[1]))
    knots = {v: np.quantile(E[v], [0, .25, .5, .75, 1]) for v in pm["vars"]}
    B = np.hstack([ramps(E[v].to_numpy(), knots[v]) for v in pm["vars"]])  # all cells
    xy = df[["x", "y"]].to_numpy() / 1000
    # pairs of sites within pm["max_km"], sampled at random
    n = pm["pairs"]
    a, b = rng.integers(0, len(sites), 4 * n), rng.integers(0, len(sites), 4 * n)
    dist = np.hypot(*(xy[sites[a]] - xy[sites[b]]).T)
    ok = (a != b) & (dist <= pm["max_km"])
    a, b, dist = a[ok][:n], b[ok][:n], dist[ok][:n]
    shared = np.asarray(P[a].multiply(P[b]).sum(1)).ravel()
    na, nb = np.asarray(P[a].sum(1)).ravel(), np.asarray(P[b].sum(1)).ravel()
    sor = 1 - 2 * shared / (na + nb)
    gk = np.array(pm["geo_knots_km"], float)
    F = np.hstack([np.abs(B[sites[a]] - B[sites[b]]), ramps(dist, gk)])
    fold = df.fold.to_numpy()[sites]
    res = []
    for f in range(bc["folds"]["n"]):  # held-out check: pairs whose two cells are both in fold f
        tr = (fold[a] != f) & (fold[b] != f)
        te = (fold[a] == f) & (fold[b] == f)
        coef = gdm_fit(F[tr], sor[tr])
        p = gdm_predict(F[te], coef)
        res.append(dict(fold=f, pairs=int(te.sum()), explained=1 - ((sor[te] - p) ** 2).sum() / ((sor[te] - sor[te].mean()) ** 2).sum()))
    coef = gdm_fit(F, sor)
    fit = 1 - ((sor - gdm_predict(F, coef)) ** 2).sum() / ((sor - sor.mean()) ** 2).sum()
    nv = 4 * len(pm["vars"])
    T = B * coef[1:nv + 1]  # turnover coordinates: city-block distance between two cells = predicted eta (climate, terrain)
    np.save(out / "plant_T.npy", T.astype("float32"))
    imp = pd.DataFrame({"var": pm["vars"], "weight": coef[1:nv + 1].reshape(-1, 4).sum(1)}).sort_values("weight", ascending=False)
    imp.to_csv(out / "plant_model_weights.csv", index=False)
    pd.DataFrame(res).to_csv(out / "plant_model_cv.csv", index=False)
    log.info("biotic plant model: %d sites of %d+ records, %d pairs; turnover explained %.3f (held-out folds %s); "
             "geographic distance weight %.3f; strongest: %s", len(sites), pm["rarefy"], len(sor), fit,
             ", ".join(f"{r['explained']:.3f}" for r in res), coef[nv + 1:].sum(),
             ", ".join(f"{r.var} {r.weight:.2f}" for r in imp.head(4).itertuples()))


# ---------------------------------------------------------------- regions
def neighbours(df: pd.DataFrame) -> sp.csr_matrix:
    """Rook adjacency of the cells on the grid."""
    pos = pd.Series(np.arange(len(df)), index=pd.MultiIndex.from_arrays([df.row, df.col]))
    i, j = [], []
    for dr, dc in ((0, 1), (1, 0)):
        nb = pd.MultiIndex.from_arrays([df.row + dr, df.col + dc])
        hit = nb.isin(pos.index)
        i.append(np.flatnonzero(hit))
        j.append(pos.loc[nb[hit]].to_numpy())
    i, j = np.concatenate(i), np.concatenate(j)
    return sp.coo_matrix((np.ones(2 * len(i)), (np.r_[i, j], np.r_[j, i])), shape=(len(df), len(df))).tocsr()


def block(Z: np.ndarray, weight: float) -> np.ndarray:
    """Centre a block of coordinates and scale it to a total variance of ``weight``."""
    Z = Z - Z.mean(0)
    return Z * np.sqrt(weight / max(Z.var(0).sum(), 1e-12))


def animal_coords(X: sp.csr_matrix, n_comp: int, seed: int) -> np.ndarray:
    """Hellinger-transformed presence, reduced by truncated SVD (cells x n_comp)."""
    from sklearn.decomposition import TruncatedSVD

    rs = np.asarray(X.sum(1)).ravel()
    H = sp.diags(1 / np.sqrt(np.maximum(rs, 1))) @ X.astype("float32")
    return TruncatedSVD(n_comp, random_state=seed).fit_transform(H)


def regions(cfg) -> None:
    """Ecoregions grown on the neighbour graph: Ward agglomeration of plants and animals (equal weight; birds, mammals,
    and reptiles with amphibians equal within animals); the whole tree is kept, cut at the counts in ``regions.k``."""
    from sklearn.cluster import AgglomerativeClustering

    bc, out, g = setup(cfg)
    rc = bc["regions"]
    df = pd.read_csv(out / "cells.csv")
    T = np.load(out / "plant_T.npy")
    X = sp.load_npz(out / "animals.npz").tocsr()
    spc = pd.read_csv(out / "animal_species.csv")
    spc = spc[~spc.clipped].reset_index(drop=True)
    groups = {"birds": ["AVES"], "mammals": ["MAMMALIA"], "reptiles and amphibians": ["REPTILIA", "AMPHIBIA"]}
    parts = [block(T, rc["weights"]["plants"])]
    for name, cls in groups.items():
        cols = np.flatnonzero(spc.cls.str.upper().isin(cls).to_numpy())
        parts.append(block(animal_coords(X[:, cols], rc["animal_components"], rc["seed"]), rc["weights"]["animals"] / len(groups)))
    Z = np.hstack(parts)
    A = neighbours(df)
    model = AgglomerativeClustering(n_clusters=None, distance_threshold=0, linkage="ward", connectivity=A,
                                    compute_full_tree=True).fit(Z)
    np.save(out / "tree_children.npy", model.children_)
    np.save(out / "tree_distances.npy", model.distances_)
    lab = pd.DataFrame({"cell": df.cell})
    for k in rc["k"]:
        lab[f"k{k}"] = cut(model.children_, len(df), k)
    lab.to_csv(out / "regions.csv", index=False)
    log.info("biotic regions: tree over %d cells (%d plant, %d animal coordinates); cut at %s", len(df), T.shape[1],
             Z.shape[1] - T.shape[1], rc["k"])


def cut(children: np.ndarray, n: int, k: int) -> np.ndarray:
    """Labels 1..k of the n leaves when the agglomeration tree is cut into k clusters."""
    parent = np.arange(2 * n - 1)
    for m, (a, b) in enumerate(children[: n - k]):
        parent[a] = parent[b] = n + m
    def root(i):
        while parent[i] != i:
            i = parent[i]
        return i
    roots = np.array([root(i) for i in range(n)])
    return pd.factorize(roots)[0] + 1


def rarefied(X: sp.csr_matrix, n: int, seed: int) -> tuple[np.ndarray, sp.csr_matrix]:
    """Cells with at least n records, each rarefied once to n records: (cell index, presence matrix)."""
    rng = np.random.default_rng(seed)
    cells_ = np.flatnonzero(np.asarray(X.sum(1)).ravel() >= n)
    rows = []
    for i in cells_:
        r = X.getrow(i)
        rows.append(r.indices[rng.multivariate_hypergeometric(r.data.astype(np.int64), n) > 0])
    P = sp.csr_matrix((np.ones(sum(map(len, rows)), "float32"), np.concatenate(rows),
                       np.cumsum([0] + [len(r) for r in rows])), shape=(len(cells_), X.shape[1]))
    return cells_, P


def brier(Y: np.ndarray, p: np.ndarray) -> float:
    return float(((Y - p) ** 2).mean())


def detail(cfg) -> None:
    """Number of ecoregions and which splits stand (preregistration section 10, Level of detail), plants on held-out
    folds: (1) the largest count worth testing is the tree cut whose regions best predict held-out cells' species
    lists (Brier loss, region frequencies from the other folds shrunk toward the overall frequency with 20
    pseudo-cells); (2) below it, from the top, a split stands when in at least 4 of 5 folds the held-out cells across
    it differ more than those within it (Jaccard, pairs 10-50 km apart, matched by 10 km bins) with at least 10 pairs
    across, and prediction is no more than 1 % worse; a split with too few pairs is kept and marked provisional."""
    bc, out, g = setup(cfg)
    dc = bc["detail"]
    df = pd.read_csv(out / "cells.csv")
    n = len(df)
    children = np.load(out / "tree_children.npy")
    X = sp.load_npz(out / "plants.npz").tocsr()
    idx, P = rarefied(X, dc["rarefy"], dc["seed"])
    keep_sp = np.asarray((P > 0).sum(0)).ravel() >= dc["min_cells_per_species"]
    Y = P[:, keep_sp].toarray()
    fold = df.fold.to_numpy()[idx]
    overall = Y.mean(0)

    def loss_at(labels, f, sub=None):
        tr, te = fold != f, fold == f
        if sub is not None:
            tr, te = tr & sub, te & sub
        lab = labels[idx]
        pred = np.empty((te.sum(), Y.shape[1]), "float32")
        for r in np.unique(lab[te]):
            m = tr & (lab == r)
            freq = (Y[m].sum(0) + 20 * overall) / (m.sum() + 20)
            pred[lab[te] == r] = freq
        return brier(Y[te], pred)

    # (1) the largest count worth testing
    scan = []
    for k in dc["k_ladder"]:
        labels = cut(children, n, k)
        scan.append(dict(k=k, **{f"fold{f}": loss_at(labels, f) for f in range(bc["folds"]["n"])}))
    scan = pd.DataFrame(scan)
    scan["mean"] = scan.filter(like="fold").mean(1)
    k_max = int(scan.loc[scan["mean"].idxmin(), "k"])
    scan.to_csv(out / "detail_scan.csv", index=False)
    log.info("biotic detail: held-out prediction best at %d regions (Brier %.5f; %d regions %.5f)", k_max,
             scan["mean"].min(), scan.k.iloc[0], scan["mean"].iloc[0])

    # (2) splits from the top, within the tree cut at k_max
    xy = df[["x", "y"]].to_numpy()[idx] / 1000
    from scipy.spatial import cKDTree
    pairs = cKDTree(xy).query_pairs(50.0, output_type="ndarray")
    d = np.hypot(*(xy[pairs[:, 0]] - xy[pairs[:, 1]]).T)
    pairs, d = pairs[d >= 10], d[d >= 10]
    inter = np.asarray(P[pairs[:, 0]].multiply(P[pairs[:, 1]]).sum(1)).ravel()
    na, nb = np.asarray(P[pairs[:, 0]].sum(1)).ravel(), np.asarray(P[pairs[:, 1]].sum(1)).ravel()
    jac = 1 - inter / (na + nb - inter)
    dbin = np.minimum((d // 10).astype(int), 4)
    pf = np.where(fold[pairs[:, 0]] == fold[pairs[:, 1]], fold[pairs[:, 0]], -1)  # pairs inside one held-out fold
    cell_pos = np.full(n, -1)
    cell_pos[idx] = np.arange(len(idx))

    # members of every tree node (leaves 0..n-1, merges n..2n-2)
    left, right = children[:, 0], children[:, 1]
    top = 2 * n - 2

    def leaves(node):
        out_, stack = [], [node]
        while stack:
            v = stack.pop()
            if v < n:
                out_.append(v)
            else:
                stack += [left[v - n], right[v - n]]
        return np.array(out_)

    stop = {2 * n - 2 - j for j in range(k_max - 1)}  # the merges undone by the cut at k_max
    # pruned from the leaves up (as a classification tree is pruned): a split is tested once both its sides are whole
    # regions, and merged when it fails; a split with structure kept below it stands. Top-down, one failed split high
    # in the tree discarded every split beneath it (2026-11-22: all of Anatolia and the Caucasus became one region)
    status_of, results = {}, []
    whole = lambda v: v < n or v not in stop or status_of.get(v) == "merged"  # noqa: E731
    for v in sorted(stop):  # children before parents
        a, b = left[v - n], right[v - n]
        if not (whole(a) and whole(b)):
            status_of[v] = "kept (structure below)"
            results.append(dict(node=v, cells_a=len(leaves(a)), cells_b=len(leaves(b)), folds_tested=-1, folds_for=-1,
                                status=status_of[v]))
            continue
        A, B = leaves(a), leaves(b)
        side = np.zeros(n, np.int8)
        side[A], side[B] = 1, 2
        s0, s1 = side[idx][pairs[:, 0]], side[idx][pairs[:, 1]]
        inside = (s0 > 0) & (s1 > 0)
        across = inside & (s0 != s1)
        within = inside & (s0 == s1)
        votes, tested = 0, 0
        for f in range(bc["folds"]["n"]):
            ac, wi = across & (pf == f), within & (pf == f)
            if ac.sum() < dc["min_pairs"]:
                continue
            diffs = [jac[ac & (dbin == k)].mean() - jac[wi & (dbin == k)].mean() for k in range(1, 5)
                     if (ac & (dbin == k)).any() and (wi & (dbin == k)).any()]
            if not diffs:
                continue
            tested += 1
            sub = side[idx] > 0  # held-out cells of this region only
            ok_loss = True
            if (sub & (fold == f)).sum() and (sub & (fold != f)).sum():
                ok_loss = loss_at(side, f, sub) <= loss_at((side > 0).astype(np.int8), f, sub) * 1.01
            votes += int(np.mean(diffs) > 0 and ok_loss)
        if tested < 4:
            status = "provisional"
        elif votes >= 4:
            status = "kept"
        else:
            status = "merged"
        status_of[v] = status
        results.append(dict(node=v, cells_a=len(A), cells_b=len(B), folds_tested=tested, folds_for=votes, status=status))
    regions_out, prov, queue = [], [], [top]
    while queue:  # the regions: nodes whose own split was merged, or not undone by the cut at k_max
        v = queue.pop()
        if v < n or v not in stop or status_of[v] == "merged":
            regions_out.append(v)
            continue
        queue += [left[v - n], right[v - n]]
        if status_of[v] == "provisional":
            prov += [left[v - n], right[v - n]]
    lab = np.zeros(n, "int32")
    is_prov = np.zeros(n, bool)
    for r, v in enumerate(sorted(regions_out, key=lambda v: -len(leaves(v))), start=1):
        lab[leaves(v)] = r
    for v in prov:
        if v in regions_out:
            is_prov[leaves(v)] = True
    res = pd.DataFrame(results)
    res.to_csv(out / "detail_splits.csv", index=False)
    labs = pd.read_csv(out / "regions.csv")
    labs["final"] = lab
    labs["provisional"] = is_prov
    labs.to_csv(out / "regions.csv", index=False)
    log.info("biotic detail: %d splits examined below %d regions: %d kept, %d provisional (too few held-out pairs), %d "
             "merged; %d ecoregions (%d provisional)", len(res), k_max, res.status.str.startswith("kept").sum(),
             (res.status == "provisional").sum(), (res.status == "merged").sum(), lab.max(),
             len(np.unique(lab[is_prov])))


def polygons(df: pd.DataFrame, labels: np.ndarray, g: Grid) -> gpd.GeoDataFrame:
    """Regions as polygons in EPSG:6933, from the cells' labels."""
    from rasterio.features import shapes

    c0, c1, r0, r1 = df.col.min(), df.col.max() + 1, df.row.min(), df.row.max() + 1
    img = np.zeros((r1 - r0, c1 - c0), "int32")
    img[df.row - r0, df.col - c0] = labels
    tr = rasterio.transform.Affine(g.res, 0, g.x0 + c0 * g.res, 0, -g.res, g.y0 - r0 * g.res)
    geo = [(shapely.geometry.shape(s), int(v)) for s, v in shapes(img, mask=img > 0, transform=tr)]
    p = gpd.GeoDataFrame({"region": [v for _, v in geo]}, geometry=[s for s, _ in geo], crs=CRS)
    return p.dissolve("region").reset_index()


def gates(cfg, k: int | None = None) -> pd.DataFrame:
    """The registered border gates (preregistration section 10) on the regions cut at k: (1) share of region boundary
    within 10 km of a border inside the mapped land, against copies of those borders shifted 30 km north, south, east
    and west; (2) no boundary on a border: a run of 10 km or more within 2.5 km whose median distance is at most
    max(0.3 km, half a cell)."""
    from shapely import affinity
    from .bordercheck import coincidence, internal_boundaries, international_borders
    from .boundaries import display_countries, load_boundaries
    from .world import on_line

    bc, out, g = setup(cfg)
    df = pd.read_csv(out / "cells.csv")
    lab = pd.read_csv(out / "regions.csv")
    countries, disputed = load_boundaries(Paths(cfg))
    cty = display_countries(countries, disputed).to_crs(CRS)
    land = gpd.GeoSeries([study_area(cfg)], crs=4326).to_crs(CRS).iloc[0]
    cty = cty[cty.intersects(land)]
    borders = international_borders(cty).intersection(land.buffer(-1000))
    rows = []
    for kk in ([k] if k else bc["regions"]["k"]):
        reg = polygons(df, lab[f"k{kk}"].to_numpy(), g)
        bound = internal_boundaries(reg)
        obs = coincidence(bound, borders, 10000)
        shifted = [coincidence(bound, affinity.translate(borders, dx, dy), 10000)
                   for dx, dy in ((0, 30000), (0, -30000), (30000, 0), (-30000, 0))]
        lines = on_line(reg.rename(columns={"region": "type"}), cty, "type", on_km=max(0.3, bc["cell_m"] / 2000))
        rows.append(dict(k=kk, boundary_km=round(bound.length / 1000), share_near_borders=round(obs, 4),
                         shifted_max=round(max(shifted), 4), share_pass=obs <= max(shifted), on_line=len(lines),
                         on_line_km=sum(r["km"] for r in lines),
                         where="; ".join(f"{r['countries']} {r['km']} km at {r['lat']} N {r['lon']} E" for r in lines)))
        log.info("biotic gates k=%d: %.1f %% of %d km of boundary within 10 km of a border (shifted copies up to %.1f %%): %s; "
                 "%d runs on a border (%d km)", kk, 100 * obs, bound.length / 1000, 100 * max(shifted),
                 "pass" if obs <= max(shifted) else "FAIL", len(lines), rows[-1]["on_line_km"])
    res = pd.DataFrame(rows)
    res.to_csv(out / "gates.csv", index=False)
    return res


def render(cfg, k: int | None = None) -> None:
    """Quick look of the regions at k: cells coloured by region, borders in thin grey (outputs in ``biotic.out``)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .boundaries import load_boundaries

    bc, out, g = setup(cfg)
    df = pd.read_csv(out / "cells.csv")
    lab = pd.read_csv(out / "regions.csv")
    ks = [k] if k else bc["regions"]["k"]
    countries = load_boundaries(Paths(cfg))[0].to_crs(CRS)
    for kk in ks:
        c0, c1, r0, r1 = df.col.min(), df.col.max() + 1, df.row.min(), df.row.max() + 1
        img = np.full((r1 - r0, c1 - c0), np.nan)
        rng = np.random.default_rng(1)
        perm = rng.permutation(kk) + 1
        img[df.row - r0, df.col - c0] = perm[lab[f"k{kk}"].to_numpy() - 1]
        ext = (g.x0 + c0 * g.res, g.x0 + c1 * g.res, g.y0 - r1 * g.res, g.y0 - r0 * g.res)
        fig, ax = plt.subplots(figsize=(18, 8))
        ax.imshow(img, extent=ext, cmap="tab20", interpolation="nearest")
        countries.boundary.plot(ax=ax, color="#444", linewidth=0.4)
        ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3]); ax.set_axis_off()
        ax.set_title(f"Ecoregions, first draft: {kk} regions from plants and animals (10 km)")
        plt.tight_layout(); plt.savefig(out / f"regions_k{kk}.png", dpi=90); plt.close(fig)
        log.info("biotic render: %s", out / f"regions_k{kk}.png")


if __name__ == "__main__":
    from .config import setup_logging

    setup_logging()
    step = sys.argv[1]
    steps = {"cells": cells, "env": env, "plants": plants, "animals": animals, "plantmodel": plantmodel,
             "regions": regions, "render": render, "gates": gates, "detail": detail}
    steps[step](load_config())
