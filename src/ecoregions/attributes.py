"""Stage 10 — polygons, zonal statistics, names, descriptions and confidence.

Polygons are vectorized from the 90 m label raster. Their outlines are
simplified *only* to remove pixel stair-steps, with a tolerance of half a cell
(45 m) via ``shapely.coverage_simplify``, which keeps shared borders identical
(no gaps or overlaps). No smoothing beyond what the 90 m data support.

Confidence score (0-1), per unit:

* ``conf_boundary`` — mean boundary-strength along the unit's border, as a
  percentile among all units (do the lines sit on real edges?)
* ``conf_homogeneity`` — 1 - within-unit / within-province feature variance
* ``conf_l2_margin`` — how unambiguously the area belongs to its L2 province
* ``conf_named`` — 1 if a gazetteer feature anchors the unit

``confidence = 0.35 b + 0.35 h + 0.15 m + 0.15 n``
"""
from __future__ import annotations

import json
import logging

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import shapely
import yaml
from pyproj import Transformer
from rasterio.features import shapes
from scipy import ndimage as ndi

from . import vectors
from .config import Paths
from .covariates import WORLDCOVER, litho_groups
from .features import dune_mask
from .grid import Grid, read
from .hydro import HAND_CLASS_CODES
from .terrain import HAMMOND_CLASSES

log = logging.getLogger(__name__)

COMPASS = ["east", "northeast", "north", "northwest", "west", "southwest", "south", "southeast"]
POSITION = {"east": "eastern", "northeast": "northeastern", "north": "northern", "northwest": "northwestern",
            "west": "western", "southwest": "southwestern", "south": "southern", "southeast": "southeastern"}


# ---------------------------------------------------------------------------
# vectorization
# ---------------------------------------------------------------------------
def polygonize(lab: np.ndarray, grid: Grid) -> gpd.GeoDataFrame:
    recs = []
    for r0 in range(0, grid.height, 4096):  # strips keep memory bounded
        r1 = min(grid.height, r0 + 4096)
        sub = lab[r0:r1]
        if not (sub > 0).any():
            continue
        tr = grid.transform * grid.transform.translation(0, r0)
        for geom, v in shapes(sub.astype("int32"), mask=sub > 0, transform=tr, connectivity=4):
            recs.append((int(v), shapely.geometry.shape(geom)))
    gdf = gpd.GeoDataFrame(recs, columns=["unit", "geometry"], crs=grid.crs)
    gdf = gdf.dissolve("unit").reset_index()  # rejoin strips
    gdf["geometry"] = shapely.coverage_simplify(gdf.geometry.values, tolerance=grid.res / 2, simplify_boundary=True)
    gdf["geometry"] = gdf.geometry.make_valid()
    return gdf


# ---------------------------------------------------------------------------
# zonal statistics
# ---------------------------------------------------------------------------
def zonal_mean(lab: np.ndarray, arr: np.ndarray, n: int) -> np.ndarray:
    v = ~np.isnan(arr) & (lab > 0)
    s = np.bincount(lab[v], weights=arr[v], minlength=n + 1)
    c = np.bincount(lab[v], minlength=n + 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return s / c


def zonal_shares(lab: np.ndarray, cls: np.ndarray, n: int, codes) -> pd.DataFrame:
    v = lab > 0
    tot = np.bincount(lab[v], minlength=n + 1).astype(float)
    out = {}
    for k in codes:
        out[k] = np.bincount(lab[v & (cls == k)], minlength=n + 1) / np.maximum(tot, 1)
    return pd.DataFrame(out)


def zonal_percentiles(lab: np.ndarray, arr: np.ndarray, qs, step: int = 3) -> pd.DataFrame:
    l = lab[::step, ::step].ravel()
    a = arr[::step, ::step].ravel()
    m = (l > 0) & ~np.isnan(a)
    return pd.Series(a[m]).groupby(l[m]).quantile(qs).unstack()


# ---------------------------------------------------------------------------
# language helpers
# ---------------------------------------------------------------------------
def texture_class(clay: float, sand: float) -> str:
    """USDA texture class from clay and sand (%)."""
    if np.isnan(clay) or np.isnan(sand):
        return "unknown texture"
    silt = 100 - clay - sand
    if clay >= 40:
        return "clay" if silt < 40 and sand <= 45 else ("silty clay" if silt >= 40 else "sandy clay")
    if clay >= 27:
        return "clay loam" if sand <= 45 and sand > 20 else ("silty clay loam" if sand <= 20 else "sandy clay loam")
    if clay >= 20 and sand > 45:
        return "sandy clay loam"
    if silt >= 80 and clay < 12:
        return "silt"
    if silt >= 50:
        return "silt loam"
    if sand >= 85 and silt + 1.5 * clay < 15:
        return "sand"
    if sand >= 70 and silt + 2 * clay < 30:
        return "loamy sand"
    if sand >= 52 or (clay < 7 and silt < 50 and sand > 43):
        return "sandy loam"
    return "loam"


def aridity_class(ai: float) -> str:
    """UNEP aridity classes on P/PET."""
    if np.isnan(ai):
        return "unknown"
    for lim, name in ((0.05, "hyper-arid"), (0.2, "arid"), (0.5, "semi-arid"), (0.65, "dry sub-humid")):
        if ai < lim:
            return name
    return "humid"


def landform_phrase(r: pd.Series) -> str:
    if r.plains_share > 0.5:
        hc = {k: r[f"hand_{k}"] for k in HAND_CLASS_CODES}
        if r.dune_share > 0.3:
            return "dune field"
        if hc["active_floodplain"] > 0.45:
            return "active river floodplain (bela/kacha)"
        if hc["meander_floodplain"] > 0.35 and hc["meander_floodplain"] >= hc["bar_upland"]:
            return "older meander and cover floodplain"
        if hc["bar_upland"] > 0.4:
            return "alluvial bar upland (old interfluve surface)"
        if r.slope_mean < 1.0:
            return "flat alluvial plain"
        return "gently sloping plain"
    return HAMMOND_CLASSES.get(int(r.hammond_dominant), "terrain")


def compass(dx: float, dy: float) -> str:
    ang = (np.degrees(np.arctan2(dy, dx)) + 360 + 22.5) % 360
    return COMPASS[int(ang // 45)]


def describe(r: pd.Series, geology: str = "USGS geo8ag") -> str:
    parts = []
    lf = landform_phrase(r)
    parts.append(f"{lf[0].upper()}{lf[1:]} in the {r.l2_name}.")
    parts.append(f"Elevation {r.elev_p10:,.0f}–{r.elev_p90:,.0f} m (median {r.elev_median:,.0f} m); "
                 f"mean slope {r.slope_mean:.1f}°, typical local relief {r.relief5k_mean:,.0f} m within 5 km.")
    if r.plains_share > 0.5:
        hc = [(HAND_CLASS_CODES[k], k.replace("_", " "), r[f"hand_{k}"]) for k in HAND_CLASS_CODES]
        mix = ", ".join(f"{n} {s * 100:.0f}%" for _, n, s in hc if s >= 0.05)
        parts.append(f"Height above the nearest major river: median {r.hand_median:.1f} m ({mix}).")
    parts.append(f"Substrate: mostly {r.litho_dominant.lower()} ({r.litho_share * 100:.0f}%, {geology} unit {r.geo_unit_dominant}).")
    parts.append(f"Topsoil (SoilGrids, 0–30 cm): {r.soil_texture}, pH {r.soil_ph:.1f}, organic carbon {r.soil_soc:.0f} g/kg.")
    seas = "strongly seasonal" if r.bio15 > 80 else ("seasonal" if r.bio15 > 50 else "fairly even")
    parts.append(f"Climate (CHELSA 1981–2010): {r.aridity_class}, mean annual temperature {r.bio1:.1f} °C, "
                 f"precipitation {r.bio12:,.0f} mm/yr ({seas}).")
    lc = [(WORLDCOVER[k], r[f"wc_{k}"]) for k in WORLDCOVER if r[f"wc_{k}"] >= 0.1]
    lc.sort(key=lambda t: -t[1])
    if lc:
        parts.append("Land cover today (ESA WorldCover 2021, context only): " +
                     ", ".join(f"{n} {s * 100:.0f}%" for n, s in lc[:3]) + ".")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# naming
# ---------------------------------------------------------------------------
def assign_names(df: pd.DataFrame, lab: np.ndarray, grid: Grid, gaz: dict) -> pd.DataFrame:
    t = Transformer.from_crs("EPSG:4326", grid.crs, always_xy=True)
    inv = ~grid.transform

    def unit_at(lon, lat):
        x, y = t.transform(lon, lat)
        c, r = inv * (x, y)
        r, c = int(r), int(c)
        if 0 <= r < grid.height and 0 <= c < grid.width:
            return int(lab[r, c])
        return 0

    from .crosscheck import expectation

    hits = []  # (unit, feature name, priority, n points)
    unresolved = {}  # unit -> features present but failing their character test
    feat_units = {}
    stats = df.set_index("unit")
    for f in gaz["features"]:
        us = [unit_at(lon, lat) for lon, lat in f["points"]]
        feat_units[f["name"]] = sorted({u for u in us if u})
        for u in set(us):
            if not u:
                continue
            if f.get("expect") and not expectation(stats.loc[u], f["expect"])[0]:
                unresolved.setdefault(u, []).append(f["name"])
                continue
            hits.append((u, f["name"], f.get("priority", 5), us.count(u)))
    h = pd.DataFrame(hits, columns=["unit", "feature", "priority", "npts"])
    df = df.copy()
    df["name"] = None
    df["name_basis"] = "provisional"
    df["also_contains"] = df.unit.map(lambda u: "; ".join(f"{n} (not resolved at this scale)" for n in unresolved.get(u, [])))
    for u, g in h.groupby("unit"):
        g = g.sort_values(["priority", "npts"], ascending=[True, False])
        top = g[g.priority == g.priority.iloc[0]]
        name = " and ".join(top.feature.tolist()[:2])
        df.loc[df.unit == u, "name"] = name
        df.loc[df.unit == u, "name_basis"] = "gazetteer"
        rest = [x for x in g.feature.tolist() if x not in top.feature.tolist()[:2]]
        prev = df.loc[df.unit == u, "also_contains"].iloc[0]
        df.loc[df.unit == u, "also_contains"] = "; ".join([x for x in rest + [prev] if x])
    # features spanning several units -> positional suffixes
    for name, grp in df[df.name_basis == "gazetteer"].groupby("name"):
        if len(grp) < 2:
            continue
        cx, cy = grp.cx.mean(), grp.cy.mean()
        pos = {i: POSITION[compass(r.cx - cx, r.cy - cy)] for i, r in grp.iterrows()}
        counts = pd.Series(pos).value_counts()
        seen: dict[str, int] = {}
        for i in grp.sort_values("cx").index:  # west to east within a shared direction
            p = pos[i]
            seen[p] = seen.get(p, 0) + 1
            suffix = p if counts[p] == 1 else f"{p} {seen[p]}"
            df.loc[i, "name"] = f"{name} ({suffix})"
    # provisional names: landform + direction from the nearest town
    places = pd.DataFrame(gaz["places"], columns=["place", "lon", "lat"])
    px, py = t.transform(places.lon.values, places.lat.values)
    for i, r in df[df.name_basis == "provisional"].iterrows():
        d = np.hypot(px - r.cx, py - r.cy)
        j = int(np.argmin(d))
        lf = landform_phrase(r)
        where = "near" if d[j] < 15000 else compass(r.cx - px[j], r.cy - py[j]) + " of"
        df.loc[i, "name"] = f"{lf[0].upper()}{lf[1:]} {where} {places.place[j]}"
    # guarantee uniqueness
    dup = df.name.duplicated(keep=False)
    for name, grp in df[dup].groupby("name"):
        for k, i in enumerate(grp.index):
            df.loc[i, "name"] = f"{name} ({k + 1})"
    df["review_flag"] = df.name_basis != "gazetteer"
    return df, feat_units


# ---------------------------------------------------------------------------
def run(cfg) -> None:
    paths = Paths(cfg)
    grid = Grid.load(paths.grid_json)
    lab = read(paths.raster("l3")).astype(np.int32)
    units = pd.read_csv(paths.interim / "l3_units.csv")
    n = int(lab.max())
    codes = json.loads((paths.interim / "province" / "l2_codes.json").read_text())
    l2name = {v["code"]: v["name"] for v in codes.values()}
    l2fam = {v["code"]: v["family"] for v in codes.values()}

    log.info("vectorizing %d units", len(units))
    poly = polygonize(lab, grid)
    df = units.copy()
    df["l2_code"] = df.l2_code.astype(str).str.zfill(2)
    df["l2_name"] = df.l2_code.map(l2name)
    df["l2_family"] = df.l2_code.map(l2fam)
    cnt = np.bincount(lab.ravel(), minlength=n + 1)
    df["area_km2"] = cnt[df.unit] * grid.cell_km2
    cen = ndi.center_of_mass(np.ones_like(lab[::3, ::3]), lab[::3, ::3], df.unit.values)
    xy = [grid.transform * (c * 3 + 1.5, r * 3 + 1.5) for r, c in cen]
    df["cx"], df["cy"] = [p[0] for p in xy], [p[1] for p in xy]

    def add_mean(col, raster, transform=None):
        a = read(paths.raster(raster)).astype("float32")
        if transform:
            a = transform(a)
        df[col] = zonal_mean(lab, a, n)[df.unit]
        return a

    dem = add_mean("elev_mean", "dem")
    q = zonal_percentiles(lab, dem, [0.0, 0.1, 0.5, 0.9, 1.0])
    for qq, name in zip([0.0, 0.1, 0.5, 0.9, 1.0], ["elev_min", "elev_p10", "elev_median", "elev_p90", "elev_max"]):
        df[name] = df.unit.map(q[qq])
    del dem
    s = add_mean("slope_mean", "slope")
    df["slope_median"] = df.unit.map(zonal_percentiles(lab, s, [0.5])[0.5])
    del s
    add_mean("relief5k_mean", "relief_5000")
    h = read(paths.raster("hand"))
    df["hand_median"] = df.unit.map(zonal_percentiles(lab, h, [0.5])[0.5])
    del h
    hc = zonal_shares(lab, read(paths.raster("hand_class")), n, HAND_CLASS_CODES.values())
    for k, code in HAND_CLASS_CODES.items():
        df[f"hand_{k}"] = hc[code].values[df.unit]
    pl = zonal_shares(lab, read(paths.raster("plains")), n, [1])
    df["plains_share"] = pl[1].values[df.unit]
    lit = read(paths.raster("geo_litho"))
    dune = dune_mask(cfg, read(paths.raster("slope")), read(paths.raster("tpi_450")),
                     read(paths.raster("plains")).astype(bool), lit).astype("uint8")
    df["dune_share"] = zonal_shares(lab, dune, n, [1])[1].values[df.unit]
    del dune
    hm = zonal_shares(lab, read(paths.raster("hammond")), n, HAMMOND_CLASSES.keys())
    df["hammond_dominant"] = hm.values[df.unit].argmax(1) + 1
    df["landform_hammond"] = df.hammond_dominant.map(HAMMOND_CLASSES)
    groups = litho_groups(cfg)
    ls = zonal_shares(lab, lit, n, groups.keys())
    df["litho_dominant"] = [groups[k] for k in ls.columns[ls.values[df.unit].argmax(1)]]
    df["litho_share"] = ls.values[df.unit].max(1)
    del lit
    with rasterio.open(paths.raster("geo_unit")) as src:
        gu = src.read(1)
        gtags = src.tags()
    gs = zonal_shares(lab, gu, n, sorted(int(k) for k in gtags if k.isdigit()))
    df["geo_unit_dominant"] = [gtags[str(k)] for k in gs.columns[gs.values[df.unit].argmax(1)]]
    del gu
    add_mean("geo_age_rank", "geo_age")
    for p, col in (("soil_clay", "soil_clay"), ("soil_sand", "soil_sand"), ("soil_phh2o", "soil_ph"), ("soil_soc", "soil_soc")):
        add_mean(col, p)
    df["soil_texture"] = [texture_class(c, s) for c, s in zip(df.soil_clay, df.soil_sand)]
    for v in ("bio1", "bio4", "bio12", "bio15", "gdd5", "ai"):
        add_mean(v, f"clim_{v}")
    df["aridity_class"] = df.ai.map(aridity_class)
    wc = zonal_shares(lab, read(paths.raster("worldcover")), n, WORLDCOVER.keys())
    for k in WORLDCOVER:
        df[f"wc_{k}"] = wc[k].values[df.unit]

    # L1 parent (majority RESOLVE ecoregion) and how nested the unit is
    res = read(paths.raster("resolve"))
    r_ids = np.unique(res[(lab > 0) & (res > 0)])
    rs = zonal_shares(lab, res, n, r_ids)
    df["l1_eco_id"] = r_ids[rs.values[df.unit].argmax(1)]
    df["l1_share"] = rs.values[df.unit].max(1)
    del res
    rv = vectors.resolve(cfg).drop_duplicates("ECO_ID").set_index("ECO_ID")
    df["l1_name"] = df.l1_eco_id.map(rv.ECO_NAME)

    # confidence components
    B = read(paths.raster("boundary_strength"))
    edge = np.zeros(lab.shape, bool)
    d = (lab[:, 1:] != lab[:, :-1]) & (lab[:, 1:] > 0) & (lab[:, :-1] > 0)
    edge[:, 1:] |= d
    edge[:, :-1] |= d
    d = (lab[1:, :] != lab[:-1, :]) & (lab[1:, :] > 0) & (lab[:-1, :] > 0)
    edge[1:, :] |= d
    edge[:-1, :] |= d
    bmean = np.bincount(lab[edge], weights=B[edge], minlength=n + 1) / np.maximum(np.bincount(lab[edge], minlength=n + 1), 1)
    del B, edge
    df["boundary_strength_mean"] = bmean[df.unit]
    df["conf_boundary"] = df.boundary_strength_mean.rank(pct=True)
    df["conf_homogeneity"] = homogeneity(paths, cfg, grid, df)
    m = read(paths.interim / "province" / "l2_margin.tif")
    f = cfg["grid"]["coarse_factor"] * cfg["grid"]["province_factor"]
    m90 = np.repeat(np.repeat(m, f, 0), f, 1)[: grid.height, : grid.width]
    df["conf_l2_margin"] = pd.Series(zonal_mean(lab, m90, n)[df.unit]).rank(pct=True).values
    del m90

    df.to_csv(paths.interim / "l3_stats.csv", index=False)
    poly.to_file(paths.vector("l3_polygons_raw"), layer="l3")
    finalize(cfg, lab)


def finalize(cfg, lab: np.ndarray | None = None) -> None:
    """Names, confidence, descriptions and codes (fast; rerun after gazetteer edits)."""
    paths = Paths(cfg)
    grid = Grid.load(paths.grid_json)
    if lab is None:
        lab = read(paths.raster("l3")).astype(np.int32)
    df = pd.read_csv(paths.interim / "l3_stats.csv", dtype={"l2_code": str})
    for c in ("conf_boundary", "conf_homogeneity", "conf_l2_margin"):
        df[c] = df[c].fillna(df[c].median())  # tiny units can lack a sample on the coarse grids
    gaz = paths.load_gazetteer()
    df, feat_units = assign_names(df, lab, grid, gaz)
    df["conf_named"] = (df.name_basis == "gazetteer").astype(float)
    df["confidence"] = (0.35 * df.conf_boundary + 0.35 * df.conf_homogeneity + 0.15 * df.conf_l2_margin
                        + 0.15 * df.conf_named).round(3)
    df["description"] = df.apply(describe, axis=1, geology=cfg["sources"]["geology"]["label"])

    # codes: L2 code + sequential number west->east within L2 (stable, readable)
    df = df.sort_values(["l2_code", "cx"]).reset_index(drop=True)
    df["seq"] = df.groupby("l2_code").cumcount() + 1
    df["code"] = df.l2_code + "." + df.seq.astype(str).str.zfill(2)
    df.to_csv(paths.interim / "l3_attributes.csv", index=False)
    (paths.interim / "gazetteer_hits.json").write_text(json.dumps(feat_units, indent=1))

    poly = gpd.read_file(paths.vector("l3_polygons_raw"), layer="l3")
    gdf = poly.merge(df, on="unit", how="inner")
    out = paths.vector("l3_polygons")
    if out.exists():
        out.unlink()
    gdf.to_file(out, layer="l3")
    log.info("names: %d units (%d named from the gazetteer, %d provisional)",
             len(df), (df.name_basis == "gazetteer").sum(), (df.name_basis != "gazetteer").sum())


def homogeneity(paths: Paths, cfg, grid: Grid, df: pd.DataFrame) -> np.ndarray:
    """1 - mean within-unit variance / within-province variance of the
    standardized default-weight features (coarse grid)."""
    f = cfg["grid"]["coarse_factor"]
    lab = read(paths.raster("l3"))[1::f, 1::f].astype(np.int64)  # centre pixel of each coarse cell
    l2 = read(paths.coarse("l2"))
    hh, ww = min(lab.shape[0], l2.shape[0]), min(lab.shape[1], l2.shape[1])
    lab, l2 = lab[:hh, :ww], l2[:hh, :ww]
    names = ["elev", "slope", "relief5k", "rem", "bio12", "bio1", "clay", "sand", "age"]
    n = int(lab.max())
    within = np.zeros(n + 1)
    total = np.zeros(n + 1)
    cnt = np.bincount(lab.ravel(), minlength=n + 1).astype(float)
    for name in names:
        a = read(paths.interim / "coarse" / f"{name}.tif")[:hh, :ww]
        a = np.where(np.isnan(a), np.nanmean(a), a)
        m = np.bincount(lab.ravel(), weights=a.ravel(), minlength=n + 1) / np.maximum(cnt, 1)
        m2 = np.bincount(lab.ravel(), weights=(a * a).ravel(), minlength=n + 1) / np.maximum(cnt, 1)
        var_u = np.maximum(m2 - m * m, 0)
        # province variance for the unit's province
        pv = {}
        for p in np.unique(l2[l2 > 0]):
            pv[p] = np.var(a[l2 == p])
        unit_l2 = np.zeros(n + 1, np.int64)
        idx = (lab > 0) & (l2 > 0)
        unit_l2[lab[idx]] = l2[idx]
        prov_var = np.array([pv.get(p, np.nan) for p in unit_l2])
        with np.errstate(invalid="ignore", divide="ignore"):
            within += var_u / np.where(prov_var > 0, prov_var, np.nan)
        total += 1
    h = 1 - within / total
    return np.clip(h[df.unit.values], 0, 1)
