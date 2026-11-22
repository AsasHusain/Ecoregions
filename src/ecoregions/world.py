"""World merge: the published areas as one map.

Areas are built as parcels, and each is clipped at the border of the countries it maps (``join``). This stage puts
them together. Each area's units are fitted to its mapped area: the thin slivers its 90 m raster edge leaves along a
seam with another area go to the unit they touch most (lakes and coasts are left alone). Where an area's overlap
strip lies in a neighbour, the neighbour's landform units on the seam take the label the strip gives them, whole
(``natural_seam``), so a change of label sits on a landform edge, not on the border. Floral Ecoregions of one
type that meet across a seam become one unit, named after its larger part, and types are dissolved over all areas.
Faunal Ecoregions of the areas clustered together (``world.joint_faunal``) share one numbering, and each is dissolved
across the seams; other areas' Faunal Ecoregions are laid side by side.

The border gate (``bordercheck.judge``) then runs on the merged map, on the six layers of the area gates mosaicked
from the areas: where a type still changes along a border at a seam, the change must be a natural break.
Borders inside one area are left to that area's own gate; the areas' reviewed exceptions apply here too.

    PYTHONPATH=src python -m ecoregions.world

Writes outputs/world/staging/ and moves the files to outputs/world/ only when the gate passes: ecoregions.gpkg
(ecoregions, ecoregion_types, faunal_regions, l2_provinces), ecoregions.csv
and bordercheck.csv. Areas: ``world.areas`` in config.yaml (published areas only).
"""
from __future__ import annotations

import logging
import re

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

from .config import ROOT, Paths, load_config

log = logging.getLogger(__name__)
CRS = "EPSG:6933"  # equal-area (EASE-Grid 2.0 global): areas in km² straight from the geometry
WEIGHTED = ["elev_p10", "elev_median", "elev_p90", "type_purity", "fallback_share", "converted_share", "rain_mm",
            "temp_c", "aridity", "humidity_pct", "vpd_pa", "summer_rain", "glacier_share", "natural_share",
            "confidence", "faunal_share"]
SHARE = re.compile(r"^(.*) ([<>]?\d+(?:\.\d+)?)%$")


def shares(text) -> dict[str, float]:
    """'Pakistan >99%; Iran 21%; Kashmir <1%' -> {name: share}; '<1' counts 0.5 and '>99' 99.5."""
    out = {}
    for part in str(text or "").split("; "):
        m = SHARE.match(part.strip())
        if m:
            v = m.group(2)
            out[m.group(1)] = float(v[1:]) + (0.5 if v[0] == ">" else -0.5) if v[0] in "<>" else float(v)
    return out


def combine(texts, weights) -> str:
    """Area-weighted mix of several 'name NN%' lists, largest first, in the same format."""
    tot: dict[str, float] = {}
    for t, w in zip(texts, weights):
        for k, v in shares(t).items():
            tot[k] = tot.get(k, 0.0) + v * w
    s = sum(weights) or 1.0
    items = [(k, v) for k, v in sorted(((k, v / s) for k, v in tot.items()), key=lambda kv: -kv[1]) if v >= 0.5]
    return "; ".join(f"{k} {'<1' if v < 1 else '>99' if v > 99 and len(items) > 1 else round(v)}%" for k, v in items)


def seams(areas: dict[str, shapely.Geometry]) -> dict[str, shapely.Geometry]:
    """Each area's outline where another area meets it."""
    out = {}
    for k, a in areas.items():
        others = [b for j, b in areas.items() if j != k]
        out[k] = a.boundary.intersection(shapely.union_all(others).buffer(1000)) if others else shapely.LineString()
    return out


def fit(units: gpd.GeoDataFrame, area, seam, max_width=500.0) -> gpd.GeoDataFrame:
    """Units clipped to ``area``; slivers narrower than ``max_width`` m along ``seam`` go to the unit they touch most."""
    units = units.copy()
    units["geometry"] = units.geometry.intersection(area)
    units = units[~units.is_empty]
    if seam.is_empty:
        return units
    gaps = gpd.GeoSeries([area.difference(units.union_all())], crs=units.crs).explode(ignore_index=True)
    gaps = gaps[~gaps.is_empty]
    thin = gaps[(2 * gaps.area / gaps.boundary.length.clip(lower=1) < max_width) & gaps.intersects(seam.buffer(1000))]
    if len(thin):
        near = gpd.sjoin(gpd.GeoDataFrame(geometry=thin.buffer(1), crs=units.crs), units[["geometry"]], predicate="intersects")
        near["shared"] = shapely.area(shapely.intersection(thin.loc[near.index].buffer(100).values,
                                                           units.geometry.loc[near.index_right].values))
        best = near.sort_values("shared").groupby(level=0).index_right.last()
        for u, idx in best.groupby(best).groups.items():
            units.loc[u, "geometry"] = shapely.union_all([units.geometry.loc[u], *thin.loc[idx]])
        log.info("world: %d slivers (%.1f km²) along the seams filled", len(best), thin.loc[best.index].area.sum() / 1e6)
    return units


def natural_seam(own: gpd.GeoDataFrame, strip: gpd.GeoDataFrame, other: gpd.GeoDataFrame, cells: gpd.GeoSeries,
                 seam, col: str, min_cover: float = 0.5) -> gpd.GeoDataFrame:
    """Move label changes off a seam onto landform edges.

    ``own``: one area's polygons with label ``col``; ``cells``: its landform units; ``strip``: the neighbour's labels
    on the same ground (its overlap strip); ``other``: the neighbour's own polygons, which lend their attributes.
    Each landform unit on the seam that the neighbour's build labels differently, over at least ``min_cover`` of the
    unit, takes that label whole, so the change of label follows the unit's inner edges, not the border."""
    if strip is None or strip.empty or seam.is_empty:
        return own
    near = gpd.GeoDataFrame(geometry=cells[cells.intersects(seam.buffer(1000))].values, crs=own.crs)
    if near.empty:
        return own
    near["cell"] = np.arange(len(near))
    near["cell_m2"] = near.area

    def majority(layer):
        ov = gpd.overlay(near[["cell", "geometry"]], layer[[col, "geometry"]], how="intersection", keep_geom_type=True)
        ov["a"] = ov.area
        s = ov.groupby(["cell", col]).a.sum().reset_index().sort_values("a").groupby("cell").last()
        return s[col], s.a

    theirs, cover = majority(strip)
    mine, _ = majority(own)
    sw = pd.DataFrame({"theirs": theirs, "mine": mine.reindex(theirs.index),
                       "cover": cover / near.set_index("cell").cell_m2.reindex(theirs.index)})
    sw = sw[(sw.theirs != sw.mine) & (sw.cover >= min_cover)]
    if sw.empty:
        return own
    cut = near.set_index("cell").loc[sw.index]
    carved = own.assign(geometry=own.geometry.difference(cut.union_all()))
    carved = carved[~carved.is_empty]
    pieces = []
    for c, r in sw.iterrows():
        donors = other[other[col] == r.theirs]
        if donors.empty:
            continue
        g = cut.geometry.loc[c]
        d = donors.iloc[[int(donors.distance(g).argmin())]].copy()
        pieces.append(d.assign(geometry=[g], area_km2=g.area / 1e6))
    log.info("world: %d landform units on the seam take the neighbour's %s (%.0f km²)", len(pieces), col,
             sum(p.area_km2.iloc[0] for p in pieces))
    if not pieces:
        return own
    out = gpd.GeoDataFrame(pd.concat([carved, *pieces], ignore_index=True), crs=own.crs)
    if "parcel" in own:  # the pieces lie in this area; the merge then joins them to the neighbour's units across the seam
        out["parcel"] = own.parcel.iloc[0]
    return out


def merge(units: gpd.GeoDataFrame, countries: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Units of one type from different areas that share at least 1 km of edge become one unit."""
    u = units.reset_index(drop=True)
    j = gpd.sjoin(u[["parcel", "type", "geometry"]], u[["parcel", "type", "geometry"]], predicate="intersects")
    j = j[(j.index < j.index_right) & (j.parcel_left != j.parcel_right) & (j.type_left == j.type_right)]
    edge = shapely.intersection(u.boundary.loc[j.index].values, u.geometry.loc[j.index_right].buffer(5).values)
    j = j[shapely.length(edge) >= 1000]
    parent = list(range(len(u)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for a, b in zip(j.index, j.index_right):
        parent[find(a)] = find(b)
    u["group"] = [find(i) for i in range(len(u))]
    u["area_km2"] = u.area / 1e6
    rows = []
    for _, g in u.groupby("group", sort=False):
        if len(g) == 1:
            rows.append(g.iloc[0].to_dict())
            continue
        w = g.area_km2.values
        r = g.loc[g.area_km2.idxmax()].to_dict()  # name, code, faunal region: the larger part's
        r["geometry"] = shapely.union_all(g.geometry.values)
        r["parcel"] = "+".join(dict.fromkeys(g.parcel))
        for c in WEIGHTED:
            if c in g:
                v = g[c].astype(float)
                r[c] = float(np.average(v.fillna(v.mean()), weights=w)) if v.notna().any() else np.nan
        r["population"] = float(g.population.sum())
        r["community_mix"] = combine(g.community_mix, w)
        r["faunal_regions"] = " ".join(dict.fromkeys(" ".join(g.faunal_regions.astype(str)).split()))
        rows.append(r)
    out = gpd.GeoDataFrame(rows, crs=u.crs).drop(columns="group")
    out["area_km2"] = out.area / 1e6
    merged = out["parcel"].str.contains("+", regex=False)
    if merged.any():  # country shares of the joined units, from the map's own country shapes
        ov = gpd.overlay(out.loc[merged, ["geometry"]].reset_index(), countries[["NAME", "geometry"]], how="intersection")
        ov["a"] = ov.area
        for i, g in ov.groupby("index"):
            s = g.groupby("NAME").a.sum().sort_values(ascending=False) / g.a.sum() * 100
            out.loc[i, "countries"] = "; ".join(f"{k} {'<1' if v < 1 else round(v)}%" for k, v in s.items() if v >= 0.05)
    from .attributes import aridity_class

    out["aridity_class"] = out.aridity.map(aridity_class)
    size = u.groupby("group").size()
    log.info("world: %d units joined across seams into %d", int(size[size > 1].sum()), int(merged.sum()))
    return out


def types(units: gpd.GeoDataFrame, per_area: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """One feature per type over all areas; statistics area-weighted over each area's type rows (exact for a type
    found in one area)."""
    rows = []
    for t, g in per_area.groupby("type", sort=False):
        w = g.area_km2.values
        r = g.loc[g.area_km2.idxmax()].drop(labels="geometry").to_dict()
        for c in WEIGHTED:
            if c in g:
                v = g[c].astype(float)
                r[c] = float(np.average(v.fillna(v.mean()), weights=w)) if v.notna().any() else np.nan
        r["population"] = float(g.population.sum())
        r["countries"] = combine(g.countries, w)
        r["community_mix"] = combine(g.community_mix, w)
        r["faunal_names"] = "; ".join(dict.fromkeys(n for s in g.faunal_names.astype(str) for n in s.split("; ") if n))
        u = units[units["type"] == t]
        r["n_ecoregions"] = len(u)
        r["geometry"] = shapely.union_all(u.geometry.values)
        r["area_km2"] = r["geometry"].area / 1e6
        rows.append(r)
    out = gpd.GeoDataFrame(rows, crs=units.crs)
    from .attributes import aridity_class

    out["aridity_class"] = out.aridity.map(aridity_class)
    return out


def rasters(bounds_lonlat, paths: list, res=1000.0, within=None):
    """The area gates' six layers (elevation, rainfall, soil sand, height above the channel, dune texture, slope),
    mosaicked from the areas' 270 m feature grids onto one equal-area grid, each with its p90-p10 spread over
    ``within`` (EPSG:6933 geometry; the mapped areas, not the whole box: with Tibet in the box the elevation spread was
    inflated). Elevation and rainfall alone could not see a river terrace's edge (the Amu Darya)."""
    from affine import Affine
    from rasterio.warp import Resampling, reproject, transform_bounds
    import rasterio

    w, s, e, n = transform_bounds(4326, CRS, *bounds_lonlat)
    width, height = int((e - w) / res) + 1, int((n - s) / res) + 1
    tr = Affine(res, 0, w, 0, -res, n)
    out = []
    srcs = [[str(p.coarse(k)) for p in paths if p.coarse(k).exists()]
            for k in ("elev", "bio12", "sand", "hand", "dune_frac", "slope")]
    for files in filter(None, srcs):
        a = np.full((height, width), np.nan, "float32")
        for f in files:
            with rasterio.open(f) as src:
                b = np.full((height, width), np.nan, "float32")
                reproject(rasterio.band(src, 1), b, dst_transform=tr, dst_crs=CRS, resampling=Resampling.bilinear,
                          dst_nodata=np.nan)
                if src.scales[0] != 1:
                    b *= src.scales[0]
                a = np.where(np.isnan(a), b, a)
        a[a < -1000] = np.nan
        v = a
        if within is not None:
            from rasterio.features import rasterize

            v = a[rasterize([(within, 1)], out_shape=a.shape, transform=tr, dtype="uint8").astype(bool)]
        out.append((a, float(np.nanpercentile(v, 90) - np.nanpercentile(v, 10)) or 1.0))
    return out, tr


def run(areas: list[str]) -> None:
    from . import bordercheck
    from .boundaries import display_countries, load_boundaries, mapped_area

    # everything is written to staging/ first and replaces the published files only when the gate passes
    pub_dir = ROOT / "outputs" / "world"
    out_dir = pub_dir / "staging"
    out_dir.mkdir(parents=True, exist_ok=True)
    cfgs = {r: load_config(region=r) for r in areas}
    countries, disputed = load_boundaries(Paths(next(iter(cfgs.values()))))
    shapes = {r: mapped_area(c, Paths(c), countries, disputed).to_crs(CRS).union_all() for r, c in cfgs.items()}
    seam = seams(shapes)
    fl, ty, fa, pv, cells = {}, [], {}, [], {}
    for r, c in cfgs.items():
        gpkg = Paths(c).outputs / "ecoregions.gpkg"
        fl[r] = fit(gpd.read_file(gpkg, layer="ecoregions").to_crs(CRS), shapes[r], seam[r]).assign(parcel=r)
        ty.append(gpd.read_file(gpkg, layer="ecoregion_types").to_crs(CRS).assign(parcel=r))
        f = gpd.read_file(gpkg, layer="faunal_regions").to_crs(CRS)
        own = "joint" if r in load_config()["world"].get("joint_faunal", []) else r  # areas clustered together share numbers
        fa[r] = fit(f, shapes[r], seam[r]).assign(parcel=r, key=lambda d: own + ":" + d.faunal_region.astype(str))
        pv.append(gpd.read_file(Paths(c).outputs / "physiographic.gpkg", layer="l2_provinces").to_crs(CRS)[["code", "geometry"]])
        cells[r] = gpd.read_file(Paths(c).outputs / "physiographic.gpkg", layer="l3_ecoregions", columns=[]).to_crs(CRS).geometry
    fl0 = list(fl.values())  # as built, for the strip agreement
    # natural seams: where an area's overlap strip lies in a neighbour, the neighbour's landform units on the seam take
    # the label the strip gives them, whole, so a change of label sits on a landform edge rather than on the border
    for r, c in cfgs.items():
        sf = Paths(c).outputs / "strip.gpkg"
        if not sf.exists():
            continue
        for q in cfgs:
            line = seam[q].intersection(shapes[r].buffer(1000)) if q != r else shapely.LineString()
            if line.is_empty:
                continue
            for layer, col, maps in (("ecoregions", "type", fl), ("faunal_regions", "faunal_region", fa)):
                st = gpd.read_file(sf, layer=layer).to_crs(CRS)
                maps[q] = natural_seam(maps[q], st[st.intersects(shapes[q])], maps[r], cells[q], line, col)
    fl, fa = list(fl.values()), list(fa.values())
    cty = display_countries(countries, disputed).to_crs(CRS)
    # strip agreement (declared 2026-11-14 before any result): where an area's 75 km strip lies in a neighbour's
    # mapped area, both classify the same ground; the type must agree on at least 90 % of it
    agree = []
    for r, c in cfgs.items():
        f = Paths(c).outputs / "strip.gpkg"
        if not f.exists():
            continue
        st = gpd.read_file(f, layer="ecoregions").to_crs(CRS)
        for q, other in zip(cfgs, fl0):
            if q == r or not st.intersects(shapes[q]).any():
                continue
            ov = gpd.overlay(st.assign(geometry=st.geometry.intersection(shapes[q]))[~st.geometry.intersection(shapes[q]).is_empty],
                             other[["type", "geometry"]].rename(columns={"type": "type_q"}), how="intersection")
            if ov.empty:
                continue
            a = ov.area
            agree.append(dict(strip_of=r, inside=q, km2=round(a.sum() / 1e6), same_type=round(a[ov["type"] == ov.type_q].sum() / a.sum(), 3)))
            log.info("world: %s's strip in %s: %.0f km², same type on %.1f %%", r, q, a.sum() / 1e6, 100 * agree[-1]["same_type"])
    pd.DataFrame(agree, columns=["strip_of", "inside", "km2", "same_type"]).to_csv(out_dir / "strip_agreement.csv", index=False)
    low = [a for a in agree if a["km2"] >= 1000 and a["same_type"] < 0.90]
    units = merge(pd.concat(fl, ignore_index=True), cty)
    units = units.sort_values(["type", "area_km2"], ascending=[True, False]).reset_index(drop=True)
    units["unit"] = np.arange(1, len(units) + 1)
    tys = types(units, pd.concat(ty, ignore_index=True))
    faunal = gpd.GeoDataFrame(pd.concat(fa, ignore_index=True), crs=CRS).sort_values("area_km2", ascending=False)
    parcels = faunal.groupby("key").parcel.agg(lambda p: "+".join(dict.fromkeys(p)))
    faunal = faunal.dissolve("key", aggfunc="first").assign(parcel=parcels).reset_index()  # attributes of the larger part
    faunal["area_km2"] = faunal.area / 1e6
    prov = gpd.GeoDataFrame(pd.concat(pv, ignore_index=True), crs=CRS).dissolve("code").reset_index()
    gpkg = out_dir / "ecoregions.gpkg"
    gpkg.unlink(missing_ok=True)
    for name, g in (("ecoregions", units), ("ecoregion_types", tys), ("faunal_regions", faunal), ("l2_provinces", prov)):
        g.to_crs(4326).to_file(gpkg, layer=name)
    units.drop(columns="geometry").to_csv(out_dir / "ecoregions.csv", index=False)
    log.info("world: %d Floral Ecoregions in %d types, %.0f km², from %s", len(units), len(tys), units.area_km2.sum(),
             ", ".join(areas))

    # the gate on the merged map, over the areas that meet another area
    joined = [r for r in areas if not seam[r].is_empty]
    if len(joined) < 2:
        promote(out_dir, pub_dir)
        return
    box = shapely.union_all([shapes[r] for r in joined])
    rs, tr = rasters(gpd.GeoSeries([box], crs=CRS).to_crs(4326).total_bounds, [Paths(cfgs[r]) for r in joined], within=box)
    bc = cfgs[joined[0]].get("bordercheck", {})
    tol, min_km = bc.get("tol_km", 10.0), bc.get("min_km", 50.0)
    sel = lambda g: g[g.intersects(box)]  # noqa: E731
    accept = [a for c in cfgs.values() for a in c.get("bordercheck", {}).get("accept", [])]
    rows = bordercheck.judge("ecoregions", sel(units), countries.to_crs(CRS), rs, tr, tol, min_km,
                             bordercheck.internal_boundaries(sel(prov)), accept=accept)
    rows += bordercheck.judge("faunal_regions", sel(faunal), countries.to_crs(CRS), rs, tr, tol, min_km, accept=accept)
    # borders inside one area were judged by that area's gate (six layers at 270 m); this one judges the seams (a stretch lies within tol of its border)
    at_seam = shapely.union_all(list(seam.values())).buffer(tol * 1000 + 1000)
    mid = gpd.GeoSeries(gpd.points_from_xy([r["lon"] for r in rows], [r["lat"] for r in rows]), crs=4326).to_crs(CRS)
    rows = [r for r, m in zip(rows, mid) if at_seam.contains(m)]
    pd.DataFrame(rows, columns=bordercheck.COLS).to_csv(out_dir / "bordercheck.csv", index=False)
    bad = [r for r in rows if r["verdict"] == "ARTEFACT"]
    if bad or low:
        raise SystemExit(f"world: {len(bad)} boundary stretch(es) follow a political border without a natural break "
                         f"(outputs/world/staging/bordercheck.csv); {len(low)} strip(s) agree on less than 90 % of their "
                         "area (outputs/world/staging/strip_agreement.csv); the published files are unchanged")
    promote(out_dir, pub_dir)


def promote(staging, pub_dir) -> None:
    """The staged files replace the published ones (only after the gate passes)."""
    for f in staging.iterdir():
        f.replace(pub_dir / f.name)
    staging.rmdir()
    log.info("world: gate passed; outputs/world updated")


if __name__ == "__main__":
    from .config import setup_logging

    setup_logging()
    run(load_config()["world"]["areas"])
