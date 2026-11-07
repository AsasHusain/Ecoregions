"""v0.2 stage — community ecoregions nested in faunal regions and biomes.

1. Communities (90 m) are grouped into ecoregion types (``communities.yaml`` ->
   ``ecoregion_types``; each type has a RESOLVE biome).
2. On the 270 m grid the type map is smoothed with a majority filter
   (``smooth_cells``) so 90 m speckle and thin elevation slivers do not become units.
3. Faunal regions (``faunal`` stage, ``faunal_k`` regions on 10 km cells) are
   carried by the physiographic L3 units: each unit takes its majority faunal
   region, so faunal boundaries follow landforms rather than 10 km blocks.
4. Ecoregions = connected areas of one type within one faunal region. Parts
   smaller than ``mmu_km2`` join the neighbour sharing the most border *within
   the same faunal region* (any neighbour only if there is none).
5. Boundaries are snapped at 90 m as in v0.1 (marker watershed on the boundary
   strength surface within ``snap_band_m``), but never across a faunal boundary.
6. Attributes, names and confidence; polygons for ecoregions, faunal regions
   (dissolved from the ecoregions, so nesting is exact) and biomes.

Confidence (0-1) = 0.6 x purity (share of the unit in its own ecoregion type at
90 m) + 0.4 x (1 - share mapped only by a province fallback rule).
"""
from __future__ import annotations

import logging

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy import ndimage as ndi

from .attributes import POSITION, compass, polygonize
from .communities import load_rules
from .config import Paths
from .grid import Grid, block_mode, read, write
from .snap import internal_edges, make_contiguous, merge_small_units, tiled_watershed, upsample

log = logging.getLogger(__name__)
WORLDCOVER_CONVERTED = (40, 50)  # cropland, built-up


def majority(lab: np.ndarray, n: int, size: int) -> np.ndarray:
    """Majority (mode) filter of labels 1..n in a size x size window; 0 stays 0."""
    best = np.zeros(lab.shape, "float32")
    out = lab.copy()
    for v in range(1, n + 1):
        c = ndi.uniform_filter((lab == v).astype("float32"), size=size, mode="nearest")
        upd = (c > best) & (lab > 0)
        out[upd] = v
        best[upd] = c[upd]
    return out


def pair_counts(lab: np.ndarray, val: np.ndarray, n_lab: int, n_val: int, rows: int = 1024) -> np.ndarray:
    """Cells of each (label, value) pair, n_lab x n_val; computed in row chunks so
    90 m rasters never need full-size int64 temporaries."""
    out = np.zeros(n_lab * n_val, np.int64)
    for r0 in range(0, lab.shape[0], rows):
        a = lab[r0:r0 + rows].ravel().astype(np.int64)
        out += np.bincount(a * n_val + val[r0:r0 + rows].ravel().astype(np.int64), minlength=n_lab * n_val)
    return out.reshape(n_lab, n_val)


def mode_per_label(lab: np.ndarray, val: np.ndarray, n_lab: int, n_val: int) -> np.ndarray:
    """Most frequent ``val`` (1..n_val-1; 0 ignored) within each label 0..n_lab-1."""
    c = pair_counts(lab, val, n_lab, n_val)
    c[:, 0] = 0
    return c.argmax(axis=1)


def merge_small_nested(lab: np.ndarray, parent: np.ndarray, min_px: int, rounds: int = 5) -> np.ndarray:
    """Merge labels smaller than ``min_px`` into the neighbour sharing the longest
    border, preferring neighbours with the same ``parent`` (smallest first)."""
    for _ in range(rounds):
        cnt = np.bincount(lab.ravel())
        small = [u for u in np.flatnonzero((cnt > 0) & (cnt < min_px)) if u > 0]
        if not small:
            break
        slices = ndi.find_objects(lab)
        changed = 0
        for u in sorted(small, key=lambda u: cnt[u]):
            sl = slices[u - 1]
            if sl is None:
                continue
            sl = tuple(slice(max(0, a.start - 1), a.stop + 1) for a in sl)
            cm = lab[sl] == u
            if not cm.any():
                continue
            ring = ndi.binary_dilation(cm, structure=np.ones((3, 3))) & ~cm
            nb, pb = lab[sl][ring], parent[sl][ring]
            ok = (nb > 0) & (nb != u)
            same = ok & (pb == parent[sl][cm][0])
            cand = nb[same] if same.any() else nb[ok]
            if cand.size:
                vals, c = np.unique(cand, return_counts=True)
                lab[sl][cm] = vals[np.argmax(c)]
                changed += 1
        if not changed:
            break
    return lab


def disambiguate(names: pd.Series, x: pd.Series, y: pd.Series) -> pd.Series:
    """Make repeated names unique: add the position relative to the namesakes'
    centroid ("(northern)"), then a number if two still coincide."""
    out = names.copy()
    for _, g in names[names.duplicated(keep=False)].groupby(names):
        mx, my = x[g.index].mean(), y[g.index].mean()
        for i in g.index:
            out[i] = f"{names[i]} ({POSITION[compass(x[i] - mx, y[i] - my)]})"
    for _, g in out[out.duplicated(keep=False)].groupby(out):
        for n, i in enumerate(g.index, 1):
            out[i] = f"{out[i]} {n}"
    return out


def run(cfg) -> None:
    paths = Paths(cfg)
    grid = Grid.load(paths.grid_json)
    f = cfg["grid"]["coarse_factor"]
    ac = cfg["assemble"]
    rules = load_rules(paths.root / cfg["communities"]["rules"])
    comms, types = rules["communities"], rules["ecoregion_types"]
    tkeys = list(types)
    tidx = {k: i + 1 for i, k in enumerate(tkeys)}
    lut = np.zeros(256, "uint8")
    for j, c in enumerate(comms):
        lut[j + 1] = tidx[c["ecoregion"]]

    # 1-2. ecoregion types, smoothed on the coarse grid
    cls90 = lut[read(paths.raster("communities"))]
    cls = majority(block_mode(cls90, f), len(tkeys), ac["smooth_cells"])
    km2_c = grid.cell_km2 * f * f

    # 3. faunal region of each L3 unit (majority of the 10 km regions over the unit)
    l3 = read(paths.raster("l3")).astype(np.int32)
    l3c = block_mode(l3, f)
    fg = Grid.load(paths.interim / "fauna" / "grid.json")
    fau = read(paths.interim / "faunal" / f"regions_k{ac['faunal_k']}.tif")
    rows = np.arange(l3c.shape[0])
    cols = np.arange(l3c.shape[1])
    ys = grid.y0 - (rows + 0.5) * grid.res * f
    xs = grid.x0 + (cols + 0.5) * grid.res * f
    fr = np.clip(((fg.y0 - ys) / fg.res).astype(int), 0, fau.shape[0] - 1)
    fc_ = np.clip(((xs - fg.x0) / fg.res).astype(int), 0, fau.shape[1] - 1)
    fau270 = fau[fr][:, fc_].astype(np.int32)
    n_units = int(l3.max()) + 1
    unit_fau = mode_per_label(l3c, fau270, n_units, int(fau.max()) + 1)
    par = unit_fau[l3c]
    par[l3c == 0] = 0

    # 4. connected areas of one type within one faunal region, small parts merged within the region
    from skimage.measure import label as cc_label

    zone = np.where((cls > 0) & (par > 0), par * 64 + cls, 0).astype(np.int32)
    comp = cc_label(zone, connectivity=2, background=0).astype(np.int32)
    log.info("assemble: %d raw parts", comp.max())
    comp = merge_small_nested(comp, par, int(ac["mmu_km2"] / km2_c))
    log.info("assemble: %d parts after the minimum mapping unit", len(np.unique(comp[comp > 0])))

    # 5. snap at 90 m, never across a faunal boundary
    study = l3 > 0
    lab = upsample(comp, f, grid.shape).astype(np.int32)
    lab[~study] = 0
    par90 = upsample(par, f, grid.shape)
    band = int(round(ac["snap_band_m"] / grid.res))
    unknown = ndi.maximum_filter(internal_edges(lab).astype("uint8"), size=2 * band + 1).astype(bool) | (study & (lab == 0))
    markers = np.where(unknown, 0, lab).astype(np.int32)
    del unknown
    snapped = tiled_watershed(read(paths.raster("boundary_strength")), markers, study, grid,
                              cfg["grid"]["tile_size"], 4 * band + 64)
    del markers
    hole = study & (snapped == 0)
    snapped[hole] = lab[hole]
    comp_par = np.zeros(int(snapped.max()) + 1, np.int32)
    comp_par[lab[study]] = par90[study]  # every part lies in one region
    crossed = study & (comp_par[snapped] != par90)
    snapped[crossed] = lab[crossed]
    log.info("snapping moved %.2f%% of study cells (%.2f%% reverted at faunal boundaries)",
             100 * ((snapped != lab) & study).sum() / study.sum(), 100 * crossed.sum() / study.sum())
    del lab, crossed
    mmu_px = int(ac["mmu_km2"] / grid.cell_km2)
    snapped, _ = make_contiguous(snapped, mmu_px)
    snapped = merge_small_units(snapped, {int(u): mmu_px for u in np.unique(snapped[snapped > 0])})
    ids = np.unique(snapped[snapped > 0])
    remap = np.zeros(int(snapped.max()) + 1, np.int32)
    remap[ids] = np.arange(1, ids.size + 1)
    eco = remap[snapped].astype("uint16")
    del snapped
    write(paths.raster("ecoregions"), eco, grid, nodata=0)
    n = int(eco.max()) + 1

    # 6. attributes
    km2 = grid.cell_km2
    tcount = pair_counts(eco, cls90, n, len(tkeys) + 1)
    cells = tcount.sum(axis=1)
    area = cells * km2
    tcount[:, 0] = 0
    etype = tcount.argmax(axis=1)
    purity = tcount.max(axis=1) / np.maximum(tcount.sum(axis=1), 1)
    del cls90
    efau = mode_per_label(eco, par90, n, int(par90.max()) + 1)
    comm = read(paths.raster("communities"))
    ccount = pair_counts(eco, comm, n, len(comms) + 1)[:, 1:]
    del comm
    fb = pair_counts(eco, read(paths.raster("communities_fallback")), n, 2)[:, 1]
    conv = pair_counts(eco, np.isin(read(paths.raster("worldcover")), WORLDCOVER_CONVERTED).astype("uint8"), n, 2)[:, 1]
    sub = (slice(None, None, 3), slice(None, None, 3))  # elevation percentiles on every 3rd cell
    e_s, d_s = eco[sub].ravel(), read(paths.raster("dem"))[sub].ravel()
    ok = e_s > 0
    elev = pd.Series(d_s[ok]).groupby(e_s[ok]).quantile([0.1, 0.5, 0.9]).unstack()
    att = pd.read_csv(paths.interim / "l3_attributes.csv", dtype={"l2_code": str})
    l2_of_unit = np.zeros(n_units, np.int32)
    l2_of_unit[att.unit.to_numpy()] = att.l2_code.astype(int).to_numpy()
    el2 = mode_per_label(eco, l2_of_unit[l3], n, 18)
    l2name = att.drop_duplicates("l2_code").set_index(att.drop_duplicates("l2_code").l2_code.astype(int)).l2_name
    short = lambda s: str(s).split(" (")[0]  # noqa: E731

    rows = []
    for u in range(1, n):
        t = tkeys[etype[u] - 1]
        top = np.argsort(-ccount[u])[:3]
        mix = "; ".join(f"{comms[j]['name']} {100 * ccount[u, j] / ccount[u].sum():.0f}%" for j in top if ccount[u, j] > 0)
        rows.append(dict(unit=u, type=t, type_name=types[t]["name"], biome=types[t]["biome"], faunal_region=int(efau[u]),
                         l2_code=f"{el2[u]:02d}", l2_name=short(l2name.get(int(el2[u]), "")), area_km2=round(area[u], 1),
                         elev_p10=round(elev.loc[u, 0.1]) if u in elev.index else np.nan,
                         elev_median=round(elev.loc[u, 0.5]) if u in elev.index else np.nan,
                         elev_p90=round(elev.loc[u, 0.9]) if u in elev.index else np.nan,
                         type_purity=round(purity[u], 3), fallback_share=round(fb[u] / max(cells[u], 1), 3),
                         converted_share=round(conv[u] / max(cells[u], 1), 3), community_mix=mix))
    df = pd.DataFrame(rows)
    df["natural_share"] = (1 - df.converted_share).round(3)
    df["confidence"] = (0.6 * df.type_purity + 0.4 * (1 - df.fallback_share)).round(2)

    poly = polygonize(eco, grid)
    cen = poly.set_index("unit").geometry.centroid
    df["cx"], df["cy"] = df.unit.map(cen.x), df.unit.map(cen.y)

    # names: type name; + province where a type occurs in several units; + compass if still repeated
    df["name"] = df.type_name
    multi = df.groupby("type").unit.transform("count") > 1
    df.loc[multi, "name"] = df.type_name + " — " + df.l2_name
    df["name"] = disambiguate(df.name, df.cx, df.cy)
    df["code"] = [f"{t:02d}.{i:02d}" for t, i in zip(df.faunal_region, df.groupby("faunal_region").cumcount() + 1)]

    # faunal region names from the provinces they cover (largest first), made unique the same way
    fr_names = (df.groupby(["faunal_region", "l2_name"]).area_km2.sum().reset_index()
                  .sort_values("area_km2", ascending=False).groupby("faunal_region").l2_name
                  .agg(lambda s: " – ".join(list(dict.fromkeys(s))[:2])))
    w = df.area_km2
    fx = (df.cx * w).groupby(df.faunal_region).sum() / w.groupby(df.faunal_region).sum()
    fy = (df.cy * w).groupby(df.faunal_region).sum() / w.groupby(df.faunal_region).sum()
    fr_names = disambiguate(fr_names, fx.reindex(fr_names.index), fy.reindex(fr_names.index))
    df["faunal_name"] = df.faunal_region.map(fr_names)
    df = df.drop(columns=["cx", "cy"])
    df.to_csv(paths.interim / "ecoregions.csv", index=False)

    gdf = poly.merge(df, on="unit")
    out = paths.vector("ecoregions")
    if out.exists():
        out.unlink()
    gdf.to_file(out, layer="ecoregions")
    fr = gdf.dissolve("faunal_region", aggfunc={"area_km2": "sum", "faunal_name": "first"}).reset_index()
    fr["n_ecoregions"] = fr.faunal_region.map(df.faunal_region.value_counts())
    fr.to_file(out, layer="faunal_regions")
    gdf.dissolve("biome", aggfunc={"area_km2": "sum"}).reset_index().to_file(out, layer="biomes")
    log.info("assemble: %d ecoregions in %d faunal regions and %d biomes (median %.0f km²)",
             len(df), df.faunal_region.nunique(), df.biome.nunique(), df.area_km2.median())
    log.info("\n%s", df.groupby("type_name").agg(units=("unit", "count"), km2=("area_km2", "sum")).round(0).to_string())

    from .qa import label_map

    label_map(eco, paths.qa / "40_ecoregions.png", f"community ecoregions ({len(df)})")
