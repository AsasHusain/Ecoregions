"""v0.2 stage — community ecoregions, with faunal regions and biomes.

1. Communities (90 m) are grouped into ecoregion types (``communities.yaml`` ->
   ``ecoregion_types``; each type has a RESOLVE biome).
2. On the 270 m grid the type map is smoothed with a majority filter
   (``smooth_cells``) so 90 m speckle and thin elevation slivers do not become units.
3. Ecoregions = connected areas of one type: a continuous landscape is one
   ecoregion, whatever faunal region or province it spans. Parts smaller than
   ``mmu_km2`` join the neighbour sharing the most border.
4. Boundaries are snapped at 90 m as in v0.1 (marker watershed on the boundary
   strength surface within ``snap_band_m``).
5. Faunal regions (``faunal`` stage, ``faunal_k`` regions on 10 km cells) are a
   separate overlay, carried by the physiographic L3 units (each unit takes its
   majority faunal region, so their boundaries follow landforms rather than
   10 km blocks). They do not cut ecoregions; each ecoregion records its main
   faunal region, the share of its area in it, and every faunal region covering
   at least 10 % of it.
6. Attributes, names and confidence; polygons for ecoregions, faunal regions
   and biomes.

Confidence (0-1) = 0.6 x purity (share of the unit in its own ecoregion type at
90 m) + 0.4 x (1 - share mapped only by a province fallback rule).
"""
from __future__ import annotations

import logging

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy import ndimage as ndi

from .attributes import POSITION, aridity_class, compass, polygonize
from .boundaries import country_parts
from .communities import load_rules
from .config import Paths
from .grid import Grid, block_mode, rasterize_geoms, read, write
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


def touching_pairs(lab: np.ndarray, rows: int = 1024) -> tuple[np.ndarray, np.ndarray]:
    """Pairs of different non-zero labels sharing a cell edge (each pair once per direction found)."""
    n = int(lab.max()) + 1
    keys = []
    for r0 in range(0, lab.shape[0], rows):
        blk = lab[r0:r0 + rows + 1].astype(np.int64)
        for a, b in ((blk[:, :-1], blk[:, 1:]), (blk[:-1], blk[1:])):
            m = (a != b) & (a > 0) & (b > 0)
            keys.append(np.unique(a[m] * n + b[m]))
    k = np.unique(np.concatenate(keys)) if keys else np.zeros(0, np.int64)
    return k // n, k % n


def join_same_type(lab: np.ndarray, typ: np.ndarray, rows: int = 1024) -> np.ndarray:
    """Give touching labels of the same type (``typ[label]``) one label: merging small
    parts into neighbours can join two parts of one type after they were labelled."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    n = int(lab.max()) + 1
    a, b = touching_pairs(lab, rows)
    same = typ[a] == typ[b]
    g = coo_matrix((np.ones(same.sum()), (a[same], b[same])), shape=(n, n))
    _, root = connected_components(g, directed=False)
    root = (root + 1).astype(np.int32)
    root[0] = 0  # background has no edges, so it is its own component
    return root[lab]


def check_invariants(eco: np.ndarray, etype: np.ndarray, area_km2: np.ndarray, mmu_km2: float,
                     codes: pd.Series, names: pd.Series) -> None:
    """Fail the build if the ecoregions break a rule of their definition: two touching
    ecoregions of one type, an ecoregion in several pieces (8-connected; islands that
    touch no other ecoregion are allowed), one below the minimum mapping unit, or
    repeated codes or names. (v0.2.1 shipped six touching same-type pairs, and v0.2.2
    one chir pine ecoregion in two distant parts, because nothing checked this.)"""
    problems = []
    a, b = touching_pairs(eco)
    same = (etype[a] == etype[b]) & (a < b)
    problems += [f"ecoregions {i} and {j} touch and are both type {etype[i]}" for i, j in zip(a[same], b[same])]
    k3 = np.ones((3, 3), bool)
    for u, sl in enumerate(ndi.find_objects(eco), 1):
        if sl is None:
            continue
        sl = tuple(slice(max(0, s.start - 1), s.stop + 1) for s in sl)
        pieces, k = ndi.label(eco[sl] == u, structure=k3)
        if k < 2:
            continue
        sizes = np.bincount(pieces.ravel())
        sizes[0] = 0
        for j in set(range(1, k + 1)) - {int(sizes.argmax())}:
            m = pieces == j
            if (eco[sl][ndi.binary_dilation(m, k3) & ~m] > 0).any():
                problems.append(f"ecoregion {u} has a detached piece of {sizes[j]} cells")
    problems += [f"ecoregion {u} is {area_km2[u]:.0f} km² (< {mmu_km2} km²)"
                 for u in range(1, len(area_km2)) if area_km2[u] < mmu_km2]
    problems += [f"repeated {k}: {v}" for k, s in (("code", codes), ("name", names)) for v in s[s.duplicated()]]
    if problems:
        raise RuntimeError("ecoregion invariants broken:\n  " + "\n  ".join(problems))


def population_cells(cfg, paths: Paths, grid: Grid) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """GHS-POP residents as (row, col, people) on the analysis grid: each 1 km cell goes to the
    grid cell under its centre (people per label = ``np.bincount(lab[row, col], people)``)."""
    import rasterio
    from rasterio.warp import transform, transform_bounds
    from rasterio.windows import Window, from_bounds

    from rasterio.windows import WindowError

    tiles = sorted((paths.raw_region / "ghs_pop").glob("*.zip"))
    if not tiles:
        raise FileNotFoundError(f"no GHS-POP tiles in {paths.raw_region / 'ghs_pop'}: run the download stage")
    out = []
    for z in tiles:
        with rasterio.open(f"/vsizip/{z}/{z.stem}.tif") as r:
            try:
                win = from_bounds(*transform_bounds(grid.rio_crs, r.crs, *grid.bounds, densify_pts=51), transform=r.transform)
                win = win.round_offsets().round_lengths().intersection(Window(0, 0, r.width, r.height))
            except WindowError:  # tile outside the grid
                continue
            pop = r.read(1, window=win).astype("float64")
            tr, crs = r.window_transform(win), r.crs
        rr, cc = np.nonzero(pop > 0)  # nodata is negative
        x, y = transform(crs, grid.rio_crs, *(tr * (cc + 0.5, rr + 0.5)))
        row = np.floor((grid.y0 - np.asarray(y)) / grid.res).astype(np.int64)
        col = np.floor((np.asarray(x) - grid.x0) / grid.res).astype(np.int64)
        ok = (row >= 0) & (row < grid.height) & (col >= 0) & (col < grid.width)
        out.append((row[ok], col[ok], pop[rr[ok], cc[ok]]))
    return tuple(np.concatenate(a) for a in zip(*out))


# climate means shown for each ecoregion, type and faunal region (CHELSA 1981-2010, area-weighted at 90 m),
# and the share of the area under glacier (RGI 7.0, c. 2000)
CLIMATE = {"rain_mm": "clim_bio12", "temp_c": "clim_bio1", "aridity": "clim_ai", "humidity_pct": "clim_hurs_mean",
           "vpd_pa": "clim_vpd_mean", "summer_rain": "clim_summer_rain", "glacier_share": "glacier"}
CLIMATE_DIGITS = {"rain_mm": 0, "temp_c": 1, "aridity": 2, "humidity_pct": 0, "vpd_pa": 0, "summer_rain": 2, "glacier_share": 3}


def label_sums(lab: np.ndarray, val: np.ndarray, n: int, rows: int = 1024) -> tuple[np.ndarray, np.ndarray]:
    """Sum and count of the finite values of ``val`` per label, in row blocks (no full-size float64 copy)."""
    s, c = np.zeros(n), np.zeros(n)
    for r0 in range(0, lab.shape[0], rows):
        lb, v = lab[r0:r0 + rows].ravel(), val[r0:r0 + rows].ravel()
        ok = np.isfinite(v)
        s += np.bincount(lb[ok], weights=v[ok], minlength=n)
        c += np.bincount(lb[ok], minlength=n)
    return s, c


def country_mix(counts: np.ndarray, names: list[str]) -> str:
    """"Pakistan 62%; Afghanistan 38%" from cell counts by country (index 0 = no country,
    index i = ``names[i - 1]``); largest first. Every country with any area is listed, so a single
    entry means the area lies wholly in that country."""
    c = np.asarray(counts, float)[1:]
    s = c.sum()
    out = []
    for j in np.argsort(-c, kind="stable"):
        if c[j] == 0:
            break
        p = 100 * c[j] / s
        txt = "100" if c[j] == s else ">99" if p >= 99.5 else "<1" if p < 0.5 else f"{p:.0f}"
        out.append(f"{names[j]} {txt}%")
    return "; ".join(out)


def place_name(counts: np.ndarray, names: dict, regions: dict | None = None) -> str:
    """Name an area after the province covering at least half of it (``counts`` by province code;
    0 = none), else after the enclosing region (``regions``: code -> name) covering most of it:
    one place name only, "&" for "and". Repeats are told apart by ``disambiguate``."""
    c = np.asarray(counts, float).copy()
    c[0] = 0
    top = int(c.argmax())
    if regions and c[top] < 0.5 * c.sum():
        by = pd.Series(c[1:], index=[regions.get(i, names.get(i, "")) for i in range(1, len(c))]).groupby(level=0).sum()
        return by.idxmax().replace(" and ", " & ")
    return names.get(top, "").replace(" and ", " & ")


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


def zone_types(cls90: np.ndarray, value: np.ndarray, src: list[int], breaks: list[float], into: list[int]) -> None:
    """In place: cells of the ``src`` types take the type of their ``value`` belt (``into[k]`` for values in
    [breaks[k-1], breaks[k])); other cells and cells without a value keep theirs."""
    hit = np.isin(cls90, src) & np.isfinite(value)
    belt = np.digitize(value, breaks)
    for k, t in enumerate(into):
        cls90[hit & (belt == k)] = t


def apply_zones(cls90: np.ndarray, zones: list, tidx: dict, paths: Paths, rows: int = 2048) -> None:
    """``assemble.zones``: regroup some ecoregion types into belts of a climate raster (e.g. rainfall), so an
    ecoregion follows the gradient species turn over along rather than a substrate band across it. Each zone:
    {raster, types (regrouped), breaks (ascending), into (one type per belt, len(breaks) + 1)}."""
    import rasterio
    from rasterio.windows import Window

    for z in zones:
        if len(z["into"]) != len(z["breaks"]) + 1:
            raise ValueError(f"assemble.zones: {len(z['breaks'])} breaks need {len(z['breaks']) + 1} belt types")
        src, into = [tidx[t] for t in z["types"]], [tidx[t] for t in z["into"]]
        with rasterio.open(paths.raster(z["raster"])) as r:
            for r0 in range(0, cls90.shape[0], rows):  # in strips: a float raster of the whole grid is large
                n = min(rows, cls90.shape[0] - r0)
                zone_types(cls90[r0:r0 + n], r.read(1, window=Window(0, r0, cls90.shape[1], n)), src, z["breaks"], into)
        log.info("zones: %s regrouped into %d belts of %s at %s", ", ".join(z["types"]), len(into), z["raster"], z["breaks"])


def run(cfg) -> None:
    paths = Paths(cfg)
    grid = Grid.load(paths.grid_json)
    f = cfg["grid"]["coarse_factor"]
    ac = cfg["assemble"]
    rules = load_rules(paths.root / cfg["communities"]["rules"])
    comms, types = rules["communities"], rules["ecoregion_types"]
    from .biomes import with_key_biomes

    types = with_key_biomes(cfg, paths, types)  # biomes from the global key (biomes stage), unless switched off
    tkeys = list(types)
    tidx = {k: i + 1 for i, k in enumerate(tkeys)}
    lut = np.zeros(256, "uint8")
    for j, c in enumerate(comms):
        lut[j + 1] = tidx[c["ecoregion"]]

    # 1-2. ecoregion types, smoothed on the coarse grid
    cls90 = lut[read(paths.raster("communities"))]
    apply_zones(cls90, ac.get("zones", []), tidx, paths)
    cls = majority(block_mode(cls90, f), len(tkeys), ac["smooth_cells"])
    km2_c = grid.cell_km2 * f * f

    # 3. faunal region of each L3 unit (majority of the 10 km regions over the unit)
    l3 = read(paths.raster("l3")).astype(np.int32)
    l3c = block_mode(l3, f)
    fg = Grid.load(paths.interim / "fauna" / "grid.json")
    faunal = ac.get("faunal", True)  # false: the region has no faunal regions yet (no range data)
    fau = read(paths.interim / "faunal" / f"regions_k{ac['faunal_k']}.tif") if faunal else np.ones((fg.height, fg.width), "uint8")
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

    # 4. connected areas of one type (faunal regions and provinces do not cut them)
    from skimage.measure import label as cc_label

    zone = np.where((cls > 0) & (l3c > 0), cls, 0).astype(np.int32)
    comp = cc_label(zone, connectivity=2, background=0).astype(np.int32)
    log.info("assemble: %d raw parts", comp.max())
    comp = merge_small_nested(comp, np.zeros_like(comp), int(ac["mmu_km2"] / km2_c))
    log.info("assemble: %d parts after the minimum mapping unit", len(np.unique(comp[comp > 0])))

    # 5. snap at 90 m
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
    log.info("snapping moved %.2f%% of study cells", 100 * ((snapped != lab) & study).sum() / study.sum())
    del lab
    mmu_px = int(ac["mmu_km2"] / grid.cell_km2)
    snapped, _ = make_contiguous(snapped, mmu_px)
    snapped = merge_small_units(snapped, {int(u): mmu_px for u in np.unique(snapped[snapped > 0])})
    ns = int(snapped.max()) + 1
    snapped = join_same_type(snapped, mode_per_label(snapped, cls90, ns, len(tkeys) + 1))
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
    nf = int(par90.max()) + 1
    fcount = pair_counts(eco, par90, n, nf)
    fcount[:, 0] = 0
    efau = fcount.argmax(axis=1)
    fshare = fcount.max(axis=1) / np.maximum(fcount.sum(axis=1), 1)
    comm = read(paths.raster("communities"))
    ccount = pair_counts(eco, comm, n, len(comms) + 1)[:, 1:]
    del comm
    fb = pair_counts(eco, read(paths.raster("communities_fallback")), n, 2)[:, 1]
    cv = np.isin(read(paths.raster("worldcover")), WORLDCOVER_CONVERTED).astype("uint8")
    conv = pair_counts(eco, cv, n, 2)[:, 1]
    cparts = country_parts(cfg, paths)
    cnames = list(dict.fromkeys(cparts.country))
    ctry = rasterize_geoms(((g, cnames.index(c) + 1) for g, c in zip(cparts.geometry, cparts.country)), grid)
    nc = len(cnames) + 1
    ecty = pair_counts(eco, ctry, n, nc)
    prow, pcol, people = population_cells(cfg, paths, grid)
    pop = np.bincount(eco[prow, pcol], weights=people, minlength=n)
    sub = (slice(None, None, 3), slice(None, None, 3))  # elevation percentiles on every 3rd cell
    e_s, d_s = eco[sub].ravel(), read(paths.raster("dem"))[sub].ravel()
    ok = e_s > 0
    elev = pd.Series(d_s[ok]).groupby(e_s[ok]).quantile([0.1, 0.5, 0.9]).unstack()
    att = pd.read_csv(paths.interim / "l3_attributes.csv", dtype={"l2_code": str})
    l2_of_unit = np.zeros(n_units, np.int32)
    l2_of_unit[att.unit.to_numpy()] = att.l2_code.astype(int).to_numpy()
    n_l2 = int(l2_of_unit.max()) + 1
    el2 = mode_per_label(eco, l2_of_unit[l3], n, n_l2)
    l2name = att.drop_duplicates("l2_code").set_index(att.drop_duplicates("l2_code").l2_code.astype(int)).l2_name
    short = lambda s: str(s).split(" (")[0]  # noqa: E731

    rows = []
    for u in range(1, n):
        t = tkeys[etype[u] - 1]
        top = np.argsort(-ccount[u])[:3]
        mix = "; ".join(f"{comms[j]['name']} {100 * ccount[u, j] / ccount[u].sum():.0f}%" for j in top if ccount[u, j] > 0)
        fr_all = [int(j) for j in np.argsort(-fcount[u]) if fcount[u, j] >= 0.1 * fcount[u].sum() and fcount[u, j] > 0]
        rows.append(dict(unit=u, type=t, type_name=types[t]["name"], biome=types[t]["biome"], faunal_region=int(efau[u]),
                         faunal_share=round(fshare[u], 3), faunal_regions=" ".join(map(str, fr_all)),
                         l2_code=f"{el2[u]:02d}", l2_name=short(l2name.get(int(el2[u]), "")), area_km2=round(area[u], 1),
                         elev_p10=round(elev.loc[u, 0.1]) if u in elev.index else np.nan,
                         elev_median=round(elev.loc[u, 0.5]) if u in elev.index else np.nan,
                         elev_p90=round(elev.loc[u, 0.9]) if u in elev.index else np.nan,
                         type_purity=round(purity[u], 3), fallback_share=round(fb[u] / max(cells[u], 1), 3),
                         converted_share=round(conv[u] / max(cells[u], 1), 3), population=int(round(pop[u])),
                         countries=country_mix(ecty[u], cnames), community_mix=mix))
    df = pd.DataFrame(rows)
    clim = {}  # column -> (sums, counts) per ecoregion, per faunal region
    for col, r in CLIMATE.items():
        v = read(paths.raster(r))
        clim[col] = (label_sums(eco, v, n), label_sums(par90, v, nf))
        del v
    mean = lambda s, c, col: np.round(np.where(c > 0, s / np.maximum(c, 1), np.nan), CLIMATE_DIGITS[col])  # noqa: E731
    for col, ((s, c), _) in clim.items():
        df[col] = mean(s, c, col)[df.unit]
    df["aridity_class"] = df.aridity.map(aridity_class)
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
    tnum = {k: i + 1 for i, k in enumerate(tkeys)}
    df = df.sort_values(["type", "area_km2"], ascending=[True, False])
    df["code"] = [f"{tnum[t]:02d}.{i:02d}" for t, i in zip(df["type"], df.groupby("type").cumcount() + 1)]
    df = df.sort_values("unit")

    # faunal regions: their own polygons, named after the provinces they cover; grouped by the
    # coarser clustering (``faunal_group_k``), which nests them exactly (UPGMA)
    frp = polygonize(par90, grid).rename(columns={"unit": "faunal_region"})
    pc = pair_counts(par90, l2_of_unit[l3], nf, n_l2)
    pnames = {int(k): short(v) for k, v in l2name.items()}
    regions = {int(p["code"]): p["region"] for p in cfg["provinces"]["units"] if "region" in p}
    fr_names = pd.Series({i: place_name(pc[i], pnames, regions) for i in frp.faunal_region})
    fc = frp.set_index("faunal_region").geometry.centroid
    fr_names = disambiguate(fr_names, fc.x.reindex(fr_names.index), fc.y.reindex(fr_names.index))
    coarse = (read(paths.interim / "faunal" / f"regions_k{ac['faunal_group_k']}.tif") if faunal else fau).astype(np.int64)
    both = (fau > 0) & (coarse > 0)
    parent = pd.crosstab(fau[both], coarse[both]).idxmax(axis=1)
    frp["group"] = frp.faunal_region.map(parent).astype(int)
    gnames = pd.Series({g: place_name(pc[frp.faunal_region[frp.group == g].to_numpy()].sum(axis=0), pnames, regions)
                        for g in frp.group.unique()})
    gc = frp.dissolve("group").geometry.centroid  # two groups can win the same place name, as regions can
    gnames = disambiguate(gnames, gc.x.reindex(gnames.index), gc.y.reindex(gnames.index))
    frp["group_name"] = frp.group.map(gnames)
    df["faunal_name"] = df.faunal_region.map(fr_names)
    if not faunal:
        df[["faunal_region", "faunal_share", "faunal_regions", "faunal_name"]] = [0, np.nan, "", ""]
        fr_names = pd.Series(dtype=str)
    df = df.drop(columns=["cx", "cy"])
    check_invariants(eco, etype, area, mmu_px * km2, df.code, df.name)  # merge_small_units works in whole cells
    df.to_csv(paths.interim / "ecoregions.csv", index=False)

    # ecoregion types as wholes (each type with all its tracts)
    typ = etype[eco].astype("uint8")
    typ[eco == 0] = 0
    nt = len(tkeys) + 1
    e_t = typ[sub].ravel()
    tel = pd.Series(d_s[e_t > 0]).groupby(e_t[e_t > 0]).quantile([0.1, 0.5, 0.9]).unstack()
    et = etype[1:]  # label 0 is outside the study area
    tcc = np.zeros((nt, ccount.shape[1]))
    np.add.at(tcc, et, ccount[1:])
    tfc = np.zeros((nt, nf))
    np.add.at(tfc, et, fcount[1:])
    tcty = np.zeros((nt, nc))
    np.add.at(tcty, et, ecty[1:])
    tpop, tconv, tcells = (np.bincount(et, weights=w[1:], minlength=nt) for w in (pop, conv, cells))
    tclim = {col: mean(np.bincount(et, weights=s[1:], minlength=nt), np.bincount(et, weights=c[1:], minlength=nt), col)
             for col, ((s, c), _) in clim.items()}
    trows = []
    for i in np.unique(typ[typ > 0]):
        t = tkeys[i - 1]
        cc_ = tcc[i]
        fr = [fr_names.get(int(j), "") for j in np.argsort(-tfc[i]) if tfc[i, j] >= 0.1 * tfc[i].sum() and tfc[i, j] > 0]
        trows.append(dict(unit=int(i), type=t, type_name=types[t]["name"], biome=types[t]["biome"],
                          area_km2=round(tcells[i] * km2, 1), population=int(round(tpop[i])),
                          converted_share=round(tconv[i] / max(tcells[i], 1), 3), countries=country_mix(tcty[i], cnames),
                          elev_p10=round(tel.loc[i, 0.1]), elev_median=round(tel.loc[i, 0.5]), elev_p90=round(tel.loc[i, 0.9]),
                          community_mix="; ".join(f"{comms[j]['name']} {100 * cc_[j] / cc_.sum():.0f}%"
                                                  for j in np.argsort(-cc_)[:4] if cc_[j] > 0),
                          faunal_names="; ".join(dict.fromkeys(fr)), n_ecoregions=int((df.type == t).sum()),
                          **{col: tclim[col][i] for col in CLIMATE}, aridity_class=aridity_class(tclim["aridity"][i])))
    tgdf = polygonize(typ, grid).merge(pd.DataFrame(trows), on="unit").drop(columns="unit")
    del typ

    gdf = poly.merge(df, on="unit")
    out = paths.vector("ecoregions")
    if out.exists():
        out.unlink()
    gdf.to_file(out, layer="ecoregions")
    if faunal:
        frp["faunal_name"] = frp.faunal_region.map(fr_names)
        frp["area_km2"] = (frp.area / 1e6).round(1)
        frp["n_ecoregions"] = frp.faunal_region.map(df.faunal_regions.str.split().explode().astype(int).value_counts()).fillna(0).astype(int)
        frp["population"] = frp.faunal_region.map(pd.Series(np.bincount(par90[prow, pcol], weights=people, minlength=nf))).round().astype(int)
        fconv = pair_counts(par90, cv, nf, 2)
        frp["converted_share"] = frp.faunal_region.map(pd.Series(fconv[:, 1] / np.maximum(fconv.sum(axis=1), 1))).round(3)
        fcty = pair_counts(par90, ctry, nf, nc)
        frp["countries"] = frp.faunal_region.map(lambda f: country_mix(fcty[f], cnames))
        for col, (_, (s, c)) in clim.items():
            frp[col] = frp.faunal_region.map(pd.Series(mean(s, c, col)))
        frp["aridity_class"] = frp.aridity.map(aridity_class)
        frp.to_file(out, layer="faunal_regions")
    del ctry
    tgdf.to_file(out, layer="ecoregion_types")
    gdf.dissolve("biome", aggfunc={"area_km2": "sum"}).reset_index().to_file(out, layer="biomes")
    log.info("assemble: %d ecoregions (median %.0f km²), %d biomes; %d faunal regions as overlay; "
             "%d ecoregions span more than one faunal region",
             len(df), df.area_km2.median(), df.biome.nunique(), len(frp) if faunal else 0,
             (df.faunal_regions.str.count(" ") > 0).sum())
    log.info("\n%s", df.groupby("type_name").agg(units=("unit", "count"), km2=("area_km2", "sum")).round(0).to_string())

    from .qa import label_map

    label_map(eco, paths.qa / "40_ecoregions.png", f"community ecoregions ({len(df)})")
