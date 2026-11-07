"""Stage 4 — hydrology derivatives on the 90 m grid.

1. **Conditioning.** HydroRIVERS reaches with a large upstream area are burned
   into the DEM, but only on low-relief terrain where a 90 m DSM cannot resolve
   flow direction by itself (the Indus plains); in the mountains the DEM is
   trusted.
2. **Flow directions** from a priority-flood (Barnes et al. 2014) that assigns
   D8 directions as it floods: depressions are filled implicitly and flats drain
   breadth-first toward their spill point. Endorheic basins are routed across
   their (filled) lowest point to the domain edge.
3. **Flow accumulation, Strahler order, stream links and catchment divides**
   from one topological ordering of the D8 graph.
4. **HAND** (height above nearest drainage, Rennó et al. 2008) along D8 flow
   paths to the nearest channel cell.
5. **REM** — on the alluvial plains flow-path HAND is ill-posed (the regional
   slope runs *parallel* to the rivers, so paths travel tens of km before
   meeting a channel). There we use a relative elevation model: DSM (median-
   filtered) minus the water-surface elevation of the nearest major river.
6. **HAND classes**: active floodplain / older meander floodplain / bar upland /
   upland (thresholds in ``config.yaml``).

All routing kernels are written in numba and operate on flat int32 indices so a
~300 M cell grid fits in ~7 GB of RAM.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from numba import njit
from scipy import ndimage as ndi

from . import vectors
from .config import Paths
from .grid import Grid, block_reduce, rasterize_geoms, read, write
from .qa import quicklook, zoom

log = logging.getLogger(__name__)

# D8 neighbour offsets, clockwise from east; opposite of k is (k + 4) % 8
DR = np.array([0, 1, 1, 1, 0, -1, -1, -1], dtype=np.int64)
DC = np.array([1, 1, 0, -1, -1, -1, 0, 1], dtype=np.int64)
OUTLET = np.uint8(254)
NODATA = np.uint8(255)

HAND_CLASS_CODES = {"active_floodplain": 1, "meander_floodplain": 2, "bar_upland": 3, "upland": 4}


# ---------------------------------------------------------------------------
# numba kernels
# ---------------------------------------------------------------------------
@njit(cache=True)
def priority_flood_d8(z, valid):
    """Priority-flood with D8 direction assignment.

    Parameters
    ----------
    z : 2-D float32, conditioned elevations (any value where ``valid`` is False)
    valid : 2-D bool, cells to route

    Returns
    -------
    dirs : 1-D uint8 (flattened), 0..7 = neighbour index, 254 = outlet, 255 = invalid
    """
    H, W = z.shape
    N = H * W
    zf = z.ravel().copy()
    vf = valid.ravel()
    dirs = np.full(N, 255, np.uint8)
    closed = np.zeros(N, np.uint8)

    cap = 1 << 22
    hk = np.empty(cap, np.float32)
    hc = np.empty(cap, np.int64)
    hv = np.empty(cap, np.int32)
    hs = 0
    counter = 0

    qcap = 1 << 22
    q = np.empty(qcap, np.int32)
    qh = 0
    qt = 0

    # seed: valid cells on the grid edge or next to an invalid cell are outlets
    for r in range(H):
        for c in range(W):
            i = r * W + c
            if not vf[i]:
                continue
            edge = r == 0 or c == 0 or r == H - 1 or c == W - 1
            if not edge:
                for k in range(8):
                    if not vf[(r + DR[k]) * W + c + DC[k]]:
                        edge = True
                        break
            if edge:
                dirs[i] = 254
                closed[i] = 1
                if hs == cap:
                    cap *= 2
                    hk2 = np.empty(cap, np.float32); hk2[:hs] = hk[:hs]; hk = hk2
                    hc2 = np.empty(cap, np.int64); hc2[:hs] = hc[:hs]; hc = hc2
                    hv2 = np.empty(cap, np.int32); hv2[:hs] = hv[:hs]; hv = hv2
                # push
                j = hs
                hk[j] = zf[i]; hc[j] = counter; hv[j] = i
                counter += 1
                hs += 1
                while j > 0:
                    p = (j - 1) >> 1
                    if hk[p] > hk[j] or (hk[p] == hk[j] and hc[p] > hc[j]):
                        hk[p], hk[j] = hk[j], hk[p]
                        hc[p], hc[j] = hc[j], hc[p]
                        hv[p], hv[j] = hv[j], hv[p]
                        j = p
                    else:
                        break

    while hs > 0 or qh < qt:
        if qh < qt:
            ci = q[qh]
            qh += 1
            if qh == qt:
                qh = 0
                qt = 0
        else:
            ci = hv[0]
            hs -= 1
            if hs > 0:
                hk[0] = hk[hs]; hc[0] = hc[hs]; hv[0] = hv[hs]
                j = 0
                while True:
                    l = 2 * j + 1
                    if l >= hs:
                        break
                    m = l
                    rr = l + 1
                    if rr < hs and (hk[rr] < hk[l] or (hk[rr] == hk[l] and hc[rr] < hc[l])):
                        m = rr
                    if hk[m] < hk[j] or (hk[m] == hk[j] and hc[m] < hc[j]):
                        hk[m], hk[j] = hk[j], hk[m]
                        hc[m], hc[j] = hc[j], hc[m]
                        hv[m], hv[j] = hv[j], hv[m]
                        j = m
                    else:
                        break
        r = ci // W
        c = ci - r * W
        zc = zf[ci]
        for k in range(8):
            rn = r + DR[k]
            cn = c + DC[k]
            if rn < 0 or cn < 0 or rn >= H or cn >= W:
                continue
            n = rn * W + cn
            if closed[n] or not vf[n]:
                continue
            closed[n] = 1
            dirs[n] = (k + 4) % 8
            if zf[n] <= zc:
                zf[n] = zc
                if qt == qcap:
                    if qh > 0:  # compact
                        q[: qt - qh] = q[qh:qt]
                        qt -= qh
                        qh = 0
                    if qt == qcap:
                        qcap *= 2
                        q2 = np.empty(qcap, np.int32)
                        q2[:qt] = q[:qt]
                        q = q2
                q[qt] = n
                qt += 1
            else:
                if hs == cap:
                    cap *= 2
                    hk2 = np.empty(cap, np.float32); hk2[:hs] = hk[:hs]; hk = hk2
                    hc2 = np.empty(cap, np.int64); hc2[:hs] = hc[:hs]; hc = hc2
                    hv2 = np.empty(cap, np.int32); hv2[:hs] = hv[:hs]; hv = hv2
                j = hs
                hk[j] = zf[n]; hc[j] = counter; hv[j] = n
                counter += 1
                hs += 1
                while j > 0:
                    p = (j - 1) >> 1
                    if hk[p] > hk[j] or (hk[p] == hk[j] and hc[p] > hc[j]):
                        hk[p], hk[j] = hk[j], hk[p]
                        hc[p], hc[j] = hc[j], hc[p]
                        hv[p], hv[j] = hv[j], hv[p]
                        j = p
                    else:
                        break
    return dirs


@njit(cache=True)
def downstream_index(i, d, W):
    return i + DR[d] * W + DC[d]


@njit(cache=True)
def topological_order(dirs, W):
    """Kahn ordering: every cell appears after all cells draining into it."""
    N = dirs.size
    indeg = np.zeros(N, np.uint8)
    for i in range(N):
        d = dirs[i]
        if d < 8:
            indeg[downstream_index(i, d, W)] += 1
    order = np.empty(N, np.int32)
    n = 0
    for i in range(N):
        if dirs[i] != 255 and indeg[i] == 0:
            order[n] = i
            n += 1
    head = 0
    while head < n:
        ci = order[head]
        head += 1
        d = dirs[ci]
        if d < 8:
            j = downstream_index(ci, d, W)
            indeg[j] -= 1
            if indeg[j] == 0:
                order[n] = j
                n += 1
    return order[:n]


@njit(cache=True)
def accumulate(dirs, order, W):
    acc = np.zeros(dirs.size, np.float32)
    for t in range(order.size):
        ci = order[t]
        acc[ci] += 1.0
        d = dirs[ci]
        if d < 8:
            acc[downstream_index(ci, d, W)] += acc[ci]
    return acc


@njit(cache=True)
def strahler_and_links(dirs, order, stream, W):
    """Strahler order and stream-link ids (a link runs between junctions)."""
    N = dirs.size
    so = np.zeros(N, np.uint8)
    maxin = np.zeros(N, np.uint8)
    nmax = np.zeros(N, np.uint8)
    nin = np.zeros(N, np.uint8)
    link = np.zeros(N, np.int32)
    inlink = np.zeros(N, np.int32)
    nlinks = 0
    for t in range(order.size):
        ci = order[t]
        if not stream[ci]:
            continue
        if maxin[ci] == 0:
            so[ci] = 1
        else:
            so[ci] = maxin[ci] + (1 if nmax[ci] >= 2 else 0)
        if nin[ci] == 1:
            link[ci] = inlink[ci]
        else:
            nlinks += 1
            link[ci] = nlinks
        d = dirs[ci]
        if d < 8:
            j = downstream_index(ci, d, W)
            if stream[j]:
                nin[j] += 1
                inlink[j] = link[ci]
                if so[ci] > maxin[j]:
                    maxin[j] = so[ci]
                    nmax[j] = 1
                elif so[ci] == maxin[j]:
                    nmax[j] += 1
    return so, link


@njit(cache=True)
def hand_flowpath(dirs, order, stream, z, W):
    """Height above the first channel cell reached along the D8 path."""
    ref = np.full(dirs.size, np.nan, np.float32)
    for t in range(order.size - 1, -1, -1):
        ci = order[t]
        if stream[ci]:
            ref[ci] = z[ci]
        else:
            d = dirs[ci]
            if d < 8:
                ref[ci] = ref[downstream_index(ci, d, W)]
            else:
                ref[ci] = z[ci]
    out = np.full(dirs.size, np.nan, np.float32)
    for i in range(dirs.size):
        if dirs[i] != 255:
            out[i] = z[i] - ref[i]
    return out


@njit(cache=True)
def basins_from_links(dirs, order, link, keep, W):
    """Label each cell with the id of the first *kept* stream link downstream."""
    bas = np.zeros(dirs.size, np.int32)
    for t in range(order.size - 1, -1, -1):
        ci = order[t]
        if link[ci] > 0 and keep[ci]:
            bas[ci] = link[ci]
        else:
            d = dirs[ci]
            if d < 8:
                bas[ci] = bas[downstream_index(ci, d, W)]
    return bas


@njit(cache=True)
def harmonic_relax(u, fixed, iters, omega):
    """Red-black SOR for Laplace's equation; ``fixed`` cells are Dirichlet,
    grid edges are no-flux (mirror)."""
    H, W = u.shape
    for _ in range(iters):
        for colour in range(2):
            for r in range(H):
                for c in range((r + colour) % 2, W, 2):
                    if fixed[r, c]:
                        continue
                    up = u[r - 1, c] if r > 0 else u[r + 1, c]
                    dn = u[r + 1, c] if r < H - 1 else u[r - 1, c]
                    lf = u[r, c - 1] if c > 0 else u[r, c + 1]
                    rt = u[r, c + 1] if c < W - 1 else u[r, c - 1]
                    u[r, c] += omega * ((up + dn + lf + rt) * 0.25 - u[r, c])
    return u


def edges_of_labels(lab: np.ndarray) -> np.ndarray:
    """True where a 4-neighbour has a different (non-zero) label."""
    e = np.zeros(lab.shape, bool)
    e[:, 1:] |= (lab[:, 1:] != lab[:, :-1]) & (lab[:, 1:] > 0) & (lab[:, :-1] > 0)
    e[1:, :] |= (lab[1:, :] != lab[:-1, :]) & (lab[1:, :] > 0) & (lab[:-1, :] > 0)
    return e


# ---------------------------------------------------------------------------
# REM for the plains
# ---------------------------------------------------------------------------
def plains_mask(dem, relief5k, pc, grid: Grid) -> np.ndarray:
    m = (dem < pc["max_elev_m"]) & (relief5k < pc["max_relief5k_m"])
    f = 5  # clean at 450 m
    small = block_reduce(m.astype("float32"), f) > 0.5
    small = ndi.binary_opening(small, iterations=2)
    small = ndi.binary_closing(small, iterations=3)
    lab, n = ndi.label(small)
    sizes = ndi.sum(small, lab, index=np.arange(1, n + 1)) * (grid.res * f / 1000) ** 2
    keep = np.isin(lab, np.nonzero(sizes >= pc["min_area_km2"])[0] + 1)
    keep = ndi.binary_fill_holes(keep)
    big = np.repeat(np.repeat(keep, f, 0), f, 1)[: grid.height, : grid.width]
    return big & ~np.isnan(dem)


def relative_elevation(cfg, dem: np.ndarray, grid: Grid, rivers, plains: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """REM = DSM(median) - water surface of the nearest major river."""
    rc = cfg["hydrology"]["rem"]
    big = rivers[rivers.UPLAND_SKM >= rc["river_min_upland_km2"]]
    riv90 = rasterize_geoms([(g, 1) for g in big.geometry], grid).astype(bool)
    f = int(round(rc["reference_resolution_m"] / grid.res))
    cg = grid.coarsen(f)
    # robust channel elevation per coarse block: low percentile of DEM on river cells
    # sample river surfaces only on the plains: upstream mountain reaches would
    # pull the membrane up along every mountain front (Daman, Kachhi, Sialkot)
    on_plains = riv90 if plains is None else riv90 & plains
    zr = np.where(on_plains, dem, np.nan)
    H, W = cg.height, cg.width
    pad = np.full((H * f, W * f), np.nan, "float32")
    pad[: grid.height, : grid.width] = zr
    blocks = pad.reshape(H, f, W, f).transpose(0, 2, 1, 3).reshape(H, W, f * f)
    with np.errstate(all="ignore"):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            surf = np.nanpercentile(blocks, rc["river_surface_percentile"], axis=2).astype("float32")
    del pad, blocks, zr
    has = ~np.isnan(surf)
    # Robust along-channel smoothing of the river samples, then *harmonic*
    # interpolation (Laplace membrane with the rivers as fixed values). Unlike
    # nearest-sample or k-NN IDW lookups, this has no seams where the "nearest
    # river" switches (those produced 5-10 m faceted steps across the Thal).
    from scipy.spatial import cKDTree

    rr, cc = np.nonzero(has)
    tree = cKDTree(np.c_[rr, cc])
    zs = surf[rr, cc]
    nb = tree.query_ball_point(np.c_[rr, cc], r=rc.get("river_median_km", 8) * 1000 / cg.res, workers=-1)
    zs = np.array([np.median(zs[j]) for j in nb], dtype="float32")
    hf = rc.get("harmonic_factor", 3)
    Hh, Wh = int(np.ceil(H / hf)), int(np.ceil(W / hf))
    fixed_v = np.full((Hh, Wh), np.nan, "float32")
    # median of river samples per harmonic cell
    df = pd.DataFrame(dict(r=rr // hf, c=cc // hf, z=zs)).groupby(["r", "c"]).z.median()
    fixed_v[df.index.get_level_values(0), df.index.get_level_values(1)] = df.values
    fixed = ~np.isnan(fixed_v)
    # initial guess: nearest river value
    _, (ir, ic) = ndi.distance_transform_edt(~fixed, return_indices=True)
    u = fixed_v[ir, ic].astype("float64")
    u = harmonic_relax(u, fixed, rc.get("harmonic_iters", 3000), 1.9)
    ref = ndi.zoom(u.astype("float32"), hf, order=1, mode="nearest", grid_mode=True)[:H, :W]
    sig = rc["smooth_sigma_km"] * 1000 / cg.res
    ref = ndi.gaussian_filter(ref, sig, mode="nearest")
    # back to 90 m by bilinear zoom
    ref90 = ndi.zoom(ref, f, order=1, mode="nearest", grid_mode=True)[: grid.height, : grid.width]
    if ref90.shape != dem.shape:  # pad if ceil() rounding left us short
        tmp = np.empty(dem.shape, "float32")
        tmp[: ref90.shape[0], : ref90.shape[1]] = ref90
        ref90 = tmp
    msize = max(1, int(round(rc["dem_median_m"] / grid.res))) | 1
    zm = ndi.median_filter(np.nan_to_num(dem, nan=0.0), size=msize)
    rem = (zm - ref90).astype("float32")
    rem[np.isnan(dem)] = np.nan
    return rem, riv90


def classify_hand(h: np.ndarray, classes: dict) -> np.ndarray:
    out = np.zeros(h.shape, "uint8")
    for name, (lo, hi) in classes.items():
        out[(h >= lo) & (h < hi)] = HAND_CLASS_CODES[name]
    return out


# ---------------------------------------------------------------------------
def run(cfg) -> None:
    paths = Paths(cfg)
    grid = Grid.load(paths.grid_json)
    hc = cfg["hydrology"]
    H, W = grid.shape
    dem = read(paths.raster("dem"))
    valid = ~np.isnan(dem)
    rivers = vectors.hydrorivers(cfg)

    # 1. conditioning
    burnable = rivers[rivers.UPLAND_SKM >= hc["burn_min_upland_km2"]]
    burn = rasterize_geoms([(g, 1) for g in burnable.geometry], grid).astype(bool)
    relief1k = read(paths.raster("relief_1000"))
    burn &= relief1k < hc["burn_max_relief_m"]
    del relief1k
    zc = np.where(valid, dem, 0).astype("float32")
    zc[burn] -= hc["burn_depth_m"]
    log.info("burned %d river cells", int(burn.sum()))
    write(paths.raster("river_burn"), burn.astype("uint8"), grid)
    del burn

    # 2. flow directions
    log.info("priority-flood on %d valid cells ...", int(valid.sum()))
    dirs = priority_flood_d8(zc, valid)
    del zc
    write(paths.raster("d8"), dirs.reshape(H, W), grid, nodata=255,
          tags=dict(encoding="0..7 clockwise from E; 254 outlet; 255 nodata"))

    # 3. accumulation, order, links, divides
    order = topological_order(dirs, W)
    log.info("topological order: %d cells", order.size)
    acc = accumulate(dirs, order, W)
    acc_km2 = acc * np.float32(grid.cell_km2)
    del acc
    write(paths.raster("flowacc_km2"), acc_km2.reshape(H, W), grid, nodata=None)
    stream = acc_km2 >= hc["stream_threshold_km2"]
    so, link = strahler_and_links(dirs, order, stream, W)
    write(paths.raster("strahler"), so.reshape(H, W), grid, nodata=0)
    log.info("max Strahler order %d", int(so.max()))
    divides = np.zeros((H, W), "uint8")
    for lvl, min_order in ((1, 2), (2, 4)):  # minor and major catchment divides
        bas = basins_from_links(dirs, order, link, so >= min_order, W).reshape(H, W)
        e = edges_of_labels(bas)
        divides[e] = lvl
        del bas, e
    write(paths.raster("divides"), divides, grid, nodata=None,
          tags=dict(values="1 divide between order>=2 catchments, 2 between order>=4 catchments"))
    del link

    # 4. flow-path HAND
    zf = np.where(valid, dem, np.nan).ravel()
    hand = hand_flowpath(dirs, order, stream, zf, W).reshape(H, W)
    del order, zf
    write(paths.raster("hand_flowpath"), hand, grid, nodata=np.nan)

    # 5-6. REM on the plains, combined HAND, classes
    del hand, dem
    rem_stage(cfg)


def rem_stage(cfg) -> None:
    """Plains mask, REM and the combined HAND layer (re-runnable without routing)."""
    paths = Paths(cfg)
    grid = Grid.load(paths.grid_json)
    hc = cfg["hydrology"]
    dem = read(paths.raster("dem"))
    valid = ~np.isnan(dem)
    relief5k = read(paths.raster("relief_5000"))
    plains = plains_mask(dem, relief5k, hc["plains"], grid)
    del relief5k
    write(paths.raster("plains"), plains.astype("uint8"), grid)
    rem, _ = relative_elevation(cfg, dem, grid, vectors.hydrorivers(cfg), plains)
    write(paths.raster("rem"), rem, grid, nodata=np.nan)
    hand = read(paths.raster("hand_flowpath"))
    # REM below the floor is implausible except behind levees; there (mostly the
    # piedmont near Sialkot, where the membrane is pulled up by rivers upstream
    # in J&K) flow-path HAND is the better measure.
    use_rem = plains & (rem >= hc["rem"].get("min_valid_m", -3))
    combo = np.where(use_rem, rem, hand).astype("float32")
    del rem, hand
    combo[~valid] = np.nan
    write(paths.raster("hand"), combo, grid, nodata=np.nan,
          tags=dict(note="REM on the alluvial plains mask, flow-path HAND elsewhere"))
    del combo
    write_hand_classes(cfg)


def write_hand_classes(cfg) -> None:
    """(Re)classify the combined HAND/REM raster — cheap, so thresholds can be
    tuned without rerunning the routing (``python -c 'from ecoregions import hydro...'``)."""
    paths = Paths(cfg)
    grid = Grid.load(paths.grid_json)
    H, W = grid.shape
    combo = read(paths.raster("hand"))
    cls = classify_hand(combo, cfg["hydrology"]["hand_classes"])
    write(paths.raster("hand_class"), cls, grid, nodata=0,
          tags={str(v): k for k, v in HAND_CLASS_CODES.items()})
    acc_km2 = read(paths.raster("flowacc_km2"))
    divides = read(paths.raster("divides"))
    mask = read(paths.raster("mask"))
    quicklook(np.log10(acc_km2 + 0.01), paths.qa / "04_flowacc.png", "log10 flow accumulation (km²)",
              cmap="Blues", vmin=-1, vmax=5)
    quicklook(combo, paths.qa / "04_hand.png", "HAND (REM on plains)", cmap="viridis", vmin=0, vmax=60,
              outline=mask == 1)
    quicklook(cls, paths.qa / "04_hand_class.png", "HAND classes", cmap="viridis", categorical=True)
    quicklook(divides, paths.qa / "04_divides.png", "catchment divides", cmap="gray_r", categorical=True)
    for name, bb in {"rechna_sandal": (72.6, 31.0, 73.9, 32.0), "kirana": (72.4, 31.6, 73.3, 32.4),
                     "bari_neeli_ganji": (72.2, 29.9, 73.9, 31.1), "lower_indus": (68.0, 25.2, 68.8, 26.2)}.items():
        zoom(combo, grid, bb, paths.qa / f"04_hand_zoom_{name}.png", f"REM/HAND — {name}", cmap="viridis",
             vmin=-2, vmax=20)
        zoom(cls, grid, bb, paths.qa / f"04_hand_class_zoom_{name}.png", f"HAND classes — {name}",
             cmap="viridis", categorical=True)
