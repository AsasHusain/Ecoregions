"""Stage 9 — snap L3 boundaries to real features at 90 m.

The first-pass units (270 m) are upsampled to 90 m. Within ``band_m`` of every
provisional boundary the labels are released and re-grown by a marker-controlled
watershed over a *boundary-strength* surface, so the final line settles on the
strongest nearby physical edge:

==================  ============================================================
ridge               positive TPI (450 m): ridgelines and crests
divide              catchment divides from the 90 m D8 network (scaled by relief)
scarp               gradient of the median-filtered REM on the plains (terrace
                    scarps, bar edges, floodplain margins)
slope_break         gradient of smoothed slope (mountain fronts, piedmont toes)
geology_contact     geo8ag contacts, Gaussian-smoothed to the map's ~1:5M accuracy
rem_edge            edges between HAND classes
==================  ============================================================

Coastlines and the LoC are hard limits (the watershed is masked to the study area).
Afterwards every unit is made contiguous: detached fragments are absorbed by the
neighbour sharing the most border (or kept as new units if above the MMU).
"""
from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage.segmentation import watershed

from .config import Paths
from .grid import Grid, read, write
from .qa import zoom

log = logging.getLogger(__name__)


def _norm(a: np.ndarray, scale: float) -> np.ndarray:
    return np.clip(a / scale, 0, 1).astype("float32")


def _grad(a: np.ndarray) -> np.ndarray:
    gy, gx = np.gradient(a)
    return np.hypot(gx, gy).astype("float32")


def boundary_strength(cfg, paths: Paths, grid: Grid) -> np.ndarray:
    w = cfg["snapping"]["strength_weights"]
    B = np.zeros(grid.shape, "float32")
    relief = read(paths.raster("relief_1000"))
    relief_f = _norm(np.nan_to_num(relief), 60.0)
    del relief

    tpi = np.nan_to_num(read(paths.raster("tpi_450")))
    B += w["ridge"] * _norm(tpi, 40.0)
    del tpi

    div = read(paths.raster("divides")).astype("float32") / 2.0
    B += w["divide"] * div * relief_f
    del div

    plains = read(paths.raster("plains")).astype(bool)
    rem = np.nan_to_num(read(paths.raster("rem")))
    rem = ndi.median_filter(rem, size=3)
    g = _grad(rem)
    del rem
    B += w["scarp"] * np.where(plains, _norm(g, 1.5), 0)  # 1.5 m per 90 m cell = clear scarp
    del g

    s = np.nan_to_num(read(paths.raster("slope")))
    s = ndi.gaussian_filter(s, 2)
    B += w["slope_break"] * _norm(_grad(s), 2.0)
    del s

    sig = cfg["snapping"]["geology_contact_sigma_m"] / grid.res
    gc = ndi.gaussian_filter(read(paths.raster("geo_contact")).astype("float32"), sig)
    B += w["geology_contact"] * gc / max(gc.max(), 1e-6)
    del gc

    hc = read(paths.raster("hand_class"))
    e = np.zeros(hc.shape, "float32")
    e[:, 1:] += hc[:, 1:] != hc[:, :-1]
    e[1:, :] += hc[1:, :] != hc[:-1, :]
    B += w["rem_edge"] * np.where(plains, np.clip(ndi.gaussian_filter(e, 1.0) * 2, 0, 1), 0)
    del e, hc, plains
    return B


def upsample(lab_c: np.ndarray, f: int, shape) -> np.ndarray:
    return np.repeat(np.repeat(lab_c, f, 0), f, 1)[: shape[0], : shape[1]]


def internal_edges(lab: np.ndarray) -> np.ndarray:
    e = np.zeros(lab.shape, bool)
    d = (lab[:, 1:] != lab[:, :-1]) & (lab[:, 1:] > 0) & (lab[:, :-1] > 0)
    e[:, 1:] |= d
    e[:, :-1] |= d
    d = (lab[1:, :] != lab[:-1, :]) & (lab[1:, :] > 0) & (lab[:-1, :] > 0)
    e[1:, :] |= d
    e[:-1, :] |= d
    return e


def tiled_watershed(B, markers, mask, grid: Grid, size: int, overlap: int) -> np.ndarray:
    out = markers.copy()
    for read_w, core, inner in grid.tiles(size, overlap):
        rs = slice(read_w.row_off, read_w.row_off + read_w.height)
        cs = slice(read_w.col_off, read_w.col_off + read_w.width)
        m = mask[rs, cs]
        mk = markers[rs, cs]
        if not (m & (mk == 0)).any():
            continue
        if not (mk > 0).any():
            continue
        lab = watershed(B[rs, cs], markers=mk, mask=m)
        ir = slice(inner.row_off, inner.row_off + inner.height)
        ic = slice(inner.col_off, inner.col_off + inner.width)
        out[core.row_off:core.row_off + core.height, core.col_off:core.col_off + core.width] = lab[ir, ic]
    return out


def make_contiguous(lab: np.ndarray, min_new_px: int) -> tuple[np.ndarray, int]:
    """Keep each unit's largest 8-connected part (queen, as in the first pass);
    absorb smaller detached parts into the neighbour sharing the most border,
    or promote parts of at least ``min_new_px`` to new units."""
    from skimage.measure import label as cc_label

    comp = cc_label(lab, connectivity=2, background=0)
    ncomp = comp.max()
    sizes = np.bincount(comp.ravel(), minlength=ncomp + 1)
    comp_unit = np.zeros(ncomp + 1, lab.dtype)
    # unit of each component (sample one pixel per component)
    flat_c = comp.ravel()
    flat_l = lab.ravel()
    idx = np.flatnonzero(flat_c)
    comp_unit[flat_c[idx]] = flat_l[idx]
    del idx
    best = pd.Series(sizes[1:], index=np.arange(1, ncomp + 1)).groupby(comp_unit[1:]).idxmax()
    keep = np.zeros(ncomp + 1, bool)
    keep[best.values] = True
    frag = np.flatnonzero(~keep[1:]) + 1
    frag = frag[np.argsort(sizes[frag])]
    slices = ndi.find_objects(comp)
    next_id = int(lab.max()) + 1
    absorbed = promoted = 0
    for c in frag:
        sl = slices[c - 1]
        sl = tuple(slice(max(0, s.start - 1), s.stop + 1) for s in sl)
        cm = comp[sl] == c
        ring = ndi.binary_dilation(cm, structure=np.ones((3, 3))) & ~cm
        nb = lab[sl][ring]
        nb = nb[(nb > 0) & (nb != comp_unit[c])]
        if sizes[c] >= min_new_px:
            lab[sl][cm] = next_id
            next_id += 1
            promoted += 1
        elif nb.size:
            vals, cnt = np.unique(nb, return_counts=True)
            lab[sl][cm] = vals[np.argmax(cnt)]
            absorbed += 1
        # else: an island with no neighbour (offshore island) keeps its unit
    log.info("contiguity: %d fragments absorbed, %d promoted to new units", absorbed, promoted)
    return lab, next_id - 1


def merge_small_units(lab: np.ndarray, min_px: dict[int, int]) -> np.ndarray:
    """Merge units smaller than their family MMU into the neighbour sharing the
    longest border (smallest first). ``min_px`` maps unit id -> threshold."""
    cnt = np.bincount(lab.ravel())
    small = [u for u, m in min_px.items() if 0 < cnt[u] < m]
    slices = ndi.find_objects(lab)
    for u in sorted(small, key=lambda u: cnt[u]):
        sl = slices[u - 1]
        if sl is None:
            continue
        sl = tuple(slice(max(0, a.start - 1), a.stop + 1) for a in sl)
        cm = lab[sl] == u
        if not cm.any():
            continue
        ring = ndi.binary_dilation(cm, structure=np.ones((3, 3))) & ~cm
        nb = lab[sl][ring]
        nb = nb[(nb > 0) & (nb != u)]
        if nb.size:
            vals, c = np.unique(nb, return_counts=True)
            lab[sl][cm] = vals[np.argmax(c)]
    log.info("MMU: %d units below their minimum mapping unit merged", len(small))
    return lab


def assign_l2(cfg, l3: np.ndarray | None = None) -> pd.DataFrame:
    """Each final L3 unit's parent L2 = majority L2 province by area (so units
    promoted from fragments, or nudged across an L2 line by snapping, are
    always assigned)."""
    paths = Paths(cfg)
    f = cfg["grid"]["coarse_factor"]
    if l3 is None:
        l3 = read(paths.raster("l3"))
    l3c = l3[1::f, 1::f].astype(np.int64)  # centre pixel of each coarse cell
    l2 = read(paths.coarse("l2"))
    h, w = min(l3c.shape[0], l2.shape[0]), min(l3c.shape[1], l2.shape[1])
    l3c, l2 = l3c[:h, :w], l2[:h, :w]
    m = (l3c > 0) & (l2 > 0)
    tab = pd.crosstab(l3c[m], l2[m])
    ids = np.unique(l3[l3 > 0])
    major = tab.idxmax(axis=1).reindex(ids)
    if major.isna().any():  # tiny units with no coarse-centre sample: nearest neighbour fill
        major = major.ffill().bfill()
    codes = json.loads((paths.interim / "province" / "l2_codes.json").read_text())
    out = pd.DataFrame(dict(unit=ids, l2_id=major.values.astype(int)))
    out["l2_code"] = out.l2_id.map(lambda i: codes[str(int(i))]["code"])
    out["l2_share"] = [tab.loc[u].max() / tab.loc[u].sum() if u in tab.index else np.nan for u in ids]
    out.to_csv(paths.interim / "l3_units.csv", index=False)
    log.info("final: %d L3 units (median L2 share %.2f)", len(out), out.l2_share.median())
    return out


def run(cfg) -> None:
    paths = Paths(cfg)
    grid = Grid.load(paths.grid_json)
    f = cfg["grid"]["coarse_factor"]
    sc = cfg["snapping"]
    study = read(paths.raster("mask")) == 1
    lab = upsample(read(paths.coarse("l3_first")), f, grid.shape).astype(np.int32)
    lab[~study] = 0
    band = int(round(sc["band_m"] / grid.res))
    edges = internal_edges(lab)
    unknown = ndi.maximum_filter(edges.astype("uint8"), size=2 * band + 1).astype(bool)
    del edges
    unknown |= study & (lab == 0)
    markers = np.where(unknown, 0, lab).astype(np.int32)
    del unknown
    B = boundary_strength(cfg, paths, grid)
    write(paths.raster("boundary_strength"), B, grid, nodata=None)
    log.info("watershed refinement in a ±%d px band", band)
    snapped = tiled_watershed(B, markers, study, grid, cfg["grid"]["tile_size"], 4 * band + 64)
    del markers
    # any study cell still unlabelled (isolated by the mask) takes the provisional label
    hole = study & (snapped == 0)
    snapped[hole] = lab[hole]
    moved = (snapped != lab) & study
    log.info("snapping moved %.2f%% of study cells", 100 * moved.sum() / study.sum())
    del moved
    del lab
    promote_px = int(sc["promote_min_km2"] / grid.cell_km2)
    snapped, _ = make_contiguous(snapped, promote_px)
    # final minimum-mapping-unit pass, per L2 family
    units = assign_l2(cfg, snapped)
    codes = json.loads((paths.interim / "province" / "l2_codes.json").read_text())
    fam = {v["code"]: v["family"] for v in codes.values()}
    mmu = cfg["segmentation"]["mmu_km2"]
    min_px = {int(u): int(mmu[fam[c]] / grid.cell_km2) for u, c in zip(units.unit, units.l2_code)}
    snapped = merge_small_units(snapped, min_px)
    # compact ids 1..n
    ids = np.unique(snapped[snapped > 0])
    lut = np.zeros(int(snapped.max()) + 1, np.int32)
    lut[ids] = np.arange(1, ids.size + 1)
    snapped = lut[snapped]
    write(paths.raster("l3"), snapped.astype("uint16"), grid, nodata=0)
    assign_l2(cfg, snapped)
    edges = internal_edges(snapped)
    hs = np.nan_to_num(read(paths.raster("tpi_450")))
    img = np.where(edges, np.nan, hs)
    for name, bb in {"punjab_bars": (72.0, 30.0, 74.0, 32.4), "salt_range": (71.6, 32.4, 73.6, 33.9),
                     "quetta": (66.5, 29.6, 67.8, 30.8), "gilgit": (73.8, 35.2, 75.8, 36.6)}.items():
        zoom(img, grid, bb, paths.qa / f"09_l3_snapped_{name}.png", f"snapped L3 — {name}", cmap="gray", vmin=-30, vmax=30)
