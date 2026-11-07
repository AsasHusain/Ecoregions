"""Stage 8 — first-pass L3 delineation on the 270 m grid.

Within each L2 province independently:

1. Features are z-scored *within the province* and weighted by the province
   family's weights (``config.yaml: segmentation.weights``), so e.g. a 5 m
   rise above the rivers matters in the Punjab but not in the Karakoram.
2. SLIC superpixels (Achanta et al. 2012) over the weighted feature cube give
   compact, feature-aligned building blocks of roughly ``superpixel_km2``.
3. Superpixels are merged by Ward agglomerative clustering with a contiguity
   constraint (only adjacent superpixels may merge) to ``target_units``.
4. Units below the minimum mapping unit are merged into their most similar
   neighbour.

Boundaries are refined at 90 m in the next stage (``snap``).
"""
from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from scipy import sparse
from skimage.segmentation import slic
from sklearn.cluster import AgglomerativeClustering

from .config import Paths
from .covariates import LITHO_GROUPS
from .grid import Grid, read, write
from .qa import quicklook

log = logging.getLogger(__name__)


def province_features(paths: Paths, weights: dict, win, pmask: np.ndarray) -> np.ndarray:
    """Weighted feature cube for one province window, standardized within the province."""
    d = paths.interim / "coarse"
    cols = []
    for name, w in weights.items():
        if not w:
            continue
        if name == "basin":
            # one-hot of the basins that cover >= 1% of the province; a change of
            # basin then contributes distance w, like a change of lithology
            b = read(d / "basin.tif", window=win)
            ids, cnt = np.unique(b[pmask & (b > 0)], return_counts=True)
            for i in ids[cnt >= 0.01 * pmask.sum()]:
                cols.append(((b == i) * (w / np.sqrt(2))).astype("float32"))
            continue
        names = [f"litho_{k}" for k in LITHO_GROUPS] if name == "litho" else [name]
        for n in names:
            a = read(d / f"{n}.tif", window=win)
            v = a[pmask]
            if name == "litho":
                z = np.nan_to_num(a, nan=0.0) * (w / np.sqrt(2))
            else:
                mu, sd = np.nanmean(v), np.nanstd(v)
                z = (a - mu) / (sd if sd > 1e-6 else 1.0)
                z = np.where(np.isnan(z), 0.0, z) * w
            cols.append(z.astype("float32"))
    X = np.stack(cols, axis=-1)
    X[~pmask] = 0
    return X


def adjacency(lab: np.ndarray, n: int) -> sparse.csr_matrix:
    """Queen-contiguity adjacency between integer labels 1..n (0 = background)."""
    pairs = []
    for a, b in ((lab[:, :-1], lab[:, 1:]), (lab[:-1, :], lab[1:, :]),
                 (lab[:-1, :-1], lab[1:, 1:]), (lab[:-1, 1:], lab[1:, :-1])):
        m = (a != b) & (a > 0) & (b > 0)
        pairs.append(np.stack([a[m], b[m]], 1))
    p = np.unique(np.concatenate(pairs), axis=0) - 1
    A = sparse.coo_matrix((np.ones(len(p)), (p[:, 0], p[:, 1])), shape=(n, n))
    return ((A + A.T) > 0).astype("int8").tocsr()


def label_means(lab: np.ndarray, X: np.ndarray, n: int) -> np.ndarray:
    idx = lab.ravel()
    cnt = np.bincount(idx, minlength=n + 1)[1:].astype("float64")
    out = np.empty((n, X.shape[-1]))
    for k in range(X.shape[-1]):
        out[:, k] = np.bincount(idx, weights=X[..., k].ravel(), minlength=n + 1)[1:] / np.maximum(cnt, 1)
    return out


def merge_small(lab: np.ndarray, X: np.ndarray, min_cells: int) -> np.ndarray:
    """Iteratively merge units smaller than ``min_cells`` into the most similar neighbour."""
    lab = lab.copy()
    while True:
        ids, counts = np.unique(lab[lab > 0], return_counts=True)
        small = ids[counts < min_cells]
        if small.size == 0 or ids.size <= 1:
            break
        # relabel to 1..n for the helpers
        remap = np.zeros(lab.max() + 1, np.int64)
        remap[ids] = np.arange(1, ids.size + 1)
        l2 = remap[lab]
        n = ids.size
        A = adjacency(l2, n).tolil()
        M = label_means(l2, X, n)
        target = {}
        for s in small[np.argsort(counts[np.isin(ids, small)])]:
            i = remap[s] - 1
            nb = A.rows[i]
            if not nb:
                continue
            d = np.linalg.norm(M[nb] - M[i], axis=1)
            target[s] = ids[nb[int(np.argmin(d))]]
            break  # merge one at a time so neighbour means stay current
        if not target:
            break
        for s, t in target.items():
            lab[lab == s] = t
    return lab


def segment_province(paths, cfg, pid: int, info: dict, l2: np.ndarray, valid: np.ndarray, cg: Grid):
    sc = cfg["segmentation"]
    fam = info["family"]
    weights = dict(sc["weights"]["default"])
    if fam in sc["weights"]:
        weights = dict(sc["weights"][fam])
    pmask_full = valid & (l2 == pid)
    rows, cols = np.nonzero(pmask_full)
    r0, r1, c0, c1 = rows.min(), rows.max() + 1, cols.min(), cols.max() + 1
    from rasterio.windows import Window

    win = Window(c0, r0, c1 - c0, r1 - r0)
    pm = pmask_full[r0:r1, c0:c1]
    X = province_features(paths, weights, win, pm)
    cell_km2 = cg.cell_km2
    area = pm.sum() * cell_km2
    n_sp = max(2, int(area / sc["superpixel_km2"][fam]))
    sp = slic(X, n_segments=n_sp, compactness=sc["slic_compactness"], mask=pm, channel_axis=-1,
              start_label=1, enforce_connectivity=True, max_num_iter=10, convert2lab=False)
    sp[~pm] = 0
    # SLIC can leave unlabeled masked cells; attach them to the nearest superpixel
    hole = pm & (sp == 0)
    if hole.any():
        _, (ir, ic) = ndi.distance_transform_edt(sp == 0, return_indices=True)
        sp[hole] = sp[ir[hole], ic[hole]]
    ids = np.unique(sp[sp > 0])
    remap = np.zeros(sp.max() + 1, np.int64)
    remap[ids] = np.arange(1, ids.size + 1)
    sp = remap[sp]
    n = ids.size
    A = adjacency(sp, n)
    M = label_means(sp, X, n)
    k = int(min(sc["target_units"][info["code"]], n))
    ncomp, comp = sparse.csgraph.connected_components(A, directed=False)
    if ncomp > 1:
        # islands / detached fragments: cluster the main body; fragments join the nearest unit later
        log.info("province %s: %d disconnected parts (largest kept for clustering)", info["code"], ncomp)
    agg = AgglomerativeClustering(n_clusters=k, linkage="ward", connectivity=A).fit(M)
    lab = np.zeros_like(sp)
    lab[sp > 0] = agg.labels_[sp[sp > 0] - 1] + 1
    # enforce contiguity: split any cluster into its connected parts (queen)
    parts = np.zeros_like(lab)
    nxt = 0
    for u in np.unique(lab[lab > 0]):
        cc, m = ndi.label(lab == u, structure=np.ones((3, 3)))
        parts[cc > 0] = cc[cc > 0] + nxt
        nxt += m
    mmu_cells = int(np.ceil(sc["mmu_km2"][fam] / cell_km2))
    lab = merge_small(parts, X, mmu_cells)
    full = np.zeros(cg.shape, np.int32)
    full[r0:r1, c0:c1][pm] = lab[pm]
    log.info("province %s %-55s %7.0f km² -> %4d superpixels -> %3d units", info["code"], info["name"][:55],
             area, n, len(np.unique(lab[lab > 0])))
    return full, sp


def run(cfg) -> None:
    paths = Paths(cfg)
    grid = Grid.load(paths.grid_json)
    cg = grid.coarsen(cfg["grid"]["coarse_factor"])
    l2 = read(paths.coarse("l2"))
    valid = (l2 > 0) & (read(paths.interim / "coarse" / "study_frac.tif") >= 0.5)
    codes = {int(k): v for k, v in json.loads((paths.interim / "province" / "l2_codes.json").read_text()).items()}
    out = np.zeros(cg.shape, np.int32)
    sp_all = np.zeros(cg.shape, np.int32)
    rows = []
    nid = 0
    sp_off = 0
    for pid, info in codes.items():
        if not (valid & (l2 == pid)).any():
            log.warning("province %s has no cells", info["code"])
            continue
        lab, sp = segment_province(paths, cfg, pid, info, l2, valid, cg)
        for u in np.unique(lab[lab > 0]):
            nid += 1
            out[lab == u] = nid
            rows.append(dict(unit=nid, l2_id=pid, l2_code=info["code"]))
        rr, cc = np.nonzero(valid & (l2 == pid))
        r0, c0 = rr.min(), cc.min()
        sub = sp_all[r0:r0 + sp.shape[0], c0:c0 + sp.shape[1]]
        sub[sp > 0] = sp[sp > 0] + sp_off
        sp_off += sp.max()
    write(paths.coarse("l3_first"), out, cg, nodata=0)
    write(paths.coarse("superpixels"), sp_all, cg, nodata=0)
    pd.DataFrame(rows).to_csv(paths.interim / "l3_first_units.csv", index=False)
    log.info("first pass: %d L3 units", nid)
    # QA: boundaries over elevation
    elev = read(paths.interim / "coarse" / "elev.tif")
    edges = np.zeros(out.shape, bool)
    edges[:, 1:] |= out[:, 1:] != out[:, :-1]
    edges[1:, :] |= out[1:, :] != out[:-1, :]
    img = np.where(edges & (out > 0), np.nan, elev)
    quicklook(img, paths.qa / "08_l3_first.png", f"L3 first pass ({nid} units) over elevation", cmap="terrain",
              vmin=0, vmax=5000, max_px=3000)
