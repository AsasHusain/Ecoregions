"""v0.2 stage — faunal biogeographic regions from species turnover (step 2).

Following Kreft & Jetz (2010) and Holt et al. (2013): presence/absence of each
species per equal-area cell (``fauna`` stage, Area of Habitat), Simpson's beta
diversity between every pair of study cells,

    beta_sim = min(b, c) / (a + min(b, c))

(a = shared species, b and c = species unique to either cell) — insensitive to
richness differences, so a species-poor desert cell is not "different" from a
rich neighbour merely for having fewer species — and average-linkage (UPGMA)
clustering of the beta_sim matrix. For each number of regions in ``k_range``
we report the share of total turnover explained (between-region / total mean
dissimilarity, as in Holt et al. 2013) so the chosen ``k`` is documented, not
arbitrary. Clusters are not forced to be contiguous; fragments smaller than
``min_fragment_cells`` are reassigned to the neighbouring region they most
resemble.

Outputs in ``interim/faunal/``: ``regions_k<k>.tif`` (10 km), ``k_scan.csv``,
``region_species.csv`` (most characteristic species per region: fidelity x
frequency, an IndVal-like score) and QA maps.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy import ndimage
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from .config import Paths
from .grid import Grid, write

log = logging.getLogger(__name__)


def beta_sim(X: sp.spmatrix) -> np.ndarray:
    """Pairwise Simpson dissimilarity between the rows (cells) of a boolean
    cells x species matrix. Rows with no species get dissimilarity 1 to all."""
    X = sp.csr_matrix(X, dtype=np.float32)
    a = (X @ X.T).toarray()
    s = np.asarray(X.sum(axis=1)).ravel()
    b = s[:, None] - a
    c = s[None, :] - a
    m = np.minimum(b, c)
    with np.errstate(invalid="ignore", divide="ignore"):
        d = np.where(a + m > 0, m / (a + m), 1.0)
    np.fill_diagonal(d, 0.0)
    return d.astype("float32")


def explained_turnover(dv: np.ndarray, i: np.ndarray, j: np.ndarray, labels: np.ndarray) -> float:
    """Share of mean pairwise dissimilarity that lies between regions; ``dv`` holds
    the dissimilarity of every pair (``i``, ``j``), computed once by the caller."""
    same = labels[i] == labels[j]
    tot = dv.mean()
    within = dv[same].mean() if same.any() else 0.0
    return float((tot - within) / tot) if tot > 0 else 0.0


def clean_fragments(lab2d: np.ndarray, min_cells: int) -> np.ndarray:
    """Reassign spatial fragments smaller than ``min_cells`` to the most common
    neighbouring label (repeated until stable)."""
    out = lab2d.copy()
    for _ in range(10):
        changed = False
        for k in np.unique(out[out > 0]):
            comp, n = ndimage.label(out == k)
            sizes = np.bincount(comp.ravel())
            for i in np.flatnonzero(sizes < min_cells):
                if i == 0:
                    continue
                m = comp == i
                ring = ndimage.binary_dilation(m) & ~m & (out > 0)
                if ring.any():
                    out[m] = np.bincount(out[ring]).argmax()
                    changed = True
        if not changed:
            break
    return out


def characteristic_species(X: sp.csc_matrix, labels: np.ndarray, names: list[str], top: int) -> pd.DataFrame:
    """Per region: species with the highest fidelity (share of the species' cells
    inside the region) x frequency (share of the region's cells occupied)."""
    rows = []
    occ = np.asarray(X.sum(axis=0)).ravel()
    for k in np.unique(labels):
        sel = labels == k
        inside = np.asarray(X[sel].sum(axis=0)).ravel()
        score = np.where(occ > 0, inside / np.maximum(occ, 1), 0) * inside / sel.sum()
        for j in np.argsort(-score)[:top]:
            if score[j] > 0:
                rows.append(dict(region=int(k), species=names[j], fidelity=round(inside[j] / occ[j], 3),
                                 frequency=round(inside[j] / sel.sum(), 3), score=round(float(score[j]), 4)))
    return pd.DataFrame(rows)


def run(cfg) -> None:
    paths = Paths(cfg)
    fc = cfg["faunal"]
    src = paths.interim / "fauna"
    out = paths.interim / "faunal"
    out.mkdir(parents=True, exist_ok=True)
    cell = Grid.load(src / "grid.json")
    cells = pd.read_csv(src / "cells.csv")
    spc = pd.read_csv(src / "species.csv")
    X = sp.load_npz(src / f"presence_{fc['presence']}.npz").tocsr()

    study = cells.study_share.to_numpy() >= fc["min_study_share"]
    Xs = X[study]
    keep = np.asarray(Xs.sum(axis=0)).ravel() > 0
    Xs = Xs[:, keep].tocsc()
    names = spc.name[keep].tolist()
    log.info("faunal: %d cells x %d species", Xs.shape[0], Xs.shape[1])
    (out / "n_species.txt").write_text(str(Xs.shape[1]))  # quoted on the maps

    d = beta_sim(Xs)
    dv = squareform(d, checks=False)  # condensed: the pairs in triu order
    del d
    i, j = np.triu_indices(Xs.shape[0], 1)
    Z = linkage(dv, method="average")

    scan = []
    shape = cell.shape
    for k in range(fc["k_range"][0], fc["k_range"][1] + 1):
        lab = fcluster(Z, k, criterion="maxclust")
        scan.append(dict(k=k, explained=round(explained_turnover(dv, i, j, lab), 4)))
    scan = pd.DataFrame(scan)
    scan.to_csv(out / "k_scan.csv", index=False)
    log.info("faunal: turnover explained by k regions\n%s", scan.to_string(index=False))

    from .qa import quicklook

    for k in fc["k_maps"]:
        lab = np.zeros(len(cells), "uint8")
        lab[study] = fcluster(Z, k, criterion="maxclust")
        lab2d = clean_fragments(lab.reshape(shape), fc["min_fragment_cells"])
        write(out / f"regions_k{k}.tif", lab2d, cell, nodata=0)
        quicklook(lab2d, paths.qa / f"31_faunal_regions_k{k}.png", f"faunal regions, k = {k} (UPGMA on beta_sim)",
                  cmap="tab20", categorical=True)
        ch = characteristic_species(Xs, lab2d.ravel()[study], names, fc["top_species"])
        ch.insert(0, "k", k)
        ch.to_csv(out / f"region_species_k{k}.csv", index=False)
