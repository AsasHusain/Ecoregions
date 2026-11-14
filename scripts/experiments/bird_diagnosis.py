"""Why did RESOLVE predict Iran's sealed bird lists better? (exploratory; docs/ledger.md, 2026-11-14)

Scores bird prediction (endpoint A, as `scorecard`: 0.25-degree GBIF API cells rarefied to 100 records, unit
frequencies from development cells shrunk with 20 pseudo-cells, Brier loss averaged over 1-degree blocks) for:
  * our ecoregions coarsened along the construction-only tree to fewer units (does region size matter?);
  * faunal clusterings (UPGMA on beta_sim, 10 km) from full ranges, elevation-trimmed ranges and Area of Habitat,
    at several numbers of regions (does trimming, or the number of regions, matter?).
Two test sets: the held-out checkerboard outside the sealed squares (exploratory, seen before) and Iran's sealed
cells (opened once on 2026-11-13; used here only to understand the failure, not to choose anything for v0.5.0).

    PYTHONPATH=src python scripts/experiments/bird_diagnosis.py
"""
import json

import numpy as np
import pandas as pd
import scipy.sparse as sp
from pyproj import Transformer
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from ecoregions.config import Paths, load_config
from ecoregions.confirm import cell_roles, load_mask
from ecoregions.evaluate import rarefy
from ecoregions.faunal import beta_sim, clean_fragments
from ecoregions.grid import Grid
from ecoregions.scorecard import (api_cells, block_mean, brier_cells, cell_points, coarsening_tree, level_map, majority,
                                  neighbours, prediction_loss, sample_array, sample_raster)
from ecoregions.vegcheck import holdout

cfg = load_config(region="pak-afg-irn")
paths = Paths(cfg)
sc, gc = cfg["scorecard"], cfg["sources"]["gbif"]
grid = json.loads(paths.grid_json.read_text())
to_xy = Transformer.from_crs("EPSG:4326", grid["crs"], always_xy=True)

cells = api_cells(cfg, paths, "birds")
lon, lat = cell_points(cells, sc["purity_points"])
x, y = (np.asarray(v) for v in to_xy.transform(lon, lat))
eco_pts = sample_raster(paths.raster("ecoregions"), x, y)
res_pts = sample_raster(paths.raster("resolve"), x, y)
rec = np.array([sum(c[3].values()) for c in cells])
nsp = np.array([len(c[3]) for c in cells])
ok = ((eco_pts > 0).mean(1) >= 0.5) & ((res_pts > 0).mean(1) >= 0.5) & (rec >= sc["min_records"]) & (nsp >= gc["min_species_cell"])
idx = np.flatnonzero(ok)
cells, eco_pts, res_pts, x, y = [cells[i] for i in idx], eco_pts[idx], res_pts[idx], x[idx], y[idx]
clon = np.array([c[0] + c[2] / 2 for c in cells])
clat = np.array([c[1] + c[2] / 2 for c in cells])
blocks = (np.floor(clon) * 1000 + np.floor(clat)).astype(int)
held = holdout(clon, clat, gc["cell_deg"])
rl = cell_roles(load_mask(cfg), cells)
dev = np.flatnonzero((rl == "development") & ~held)
tests = {"held-out (exploratory)": np.flatnonzero((rl == "development") & held),
         "Iran sealed (opened)": np.flatnonzero(rl == "confirmation")}
print({k: len(v) for k, v in tests.items()}, "development", len(dev))

# ---- partitions
parts = {"ours (165)": majority(eco_pts)[0], "RESOLVE": majority(res_pts)[0]}
eco_df = pd.read_csv(paths.interim / "ecoregions.csv")
tree = coarsening_tree(eco_df, neighbours(paths))
for k in (120, 80, 60, 39, 25, 15, 8):
    lm = level_map(tree, eco_df.unit.astype(int).tolist(), k)
    if lm is not None:
        parts[f"ours coarsened ({k})"] = majority(np.vectorize(lambda u: lm.get(int(u), int(u)))(eco_pts))[0]

fsrc = paths.interim / "fauna"
fgrid = Grid.load(fsrc / "grid.json")
fcells = pd.read_csv(fsrc / "cells.csv")
study = fcells.study_share.to_numpy() >= cfg["faunal"]["min_study_share"]
for kind in ("range", "elev", "aoh"):
    X = sp.load_npz(fsrc / f"presence_{kind}.npz").tocsr()[study]
    X = X[:, np.asarray(X.sum(0)).ravel() > 0].tocsc()
    Z = linkage(squareform(beta_sim(X), checks=False), method="average")
    for k in (8, 20, 30, 40):
        lab = np.zeros(len(fcells), "uint16")
        lab[study] = fcluster(Z, k, criterion="maxclust")
        lab2d = clean_fragments(lab.reshape(fgrid.shape), cfg["faunal"]["min_fragment_cells"])
        parts[f"faunal {kind} ({k})"] = majority(sample_array(lab2d, fgrid.transform, x, y))[0]
    print("clustered", kind, flush=True)

# ---- endpoint A over rarefaction draws
vocab = {s: n for n, s in enumerate(sorted({s for c in cells for s in c[3]}))}


def pres(lists):
    M = np.zeros((len(lists), len(vocab)), bool)
    for a, s in enumerate(lists):
        M[a, [vocab[k] for k in s]] = True
    return M


out = []
for tname, test in tests.items():
    losses = {}
    for draw in range(sc["draws"]):
        drng = np.random.default_rng([sc["seed"], draw])
        P = pres([rarefy(c[3], sc["depth"], drng) if sum(c[3].values()) >= sc["depth"] else list(c[3]) for c in cells])
        sp_ = P[dev].sum(0) >= 1
        Pd, Pt = P[dev][:, sp_].astype(float), P[test][:, sp_].astype(float)
        prior = Pd.mean(0)
        losses.setdefault("overall-frequency prior", []).append(block_mean(brier_cells(Pt, np.broadcast_to(prior, Pt.shape)), blocks[test]))
        for name, g in parts.items():
            losses.setdefault(name, []).append(block_mean(prediction_loss(Pd, g[dev], Pt, g[test], prior, sc["prior_cells"]), blocks[test]))
    L = {k: float(np.mean(v)) for k, v in losses.items()}
    for name, l in L.items():
        g = parts.get(name)
        out.append(dict(test=tname, partition=name, units_in_test=len(np.unique(g[test])) if g is not None else 0,
                        units_in_dev=len(np.unique(g[dev])) if g is not None else 0,
                        loss=round(l, 5), vs_RESOLVE_pct=round(100 * (L["RESOLVE"] - l) / L["RESOLVE"], 2)))
df = pd.DataFrame(out)
df.to_csv(paths.outputs / "bird_diagnosis.csv", index=False)
pd.set_option("display.width", 200)
print(df.to_string(index=False))
