# Exploratory experiment kept for reproducibility (see docs/ledger.md); run from the repository root.
"""Finer assemblage test for Texas plants: 0.1-degree cells from the local GBIF plant download.

Records: wild, non-fossil, not cultivated, coordinates good to `vegcheck.max_uncertainty_m` (as `vegcheck`).
A 0.1-degree cell is held out when its parent 0.25-degree cell is held out (the calibration/holdout checkerboard),
so no record used in calibration or the audit is scored. Scored on held-out cells, rarefied to 100 records x 50 runs:
ours (full), RESOLVE, ours merged to RESOLVE's count (merges chosen on calibration cells, maximising ANOSIM R),
and, within the central band only, ours vs RESOLVE.
"""
import json
import sys
import time

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "src"))
from ecoregions.config import Paths, load_config  # noqa: E402
from ecoregions.evaluate import presence, rarefy, summarise_runs  # noqa: E402
from ecoregions.validate import anosim_r, compare_partitions, jaccard_matrix, ranks_of  # noqa: E402
from ecoregions.vegcheck import holdout  # noqa: E402

FINE = 0.1
cfg = load_config(None, "texas")
paths = Paths(cfg)
gc, vc = cfg["sources"]["gbif"], cfg["validation"]
rc = vc["rarefy"]

cols = ["gbifID", "kingdom", "species", "basisOfRecord", "cultivated", "decimalLatitude", "decimalLongitude",
        "coordinateUncertaintyInMeters"]
recs = pd.concat([pd.read_csv(paths.raw / "gbif_dwca" / f"occurrences_{d['key']}.csv.gz", dtype=str, usecols=cols)
                  for d in cfg["sources"]["gbif_dwca"]["downloads"]]).drop_duplicates("gbifID")
recs = recs[(recs.kingdom == "Plantae") & (recs.cultivated != "True") & (recs.basisOfRecord != "FOSSIL_SPECIMEN")]
unc = pd.to_numeric(recs.coordinateUncertaintyInMeters, errors="coerce")
recs = recs[~(unc > cfg["vegcheck"]["max_uncertainty_m"])][["species", "decimalLatitude", "decimalLongitude"]].dropna()
lat, lon = recs.decimalLatitude.astype(float).to_numpy(), recs.decimalLongitude.astype(float).to_numpy()
recs = recs.assign(ci=np.floor(lon / FINE).astype(int), cj=np.floor(lat / FINE).astype(int))
cnt = recs.groupby(["ci", "cj", "species"]).size()
cells = cnt.groupby(level=[0, 1]).sum()
cells = cells[cells >= rc["records"]]
keys = list(cells.index)
keep_k = set(keys)
lists = {k: s.droplevel([0, 1]).to_dict() for k, s in cnt.groupby(level=[0, 1]) if k in keep_k}
lon_c = np.array([(i + 0.5) * FINE for i, _ in keys])
lat_c = np.array([(j + 0.5) * FINE for _, j in keys])
nsp = np.array([len(lists[k]) for k in keys])

grid = json.loads(paths.grid_json.read_text())
t = Transformer.from_crs("EPSG:4326", grid["crs"], always_xy=True)
xy = np.array(t.transform(lon_c, lat_c)).T
with rasterio.open(paths.raster("ecoregions")) as e, rasterio.open(paths.raster("resolve")) as r:
    g_eco = np.array([v[0] for v in e.sample(xy)]).astype(int)
    g_res = np.array([v[0] for v in r.sample(xy)]).astype(int)
ok = (g_eco > 0) & (g_res > 0) & (nsp >= gc["min_species_cell"])
held = holdout(lon_c, lat_c, gc["cell_deg"])  # the parent 0.25-degree checkerboard
cal, test = np.flatnonzero(ok & ~held), np.flatnonzero(ok & held)
print(f"0.1-degree cells with >= {rc['records']} wild plant records: {len(keys)}; in study area: {ok.sum()}; "
      f"calibration {len(cal)}, held-out {len(test)}", flush=True)

eco = gpd.read_file(paths.vector("ecoregions"), layer="ecoregions")[["unit", "type_name", "geometry"]]
names = eco.set_index("unit").type_name
buf = eco.assign(geometry=eco.buffer(100))
pairs = gpd.sjoin(buf, buf, predicate="intersects")
adj = {(a, b) for a, b in zip(pairs.unit_left, pairs.unit_right) if a != b}


def tested_groups(labels, idx):
    v = pd.Series(labels[idx]).value_counts()
    return int((v >= 2).sum())


def merge(g, k_target):
    """Greedy merges of neighbouring ecoregions, each the one that most raises ANOSIM R on calibration cells."""
    lab = {int(u): int(u) for u in eco.unit}
    find = lambda u: lab.get(int(u), int(u))  # noqa: E731
    rank = ranks_of(jaccard_matrix(presence([list(lists[keys[i]]) for i in cal])))
    log = []
    while True:
        cur = np.array([find(u) for u in g])
        if tested_groups(cur, test) <= k_target:
            return cur, log
        cl = cur[cal]
        present = set(np.unique(cl))
        gadj = {(find(a), find(b)) for a, b in adj if find(a) != find(b)}
        best, bs = None, -np.inf
        for a, b in gadj:
            if a < b and a in present and b in present:
                trial = np.where(cl == b, a, cl)
                vcn = pd.Series(trial).value_counts()
                keep = np.isin(trial, vcn[vcn >= 2].index)
                s = anosim_r(rank[np.ix_(keep, keep)], trial[keep])
                if s > bs:
                    best, bs = (a, b), s
        if best is None:
            cand = [(a, b) for a, b in gadj if a not in present]
            if not cand:
                return cur, log
            best = (cand[0][1], cand[0][0])
        a, b = best
        for u in lab:
            if lab[u] == b:
                lab[u] = a
        log.append((names.get(a, a), names.get(b, b), bs))


k_res = tested_groups(g_res, test)
merged, log = merge(g_eco, k_res)
print(f"groups tested: ours {tested_groups(g_eco, test)}, RESOLVE {k_res}, ours merged {tested_groups(merged, test)}",
      flush=True)
for a, b, s in log:
    print(f"  merge: {a} <- {b}", flush=True)

# the central band: cells in the Blackland, Post Oak, Grand Prairie, Cross Timbers and Llano ecoregions
band_types = ["Blackland Tallgrass Prairie", "Post Oak Savannas & Woodlands", "Grand Prairie Midgrass Prairie & Oak–Juniper Hills",
              "Cross Timbers Oak Woodlands", "Llano Uplift Oak Woodlands & Granite Outcrops",
              "Central Texas Oak Savannas & Prairies", "East Texas Oak Woodlands & Tallgrass Prairies"]
band = np.isin(pd.Series(g_eco).map(names).to_numpy(), band_types)
tband = np.flatnonzero(ok & held & band)
print(f"central band held-out cells: {len(tband)}", flush=True)

out = []
t0 = time.time()
for r in range(rc["runs"]):
    rng = np.random.default_rng([rc["seed"], r])
    P = presence([rarefy(lists[keys[i]], rc["records"], rng) for i in test])
    df = compare_partitions(P, {"ours (full)": g_eco[test], "ours (merged to RESOLVE's count)": merged[test],
                                "RESOLVE": g_res[test]}, xy[test], vc["random_partitions"], 0, rc["seed"] + r, rc["dissimilarity"])
    df.insert(0, "run", r); df.insert(0, "cells", "holdout 0.1 deg"); df.insert(0, "taxon", "plants")
    out.append(df)
    pos = {c: n for n, c in enumerate(test)}
    sub = [pos[c] for c in tband]
    db = compare_partitions(P[sub], {"ours, central band": g_eco[tband], "RESOLVE, central band": g_res[tband]},
                            xy[tband], vc["random_partitions"], 0, rc["seed"] + r, rc["dissimilarity"])
    db.insert(0, "run", r); db.insert(0, "cells", "holdout 0.1 deg, central band"); db.insert(0, "taxon", "plants")
    out.append(db)
print(f"scored in {time.time() - t0:.0f} s", flush=True)
s = summarise_runs(pd.concat(out, ignore_index=True))
s["excess"] = s.R_median - s.null_R_mean
s.to_csv(sys.argv[1], index=False)
print(s[["cells", "partition", "n_cells", "n_groups", "R_median", "null_R_mean", "excess", "pct_median", "pct_p5", "pct_p95"]]
      .round(3).to_string(index=False))
