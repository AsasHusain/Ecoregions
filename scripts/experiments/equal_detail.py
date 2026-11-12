# Exploratory experiment kept for reproducibility (see docs/ledger.md); run from the repository root.
"""Equal-detail test: our ecoregions merged down to RESOLVE's number of units, then both scored alike.

Merging uses only the calibration half of the GBIF cells (never the held-out half): neighbouring ecoregions are
joined, most similar first (mean Jaccard similarity of their cells' species lists, average linkage), until our map
has as many units among the held-out test cells as RESOLVE has. No line is drawn: every boundary left is one of
ours. Then the rarefied ANOSIM test (as `evaluate`) scores ours (full), ours (merged) and RESOLVE on held-out cells.
Usage: python equal_detail.py <region> <out_csv>
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
from ecoregions.validate import GBIF, TAXA, anosim_r, assemblage_cells, compare_partitions, fetch_cells, jaccard_matrix, ranks_of  # noqa: E402
from ecoregions.vegcheck import holdout  # noqa: E402

region, out_csv = sys.argv[1], sys.argv[2]
cfg = load_config(None, region)
paths = Paths(cfg)
gc, vc = cfg["sources"]["gbif"], cfg["validation"]
rc = vc["rarefy"]
client = GBIF(gc["api"], paths.raw / "gbif", gc["request_pause_s"])
cells = assemblage_cells(cfg)
cd = gc["cell_deg"]
grid = json.loads(paths.grid_json.read_text())
t = Transformer.from_crs("EPSG:4326", grid["crs"], always_xy=True)
xy = np.array([t.transform(lon + cd / 2, lat + cd / 2) for lon, lat, _ in cells])
with rasterio.open(paths.raster("ecoregions")) as e, rasterio.open(paths.raster("resolve")) as r:
    g_eco = np.array([v[0] for v in e.sample(xy)]).astype(int)
    g_res = np.array([v[0] for v in r.sample(xy)]).astype(int)
held = holdout([lon for lon, _, _ in cells], [lat for _, lat, _ in cells], cd)

# neighbouring ecoregions (polygons touching, with a 100 m tolerance for the snapped boundaries)
eco = gpd.read_file(paths.vector("ecoregions"), layer="ecoregions")[["unit", "type_name", "geometry"]]
names = eco.set_index("unit").type_name
buf = eco.assign(geometry=eco.buffer(100))
pairs = gpd.sjoin(buf, buf, predicate="intersects")
adj = {(a, b) for a, b in zip(pairs.unit_left, pairs.unit_right) if a != b}


def tested_groups(labels, idx):
    """Groups the ANOSIM test keeps among cells idx (at least 2 cells)."""
    vcnt = pd.Series(labels[idx]).value_counts()
    return int((vcnt >= 2).sum())


def merge(g, cal_idx, test_idx, P_cal, k_target):
    """Adjacency-constrained average-linkage merging of units on calibration cells; returns new labels per cell
    and the merge log. Units with no calibration cells join their neighbour with the most calibration cells."""
    lab = {int(u): int(u) for u in eco.unit}       # unit -> current group (all units: adjacency runs through unsampled ones)
    global RANK
    RANK = ranks_of(jaccard_matrix(P_cal))         # ranked dissimilarities of calibration cells
    log = []
    find = lambda u: lab.get(int(u), int(u))  # noqa: E731  (0: outside, never merged)
    while True:
        cur = np.array([find(u) for u in g])
        if tested_groups(cur, test_idx) <= k_target:
            return cur, log
        groups = {}
        for i, c in enumerate(cal_idx):
            groups.setdefault(cur[c], []).append(i)
        gadj = {(find(a), find(b)) for a, b in adj if a in lab and b in lab and find(a) != find(b)}
        best, bs = None, -np.inf
        cal_lab = cur[cal_idx]
        for a, b in gadj:
            if a < b and a in groups and b in groups:
                trial = np.where(cal_lab == b, a, cal_lab)
                vc_ = pd.Series(trial).value_counts()
                keep = np.isin(trial, vc_[vc_ >= 2].index)
                s = anosim_r(RANK[np.ix_(keep, keep)], trial[keep]) if len(np.unique(trial[keep])) > 1 else -1
                if s > bs:
                    best, bs = (a, b), s
        if best is None:  # no neighbouring pair with calibration data: join a data-less group to a neighbour
            cand = [(a, b) for a, b in gadj if a not in groups]
            if not cand:
                return cur, log
            a, b = cand[0]
            best, bs = (b, a), np.nan
        a, b = best
        for u in lab:
            if lab[u] == b:
                lab[u] = a
        log.append((a, b, bs))


out, merges = [], []
for taxon in ("plants", "birds"):
    res = fetch_cells(client, cells, TAXA[taxon], gc.get("threads", 2))
    cnt = np.array([x["count"] for x in res])
    nsp = np.array([len(x["species"]) for x in res])
    ok = (cnt >= gc["min_records_cell"]) & (nsp >= gc["min_species_cell"]) & (g_eco > 0) & (g_res > 0)
    ok &= np.array([sum(x["species"].values()) for x in res]) >= rc["records"]
    cal, test = np.flatnonzero(ok & ~held), np.flatnonzero(ok & held)
    k_res = tested_groups(g_res, test)
    k_ours = tested_groups(g_eco, test)
    P_cal = presence([list(res[i]["species"]) for i in cal])
    merged, log = merge(g_eco, cal, test, P_cal, k_res)
    for a, b, s in log:
        merges.append(dict(region=region, taxon=taxon, kept=names.get(a, a), joined=names.get(b, b), similarity=s))
    print(f"{region} {taxon}: held-out cells {len(test)}, calibration cells {len(cal)}; groups tested: ours {k_ours}, "
          f"RESOLVE {k_res}, ours merged {tested_groups(merged, test)} ({len(log)} merges)", flush=True)
    parts = {"ours (full)": g_eco[test], "ours (merged to RESOLVE's count)": merged[test], "RESOLVE": g_res[test]}
    t0 = time.time()
    for r in range(rc["runs"]):
        rng = np.random.default_rng([rc["seed"], r])
        P = presence([rarefy(res[i]["species"], rc["records"], rng) for i in test])
        df = compare_partitions(P, parts, xy[test], vc["random_partitions"], 0, rc["seed"] + r, rc["dissimilarity"])
        df.insert(0, "run", r)
        df.insert(0, "cells", "holdout")
        df.insert(0, "taxon", taxon)
        out.append(df)
    print(f"  scored in {time.time() - t0:.0f} s", flush=True)

s = summarise_runs(pd.concat(out, ignore_index=True))
s["excess"] = s.R_median - s.null_R_mean
s.insert(0, "region", region)
s.to_csv(out_csv, index=False)
pd.DataFrame(merges).to_csv(out_csv.replace(".csv", "_merges.csv"), index=False)
print(s[["taxon", "partition", "n_groups", "R_median", "null_R_mean", "excess", "pct_median", "pct_p5", "pct_p95"]].round(3).to_string(index=False))
