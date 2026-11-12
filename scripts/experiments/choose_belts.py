# Exploratory experiment kept for reproducibility (see docs/ledger.md); run from the repository root.
"""Choose rainfall belts for the central Texas band on CALIBRATION 0.1-degree cells only.

Candidates: the current ecoregions; 2 or 3 rainfall belts (breaks on a 25 mm grid); belts x current type.
Score: ANOSIM R minus the mean R of random contiguous partitions with the same number of groups (the test's
own measure), median over 10 rarefied draws. Held-out cells are not read.
"""
import json
import sys
from itertools import combinations

import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "src"))
from ecoregions.config import Paths, load_config  # noqa: E402
from ecoregions.evaluate import presence, rarefy  # noqa: E402
from ecoregions.validate import compare_partitions  # noqa: E402
from ecoregions.vegcheck import holdout  # noqa: E402

FINE = 0.1
BAND = ["blackland", "cross_timbers", "grand_prairie", "post_oak"]
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
# calibration records only (the parent 0.25-degree checkerboard), before anything else is computed
recs = recs[~holdout((recs.ci + 0.5) * FINE, (recs.cj + 0.5) * FINE, gc["cell_deg"])]
cnt = recs.groupby(["ci", "cj", "species"]).size()
tot = cnt.groupby(level=[0, 1]).sum()
keys = [k for k in tot.index if tot[k] >= rc["records"]]
kk = set(keys)
lists = {k: s.droplevel([0, 1]).to_dict() for k, s in cnt.groupby(level=[0, 1]) if k in kk}
lon_c = np.array([(i + 0.5) * FINE for i, _ in keys]); lat_c = np.array([(j + 0.5) * FINE for _, j in keys])
grid = json.loads(paths.grid_json.read_text())
xy = np.array(Transformer.from_crs("EPSG:4326", grid["crs"], always_xy=True).transform(lon_c, lat_c)).T
with rasterio.open(paths.raster("ecoregions")) as e, rasterio.open(paths.raster("clim_bio12")) as r:
    g_eco = np.array([v[0] for v in e.sample(xy)]).astype(int)
    rain = np.array([v[0] for v in r.sample(xy)]).astype(float)
eco = pd.read_csv(paths.interim / "ecoregions.csv").set_index("unit")
typ = pd.Series(g_eco).map(eco.type).to_numpy()
nsp = np.array([len(lists[k]) for k in keys])
sel = np.flatnonzero(np.isin(typ, BAND) & (nsp >= gc["min_species_cell"]) & np.isfinite(rain))
print(f"calibration cells in the band: {len(sel)}; rain {rain[sel].min():.0f}-{rain[sel].max():.0f} mm", flush=True)

cands = {"current ecoregions": g_eco[sel], "current types": typ[sel]}
grid_b = list(range(775, 1176, 25))
for b in grid_b:
    cands[f"2 belts @{b}"] = (rain[sel] >= b).astype(int)
for b1, b2 in combinations(grid_b, 2):
    if b2 - b1 >= 100:
        cands[f"3 belts @{b1},{b2}"] = np.digitize(rain[sel], [b1, b2])
rows = []
for r_ in range(10):
    rng = np.random.default_rng([rc["seed"], 1000 + r_])
    P = presence([rarefy(lists[keys[i]], rc["records"], rng) for i in sel])
    df = compare_partitions(P, cands, xy[sel], 49, 0, 7 + r_, rc["dissimilarity"])
    rows.append(df)
res = pd.concat(rows).groupby("partition").agg(n_groups=("n_groups", "median"), R=("R", "median"),
                                                null=("null_R_mean", "median"), pct=("pct_of_null_below", "median"))
res["excess"] = res.R - res.null
res = res.sort_values("excess", ascending=False)
best3 = res[res.index.str.startswith("3 belts")].index[0]
b1, b2 = map(int, best3.split("@")[1].split(","))
z = np.digitize(rain[sel], [b1, b2])
cross = pd.Series(z).astype(str).str.cat(pd.Series(typ[sel]), sep="|").to_numpy()
rows = []
for r_ in range(10):
    rng = np.random.default_rng([rc["seed"], 1000 + r_])
    P = presence([rarefy(lists[keys[i]], rc["records"], rng) for i in sel])
    rows.append(compare_partitions(P, {f"{best3} x type": cross}, xy[sel], 49, 0, 7 + r_, rc["dissimilarity"]))
x = pd.concat(rows).groupby("partition").agg(n_groups=("n_groups", "median"), R=("R", "median"),
                                              null=("null_R_mean", "median"), pct=("pct_of_null_below", "median"))
x["excess"] = x.R - x.null
res = pd.concat([res, x]).sort_values("excess", ascending=False)
print(res.round(3).head(12).to_string())
print(res.loc[["current ecoregions", "current types"]].round(3).to_string())
res.to_csv(sys.argv[1])
