"""Merges of over-split ecoregion types (stage ``merges``; proposals only, nothing on the map changes).

One global rule, decided on construction and calibration data, never on held-out cells or reference maps:
1. A merge tree of the TYPES from construction data only: each type's area-weighted community mix; the two
   neighbouring types whose merge least raises the within-group spread (Ward) are joined first.
2. Every level of the tree is scored by cross-validation on calibration (development) cells only: 1-degree blocks
   in ``merges.folds`` folds; a block's plant lists are predicted from the other folds (scorecard endpoint A).
3. The level chosen is the smallest number of types whose cross-validated loss is within one standard error of the
   best level (the one-standard-error rule; ``merges.rule``).
Writes outputs/type_merge_tree.csv (every level, its loss and standard error) and outputs/type_merges.csv (the merges
proposed) and interim/type_merges.json (type -> merged group), which the scorecard reads as "ours, types merged".
"""
from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd
from pyproj import Transformer

from .config import Paths
from .evaluate import rarefy
from .scorecard import (block_mean, cell_points, coarsening_tree, level_map, local_plant_cells, majority, neighbours,
                        prediction_loss, sample_raster)
from .vegcheck import holdout

log = logging.getLogger(__name__)


def type_table(eco_df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """One row per type: area and area-weighted community mix (as 'Name pct%; ...'), and type -> integer id."""
    rows = []
    for t, g in eco_df.groupby("type"):
        mix = {}
        for s, a in zip(g.community_mix.fillna(""), g.area_km2):
            for part in s.split("; "):
                if part.strip():
                    name, pct = part.rsplit(" ", 1)
                    mix[name] = mix.get(name, 0.0) + a * float(pct.rstrip("%"))
        area = float(g.area_km2.sum())
        rows.append(dict(type=t, area_km2=area, community_mix="; ".join(f"{k} {v / area:.4f}%" for k, v in mix.items())))
    df = pd.DataFrame(rows)
    df.insert(0, "unit", np.arange(1, len(df) + 1))
    return df, dict(zip(df.type, df.unit))


def run(cfg) -> None:
    paths = Paths(cfg)
    mc, sc, gc = cfg["merges"], cfg["scorecard"], cfg["sources"]["gbif"]
    eco_df = pd.read_csv(paths.interim / "ecoregions.csv")
    types, tid = type_table(eco_df)
    unit_type = eco_df.set_index("unit").type
    uadj = neighbours(paths)
    tadj = {(tid[unit_type[a]], tid[unit_type[b]]) for a, b in uadj
            if a in unit_type.index and b in unit_type.index and unit_type[a] != unit_type[b]}
    tree = coarsening_tree(types, tadj)
    tnames = dict(zip(types.unit, types.type))
    grid = json.loads(paths.grid_json.read_text())
    to_xy = Transformer.from_crs("EPSG:4326", grid["crs"], always_xy=True)
    rng = np.random.default_rng(mc["seed"])
    levels = list(range(len(types), 0, -1))
    losses = {k: [] for k in levels}  # per (scale, draw, fold): one loss per level
    for deg in sc["scales"]:
        cells = [c for c in local_plant_cells(cfg, paths, deg)
                 if sum(c[3].values()) >= sc["min_records"] and len(c[3]) >= gc["min_species_cell"]]
        lon, lat = cell_points(cells, sc["purity_points"])
        x, y = (np.asarray(v) for v in to_xy.transform(lon, lat))
        eco_pts = sample_raster(paths.raster("ecoregions"), x, y)
        units, _ = majority(eco_pts)
        clon = np.array([c[0] + c[2] / 2 for c in cells])
        clat = np.array([c[1] + c[2] / 2 for c in cells])
        keep = ((eco_pts > 0).mean(1) >= 0.5) & ~holdout(clon, clat, gc["cell_deg"]) & np.isin(units, unit_type.index)
        idx = np.flatnonzero(keep)  # calibration cells only
        units, blocks = units[idx], (np.floor(clon[idx]) * 1000 + np.floor(clat[idx])).astype(int)
        ub = np.unique(blocks)
        fold_of = dict(zip(ub, rng.permutation(len(ub)) % mc["folds"]))
        fold = np.array([fold_of[b] for b in blocks])
        vocab = {s: n for n, s in enumerate(sorted({s for i in idx for s in cells[i][3]}))}
        ttype = np.array([tid[unit_type[u]] for u in units])
        for draw in range(mc["draws"]):
            drng = np.random.default_rng([mc["seed"], draw])
            P = np.zeros((len(idx), len(vocab)))
            for a, i in enumerate(idx):
                c = cells[i][3]
                sp = rarefy(c, sc["depth"], drng) if sum(c.values()) >= sc["depth"] else list(c)
                P[a, [vocab[s] for s in sp]] = 1
            for f in range(mc["folds"]):
                tr_, te = fold != f, fold == f
                prior = P[tr_].mean(0)
                for k in levels:
                    m = level_map(tree, list(tnames), k)
                    if m is None:
                        continue
                    g = np.array([m[t] for t in ttype])
                    losses[k].append(block_mean(prediction_loss(P[tr_], g[tr_], P[te], g[te], prior, sc["prior_cells"]), blocks[te]))
        log.info("merges: %.3f-degree calibration cells %d in %d blocks", deg, len(idx), len(ub))
    lv = pd.DataFrame([dict(types=k, cv_loss=np.mean(v), se=np.std(v, ddof=1) / np.sqrt(len(v)), runs=len(v))
                       for k, v in losses.items() if v])
    best = lv.loc[lv.cv_loss.idxmin()]
    chosen = int(lv[lv.cv_loss <= best.cv_loss + best.se].types.min()) if mc["rule"] == "one_se" else int(best.types)
    lv["chosen"] = lv.types == chosen
    lv.to_csv(paths.outputs / "type_merge_tree.csv", index=False)
    m = level_map(tree, list(tnames), chosen)
    groups = {}
    for u, g in m.items():
        groups.setdefault(g, []).append(tnames[u])
    prop = pd.DataFrame([dict(group=tnames[g], types=" + ".join(sorted(v)), n=len(v)) for g, v in groups.items() if len(v) > 1])
    prop.to_csv(paths.outputs / "type_merges.csv", index=False)
    (paths.interim / "type_merges.json").write_text(json.dumps({tnames[u]: tnames[g] for u, g in m.items()}, indent=1))
    log.info("merges: %d types; best cross-validated level %d types, one-SE level %d types; proposed merges:\n%s",
             len(types), int(best.types), chosen, prop.to_string(index=False) if len(prop) else "(none)")
