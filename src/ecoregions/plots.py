"""Confirmatory tests on vegetation plots (sPlotOpen v2.0), exactly as pre-registered (docs/preregistration.md):
A prediction of plot species lists from unit frequencies fitted on GBIF development cells; B boundary contrast between
plots 1-50 km apart; C agreement of plot structure with the biome's defining formations. 2,000 bootstrap resamples of
1-degree blocks of plots for every paired difference. Writes outputs/plots_tests.csv and plots_decisions.csv.

Stage ``plots``; it refuses to run without ``plots.map_sha256`` (the frozen map's hash, recorded before opening).
"""
from __future__ import annotations

import hashlib
import json
import logging

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer

from .config import Paths
from .evaluate import rarefy
from .scorecard import (cell_points, climate_kmeans, coarse_labels, coarsening_tree, graph_partition, land_grid, level_map,
                        local_plant_cells, majority, neighbours, sample_array, sample_raster, unit_frequencies)
from .vegcheck import holdout

log = logging.getLogger(__name__)
FORMS = {"forest": ["Forest", "is_forest"], "shrubland": ["Shrubland"], "grassland": ["Grassland"], "wetland": ["Wetland"],
         "sparse": ["Sparse_vegetation"]}
BIOME_FORMS = {"Mediterranean Forests, Woodlands & Scrub": {"forest", "shrubland"},
               "Temperate Grasslands, Savannas & Shrublands": {"grassland", "shrubland"},
               "Tropical & Subtropical Grasslands, Savannas & Shrublands": {"grassland", "shrubland"},
               "Deserts & Xeric Shrublands": {"shrubland", "sparse"},
               "Montane Grasslands & Shrublands": {"grassland", "shrubland", "sparse"},
               "Tundra": {"grassland", "shrubland", "sparse"},
               "Flooded Grasslands & Savannas": {"wetland", "grassland"},
               "Mangroves": {"forest", "wetland"}, "Rock & Ice": {"sparse"}}


def forms_of(biome: str) -> set:
    return BIOME_FORMS.get(biome, {"forest"} if "Forest" in biome else set())


def truthy(s: pd.Series) -> pd.Series:
    return s.astype(str).str.upper().isin(["TRUE", "1", "T", "YES"])


def read_plots(cfg, paths: Paths) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Eligible plots inside the study area (pre-registered filters) and their species lists."""
    d = paths.raw / "splotopen" / "unzipped"
    h = pd.read_csv(next(d.glob("sPlotOpen_header*.txt")), sep="\t", low_memory=False)
    study = gpd.read_file(paths.vector("boundaries"), layer="study_area").to_crs(4326).union_all()
    w, s, e, n = study.bounds
    h = h[h.Longitude.between(w, e) & h.Latitude.between(s, n)]
    import shapely

    h = h[shapely.contains_xy(study, h.Longitude.to_numpy(), h.Latitude.to_numpy())]
    yr = pd.to_numeric(h.Date_of_recording.astype(str).str[:4], errors="coerce")
    unc = pd.to_numeric(h.Location_uncertainty, errors="coerce")
    nat = pd.to_numeric(h.Naturalness, errors="coerce")
    h = h[unc.notna() & (unc <= 1000) & (nat != 3) & (yr >= 1950)].copy()
    dt = pd.read_csv(next(d.glob("sPlotOpen_DT*.txt")), sep="\t", usecols=["PlotObservationID", "Species"])
    dt = dt[dt.PlotObservationID.isin(h.PlotObservationID)].dropna()
    dt["Species"] = dt.Species.str.split().str[:2].str.join(" ")
    h["n_species"] = h.PlotObservationID.map(dt.groupby("PlotObservationID").Species.nunique()).fillna(0).astype(int)
    for f, cols in FORMS.items():
        h[f"form_{f}"] = np.any([truthy(h[c]) for c in cols if c in h], axis=0)
    h["species_ok"] = (h.n_species >= 5) & h.Plant_recorded.astype(str).isin(["All vascular plants", "Not specified"])
    return h, dt


def boot_weights(blocks: np.ndarray, n: int, seed: int) -> tuple[np.ndarray, list]:
    ub = np.unique(blocks)
    rng = np.random.default_rng(seed)
    return ub, [np.bincount(rng.choice(len(ub), len(ub)), minlength=len(ub)).astype(float) for _ in range(n)]


def wmean_blocks(v: np.ndarray, bidx: np.ndarray, w: np.ndarray) -> float:
    per = np.bincount(bidx, weights=v, minlength=len(w)) / np.maximum(np.bincount(bidx, minlength=len(w)), 1)
    has = np.bincount(bidx, minlength=len(w)) > 0
    return float((per * w)[has].sum() / w[has].sum())


def contrast(D, across, bins, w=None) -> float:
    w = np.ones(len(D)) if w is None else w
    out = []
    for b in np.unique(bins):
        m = bins == b
        a, i = m & across & (w > 0), m & ~across & (w > 0)
        if a.any() and i.any():
            out.append((D[a] * w[a]).sum() / w[a].sum() - (D[i] * w[i]).sum() / w[i].sum())
    return float(np.mean(out)) if out else np.nan


def run(cfg) -> None:
    paths = Paths(cfg)
    pc, sc, gc = cfg["plots"], cfg["scorecard"], cfg["sources"]["gbif"]
    gpkg = paths.outputs / "ecoregions.gpkg"
    sha = hashlib.sha256(gpkg.read_bytes()).hexdigest()
    if sha != pc.get("map_sha256"):
        raise SystemExit(f"plots: the map's hash {sha} differs from the frozen plots.map_sha256; refusing to run")
    grid = json.loads(paths.grid_json.read_text())
    to_xy = Transformer.from_crs("EPSG:4326", grid["crs"], always_xy=True)
    tr_c, eco_c, res_c, refs, (cok, Z) = coarse_labels(cfg, paths)
    eco_df = pd.read_csv(paths.interim / "ecoregions.csv")
    h, dt = read_plots(cfg, paths)
    px, py = (np.asarray(v) for v in to_xy.transform(h.Longitude.to_numpy(), h.Latitude.to_numpy()))
    blocks = (np.floor(h.Longitude) * 1000 + np.floor(h.Latitude)).astype(int).to_numpy()
    log.info("plots: %d eligible plots (%d with species lists) in %d 1-degree blocks", len(h), int(h.species_ok.sum()),
             len(np.unique(blocks)))
    # ---- partitions at the plots
    ours = sample_raster(paths.raster("ecoregions"), px[:, None], py[:, None])[:, 0]
    res = sample_raster(paths.raster("resolve"), px[:, None], py[:, None])[:, 0]
    parts = {"ours": ours, "RESOLVE": res}
    for name, arr in refs.items():
        if not name.startswith("our "):
            parts[name] = sample_array(arr, tr_c, px[:, None], py[:, None])[:, 0]
    adj = neighbours(paths)
    tree = coarsening_tree(eco_df, adj)
    k_res = len(np.unique(res_c[(eco_c > 0) & (res_c > 0)]))
    lmap = level_map(tree, eco_df.unit.astype(int).tolist(), k_res)
    if lmap is not None:
        parts[f"ours coarsened to RESOLVE's {k_res} units"] = np.array([lmap.get(int(u), int(u)) for u in ours])
    for k in sorted({len(eco_df), k_res}):
        parts[f"climate k-means ({k})"] = sample_array(climate_kmeans(cok, Z, k, sc["seed"]), tr_c, px[:, None], py[:, None])[:, 0]
    # random contiguous maps on the 0.125-degree land graph
    lij, lpos = land_grid(cfg, 0.125, to_xy, eco_c, tr_c)
    at = np.array([lpos.get((int(np.floor(lo / 0.125)), int(np.floor(la / 0.125))), -1) for lo, la in zip(h.Longitude, h.Latitude)])
    rng = np.random.default_rng(pc["seed"])
    nulls = {k: [graph_partition(lij, lpos, k, rng)[np.maximum(at, 0)] for _ in range(sc["random_partitions"])]
             for k in sorted({len(eco_df), k_res})}
    # ---- A: species prediction from GBIF development cells (0.125 degrees, rarefied as the scorecard)
    rows, dec = [], []
    sp_ok = h.species_ok.to_numpy()
    cells = [c for c in local_plant_cells(cfg, paths, 0.125) if sum(c[3].values()) >= sc["min_records"] and len(c[3]) >= gc["min_species_cell"]]
    lon, lat = cell_points(cells, sc["purity_points"])
    x, y = (np.asarray(v) for v in to_xy.transform(lon, lat))
    eco_pts = sample_raster(paths.raster("ecoregions"), x, y)
    clon = np.array([c[0] + c[2] / 2 for c in cells])
    clat = np.array([c[1] + c[2] / 2 for c in cells])
    dev = ((eco_pts > 0).mean(1) >= 0.5) & ~holdout(clon, clat, gc["cell_deg"])
    cells = [c for c, k in zip(cells, dev) if k]
    eco_pts, x, y = eco_pts[dev], x[dev], y[dev]
    cell_lab = {"ours": majority(eco_pts)[0], "RESOLVE": majority(sample_raster(paths.raster("resolve"), x, y))[0]}
    for name, arr in refs.items():
        if not name.startswith("our "):
            cell_lab[name] = majority(sample_array(arr, tr_c, x, y))[0]
    if lmap is not None:
        cell_lab[f"ours coarsened to RESOLVE's {k_res} units"] = np.array([lmap.get(int(u), int(u)) for u in cell_lab["ours"]])
    for k in sorted({len(eco_df), k_res}):
        cell_lab[f"climate k-means ({k})"] = majority(sample_array(climate_kmeans(cok, Z, k, sc["seed"]), tr_c, x, y))[0]
    cat_ = np.array([lpos.get((int(round(c[0] / 0.125)), int(round(c[1] / 0.125))), -1) for c in cells])
    plot_sp = dt[dt.PlotObservationID.isin(h.PlotObservationID[sp_ok])].groupby("PlotObservationID").Species.agg(set)
    dev_sp = {s for c in cells for s in c[3]}
    universe = sorted(dev_sp & set().union(*plot_sp.values))
    vpos = {s: i for i, s in enumerate(universe)}
    Y = np.zeros((int(sp_ok.sum()), len(universe)))
    for a, pid in enumerate(h.PlotObservationID[sp_ok]):
        for s in plot_sp.get(pid, ()):
            if s in vpos:
                Y[a, vpos[s]] = 1
    preds = {}
    for draw in range(sc["draws"]):
        drng = np.random.default_rng([sc["seed"], draw])
        P = np.zeros((len(cells), len(universe)))
        for a, c in enumerate(cells):
            lst = rarefy(c[3], sc["depth"], drng) if sum(c[3].values()) >= sc["depth"] else list(c[3])
            idx = [vpos[s] for s in lst if s in vpos]
            P[a, idx] = 1
        prior = P.mean(0)
        for name, lab in [*cell_lab.items(), *[(f"random contiguous ({k}) #{i}", None) for k in nulls for i in range(len(nulls[k]))]]:
            if lab is None:
                k = int(name.split("(")[1].split(")")[0]); i = int(name.split("#")[1])
                lab_c = graph_partition(lij, lpos, k, np.random.default_rng([pc["seed"], k, i]))[np.maximum(cat_, 0)]
                lab_p = graph_partition(lij, lpos, k, np.random.default_rng([pc["seed"], k, i]))[np.maximum(at, 0)][sp_ok]
            else:
                lab_c, lab_p = lab, parts[name][sp_ok]
            F, pos = unit_frequencies(P, lab_c, prior, sc["prior_cells"])
            r_ = np.array([pos.get(u, -1) for u in lab_p])
            pr = np.where(r_[:, None] >= 0, F[np.maximum(r_, 0)], prior[None, :])
            preds.setdefault(name, []).append(pr)
        preds.setdefault("overall-frequency prior", []).append(np.broadcast_to(prior, Y.shape))
        from scipy.spatial import cKDTree

        nn = cKDTree(np.column_stack([x.mean(1), y.mean(1)])).query(np.column_stack([px, py])[sp_ok], k=sc["knn"])[1]
        preds.setdefault("spatial-only (nearest development cells)", []).append(P[nn].mean(1))
    loss = {k: ((Y - np.mean(v, 0)) ** 2).mean(1) for k, v in preds.items()}
    ub, bw = boot_weights(blocks[sp_ok], pc["bootstrap"], pc["seed"])
    bidx = np.searchsorted(ub, blocks[sp_ok])
    one = np.ones(len(ub))
    base = wmean_blocks(loss["overall-frequency prior"], bidx, one)
    rnd = {}
    for kk in [k for k in loss if " #" in k]:  # random contiguous maps: reported as the median of their block means
        rnd.setdefault(kk.split(" #")[0], []).append(wmean_blocks(loss.pop(kk), bidx, one))
    for k, l_ in loss.items():
        rows.append(dict(endpoint="A prediction", name=k, value=wmean_blocks(l_, bidx, one), skill=1 - wmean_blocks(l_, bidx, one) / base))
    for k, v in rnd.items():
        rows.append(dict(endpoint="A prediction", name=k, value=float(np.median(v)), skill=1 - float(np.median(v)) / base,
                         p5=float(np.percentile(v, 5)), p95=float(np.percentile(v, 95))))
    # ---- B: boundary contrast between plots 1-50 km apart
    from scipy.spatial import cKDTree

    xy = np.column_stack([px, py])[sp_ok]
    pr = np.array(sorted(cKDTree(xy).query_pairs(50e3)), dtype=int).reshape(-1, 2)
    dd = np.linalg.norm(xy[pr[:, 0]] - xy[pr[:, 1]], axis=1)
    keep = dd >= 1e3
    pr, dd = pr[keep], dd[keep]
    bins = np.minimum((dd // 10e3).astype(int), 4)
    Yb = Y.astype(bool)
    D = 1 - (Yb[pr[:, 0]] & Yb[pr[:, 1]]).sum(1) / np.maximum((Yb[pr[:, 0]] | Yb[pr[:, 1]]).sum(1), 1)
    labs_b = {k: v[sp_ok] for k, v in parts.items()}
    for name, lab in labs_b.items():
        rows.append(dict(endpoint="B boundary contrast", name=name, value=contrast(D, lab[pr[:, 0]] != lab[pr[:, 1]], bins)))
    for k in nulls:
        vals = [contrast(D, n_[sp_ok][pr[:, 0]] != n_[sp_ok][pr[:, 1]], bins) for n_ in nulls[k]]
        rows.append(dict(endpoint="B boundary contrast", name=f"random contiguous ({k})", value=float(np.nanmedian(vals))))
    pw = [w[bidx[pr[:, 0]]] * w[bidx[pr[:, 1]]] for w in bw]
    # ---- C: biome structure
    from .biomes import key_biomes
    from .communities import load_rules
    from . import vectors
    from .evaluate import _biome_key

    types = load_rules(paths.root / cfg["communities"]["rules"])["ecoregion_types"]
    kb = key_biomes(paths)
    ut = eco_df.set_index("unit").type
    rt = vectors.resolve(cfg)[["ECO_ID", "BIOME_NAME"]].drop_duplicates("ECO_ID").set_index("ECO_ID").BIOME_NAME.map(_biome_key)
    form_ok = h[[f"form_{f}" for f in FORMS]].any(axis=1).to_numpy()
    forms = [{f for f in FORMS if r[f"form_{f}"]} for _, r in h.iterrows()]
    biome_maps = {"biomes by the key": [kb.get(ut.get(int(u), ""), "") for u in ours],
                  "biomes as hand-set": [types.get(ut.get(int(u), ""), {}).get("biome", "") for u in ours],
                  "RESOLVE biomes": [rt.get(int(r), "") for r in res]}
    agree = {k: np.array([bool(forms_of(b) & f) for b, f in zip(v, forms)], float) for k, v in biome_maps.items()}
    ubc, bwc = boot_weights(blocks[form_ok], pc["bootstrap"], pc["seed"] + 1)
    bic = np.searchsorted(ubc, blocks[form_ok])
    for k, a in agree.items():
        rows.append(dict(endpoint="C biome structure", name=k, value=wmean_blocks(a[form_ok], bic, np.ones(len(ubc)))))
    # ---- paired differences and decisions
    def interval(fn, ws):
        v = np.array([fn(w) for w in ws])
        return float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))

    cands = ["ours", f"ours coarsened to RESOLVE's {k_res} units"]
    comps = [c for c in parts if c not in cands] + ["spatial-only (nearest development cells)"]
    for cand in [c for c in cands if c in loss]:
        for comp in comps:
            if comp not in loss:
                continue
            fa = lambda w: (wmean_blocks(loss[comp], bidx, w) - wmean_blocks(loss[cand], bidx, w)) / wmean_blocks(loss[comp], bidx, w)  # noqa: E731
            lo_a, hi_a = interval(fa, bw)
            est_a = fa(one)
            if comp in labs_b:
                fb = lambda w: contrast(D, labs_b[cand][pr[:, 0]] != labs_b[cand][pr[:, 1]], bins, w) - contrast(D, labs_b[comp][pr[:, 0]] != labs_b[comp][pr[:, 1]], bins, w)  # noqa: E731
                lo_b, hi_b = interval(fb, pw)
                est_b = fb(np.ones(len(pr)))
            else:
                lo_b = hi_b = est_b = np.nan
            equal = cand != "ours"
            if comp == "RESOLVE":
                verdict = ("improvement" if equal and lo_a > 0.02 and lo_b > 0.02 else
                           "non-inferior" if lo_a > -0.01 and (np.isnan(lo_b) or lo_b > -0.01) else "not demonstrated")
            else:
                verdict = ""
            dec.append(dict(endpoint="A+B", candidate=cand, comparator=comp, prediction_improvement=est_a, ci=f"{lo_a:.3f} to {hi_a:.3f}",
                            boundary_difference=est_b, ci_b=f"{lo_b:.3f} to {hi_b:.3f}", verdict=verdict))
    for cand in ("biomes by the key", "biomes as hand-set"):
        fc = lambda w: wmean_blocks(agree[cand][form_ok], bic, w) - wmean_blocks(agree["RESOLVE biomes"][form_ok], bic, w)  # noqa: E731
        lo, hi = interval(fc, bwc)
        dec.append(dict(endpoint="C", candidate=cand, comparator="RESOLVE biomes", prediction_improvement=fc(np.ones(len(ubc))),
                        ci=f"{lo:.3f} to {hi:.3f}", verdict="improvement" if lo > 0 else "non-inferior" if lo > -0.02 else "not demonstrated"))
    n_blocks = dict(A=len(ub), C=len(ubc))
    out = pd.DataFrame(rows)
    d = pd.DataFrame(dec).assign(blocks_A=n_blocks["A"], blocks_C=n_blocks["C"], plots_A=int(sp_ok.sum()), plots_C=int(form_ok.sum()),
                                 pairs_B=len(pr), map_sha256=sha)
    out.to_csv(paths.outputs / "plots_tests.csv", index=False)
    d.to_csv(paths.outputs / "plots_decisions.csv", index=False)
    log.info("plots tests:\n%s\n%s", out.round(4).to_string(index=False), d.round(4).to_string(index=False))
