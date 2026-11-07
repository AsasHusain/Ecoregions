"""v0.2 stage — validation of the community ecoregions.

1. **Assemblages (independent data).** GBIF occurrence species lists on 0.25°
   cells (as in ``validate``; cached responses), ANOSIM R for each partition and
   the share of random contiguous partitions with the same number of groups that
   it beats: v0.2 ecoregions, ecoregion types, faunal regions and biomes against
   the physiographic L3/L2 layers, RESOLVE ecoregions and RESOLVE biomes. GBIF
   records are not used to build any v0.2 layer.
2. **Taxon congruence.** Faunal regions rebuilt from birds alone and from
   mammals + reptiles + amphibians alone (same method and number of regions);
   adjusted Rand index and normalized mutual information between the two and
   with the combined regions. Boundaries shared by groups that disperse very
   differently are more likely real.
3. **Sensitivity.** Every numeric rule threshold perturbed at random
   (``evaluate.sigma``, relative unless noted), communities re-classified on the
   270 m grid and grouped into ecoregion types; stability of a cell = share of
   runs in which its type is unchanged; per ecoregion = mean over its cells.
4. **Correspondence with RESOLVE.** Main RESOLVE ecoregion of each type and
   ecoregion, and agreement of biomes cell by cell.

Outputs: ``outputs/validation_*.csv`` and a stability column for the ecoregions.
"""
from __future__ import annotations

import copy
import json
import logging

import numpy as np
import pandas as pd
import rasterio
import scipy.sparse as sp
from rasterio.enums import Resampling
from scipy import ndimage as ndi
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from . import vectors
from .assemble import pair_counts
from .communities import CATEGORICAL, NUMERIC, classify, load_rules, northness
from .config import Paths
from .faunal import beta_sim
from .features import dune_mask
from .grid import Grid, block_mode, block_reduce, read
from .validate import GBIF, TAXA, assemblage_cells, compare_partitions, fetch_cells

log = logging.getLogger(__name__)


def _biome_key(s: str) -> str:
    s = str(s).replace(" and ", " & ")
    return "Rock & Ice" if s in ("N/A", "nan") else s


# ---------------------------------------------------------------------------
# 1. assemblages
# ---------------------------------------------------------------------------
def assemblages(cfg, paths: Paths, eco_df: pd.DataFrame, res_tab: pd.DataFrame) -> pd.DataFrame:
    from pyproj import Transformer

    gc, vc = cfg["sources"]["gbif"], cfg["validation"]
    client = GBIF(gc["api"], paths.raw / "gbif", gc["request_pause_s"])
    cells = assemblage_cells(cfg)
    cd = gc["cell_deg"]
    grid = json.loads(paths.grid_json.read_text())
    t = Transformer.from_crs("EPSG:4326", grid["crs"], always_xy=True)
    xy = np.array([t.transform(lon + cd / 2, lat + cd / 2) for lon, lat, _ in cells])
    with rasterio.open(paths.raster("ecoregions")) as e, rasterio.open(paths.raster("l3")) as l3, \
            rasterio.open(paths.raster("resolve")) as r:
        g_eco = np.array([v[0] for v in e.sample(xy)])
        g_l3 = np.array([v[0] for v in l3.sample(xy)])
        g_res = np.array([v[0] for v in r.sample(xy)])
    e = eco_df.set_index("unit")
    l2map = pd.read_csv(paths.interim / "l3_units.csv", dtype={"l2_code": str}).set_index("unit").l2_code
    rb = res_tab.set_index("ECO_ID").BIOME_NAME.map(_biome_key)
    parts_all = {
        "v0.2 ecoregions": g_eco,
        "v0.2 ecoregion types": pd.Series(g_eco).map(e.type).fillna("").to_numpy(),
        "v0.2 faunal regions": pd.Series(g_eco).map(e.faunal_region).fillna(0).to_numpy(),
        "v0.2 biomes": pd.Series(g_eco).map(e.biome).map(_biome_key).fillna("").to_numpy(),
        "physiographic L3": g_l3,
        "physiographic L2": pd.Series(g_l3).map(l2map).fillna("").to_numpy(),
        "RESOLVE ecoregions": g_res,
        "RESOLVE biomes": pd.Series(g_res).map(rb).fillna("").to_numpy(),
    }
    out = []
    for taxon, extra in TAXA.items():
        res = fetch_cells(client, cells, extra, gc.get("threads", 2))
        cnt = np.array([x["count"] for x in res])
        nsp = np.array([len(x["species"]) for x in res])
        ok = (cnt >= gc["min_records_cell"]) & (nsp >= gc["min_species_cell"]) & (g_eco > 0) & (g_res > 0)
        idx = np.flatnonzero(ok)
        if len(idx) < 30:
            continue
        keys = sorted({k for i in idx for k in res[i]["species"]})
        kpos = {k: j for j, k in enumerate(keys)}
        P = np.zeros((len(idx), len(keys)), bool)
        for a, i in enumerate(idx):
            P[a, [kpos[k] for k in res[i]["species"]]] = True
        df = compare_partitions(P, {k: v[idx] for k, v in parts_all.items()}, xy[idx],
                                vc["random_partitions"], vc["anosim_permutations"], vc["seed"])
        df.insert(0, "taxon", taxon)
        out.append(df)
        log.info("assemblages %s: %d cells", taxon, len(idx))
    return pd.concat(out, ignore_index=True)


# ---------------------------------------------------------------------------
# 2. taxon congruence of the faunal regions
# ---------------------------------------------------------------------------
def congruence(cfg, paths: Paths) -> pd.DataFrame:
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    src = paths.interim / "fauna"
    cells = pd.read_csv(src / "cells.csv")
    spc = pd.read_csv(src / "species.csv")
    X = sp.load_npz(src / f"presence_{cfg['faunal']['presence']}.npz").tocsr()
    study = cells.study_share.to_numpy() >= cfg["faunal"]["min_study_share"]
    k = cfg["assemble"]["faunal_k"]
    subsets = {"all": np.ones(len(spc), bool), "birds": (spc.cls == "AVES").to_numpy(),
               "mammals+reptiles+amphibians": (spc.cls != "AVES").to_numpy()}
    labels = {}
    for name, sel in subsets.items():
        Xs = X[study][:, sel]
        Xs = Xs[:, np.asarray(Xs.sum(axis=0)).ravel() > 0]
        labels[name] = fcluster(linkage(squareform(beta_sim(Xs), checks=False), "average"), k, "maxclust")
        log.info("congruence: %s, %d species", name, Xs.shape[1])
    rows = []
    names = list(labels)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            rows.append(dict(a=a, b=b, k=k, adjusted_rand=round(adjusted_rand_score(labels[a], labels[b]), 3),
                             nmi=round(normalized_mutual_info_score(labels[a], labels[b]), 3)))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 3. sensitivity of the ecoregion types to the rule thresholds
# ---------------------------------------------------------------------------
def coarse_layers(cfg, paths: Paths, grid: Grid, f: int) -> dict:
    """Rule inputs on the coarse grid (decimated reads; derived layers recomputed)."""
    shape = (int(np.ceil(grid.height / f)), int(np.ceil(grid.width / f)))

    def rd(name, how=Resampling.nearest):
        with rasterio.open(paths.raster(name)) as s:
            return s.read(1, out_shape=shape, resampling=how)

    cc = cfg["communities"]
    L = {k: rd(v) for k, v in (NUMERIC | CATEGORICAL).items() if v}
    res = grid.res * f
    mask = rd("mask")
    L["coast_km"] = (ndi.distance_transform_edt(mask != 0) * res / 1000).astype("float32")
    # channels are one cell wide: any channel cell marks its block (as in the communities stage)
    chan = block_reduce((read(paths.raster("flowacc_km2")) >= cc["river_min_upland_km2"]).astype("float32"), f)
    chan = chan[: shape[0], : shape[1]] > 0
    L["river_km"] = (ndi.distance_transform_edt(~chan) * res / 1000).astype("float32")
    water = (rd("worldcover") == 80) | (mask == 0)
    L["creek_m"] = np.minimum(ndi.distance_transform_edt(~water) * res, cc["creek_max_m"] + res).astype("float32")
    L["northness"] = northness(L["elev"])
    tex = dune_mask(cfg, L["slope"], rd("tpi_450"), rd("plains") > 0, L["litho"])
    win = max(1, round(cc["dune_window_m"] / res)) | 1
    L["dune"] = ndi.uniform_filter(tex.astype("float32"), size=win) >= cc["dune_min_share"]
    att = pd.read_csv(paths.interim / "l3_attributes.csv", dtype={"l2_code": str})
    l3 = rd("l3").astype(np.int64)
    lut = np.zeros(int(att.unit.max()) + 1, "uint8")
    lut[att.unit.to_numpy()] = att.l2_code.astype(int).to_numpy()
    L["province"] = np.where(l3 > 0, lut[np.minimum(l3, len(lut) - 1)], 0)
    L["_study"] = l3 > 0
    return L


def perturb(rules: dict, sigma: dict, rng) -> dict:
    """Independent random shift of every numeric threshold of every rule."""
    r = copy.deepcopy(rules)
    for c in r["communities"]:
        for k, v in c["where"].items():
            if k not in NUMERIC or k not in sigma:
                continue
            s = sigma[k]
            new = []
            for b in v:
                if b is None:
                    new.append(None)
                elif k == "northness":  # bounded index: additive shift
                    new.append(float(b + rng.normal(0, s)))
                else:
                    new.append(float(b * (1 + rng.normal(0, s))))
            c["where"][k] = new
    return r


def sensitivity(cfg, paths: Paths, eco_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    ec = cfg["evaluate"]
    grid = Grid.load(paths.grid_json)
    f = cfg["grid"]["coarse_factor"]
    rules = load_rules(paths.root / cfg["communities"]["rules"])
    tkeys = list(rules["ecoregion_types"])
    lut = np.zeros(256, "uint8")
    for j, c in enumerate(rules["communities"]):
        lut[j + 1] = tkeys.index(c["ecoregion"]) + 1
    L = coarse_layers(cfg, paths, grid, f)
    study = L.pop("_study")
    base = lut[classify(L, rules)[0]]
    rng = np.random.default_rng(ec["seed"])
    agree = np.zeros(base.shape, "float32")
    for i in range(ec["runs"]):
        agree += lut[classify(L, perturb(rules, ec["sigma"], rng))[0]] == base
        log.info("sensitivity run %d/%d", i + 1, ec["runs"])
    agree /= ec["runs"]
    eco = block_mode(read(paths.raster("ecoregions")), f)[: base.shape[0], : base.shape[1]]
    ok = study & (eco > 0)
    n = int(eco.max()) + 1
    s_sum = np.bincount(eco[ok], weights=agree[ok], minlength=n)
    s_cnt = np.bincount(eco[ok], minlength=n)
    per_eco = pd.DataFrame(dict(unit=np.arange(n), stability=s_sum / np.maximum(s_cnt, 1)))[1:]
    per_eco = eco_df[["unit", "code", "name", "type_name", "area_km2"]].merge(per_eco, on="unit")
    per_eco["stability"] = per_eco.stability.round(3)
    per_type = per_eco.groupby("type_name").apply(
        lambda g: np.average(g.stability, weights=g.area_km2), include_groups=False).round(3).rename("stability")
    return per_eco, per_type.reset_index()


# ---------------------------------------------------------------------------
# 4. correspondence with RESOLVE
# ---------------------------------------------------------------------------
def correspondence(paths: Paths, eco_df: pd.DataFrame, res_tab: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    eco = read(paths.raster("ecoregions"))
    res = read(paths.raster("resolve"))
    rid = np.unique(res)
    pos = np.zeros(int(rid.max()) + 1, np.int64)
    pos[rid] = np.arange(len(rid))
    pc = pair_counts(eco, pos[res], int(eco.max()) + 1, len(rid))
    names = res_tab.set_index("ECO_ID").ECO_NAME
    biomes = res_tab.set_index("ECO_ID").BIOME_NAME.map(_biome_key)
    e = eco_df.set_index("unit")
    rows, agree, total = [], 0, 0
    for u in e.index:
        c = pc[u].copy()
        c[rid == 0] = 0
        if c.sum() == 0:
            continue
        j = int(c.argmax())
        rows.append(dict(unit=u, code=e.code[u], name=e.name[u], resolve_main=names.get(int(rid[j]), ""),
                         resolve_share=round(c[j] / c.sum(), 3)))
        same = [i for i, r in enumerate(rid) if r and biomes.get(int(r), "") == _biome_key(e.biome[u])]
        agree += c[same].sum()
        total += c.sum()
    return pd.DataFrame(rows), agree / max(total, 1)


def run(cfg) -> None:
    paths = Paths(cfg)
    eco_df = pd.read_csv(paths.interim / "ecoregions.csv")
    res_tab = vectors.resolve(cfg)[["ECO_ID", "ECO_NAME", "BIOME_NAME"]].drop_duplicates("ECO_ID")
    out = paths.outputs

    corr, biome_agree = correspondence(paths, eco_df, res_tab)
    corr.to_csv(out / "validation_resolve.csv", index=False)
    log.info("RESOLVE: median ecoregion %.0f%% inside one RESOLVE unit; %d of %d straddle (<60%%); "
             "biome agreement %.0f%% of the area", 100 * corr.resolve_share.median(),
             (corr.resolve_share < 0.6).sum(), len(corr), 100 * biome_agree)

    per_eco, per_type = sensitivity(cfg, paths, eco_df)
    per_eco.to_csv(out / "validation_stability.csv", index=False)
    per_type.to_csv(out / "validation_stability_types.csv", index=False)
    log.info("stability by type:\n%s", per_type.sort_values("stability").to_string(index=False))

    cg = congruence(cfg, paths)
    cg.to_csv(out / "validation_congruence.csv", index=False)
    log.info("taxon congruence:\n%s", cg.to_string(index=False))

    an = assemblages(cfg, paths, eco_df, res_tab)
    an.to_csv(out / "validation_assemblages.csv", index=False)
    log.info("assemblages:\n%s", an.round(3).to_string(index=False))
