"""v0.2 stage — validation of the community ecoregions.

1. **Assemblages (independent data).** GBIF occurrence species lists on 0.25°
   cells (as in ``validate``; cached responses), ANOSIM R for each partition and
   the share of random contiguous partitions with the same number of groups that
   it beats: ecoregions, ecoregion types, faunal regions and biomes against
   the physiographic L3/L2 layers, RESOLVE ecoregions and RESOLVE biomes. GBIF
   records are not used to build any layer of the map; calibration and the audit use
   only half the cells (``vegcheck.holdout``), so the test is repeated on the
   held-out half.
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
5. **Reference maps** (``validation.references``, e.g. EPA Level III/IV in Texas). Compared with the finished
   map only, never used to build it: the reference unit covering most of each ecoregion, and the V-measure
   between ecoregion types and reference units (1 = the same partition, 0 = unrelated).

Outputs: ``outputs/validation_*.csv`` and a stability column for the ecoregions.
"""
from __future__ import annotations

import copy
import json
import logging
import time

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
from .vegcheck import holdout

log = logging.getLogger(__name__)


def _biome_key(s: str) -> str:
    s = str(s).replace(" and ", " & ")
    return "Rock & Ice" if s in ("N/A", "nan") else s


# ---------------------------------------------------------------------------
# 1. assemblages
# ---------------------------------------------------------------------------
def rarefy(counts: dict, n: int, rng) -> list:
    """Species in ``n`` records drawn without replacement from a cell's records (``counts``: species -> records)."""
    names = list(counts)
    draw = rng.multivariate_hypergeometric(np.fromiter(counts.values(), np.int64, len(names)), n)
    return [names[i] for i in np.flatnonzero(draw)]


def presence(lists: list) -> np.ndarray:
    """Cells x species boolean matrix from per-cell species lists."""
    kpos = {k: j for j, k in enumerate(sorted({k for s in lists for k in s}))}
    P = np.zeros((len(lists), len(kpos)), bool)
    for a, s in enumerate(lists):
        P[a, [kpos[k] for k in s]] = True
    return P


def summarise_runs(runs: pd.DataFrame) -> pd.DataFrame:
    """Median and 5-95 % range of R and pct_of_null_below over the rarefaction draws."""
    q = lambda p: (lambda s: float(np.nanpercentile(s, p)))  # noqa: E731
    g = runs.groupby(["taxon", "cells", "partition"], sort=False)
    out = g.agg(n_cells=("n_cells", "median"), n_groups=("n_groups", "median"),
                R_median=("R", "median"), R_p5=("R", q(5)), R_p95=("R", q(95)),
                null_R_mean=("null_R_mean", "median"),
                pct_median=("pct_of_null_below", "median"), pct_p5=("pct_of_null_below", q(5)),
                pct_p95=("pct_of_null_below", q(95)))
    return out.reset_index()


def assemblages(cfg, paths: Paths, eco_df: pd.DataFrame, res_tab: pd.DataFrame, rarefied: bool = False,
                parts_extra: dict | None = None) -> pd.DataFrame:
    """ANOSIM test of every partition on GBIF species lists of 0.25-degree cells, all cells and held-out cells.
    ``rarefied``: each cell's list from ``validation.rarefy.records`` records drawn at random (repeated
    ``runs`` times; returns every run). ``parts_extra``: more partitions, as one label per cell."""
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
    # faunal regions are an overlay (ecoregions can span several): sample their own polygons
    import geopandas as gpd

    faunal = cfg["assemble"].get("faunal", True)
    fr = gpd.read_file(paths.vector("ecoregions"), layer="faunal_regions" if faunal else "biomes")
    pts = gpd.GeoDataFrame(geometry=gpd.points_from_xy(xy[:, 0], xy[:, 1]), crs=fr.crs)
    g_fau = None if not faunal else gpd.sjoin(pts, fr[["faunal_region", "geometry"]], how="left", predicate="within")         .groupby(level=0).faunal_region.first().reindex(range(len(pts))).fillna(0).astype(int).to_numpy()
    e = eco_df.set_index("unit")
    l2map = pd.read_csv(paths.interim / "l3_units.csv", dtype={"l2_code": str}).set_index("unit").l2_code
    rb = res_tab.set_index("ECO_ID").BIOME_NAME.map(_biome_key)
    parts_all = {
        "ecoregions": g_eco,
        "ecoregion types": pd.Series(g_eco).map(e.type).fillna("").to_numpy(),
        "faunal regions": g_fau,
        "biomes": pd.Series(g_eco).map(e.biome).map(_biome_key).fillna("").to_numpy(),
        "physiographic L3": g_l3,
        "physiographic L2": pd.Series(g_l3).map(l2map).fillna("").to_numpy(),
        "RESOLVE ecoregions": g_res,
        "RESOLVE biomes": pd.Series(g_res).map(rb).fillna("").to_numpy(),
    }
    if g_fau is None:
        del parts_all["faunal regions"]
    parts_all.update(parts_extra or {})
    held = holdout([lon for lon, _, _ in cells], [lat for _, lat, _ in cells], cd)
    rc = vc["rarefy"]
    out = []
    for taxon, extra in TAXA.items():
        res = fetch_cells(client, cells, extra, gc.get("threads", 2))
        cnt = np.array([x["count"] for x in res])
        nsp = np.array([len(x["species"]) for x in res])
        ok = (cnt >= gc["min_records_cell"]) & (nsp >= gc["min_species_cell"]) & (g_eco > 0) & (g_res > 0)
        if rarefied:  # records identified to species (the facet counts) must cover the draw
            ok &= np.array([sum(x["species"].values()) for x in res]) >= rc["records"]
        # all cells, and held-out cells only (never used by calibration or the audit)
        for tag, sel in (("all", ok), ("holdout", ok & held)):
            idx = np.flatnonzero(sel)
            if len(idx) < 30:
                continue
            parts = {k: v[idx] for k, v in parts_all.items()}
            if rarefied:
                t0 = time.time()
                for r in range(rc["runs"]):
                    rng = np.random.default_rng([rc["seed"], r])
                    P = presence([rarefy(res[i]["species"], rc["records"], rng) for i in idx])
                    df = compare_partitions(P, parts, xy[idx], vc["random_partitions"], 0, rc["seed"] + r,
                                            rc["dissimilarity"])
                    df.insert(0, "run", r)
                    df.insert(0, "dissimilarity", rc["dissimilarity"])
                    df.insert(0, "cells", tag)
                    df.insert(0, "taxon", taxon)
                    out.append(df)
                log.info("assemblages %s (%s), rarefied to %d records x %d runs: %d cells, %.0f s",
                         taxon, tag, rc["records"], rc["runs"], len(idx), time.time() - t0)
                continue
            P = presence([list(res[i]["species"]) for i in idx])
            for kind in vc["dissimilarities"]:
                df = compare_partitions(P, parts, xy[idx], vc["random_partitions"], vc["anosim_permutations"],
                                        vc["seed"], kind)
                df.insert(0, "dissimilarity", kind)
                df.insert(0, "cells", tag)
                df.insert(0, "taxon", taxon)
                out.append(df)
            log.info("assemblages %s (%s): %d cells", taxon, tag, len(idx))
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


def v_measure(c: np.ndarray) -> tuple[float, float, float]:
    """Homogeneity, completeness and V-measure of a contingency table (rows: our units, columns: reference units)."""
    c = c[c.sum(1) > 0][:, c.sum(0) > 0].astype(float)
    p = c / c.sum()
    ent = lambda q: -np.sum(q[q > 0] * np.log(q[q > 0]))  # noqa: E731
    h_ref, h_our, h_joint = ent(p.sum(0)), ent(p.sum(1)), ent(p)
    hom = 1 - (h_joint - h_our) / h_ref if h_ref else 1.0  # each of our units holds one reference unit
    com = 1 - (h_joint - h_ref) / h_our if h_our else 1.0  # each reference unit lies in one of ours
    return hom, com, (2 * hom * com / (hom + com) if hom + com else 0.0)


def reference_comparison(cfg, paths: Paths, eco_df: pd.DataFrame, ref: dict) -> tuple[pd.DataFrame, dict]:
    """One reference map against the ecoregions: per ecoregion its main reference unit and that unit's share;
    for the map, the V-measure of ecoregion types and of ecoregions against the reference units."""
    import geopandas as gpd

    from .grid import rasterize_geoms

    grid = Grid.load(paths.grid_json)
    g = gpd.read_file(f"zip://{paths.raw / 'references' / ref['file']}").to_crs(grid.crs)
    names = sorted(g[ref["field"]].unique())
    ids = {n: i + 1 for i, n in enumerate(names)}
    lab = rasterize_geoms(((geom, ids[n]) for geom, n in zip(g.geometry, g[ref["field"]])), grid, dtype="uint16")
    eco = read(paths.raster("ecoregions"))
    pc = pair_counts(eco, lab, int(eco.max()) + 1, len(names) + 1)
    pc[0], pc[:, 0] = 0, 0  # outside the map or the reference
    e = eco_df.set_index("unit")
    rows = [dict(unit=u, code=e.code[u], name=e.name[u], type_name=e.type_name[u], reference_main=names[pc[u].argmax() - 1],
                 reference_share=round(pc[u].max() / pc[u].sum(), 3)) for u in e.index if u < len(pc) and pc[u].sum()]
    types = pd.DataFrame(pc[e.index]).groupby(e.type.to_numpy()).sum().to_numpy()
    out = {"reference": ref["name"]}
    for k, c in (("types", types), ("ecoregions", pc)):
        out.update(zip([f"{k}_homogeneity", f"{k}_completeness", f"{k}_v_measure"], np.round(v_measure(c), 3)))
    return pd.DataFrame(rows), out


def run(cfg) -> None:
    paths = Paths(cfg)
    eco_df = pd.read_csv(paths.interim / "ecoregions.csv")
    refs = cfg.get("validation", {}).get("references", [])
    if refs:
        summary = []
        for ref in refs:
            per, s = reference_comparison(cfg, paths, eco_df, ref)
            per.to_csv(paths.outputs / f"validation_reference_{ref['name'].lower().replace(' ', '_')}.csv", index=False)
            summary.append(s)
            log.info("%s: V-measure %.2f (types), %.2f (ecoregions)", ref["name"], s["types_v_measure"], s["ecoregions_v_measure"])
        pd.DataFrame(summary).to_csv(paths.outputs / "validation_references.csv", index=False)
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

    if cfg["assemble"].get("faunal", True):
        cg = congruence(cfg, paths)
        cg.to_csv(out / "validation_congruence.csv", index=False)
        log.info("taxon congruence:\n%s", cg.to_string(index=False))

    if not cfg["validation"].get("legacy_assemblages", False):
        log.info("assemblage tests: see the scorecard stage (legacy ANOSIM tests off: validation.legacy_assemblages)")
        return
    an = assemblages(cfg, paths, eco_df, res_tab)
    an.to_csv(out / "validation_assemblages.csv", index=False)
    runs = assemblages(cfg, paths, eco_df, res_tab, rarefied=True)
    runs.to_csv(paths.interim / "validation_assemblages_rarefied_runs.csv", index=False)
    summarise_runs(runs).to_csv(out / "validation_assemblages_rarefied.csv", index=False)
    log.info("assemblages:\n%s", an.round(3).to_string(index=False))
