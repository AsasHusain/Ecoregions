"""Stage 11 — GBIF-based validation.

1. **Sampling density per L3 unit.** For every unit polygon (simplified to fit
   a URL, see ``_wkt``) we ask the GBIF occurrence API for the record count and
   the species facet. This gives records/100 km² and the number of species, and
   flags units that are effectively unsampled.
2. **Do L3 boundaries separate species assemblages?** On a 0.25° grid we build
   presence/absence species lists per cell (all taxa, birds, plants) and run
   ANOSIM (Clarke 1993) on Jaccard dissimilarities with cells grouped by
   (a) L3, (b) L2 and (c) RESOLVE. Because ANOSIM R rises with the number of
   groups and with spatial autocorrelation alone, each partition is compared
   with a null distribution of *random contiguous partitions with the same
   number of groups* (seeded Voronoi on the same cells). A partition separates
   assemblages better than chance if its R exceeds most null R values.

All API responses are cached in ``data/raw/gbif`` so reruns are offline.
Only aggregated counts are stored; no occurrence records are redistributed.
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
from urllib.parse import quote_plus
from shapely.geometry import box
from shapely.geometry.polygon import orient

from .config import Paths

log = logging.getLogger(__name__)

BASIS = ["HUMAN_OBSERVATION", "PRESERVED_SPECIMEN", "OBSERVATION", "MACHINE_OBSERVATION",
         "MATERIAL_SAMPLE", "OCCURRENCE"]
TAXA = {"all": {}, "birds": {"classKey": 212}, "plants": {"kingdomKey": 6}}
FACET_CAP = 14999


class GBIF:
    def __init__(self, api: str, cache: Path, pause: float):
        self.api = api
        self.cache = cache
        self.cache.mkdir(parents=True, exist_ok=True)
        self.pause = pause
        self.s = requests.Session()
        self.s.headers["User-Agent"] = "pakistan-ecoregions/0.1 (research; aggregated counts only)"

    def species_facet(self, wkt: str, extra: dict) -> dict:
        params = dict(geometry=wkt, limit=0, facet="speciesKey", facetLimit=FACET_CAP + 1,
                      hasCoordinate="true", hasGeospatialIssue="false", occurrenceStatus="PRESENT",
                      basisOfRecord=BASIS, **extra)
        key = hashlib.sha1(json.dumps(params, sort_keys=True).encode()).hexdigest()
        f = self.cache / f"{key}.json"
        if f.exists():
            return json.loads(f.read_text())
        for attempt in range(10):
            try:
                r = self.s.get(f"{self.api}/occurrence/search", params=params, timeout=120)
                if r.status_code == 429 or r.status_code >= 500:
                    ra = r.headers.get("Retry-After")
                    if ra and ra.isdigit():
                        time.sleep(int(ra))
                    raise requests.HTTPError(f"HTTP {r.status_code}")
                if 400 <= r.status_code < 500:  # malformed query: retrying will not help
                    raise ValueError(f"GBIF rejected the query (HTTP {r.status_code})")
                r.raise_for_status()
                d = r.json()
                out = dict(count=d["count"],
                           species={c["name"]: c["count"] for c in (d["facets"][0]["counts"] if d["facets"] else [])})
                f.write_text(json.dumps(out))
                time.sleep(self.pause)
                return out
            except requests.RequestException as e:
                wait = min(2 ** (attempt + 1), 120)
                log.warning("GBIF retry in %ss: %s", wait, e)
                time.sleep(wait)
        raise RuntimeError("GBIF API unavailable after retries")


def _wkt(geom, max_chars: int = 2800) -> str:
    """WGS84 WKT acceptable to the GBIF API: counter-clockwise shells, no holes,
    coordinates rounded to 1e-3 deg (~100 m) and short enough for a GET URL
    (the API's server rejects URLs much over ~3 kB of encoded WKT).
    Simplification is for the query only; spot checks showed record counts
    within a few percent of the full-resolution polygon."""
    from shapely import set_precision
    from shapely.geometry import MultiPolygon, Polygon

    tol, keep = 0.001, None
    while True:
        s = geom.simplify(tol, preserve_topology=True)
        polys = list(s.geoms) if s.geom_type == "MultiPolygon" else [s]
        polys = sorted((p for p in polys if p.area > 1e-5), key=lambda p: -p.area)[:keep] or polys[:1]
        polys = [Polygon(orient(p if p.is_valid else p.buffer(0), 1.0).exterior) for p in polys]
        g = polys[0] if len(polys) == 1 else MultiPolygon(polys)
        g = set_precision(g, 1e-3)
        g = orient(g, 1.0) if g.geom_type == "Polygon" else MultiPolygon([orient(p, 1.0) for p in g.geoms])
        w = g.wkt
        if len(quote_plus(w)) <= max_chars:  # the limit applies to the URL-encoded form
            return w
        if tol < 1.0:
            tol *= 1.5
        elif len(polys) > 1:  # every part already at ~4 vertices (many-islet units): drop the smallest
            keep = len(polys) - 1
        else:
            raise ValueError("geometry cannot be shortened to a GBIF query")


# ---------------------------------------------------------------------------
# ANOSIM
# ---------------------------------------------------------------------------
def jaccard_matrix(P: np.ndarray) -> np.ndarray:
    P = P.astype("float32")
    inter = P @ P.T
    n = P.sum(1)
    union = n[:, None] + n[None, :] - inter
    with np.errstate(invalid="ignore", divide="ignore"):
        d = 1 - inter / union
    np.fill_diagonal(d, 0)
    return np.nan_to_num(d, nan=1.0)


def dissimilarity(P: np.ndarray, kind: str) -> np.ndarray:
    """Cell x cell dissimilarity of species lists: ``jaccard``, or ``simpson`` (beta_sim, which does
    not count a short list as unlike a long one that contains it, so it is less sensitive to effort)."""
    if kind == "jaccard":
        return jaccard_matrix(P)
    if kind == "simpson":
        from .faunal import beta_sim

        return beta_sim(P)
    raise ValueError(f"unknown dissimilarity {kind!r}")


def anosim_r(rank: np.ndarray, groups: np.ndarray) -> float:
    iu = np.triu_indices(len(groups), 1)
    same = groups[iu[0]] == groups[iu[1]]
    r = rank[iu]
    if same.all() or not same.any():
        return np.nan
    M = len(r)
    return float((r[~same].mean() - r[same].mean()) / (M / 2))


def ranks_of(d: np.ndarray) -> np.ndarray:
    iu = np.triu_indices(d.shape[0], 1)
    from scipy.stats import rankdata

    rk = np.zeros_like(d)
    rk[iu] = rankdata(d[iu])
    return rk + rk.T


def random_partitions(xy: np.ndarray, k: int, n: int, rng) -> list[np.ndarray]:
    """Random contiguous partitions by Voronoi around k random cells."""
    from scipy.spatial import cKDTree

    out = []
    for _ in range(n):
        seeds = xy[rng.choice(len(xy), size=min(k, len(xy)), replace=False)]
        out.append(cKDTree(seeds).query(xy)[1])
    return out


def compare_partitions(P: np.ndarray, partitions: dict, xy: np.ndarray, n_null: int, n_perm: int, seed: int,
                       dissim: str = "jaccard") -> pd.DataFrame:
    """ANOSIM R of each partition, its permutation p (skipped when ``n_perm`` is 0) and its rank among
    ``n_null`` random contiguous partitions with the same number of groups."""
    rng = np.random.default_rng(seed)
    d = dissimilarity(P, dissim)
    rows = []
    for name, g in partitions.items():
        # drop groups with a single sampled cell (no within-group pairs)
        vc = pd.Series(g).value_counts()
        keep = np.isin(g, vc[vc >= 2].index)
        gg = g[keep]
        if len(np.unique(gg)) < 2:
            rows.append(dict(partition=name, n_cells=int(keep.sum()), n_groups=len(np.unique(gg)), R=np.nan))
            continue
        sub = ranks_of(d[np.ix_(keep, keep)])  # rank within the analysed subset (R in [-1, 1])
        R = anosim_r(sub, gg)
        perm = np.array([anosim_r(sub, rng.permutation(gg)) for _ in range(n_perm)]) if n_perm else None
        k = len(np.unique(gg))
        null = np.array([anosim_r(sub, p) for p in random_partitions(xy[keep], k, n_null, rng)])
        rows.append(dict(partition=name, n_cells=int(keep.sum()), n_groups=k, R=R,
                         p_perm=float((np.sum(perm >= R) + 1) / (n_perm + 1)) if n_perm else np.nan,
                         null_R_mean=float(np.nanmean(null)), null_R_p95=float(np.nanpercentile(null, 95)),
                         pct_of_null_below=float(np.mean(null < R) * 100)))
    return pd.DataFrame(rows)


def assemblage_cells(cfg) -> list:
    """0.25-degree cells covering at least half their area of the study area."""
    paths = Paths(cfg)
    cd = cfg["sources"]["gbif"]["cell_deg"]
    study = gpd.read_file(paths.vector("boundaries"), layer="study_area").to_crs(4326).union_all()
    w, s, e, n = study.bounds
    from .confirm import load_mask

    m = load_mask(cfg)
    sealed = None if m is None or m.get("open") else m["_conf"].union(m["_buf"])  # confirmation and buffer cells left out
    cells = []
    for lat in np.arange(np.floor(s / cd) * cd, n, cd):
        for lon in np.arange(np.floor(w / cd) * cd, e, cd):
            b = box(lon, lat, lon + cd, lat + cd)
            if b.intersection(study).area / b.area >= 0.5 and not (sealed is not None and b.intersects(sealed)):
                cells.append((round(lon, 4), round(lat, 4), b))
    log.info("GBIF assemblage grid: %d cells of %.2f°", len(cells), cd)
    return cells


def fetch_cells(client: "GBIF", cells: list, extra: dict, threads: int) -> list:
    with ThreadPoolExecutor(threads) as ex:
        return list(ex.map(lambda c: client.species_facet(orient(c[2], 1.0).wkt, extra), cells))


def prefetch(cfg) -> None:
    """Warm the cache for the grid-cell queries (independent of the L3 units)."""
    gc = cfg["sources"]["gbif"]
    client = GBIF(gc["api"], Paths(cfg).raw / "gbif", gc["request_pause_s"])
    cells = assemblage_cells(cfg)
    for taxon, extra in TAXA.items():
        fetch_cells(client, cells, extra, gc.get("threads", 2))
        log.info("prefetched %s", taxon)


def run(cfg) -> None:
    paths = Paths(cfg)
    gc = cfg["sources"]["gbif"]
    vc = cfg["validation"]
    client = GBIF(gc["api"], paths.raw / "gbif", gc["request_pause_s"])
    units = gpd.read_file(paths.vector("l3_polygons"), layer="l3").to_crs(4326)

    # --- 1. per-unit density -------------------------------------------------
    def unit_stats(row):
        r = client.species_facet(_wkt(row.geometry), {})
        return dict(unit=row.unit, gbif_records=r["count"], gbif_species=len(r["species"]),
                    gbif_species_capped=len(r["species"]) >= FACET_CAP)

    with ThreadPoolExecutor(gc.get("threads", 2)) as ex:
        st = pd.DataFrame(list(ex.map(unit_stats, [r for _, r in units.iterrows()])))
    st = st.merge(units[["unit", "area_km2"]], on="unit")
    st["gbif_records_per_100km2"] = st.gbif_records / st.area_km2 * 100
    st["gbif_species_per_100km2"] = st.gbif_species / st.area_km2 * 100
    st["gbif_sampling"] = pd.cut(st.gbif_records_per_100km2, [-1, 1, 10, 100, np.inf],
                                 labels=["unsampled (<1/100 km²)", "sparse (1-10)", "moderate (10-100)", "well sampled (>100)"]).astype(str)
    st.drop(columns="area_km2").to_csv(paths.outputs / "gbif_unit_stats.csv", index=False)
    log.info("GBIF per-unit: median %.1f records/100 km²; %d of %d units unsampled",
             st.gbif_records_per_100km2.median(), (st.gbif_records_per_100km2 < 1).sum(), len(st))

    # --- 2. assemblage test ----------------------------------------------------
    import rasterio
    from pyproj import Transformer

    cells = assemblage_cells(cfg)
    cd = gc["cell_deg"]
    grid = json.loads(paths.grid_json.read_text())
    t = Transformer.from_crs("EPSG:4326", grid["crs"], always_xy=True)
    with rasterio.open(paths.raster("l3")) as l3r, rasterio.open(paths.raster("resolve")) as rr:
        xy = np.array([t.transform(lon + cd / 2, lat + cd / 2) for lon, lat, _ in cells])
        g_l3 = np.array([v[0] for v in l3r.sample(xy)])
        g_res = np.array([v[0] for v in rr.sample(xy)])
    l2map = pd.read_csv(paths.interim / "l3_units.csv", dtype={"l2_code": str}).set_index("unit").l2_code.to_dict()
    g_l2 = np.array([l2map.get(int(u), "") for u in g_l3])

    results = []
    density = []
    for taxon, extra in TAXA.items():
        res = fetch_cells(client, cells, extra, gc.get("threads", 2))
        cnt = np.array([r["count"] for r in res])
        nsp = np.array([len(r["species"]) for r in res])
        ok = (cnt >= gc["min_records_cell"]) & (nsp >= gc["min_species_cell"]) & (g_l3 > 0) & (g_res > 0)
        density.append(pd.DataFrame(dict(taxon=taxon, lon=[c[0] for c in cells], lat=[c[1] for c in cells],
                                         records=cnt, species=nsp, used=ok)))
        log.info("taxon %s: %d of %d cells pass the sampling threshold", taxon, ok.sum(), len(cells))
        if ok.sum() < 30:
            results.append(pd.DataFrame([dict(taxon=taxon, partition="(too few sampled cells)", n_cells=int(ok.sum()))]))
            continue
        idx = np.flatnonzero(ok)
        keys = sorted({k for i in idx for k in res[i]["species"]})
        kpos = {k: j for j, k in enumerate(keys)}
        P = np.zeros((len(idx), len(keys)), bool)
        for a, i in enumerate(idx):
            P[a, [kpos[k] for k in res[i]["species"]]] = True
        parts = {"L3 (this study)": g_l3[idx], "L2 (this study)": g_l2[idx], "L1 RESOLVE": g_res[idx]}
        df = compare_partitions(P, parts, xy[idx], vc["random_partitions"], vc["anosim_permutations"], vc["seed"])
        df.insert(0, "taxon", taxon)
        results.append(df)
    out = pd.concat(results, ignore_index=True)
    out.to_csv(paths.outputs / "gbif_anosim.csv", index=False)
    pd.concat(density).to_csv(paths.outputs / "gbif_cells.csv", index=False)
    log.info("\n%s", out.to_string())
