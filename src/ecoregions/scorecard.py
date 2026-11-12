"""Scorecard: held-out species tests for every mapped area (structure after the independent review, docs/review).

Status of results from the areas built so far: EXPLORATORY. Their held-out cells have been viewed while the maps were
developed, so no relabelling makes them an independent confirmation; that needs fresh, spatially separated data.

Design
* Cells: degree grids that nest in the 0.25-degree calibration/held-out checkerboard (``scales``: 0.25, 0.125), so no
  cell mixes records of the two roles. Eligibility (land >= half the cell, enough records and species) is fixed before
  any map is scored, and every partition is scored on the SAME cells; units with a single cell are kept.
* Each cell takes the unit covering most of its area (``purity_points``^2 sample points); purity is recomputed for
  every partition. The old test's centre point is sampled at the true cell centre.
* Co-primary endpoints, each paired against RESOLVE (and the other comparators) on identical cells:
  A. Prediction loss: a held-out cell's rarefied species list predicted by its unit's species frequencies on the
     development (calibration) cells, shrunk toward the overall frequency (beta-binomial, prior strength
     ``prior_cells``); Brier loss averaged over species, then cells within 1-degree blocks, then blocks. Lower is
     better. Skill = 1 - loss / loss of the overall-frequency prior.
  B. Boundary contrast: mean Jaccard dissimilarity of held-out cell pairs 10-50 km apart that lie across a unit
     boundary minus that of pairs inside one unit, matched by 10 km distance bins (bins weighted equally).
     (Map-level; per-boundary contrasts are a later step.)
  Secondary: ANOSIM R on the full lists with Simpson's beta, as a z-score against Voronoi random partitions.
* Baselines: RESOLVE, reference maps (comparison only), climate-only k-means, a spatial-only predictor (frequencies of
  the ``knn`` nearest development cells, endpoint A only), and ``random_partitions`` contiguous random maps grown on
  the land-cell graph (independent of where records are).
* Equal unit count: our map coarsened along a tree built from construction data only (each ecoregion's community
  mix, area-weighted Ward merges of neighbours; archived), at RESOLVE's full-domain unit count. No GBIF record and no
  reference boundary chooses a merge.
* Controls the test must pass: renaming units changes nothing; moving patches of cells across boundaries (``damage``)
  must score worse; cutting every unit in two at random must not score better.
* Uncertainty: ``bootstrap`` resamples of 1-degree blocks of held-out cells for every paired difference.
* Coverage: land area of the eligible development and held-out cells against the study area.
Writes outputs/scorecard.csv (every draw), scorecard_summary.csv, scorecard_contrasts.csv, scorecard_coverage.csv and
coarsening_tree.csv.
"""
from __future__ import annotations

import json
import logging
import time

import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer

from .config import Paths
from .evaluate import presence, rarefy
from .validate import GBIF, TAXA, assemblage_cells, fetch_cells, random_partitions
from .vegcheck import holdout

log = logging.getLogger(__name__)
STATUS = "exploratory"  # held-out cells of these areas were viewed during development
FAUNAL = "our faunal ecoregions"  # candidate for vertebrates (birds); floral ecoregions are the candidate for plants


# ---------------------------------------------------------------- statistics
class Ranked:
    """Upper-triangle ranks of a cell x cell dissimilarity, for many ANOSIM R evaluations."""

    def __init__(self, d: np.ndarray):
        from scipy.stats import rankdata

        self.iu = np.triu_indices(d.shape[0], 1)
        self.r = rankdata(d[self.iu])
        self.half = len(self.r) / 2

    def R(self, g: np.ndarray) -> float:
        same = g[self.iu[0]] == g[self.iu[1]]
        if same.all() or not same.any():
            return np.nan
        return float((self.r[~same].mean() - self.r[same].mean()) / self.half)


def zscore(x: float, null: np.ndarray) -> float:
    null = null[np.isfinite(null)]
    if not np.isfinite(x) or len(null) < 5 or null.std() == 0:
        return np.nan
    return float((x - null.mean()) / null.std())


def unit_frequencies(Pd: np.ndarray, gd: np.ndarray, prior: np.ndarray, strength: float) -> tuple[np.ndarray, dict]:
    """Species frequencies per unit on development cells, shrunk toward ``prior`` with ``strength`` pseudo-cells."""
    labs = np.unique(gd)
    F = np.stack([(Pd[gd == u].sum(0) + strength * prior) / ((gd == u).sum() + strength) for u in labs])
    return F, {u: i for i, u in enumerate(labs)}


def brier_cells(Pt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Per held-out cell: mean over species of (y - p)^2 for presence y and predicted probability p (row per cell)."""
    S = Pt.shape[1]
    return ((Pt ** 2).sum(1) - 2 * (Pt * pred).sum(1) + (pred ** 2).sum(1)) / S


def prediction_loss(Pd, gd, Pt, gt, prior, strength) -> np.ndarray:
    """Per held-out cell Brier loss of a partition (units absent from development cells get the prior)."""
    F, pos = unit_frequencies(Pd, gd, prior, strength)
    rows = np.array([pos.get(u, -1) for u in gt])
    pred = np.where(rows[:, None] >= 0, F[np.maximum(rows, 0)], prior[None, :])
    return brier_cells(Pt, pred)


def block_mean(loss: np.ndarray, blocks: np.ndarray, w: np.ndarray | None = None) -> float:
    """Mean over blocks of the mean over cells (``w``: block weights from a bootstrap draw)."""
    ub, inv = np.unique(blocks, return_inverse=True)
    per = np.bincount(inv, weights=loss) / np.bincount(inv)
    w = np.ones(len(ub)) if w is None else w  # bootstrap: one weight per unique block, in sorted order
    return float((per * w).sum() / w.sum())


def pair_table(xy: np.ndarray, blocks: np.ndarray, lo: float = 10e3, hi: float = 50e3, binw: float = 10e3):
    """Held-out cell pairs lo-hi metres apart: indices, distance bin and the two blocks."""
    from scipy.spatial import cKDTree

    pr = np.array(sorted(cKDTree(xy).query_pairs(hi)), dtype=int).reshape(-1, 2)
    d = np.linalg.norm(xy[pr[:, 0]] - xy[pr[:, 1]], axis=1)
    keep = d >= lo
    pr, d = pr[keep], d[keep]
    return pr, ((d - lo) // binw).astype(int), blocks[pr[:, 0]], blocks[pr[:, 1]]


def boundary_contrast(D_pairs: np.ndarray, across: np.ndarray, bins: np.ndarray, w: np.ndarray | None = None) -> float:
    """Mean over distance bins of (mean dissimilarity across boundaries - mean within units); bins need both kinds."""
    out = []
    w = np.ones(len(D_pairs)) if w is None else w
    for b in np.unique(bins):
        m = bins == b
        a, i = m & across, m & ~across
        if w[a].sum() > 0 and w[i].sum() > 0:
            out.append((D_pairs[a] * w[a]).sum() / w[a].sum() - (D_pairs[i] * w[i]).sum() / w[i].sum())
    return float(np.mean(out)) if out else np.nan


def jaccard_pairs(P: np.ndarray, pr: np.ndarray) -> np.ndarray:
    a, b = P[pr[:, 0]], P[pr[:, 1]]
    inter = (a & b).sum(1)
    union = (a | b).sum(1)
    return 1 - inter / np.maximum(union, 1)


# ---------------------------------------------------------------- partitions sampled over each cell's area
def cell_points(cells: list, n: int) -> tuple[np.ndarray, np.ndarray]:
    """n x n lon/lat sample points inside each cell (lon0, lat0, deg); returns lon, lat of shape (cells, n*n)."""
    f = (np.arange(n) + 0.5) / n
    fx, fy = np.meshgrid(f, f)
    lon = np.array([c[0] + fx.ravel() * c[2] for c in cells])
    lat = np.array([c[1] + fy.ravel() * c[2] for c in cells])
    return lon, lat


def majority(lab: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per row: the most frequent label (0 = outside) and its share."""
    out, share = np.zeros(len(lab), int), np.zeros(len(lab))
    for i, row in enumerate(lab):
        v, c = np.unique(row, return_counts=True)
        out[i], share[i] = v[c.argmax()], c.max() / len(row)
    return out, share


def sample_raster(path, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    with rasterio.open(path) as r:
        return np.array([v[0] for v in r.sample(zip(x.ravel(), y.ravel()))]).reshape(x.shape)


def sample_array(arr: np.ndarray, transform, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    col = np.floor((x - transform.c) / transform.a).astype(int)
    row = np.floor((y - transform.f) / transform.e).astype(int)
    ok = (row >= 0) & (row < arr.shape[0]) & (col >= 0) & (col < arr.shape[1])
    out = np.zeros(x.shape, arr.dtype)
    out[ok] = arr[row[ok], col[ok]]
    return out


def coarse_labels(cfg, paths: Paths, step: int = 10):
    """Coarse rasters on the analysis grid (every ``step``-th pixel): our map, RESOLVE, reference maps, climate."""
    import geopandas as gpd
    from rasterio.features import rasterize
    from rasterio.transform import Affine

    with rasterio.open(paths.raster("ecoregions")) as e:
        eco = e.read(1, out_shape=(e.height // step, e.width // step), resampling=rasterio.enums.Resampling.nearest)
        tr = e.transform * Affine.scale(e.width / eco.shape[1], e.height / eco.shape[0])
        crs = e.crs
    with rasterio.open(paths.raster("resolve")) as r:
        res = r.read(1, out_shape=eco.shape, resampling=rasterio.enums.Resampling.nearest)
    refs = {}
    import pyogrio

    if "faunal_regions" in [n for n, _ in pyogrio.list_layers(paths.vector("ecoregions"))]:  # our own faunal map
        fr = gpd.read_file(paths.vector("ecoregions"), layer="faunal_regions").to_crs(crs)
        refs[FAUNAL] = rasterize(((geom, int(v)) for geom, v in zip(fr.geometry, fr.faunal_region)),
                                 out_shape=eco.shape, transform=tr, dtype="uint16")
    for ref in cfg.get("validation", {}).get("references", []):
        g = gpd.read_file(f"zip://{paths.raw / 'references' / ref['file']}").to_crs(crs)
        ids = {n: i + 1 for i, n in enumerate(sorted(g[ref["field"]].unique()))}
        refs[ref["name"]] = rasterize(((geom, ids[n]) for geom, n in zip(g.geometry, g[ref["field"]])),
                                      out_shape=eco.shape, transform=tr, dtype="uint16")
    layers = []
    for name in cfg["scorecard"]["climate_layers"]:
        with rasterio.open(paths.raster(name)) as r:
            layers.append(r.read(1, out_shape=eco.shape, resampling=rasterio.enums.Resampling.average).astype(float))
    X = np.stack(layers, -1)
    ok = (eco > 0) & np.isfinite(X).all(-1)
    Z = X[ok]
    Z = (Z - Z.mean(0)) / Z.std(0)
    return tr, eco, res, refs, (ok, Z)


def climate_kmeans(ok: np.ndarray, Z: np.ndarray, k: int, seed: int) -> np.ndarray:
    from sklearn.cluster import KMeans

    lab = np.zeros(ok.shape, "int32")
    lab[ok] = KMeans(k, n_init=4, random_state=seed).fit_predict(Z) + 1
    return lab


# ---------------------------------------------------------------- species lists per cell
def local_plant_cells(cfg, paths: Paths, deg: float, roles=("development",)) -> list:
    """Wild plant records of this area's GBIF downloads (filters as vegcheck), as cells (lon0, lat0, deg, counts)."""
    cols = ["gbifID", "kingdom", "species", "basisOfRecord", "cultivated", "decimalLatitude", "decimalLongitude",
            "coordinateUncertaintyInMeters"]
    recs = pd.concat([pd.read_csv(paths.raw / "gbif_dwca" / f"occurrences_{d['key']}.csv.gz", dtype=str, usecols=cols)
                      for d in cfg["sources"]["gbif_dwca"]["downloads"]]).drop_duplicates("gbifID")
    recs = recs[(recs.kingdom == "Plantae") & (recs.cultivated != "True") & (recs.basisOfRecord != "FOSSIL_SPECIMEN")]
    unc = pd.to_numeric(recs.coordinateUncertaintyInMeters, errors="coerce")
    recs = recs[~(unc > cfg["vegcheck"]["max_uncertainty_m"])][["species", "decimalLatitude", "decimalLongitude"]].dropna()
    from .confirm import load_mask, role

    r = role(load_mask(cfg), recs.decimalLongitude.astype(float), recs.decimalLatitude.astype(float))
    recs = recs[np.isin(r, roles)]  # development only while a map is built; confirmation only when it is opened
    i = np.floor(recs.decimalLongitude.astype(float) / deg).astype(int)
    j = np.floor(recs.decimalLatitude.astype(float) / deg).astype(int)
    cnt = recs.assign(i=i, j=j).groupby(["i", "j", "species"]).size()
    return [(round(a * deg, 6), round(b * deg, 6), deg, s.droplevel([0, 1]).to_dict())
            for (a, b), s in cnt.groupby(level=[0, 1])]


def api_cells(cfg, paths: Paths, taxon: str) -> list:
    gc = cfg["sources"]["gbif"]
    client = GBIF(gc["api"], paths.raw / "gbif", gc["request_pause_s"])
    cells = assemblage_cells(cfg)
    res = fetch_cells(client, cells, TAXA[taxon], gc.get("threads", 2))
    return [(lon, lat, gc["cell_deg"], r["species"]) for (lon, lat, _), r in zip(cells, res)]


# ---------------------------------------------------------------- equal-count coarsening (construction data only)
def neighbours(paths: Paths) -> set:
    import geopandas as gpd

    eco = gpd.read_file(paths.vector("ecoregions"), layer="ecoregions")[["unit", "geometry"]]
    b = eco.assign(geometry=eco.buffer(100))
    p = gpd.sjoin(b, b, predicate="intersects")
    return {(int(a), int(c)) for a, c in zip(p.unit_left, p.unit_right) if a != c}


def coarsening_tree(eco_df: pd.DataFrame, adj: set) -> pd.DataFrame:
    """Merge neighbouring ecoregions until one remains, each time the pair whose merge least raises the area-weighted
    within-group spread of community mixes (Ward). Construction data only: each ecoregion's 90 m community mix.
    Rows: step, kept, joined, groups left after the merge, Ward cost."""
    mix = eco_df.community_mix.fillna("").map(lambda s: dict((p.rsplit(" ", 1)[0], float(p.rsplit(" ", 1)[1].rstrip("%")))
                                                             for p in s.split("; ") if p.strip()))
    names = sorted({k for m in mix for k in m})
    V = np.array([[m.get(n, 0.0) / 100 for n in names] for m in mix])
    units = eco_df.unit.astype(int).tolist()
    cen = {u: V[i] for i, u in enumerate(units)}
    w = dict(zip(units, eco_df.area_km2.astype(float)))
    group = {u: u for u in units}
    rows, n = [], len(units)
    while n > 1:
        gadj = {(group[a], group[b]) for a, b in adj if a in group and b in group and group[a] != group[b]}
        best, bc = None, np.inf
        for a, b in gadj:
            if a < b:
                c = w[a] * w[b] / (w[a] + w[b]) * float(((cen[a] - cen[b]) ** 2).sum())
                if c < bc or (c == bc and (a, b) < best):
                    best, bc = (a, b), c
        if best is None:
            break  # disconnected pieces (islands) stay apart
        a, b = best
        cen[a] = (w[a] * cen[a] + w[b] * cen[b]) / (w[a] + w[b])
        w[a] += w[b]
        for u in group:
            if group[u] == b:
                group[u] = a
        n -= 1
        rows.append(dict(step=len(rows) + 1, kept=a, joined=b, groups=n, ward_cost=bc))
    return pd.DataFrame(rows)


def level_map(tree: pd.DataFrame, units: list, k: int) -> dict | None:
    """unit -> group at the level with k groups (None when the tree never reaches k)."""
    group = {u: u for u in units}
    if len(units) == k:
        return group
    for r in tree.itertuples():
        for u in group:
            if group[u] == r.joined:
                group[u] = r.kept
        if r.groups == k:
            return group
    return None


# ---------------------------------------------------------------- random maps and controls
def land_cells(cells_all: list, inside: np.ndarray) -> tuple[np.ndarray, dict]:
    """Grid index (i, j) of the given cells and a lookup (i, j) -> position."""
    ij = np.array([(round(c[0] / c[2]), round(c[1] / c[2])) for c in cells_all])[inside]
    return ij, {tuple(t): n for n, t in enumerate(ij)}


def land_grid(cfg, deg: float, to_xy, eco_c: np.ndarray, tr_c, n: int = 6) -> tuple[np.ndarray, dict]:
    """Every cell of the area's grid with at least half its area on the map (coarse map, n x n points): the land
    graph random maps grow on, independent of where records are."""
    w, s, e, nn = cfg["grid"]["bbox_lonlat"]
    I, J = np.meshgrid(np.arange(np.floor(w / deg), np.ceil(e / deg)), np.arange(np.floor(s / deg), np.ceil(nn / deg)))
    cells = [(i * deg, j * deg, deg, None) for i, j in zip(I.ravel(), J.ravel())]
    lon, lat = cell_points(cells, n)
    x, y = (np.asarray(v) for v in to_xy.transform(lon, lat))
    keep = (sample_array(eco_c, tr_c, x, y) > 0).mean(1) >= 0.5
    return land_cells(cells, keep)


def graph_partition(ij: np.ndarray, pos: dict, k: int, rng) -> np.ndarray:
    """A random contiguous partition of the land-cell graph (4-neighbour): k seeds drawn uniformly over land cells,
    grown one random frontier cell at a time; land the seeds cannot reach (islands) gets seeds of its own."""
    lab = np.full(len(ij), -1)
    seeds = list(rng.choice(len(ij), size=min(k, len(ij)), replace=False))
    nxt = 0
    while True:
        for s in seeds:
            lab[s] = nxt
            nxt += 1
        frontier = [(s, t) for s in seeds for t in nbrs(ij[s], pos) if lab[t] < 0]
        while frontier:
            s, t = frontier.pop(rng.integers(len(frontier)))
            if lab[t] >= 0:
                continue
            lab[t] = lab[s]
            frontier.extend((t, u) for u in nbrs(ij[t], pos) if lab[u] < 0)
        rest = np.flatnonzero(lab < 0)
        if not len(rest):
            return lab
        seeds = [int(rng.choice(rest))]


def nbrs(c, pos):
    i, j = c
    return [pos[q] for q in ((i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)) if q in pos]


def damaged(g: np.ndarray, ij: np.ndarray, pos: dict, share: float, rng) -> np.ndarray:
    """Control: patches of cells (a cell and its 4 neighbours) on unit boundaries moved into the neighbouring unit,
    until ``share`` of the cells changed. Keeps the number of units (as long as no unit vanishes)."""
    out = g.copy()
    edge = [n for n in range(len(g)) if any(g[m] != g[n] for m in nbrs(ij[n], pos))]
    changed, target = set(), share * len(g)
    for n in rng.permutation(edge):
        if len(changed) >= target:
            break
        other = [g[m] for m in nbrs(ij[n], pos) if g[m] != g[n]]
        if not other:
            continue
        new = other[rng.integers(len(other))]
        for m in [n, *nbrs(ij[n], pos)]:
            if out[m] == g[n] and (out == g[n]).sum() > 1:
                out[m] = new
                changed.add(m)
    return out


def split_in_two(g: np.ndarray, xy: np.ndarray, rng) -> np.ndarray:
    """Control: every unit cut in two around two of its cells chosen at random (detail with no information)."""
    out = g * 2
    for u in np.unique(g):
        idx = np.flatnonzero(g == u)
        if len(idx) < 2:
            continue
        s = xy[rng.choice(idx, 2, replace=False)]
        out[idx] += (np.linalg.norm(xy[idx] - s[1], axis=1) < np.linalg.norm(xy[idx] - s[0], axis=1))
    return out


def renamed(g: np.ndarray, rng) -> np.ndarray:
    """Control: units renamed at random (one-to-one); every score must be identical."""
    u = np.unique(g)
    m = dict(zip(u, rng.permutation(u) + 10 ** 6))
    return np.array([m[x] for x in g])


# ---------------------------------------------------------------- run
def run(cfg) -> None:
    paths = Paths(cfg)
    sc, gc = cfg["scorecard"], cfg["sources"]["gbif"]
    grid = json.loads(paths.grid_json.read_text())
    to_xy = Transformer.from_crs("EPSG:4326", grid["crs"], always_xy=True)
    tr_c, eco_c, res_c, refs, (cok, Z) = coarse_labels(cfg, paths)
    eco_df = pd.read_csv(paths.interim / "ecoregions.csv")
    adj = neighbours(paths)
    tree = coarsening_tree(eco_df, adj)
    tree.to_csv(paths.outputs / "coarsening_tree.csv", index=False)
    k_res_domain = len(np.unique(res_c[(eco_c > 0) & (res_c > 0)]))  # RESOLVE units within the study area
    lmap = level_map(tree, eco_df.unit.astype(int).tolist(), k_res_domain)
    study_km2 = float(eco_df.area_km2.sum())
    rows, contrasts, cover = [], [], []
    for taxon, deg in [(t, s) for t in sc["taxa"] for s in sc["scales"]]:
        if taxon == "plants" and cfg["sources"].get("gbif_dwca", {}).get("downloads"):
            cells = local_plant_cells(cfg, paths, deg)
        elif deg == gc["cell_deg"]:
            cells = api_cells(cfg, paths, taxon)
        else:
            continue  # no local records for this taxon: only the API's 0.25-degree cells
        t0 = time.time()
        # land support and eligibility, fixed before any partition is scored
        lon, lat = cell_points(cells, sc["purity_points"])
        x, y = (np.asarray(v) for v in to_xy.transform(lon, lat))
        eco_pts = sample_raster(paths.raster("ecoregions"), x, y)
        res_pts = sample_raster(paths.raster("resolve"), x, y)
        land = (eco_pts > 0).mean(1)
        rec = np.array([sum(c[3].values()) for c in cells])
        nsp = np.array([len(c[3]) for c in cells])
        elig = (land >= 0.5) & ((res_pts > 0).mean(1) >= 0.5) & (rec >= sc["min_records"]) & (nsp >= gc["min_species_cell"])
        idx = np.flatnonzero(elig)
        cells = [cells[i] for i in idx]
        eco_pts, res_pts, x, y, land = eco_pts[idx], res_pts[idx], x[idx], y[idx], land[idx]
        clon = np.array([c[0] + c[2] / 2 for c in cells])
        clat = np.array([c[1] + c[2] / 2 for c in cells])
        cx, cy = (np.asarray(v) for v in to_xy.transform(clon, clat))  # true cell centres
        xy = np.column_stack([cx, cy])
        held = holdout(clon, clat, gc["cell_deg"])  # cells nest in the checkerboard: one role per cell
        dev, test = np.flatnonzero(~held), np.flatnonzero(held)
        if len(test) < 30 or len(dev) < 30:
            log.info("scorecard %s %.3f deg: %d development / %d held-out cells; skipped", taxon, deg, len(dev), len(test))
            continue
        blocks = (np.floor(clon) * 1000 + np.floor(clat)).astype(int)  # 1-degree blocks
        cell_km2 = (deg * 111.32) ** 2 * np.cos(np.radians(clat))
        for role, sel in (("development", dev), ("held-out", test)):
            cover.append(dict(taxon=taxon, cell_deg=deg, role=role, cells=len(sel), blocks=len(np.unique(blocks[sel])),
                              land_km2=round(float((cell_km2[sel] * land[sel]).sum())), study_km2=round(study_km2),
                              share=round(float((cell_km2[sel] * land[sel]).sum()) / study_km2, 3)))
        # partitions on the common cells (purity recomputed for each)
        ours, p_ours = majority(eco_pts)
        parts = {"ours": (ours, p_ours), "RESOLVE": majority(res_pts)}
        for name, arr in refs.items():
            parts[name] = majority(sample_array(arr, tr_c, x, y))
        if lmap is not None:
            parts[f"ours coarsened to RESOLVE's {k_res_domain} units"] = majority(np.vectorize(lambda u: lmap.get(int(u), int(u)))(eco_pts))
        k_ours_domain = len(eco_df)
        for k in sorted({k_ours_domain, k_res_domain}):
            parts[f"climate k-means ({k})"] = majority(sample_array(climate_kmeans(cok, Z, k, sc["seed"]), tr_c, x, y))
        tm = paths.interim / "type_merges.json"  # proposed type merges (merges stage): adjacent units of one merged group join
        if tm.exists():
            gmap, ut = json.loads(tm.read_text()), eco_df.set_index("unit").type
            root = {int(u): int(u) for u in ut.index}

            def find(u):
                while root[u] != u:
                    u = root[u]
                return u

            for a, b in adj:
                if a in root and b in root and gmap.get(ut[a], ut[a]) == gmap.get(ut[b], ut[b]):
                    root[find(a)] = find(b)
            parts["ours, types merged"] = majority(np.vectorize(lambda u: find(int(u)) if int(u) in root else int(u))(eco_pts))
        for name, layer in (("ours", "ecoregions"), ("RESOLVE", "resolve")):
            parts[f"{name}, centre point (old test)"] = (sample_raster(paths.raster(layer), cx[:, None], cy[:, None])[:, 0],
                                                         np.full(len(cx), np.nan))
        rng = np.random.default_rng(sc["seed"])
        ij, pos = land_cells(cells, np.ones(len(cells), bool))  # eligible cells (controls)
        lij, lpos = land_grid(cfg, deg, to_xy, eco_c, tr_c)     # every land cell (random maps)
        at = np.array([lpos.get(tuple(t), -1) for t in ij])

        def random_map(k):
            lab = graph_partition(lij, lpos, k, rng)
            return np.where(at >= 0, lab[np.maximum(at, 0)], -1)

        parts["control: ours, units renamed"] = (renamed(ours, rng), p_ours)
        parts["control: ours, boundary patches moved"] = (damaged(ours, ij, pos, sc["damage_share"], rng), np.full(len(ours), np.nan))
        parts["control: ours, units split in two"] = (split_in_two(ours, xy, rng), np.full(len(ours), np.nan))
        k_o = len(np.unique(eco_c[eco_c > 0]))
        nulls = [random_map(k_o) for _ in range(sc["random_partitions"])]
        null_res = [random_map(k_res_domain) for _ in range(sc["random_partitions"])]
        pr, pbin, b1, b2 = pair_table(xy[test], blocks[test])
        vocab = {s: n for n, s in enumerate(sorted({s for c in cells for s in c[3]}))}

        def pres(lists):
            M = np.zeros((len(lists), len(vocab)), bool)
            for a, s in enumerate(lists):
                M[a, [vocab[k] for k in s]] = True
            return M

        Pfull = pres([list(c[3]) for c in cells])
        sp = Pfull[dev].sum(0) >= 1  # species universe: seen on development cells
        # ---- endpoints over rarefaction draws
        boot = np.random.default_rng([sc["seed"], 99])
        ub = np.unique(blocks[test])
        bw = [np.bincount(boot.choice(len(ub), len(ub)), minlength=len(ub)).astype(float) for _ in range(sc["bootstrap"])]
        loss_draws, bc_draws = {}, {}
        for draw in range(sc["draws"]):
            drng = np.random.default_rng([sc["seed"], draw])
            P = pres([rarefy(c[3], sc["depth"], drng) if sum(c[3].values()) >= sc["depth"] else list(c[3]) for c in cells])
            Pd, Pt = P[dev][:, sp].astype(float), P[test][:, sp].astype(float)
            prior = Pd.mean(0)
            base = brier_cells(Pt, np.broadcast_to(prior, Pt.shape))
            from scipy.spatial import cKDTree

            nn = cKDTree(xy[dev]).query(xy[test], k=sc["knn"])[1]
            spatial = brier_cells(Pt, Pd[nn].mean(1))
            Dp = jaccard_pairs(P[test], pr)
            loss_draws.setdefault("overall-frequency prior", []).append(base)
            loss_draws.setdefault("spatial-only (nearest development cells)", []).append(spatial)
            for pname, (g, pur) in parts.items():
                loss = prediction_loss(Pd, g[dev], Pt, g[test], prior, sc["prior_cells"])
                loss_draws.setdefault(pname, []).append(loss)
                across = g[test][pr[:, 0]] != g[test][pr[:, 1]]
                bc_draws.setdefault(pname, []).append((Dp, across))
                rows.append(dict(taxon=taxon, cell_deg=deg, draw=draw, partition=pname, units=len(np.unique(g[test])),
                                 purity=float(np.nanmean(pur[test])) if np.isfinite(pur).any() else np.nan,
                                 loss=block_mean(loss, blocks[test]), skill=1 - block_mean(loss, blocks[test]) / block_mean(base, blocks[test]),
                                 boundary_contrast=boundary_contrast(Dp, across, pbin)))
            for name, sel in (("random contiguous (our count)", nulls), ("random contiguous (RESOLVE's count)", null_res)):
                for n_ in sel[draw::sc["draws"]]:
                    loss = prediction_loss(Pd, n_[dev], Pt, n_[test], prior, sc["prior_cells"])
                    across = n_[test][pr[:, 0]] != n_[test][pr[:, 1]]
                    rows.append(dict(taxon=taxon, cell_deg=deg, draw=draw, partition=name, units=len(np.unique(n_[test])),
                                     loss=block_mean(loss, blocks[test]), skill=1 - block_mean(loss, blocks[test]) / block_mean(base, blocks[test]),
                                     boundary_contrast=boundary_contrast(Dp, across, pbin)))
            for name in ("overall-frequency prior", "spatial-only (nearest development cells)"):
                l_ = loss_draws[name][-1]
                rows.append(dict(taxon=taxon, cell_deg=deg, draw=draw, partition=name, loss=block_mean(l_, blocks[test]),
                                 skill=1 - block_mean(l_, blocks[test]) / block_mean(base, blocks[test])))
        # ---- paired contrasts with block bootstrap (losses averaged over draws per cell)
        L = {k: np.mean(v, 0) for k, v in loss_draws.items()}
        for cand in ["ours", f"ours coarsened to RESOLVE's {k_res_domain} units", FAUNAL, "ours, types merged"]:
            if cand not in L:
                continue
            for comp in [c for c in L if c != cand and not c.startswith(("control", "ours", FAUNAL))]:
                d = L[comp] - L[cand]  # positive: the candidate predicts better
                est = block_mean(d, blocks[test]) / block_mean(L[comp], blocks[test])
                bs = [block_mean(d, blocks[test], w) / block_mean(L[comp], blocks[test], w) for w in bw]
                bcc = [boundary_contrast(*bc_draws[cand][k_], pbin) - boundary_contrast(*bc_draws[comp][k_], pbin)
                       for k_ in range(sc["draws"])] if comp in bc_draws else []
                contrasts.append(dict(taxon=taxon, cell_deg=deg, candidate=cand, comparator=comp,
                                      relative_loss_improvement=est, ci_low=np.percentile(bs, 2.5), ci_high=np.percentile(bs, 97.5),
                                      boundary_contrast_difference=float(np.median(bcc)) if bcc else np.nan))
        # ---- secondary: ANOSIM on full lists (Simpson), z against Voronoi partitions
        from .faunal import beta_sim

        rk = Ranked(beta_sim(Pfull[test]))
        for pname, (g, _) in parts.items():
            k = len(np.unique(g[test]))
            if k < 2:
                continue
            R = rk.R(g[test])
            zr = zscore(R, np.array([rk.R(n_) for n_ in random_partitions(xy[test], k, sc["random_partitions"], rng)]))
            rows.append(dict(taxon=taxon, cell_deg=deg, draw=-1, partition=pname, anosim_R_simpson=R, anosim_z_voronoi=zr))
        log.info("scorecard %s %.3f deg: %d development, %d held-out cells, %d pairs, %.0f s", taxon, deg, len(dev), len(test),
                 len(pr), time.time() - t0)
    df = pd.DataFrame(rows).assign(status=STATUS)
    df.to_csv(paths.outputs / "scorecard.csv", index=False)
    summ = df.groupby(["taxon", "cell_deg", "partition"], sort=False).median(numeric_only=True).drop(columns="draw")
    summ.round(4).to_csv(paths.outputs / "scorecard_summary.csv")
    pd.DataFrame(contrasts).assign(status=STATUS).round(4).to_csv(paths.outputs / "scorecard_contrasts.csv", index=False)
    pd.DataFrame(cover).to_csv(paths.outputs / "scorecard_coverage.csv", index=False)
    log.info("scorecard (%s):\n%s\n%s", STATUS, summ[["units", "purity", "loss", "skill", "boundary_contrast", "anosim_z_voronoi"]]
             .round(3).to_string(), pd.DataFrame(contrasts).round(3).to_string())


# ---------------------------------------------------------------- biomes: vegetation structure (ESA WorldCover)
NATURAL = [10, 20, 30, 60, 70, 90, 95, 100]  # tree, shrub, grass, bare/sparse, snow/ice, herbaceous wetland, mangrove, moss


def biome_structure(cfg) -> pd.DataFrame:
    """Biomes are classes of vegetation structure, so they are tested on structure: on natural land (WorldCover
    cropland, built-up and water left out; WorldCover's vegetation classes are used by no rule), how well does each
    biome map predict the mix of tree, shrub, grass, bare, snow, wetland, mangrove and moss cover of held-out cells
    from development cells? Same cells, split, shrinkage, blocks and bootstrap as the species endpoints.
    Writes outputs/biome_structure.csv and biome_structure_contrasts.csv."""
    from rasterio.transform import Affine
    from scipy.spatial import cKDTree

    from . import vectors
    from .biomes import key_biomes
    from .communities import load_rules
    from .evaluate import _biome_key

    paths = Paths(cfg)
    sc, gc = cfg["scorecard"], cfg["sources"]["gbif"]
    grid = json.loads(paths.grid_json.read_text())
    to_xy = Transformer.from_crs("EPSG:4326", grid["crs"], always_xy=True)
    tr_c, eco_c, res_c, _, (cok, Z) = coarse_labels(cfg, paths)
    with rasterio.open(paths.raster("worldcover")) as r:
        wc = r.read(1, out_shape=(r.height // 3, r.width // 3), resampling=rasterio.enums.Resampling.nearest)
        tr_w = r.transform * Affine.scale(r.width / wc.shape[1], r.height / wc.shape[0])
    eco_df = pd.read_csv(paths.interim / "ecoregions.csv")
    types = load_rules(paths.root / cfg["communities"]["rules"])["ecoregion_types"]
    kb = key_biomes(paths)
    names = sorted({*kb.values(), *(v["biome"] for v in types.values())})
    code = {n: i + 1 for i, n in enumerate(names)}
    unit_type = eco_df.set_index("unit").type
    lut_key = np.zeros(int(eco_c.max()) + 1, int)
    lut_hand = np.zeros_like(lut_key)
    for u, t in unit_type.items():
        if u < len(lut_key):
            lut_key[u] = code[kb.get(t, types[t]["biome"])]
            lut_hand[u] = code[types[t]["biome"]]
    rt = vectors.resolve(cfg)[["ECO_ID", "BIOME_NAME"]].drop_duplicates("ECO_ID").set_index("ECO_ID").BIOME_NAME.map(_biome_key)
    rcode = {n: i + 1 for i, n in enumerate(sorted(rt.unique()))}
    lut_res = np.zeros(int(max(res_c.max(), rt.index.max())) + 1, int)
    for e, b in rt.items():
        lut_res[int(e)] = rcode[b]
    maps = {"biomes by the key": lut_key[eco_c], "biomes as hand-set": lut_hand[eco_c], "RESOLVE biomes": lut_res[res_c] * (eco_c > 0)}
    k_key = len(np.unique(maps["biomes by the key"][eco_c > 0]))
    k_res = len(np.unique(maps["RESOLVE biomes"][eco_c > 0]))
    for k in sorted({k_key, k_res}):
        maps[f"climate k-means ({k})"] = climate_kmeans(cok, Z, k, sc["seed"])
    out, contrasts = [], []
    for deg in sc["scales"]:
        w, s, e, n = cfg["grid"]["bbox_lonlat"]
        I, J = np.meshgrid(np.arange(np.floor(w / deg), np.ceil(e / deg)), np.arange(np.floor(s / deg), np.ceil(n / deg)))
        cells = [(i * deg, j * deg, deg, None) for i, j in zip(I.ravel(), J.ravel())]
        lon, lat = cell_points(cells, 10)
        x, y = (np.asarray(v) for v in to_xy.transform(lon, lat))
        land = (sample_array(eco_c, tr_c, x, y) > 0).mean(1)
        cover = sample_array(wc, tr_w, x, y)
        valid = (cover > 0).sum(1)
        nat = np.isin(cover, NATURAL)
        ok = (land >= 0.5) & (valid > 0) & (nat.sum(1) >= 0.5 * np.maximum(valid, 1))
        idx = np.flatnonzero(ok)
        Y = np.stack([(cover[idx] == c).sum(1) for c in NATURAL], 1) / nat[idx].sum(1, keepdims=True)
        clon = np.array([cells[i][0] + deg / 2 for i in idx])
        clat = np.array([cells[i][1] + deg / 2 for i in idx])
        held = holdout(clon, clat, gc["cell_deg"])
        dev, test = np.flatnonzero(~held), np.flatnonzero(held)
        blocks = (np.floor(clon) * 1000 + np.floor(clat)).astype(int)
        labels = {nm: majority(sample_array(m, tr_c, x[idx], y[idx]))[0] for nm, m in maps.items()}
        ij, pos = land_cells([cells[i] for i in idx], np.ones(len(idx), bool))
        rng = np.random.default_rng(sc["seed"])
        prior = Y[dev].mean(0)

        def loss_of(g):
            F, p_ = unit_frequencies(Y[dev], g[dev], prior, sc["prior_cells"])
            rows = np.array([p_.get(u, -1) for u in g[test]])
            pred = np.where(rows[:, None] >= 0, F[np.maximum(rows, 0)], prior[None, :])
            return ((Y[test] - pred) ** 2).mean(1)

        L = {nm: loss_of(g) for nm, g in labels.items()}
        L["overall prior"] = ((Y[test] - prior) ** 2).mean(1)
        xy = np.column_stack(to_xy.transform(clon, clat))
        nn = cKDTree(xy[dev]).query(xy[test], k=sc["knn"])[1]
        L["spatial-only (nearest development cells)"] = ((Y[test] - Y[dev][nn].mean(1)) ** 2).mean(1)
        base = block_mean(L["overall prior"], blocks[test])
        for k in sorted({k_key, k_res}):
            ls = [block_mean(loss_of(graph_partition(ij, pos, k, rng)), blocks[test]) for _ in range(sc["random_partitions"])]
            out.append(dict(cell_deg=deg, map=f"random contiguous ({k})", units=k, loss=float(np.median(ls)),
                            skill=1 - float(np.median(ls)) / base, loss_p5=float(np.percentile(ls, 5))))
        for nm, l_ in L.items():
            g = labels.get(nm)
            out.append(dict(cell_deg=deg, map=nm, units=len(np.unique(g[test])) if g is not None else np.nan,
                            loss=block_mean(l_, blocks[test]), skill=1 - block_mean(l_, blocks[test]) / base,
                            cells_dev=len(dev), cells_test=len(test)))
        ub = np.unique(blocks[test])
        boot = np.random.default_rng([sc["seed"], 7])
        bw = [np.bincount(boot.choice(len(ub), len(ub)), minlength=len(ub)).astype(float) for _ in range(sc["bootstrap"])]
        for cand in ("biomes by the key", "biomes as hand-set"):
            for comp in [c for c in L if c not in (cand, "biomes as hand-set", "biomes by the key")]:
                d = L[comp] - L[cand]
                est = block_mean(d, blocks[test]) / block_mean(L[comp], blocks[test])
                bs = [block_mean(d, blocks[test], w_) / block_mean(L[comp], blocks[test], w_) for w_ in bw]
                contrasts.append(dict(cell_deg=deg, candidate=cand, comparator=comp, relative_loss_improvement=est,
                                      ci_low=np.percentile(bs, 2.5), ci_high=np.percentile(bs, 97.5)))
        log.info("biome structure %.3f deg: %d development, %d held-out natural cells", deg, len(dev), len(test))
    df = pd.DataFrame(out).assign(status=STATUS)
    df.round(4).to_csv(paths.outputs / "biome_structure.csv", index=False)
    pd.DataFrame(contrasts).assign(status=STATUS).round(4).to_csv(paths.outputs / "biome_structure_contrasts.csv", index=False)
    log.info("biome structure:\n%s\n%s", df.round(4).to_string(), pd.DataFrame(contrasts).round(3).to_string())
    return df
