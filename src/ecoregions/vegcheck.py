"""v0.2 stage — check the vegetation-community layer against independent data.

None of these layers is an input to the community rules; they are used only to
calibrate thresholds and to report agreement.

1. Validation layers on the 90 m grid (``interim/vegcheck/``):
   * ``gmw_union_pct`` / ``gmw_2020_pct`` — % mangrove cover (Global Mangrove
     Watch v3, ~25 m; union of 1996-2020 = best proxy for potential extent);
   * ``jung_lvl2`` — Jung et al. (2020) IUCN habitat class, level 2 (mode);
   * ``pnv_biome`` — Hengl et al. (2018) potential natural vegetation biome.
2. Agreement:
   * mangroves: GMW area captured by the ``mangrove`` community (recall) and
     mapped mangrove confirmed by GMW (precision), at 90 m and within 1 km;
   * community x Jung lvl2 and community x PNV biome area tables (km²);
   * indicator species: for each community, the share of GBIF records of its
     indicator species falling in it, at the cell and within ``indicator_radius_km``
     (Himalayan Uplands specimens are located to ~10 km).
"""
from __future__ import annotations

import logging
import re
import zipfile

import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject, transform_bounds
from scipy import ndimage

from .communities import load_rules
from .config import Paths
from .grid import Grid, block_reduce, is_current, read, reproject_file, write

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# validation layers on the analysis grid
# ---------------------------------------------------------------------------
def gmw_pct(zip_path, grid: Grid) -> np.ndarray:
    """% mangrove per 90 m cell from the GMW 1-degree tiles that touch the grid.
    The tiles flag non-mangrove as nodata (0); here 0 counts as 'not mangrove'."""
    w, s, e, n = transform_bounds(grid.rio_crs, "EPSG:4326", *grid.bounds, densify_pts=41)
    out = np.zeros(grid.shape, "float32")
    names = [x for x in zipfile.ZipFile(zip_path).namelist() if x.endswith(".tif")]
    used = 0
    for name in names:
        m = re.search(r"_([NS])(\d{2})([EW])(\d{3})_", name)
        lat = int(m[2]) * (1 if m[1] == "N" else -1)
        lon = int(m[4]) * (1 if m[3] == "E" else -1)
        if not (s - 1 <= lat <= n + 1 and w - 1 <= lon <= e + 1):
            continue
        with rasterio.open(f"/vsizip/{zip_path}/{name}") as src:
            a = (src.read(1) == 1).astype("float32")
            reproject(a, out, src_transform=src.transform, src_crs=src.crs, src_nodata=None,
                      dst_transform=grid.transform, dst_crs=grid.rio_crs, dst_nodata=-1,
                      init_dest_nodata=False, resampling=Resampling.average, num_threads=4)
        used += 1
    log.info("GMW %s: %d tiles", zip_path.name, used)
    return np.clip(out * 100, 0, 100).astype("uint8")


def prepare(cfg, paths: Paths, grid: Grid) -> None:
    out = paths.interim / "vegcheck"
    raw = paths.raw
    for tag in ("union", "2020"):
        f = out / f"gmw_{tag}_pct.tif"
        if not is_current(f, grid):
            write(f, gmw_pct(raw / "gmw" / f"gmw_v3_{tag}_gtiff.zip", grid), grid, nodata=None)
    f = out / "jung_lvl2.tif"
    if not is_current(f, grid):
        tif = raw / "jung_habitat" / "iucn_habitatclassification_composite_lvl2_ver004.tif"
        if not tif.exists():  # striped GeoTIFF inside a deflated zip: extract once for fast windowed reads
            with zipfile.ZipFile(raw / "jung_habitat" / "iucn_habitatclassification_composite_lvl2_ver004.zip") as z:
                z.extract(tif.name, tif.parent)
        write(f, reproject_file(tif, grid, Resampling.mode, dtype="uint16", dst_nodata=0), grid, nodata=0)
    f = out / "pnv_biome.tif"
    if not is_current(f, grid):
        write(f, reproject_file(raw / "pnv" / "pnv_biome_1km.tif", grid, Resampling.nearest, dtype="uint8",
                                dst_nodata=255), grid, nodata=255)


def jung_legend(paths: Paths) -> dict[int, str]:
    """Level-2 class names from the QGIS colour table shipped with the map."""
    with zipfile.ZipFile(paths.raw / "jung_habitat" / "iucn_habitatclassification_code_ver004.zip") as z:
        txt = z.read("Habitatmapping-004/styles/level2.clr").decode("utf-8")
    return {int(p[0]): p[5] for p in (line.split(" ", 5) for line in txt.splitlines() if line.strip())}


def pnv_legend(paths: Paths) -> dict[int, str]:
    d = pd.read_csv(paths.raw / "pnv" / "pnv_biome_1km_legend.csv")
    return dict(zip(d.Number, d["New.global.consolidated.biome.scheme"]))


# ---------------------------------------------------------------------------
# agreement
# ---------------------------------------------------------------------------
def crosstab_km2(a: np.ndarray, b: np.ndarray, valid: np.ndarray, km2: float) -> pd.DataFrame:
    """Area (km²) of every (a, b) pair among valid cells."""
    s = pd.Series(1, index=pd.MultiIndex.from_arrays([a[valid], b[valid]]))
    return (s.groupby(level=[0, 1]).size() * km2).unstack(fill_value=0)


def precision_recall(pred: np.ndarray, truth: np.ndarray, tol_cells: int = 0) -> tuple[float, float]:
    """Share of predicted cells that are true (precision) and of true cells
    that are predicted (recall), each counting a match within ``tol_cells``."""
    if tol_cells:
        st = np.ones((2 * tol_cells + 1,) * 2, bool)
        truth_n, pred_n = ndimage.binary_dilation(truth, st), ndimage.binary_dilation(pred, st)
    else:
        truth_n, pred_n = truth, pred
    p = (pred & truth_n).sum() / max(pred.sum(), 1)
    r = (truth & pred_n).sum() / max(truth.sum(), 1)
    return float(p), float(r)


def binomial(name) -> str:
    """'Juniperus excelsa subsp. polycarpos' -> 'juniperus excelsa'."""
    return " ".join(str(name).lower().split()[:2])


def indicator_hits(comm: np.ndarray, grid: Grid, rules: dict, rec: pd.DataFrame, radius_km: float,
                   factor: int) -> pd.DataFrame:
    """For each community: GBIF records of its indicator species, the share in
    the community's cells, and the share within ``radius_km`` of them."""
    from pyproj import Transformer

    tr = Transformer.from_crs("EPSG:4326", grid.rio_crs, always_xy=True)
    x, y = tr.transform(rec.decimalLongitude.to_numpy(), rec.decimalLatitude.to_numpy())
    col = ((x - grid.bounds[0]) / grid.res).astype(int)
    row = ((grid.bounds[3] - y) / grid.res).astype(int)
    inside = (row >= 0) & (row < grid.shape[0]) & (col >= 0) & (col < grid.shape[1])
    rec = rec.assign(row=row, col=col)[inside]
    rec = rec[comm[rec.row, rec.col] > 0]  # study area only
    rec_sp = rec.species.map(binomial)
    rad = max(1, round(radius_km * 1000 / (grid.res * factor)))
    rows = []
    for j, c in enumerate(rules["communities"]):
        sp = {binomial(s) for s in c["indicators"]}
        r = rec[rec_sp.isin(sp)]
        if r.empty:
            rows.append(dict(id=c["id"], records=0))
            continue
        here = comm[r.row, r.col] == j + 1
        near_mask = ndimage.maximum_filter(block_reduce((comm == j + 1).astype("float32"), factor) > 0, size=2 * rad + 1)
        rr = np.minimum(r.row.to_numpy() // factor, near_mask.shape[0] - 1)
        cc = np.minimum(r.col.to_numpy() // factor, near_mask.shape[1] - 1)
        rows.append(dict(id=c["id"], records=len(r), species_found=r.species.map(binomial).nunique(),
                         in_cell_pct=round(100 * here.mean(), 1), within_radius_pct=round(100 * near_mask[rr, cc].mean(), 1)))
    return pd.DataFrame(rows)


def run(cfg) -> None:
    paths = Paths(cfg)
    grid = Grid.load(paths.grid_json)
    vc = cfg["vegcheck"]
    prepare(cfg, paths, grid)
    out = paths.interim / "vegcheck"
    rules = load_rules(paths.root / cfg["communities"]["rules"])
    names = {j + 1: c["id"] for j, c in enumerate(rules["communities"])}
    comm = read(paths.raster("communities"))
    valid = comm > 0
    km2 = grid.cell_km2

    # mangroves
    mid = next(j + 1 for j, c in enumerate(rules["communities"]) if c["id"] == "mangrove")
    pred = comm == mid
    rows = []
    for tag in ("union", "2020"):
        truth = (read(out / f"gmw_{tag}_pct.tif") >= vc["gmw_min_pct"]) & valid
        for tol_m in (0, 1000):
            p, r = precision_recall(pred, truth, round(tol_m / grid.res))
            rows.append(dict(gmw=tag, tolerance_m=tol_m, gmw_km2=truth.sum() * km2, mapped_km2=pred.sum() * km2,
                             precision=round(p, 3), recall=round(r, 3)))
    mg = pd.DataFrame(rows)
    mg.to_csv(out / "mangrove_agreement.csv", index=False)
    log.info("mangroves vs GMW:\n%s", mg.to_string(index=False))

    # area tables
    for tag, layer, legend in (("jung", "jung_lvl2", jung_legend(paths)), ("pnv", "pnv_biome", pnv_legend(paths))):
        ct = crosstab_km2(comm, read(out / f"{layer}.tif"), valid, km2)
        ct.index = ct.index.map(names)
        ct.columns = [legend.get(int(k), str(k)) for k in ct.columns]
        ct = ct.loc[:, ct.sum() >= vc["min_table_km2"]].round(0)
        ct.to_csv(out / f"communities_x_{tag}.csv")
        log.info("wrote communities_x_%s.csv (%d classes)", tag, ct.shape[1])

    # indicator species in GBIF records
    recs = pd.concat([pd.read_csv(f, usecols=["species", "decimalLatitude", "decimalLongitude"])
                      for f in sorted((paths.raw / "gbif_records").glob("*.csv"))]).dropna()
    ind = indicator_hits(comm, grid, rules, recs, vc["indicator_radius_km"], cfg["grid"]["coarse_factor"])
    ind.to_csv(out / "indicator_species.csv", index=False)
    log.info("indicator species:\n%s", ind.to_string(index=False))
