"""Stage 6 — feature stack for regionalization.

The 90 m layers are aggregated to the segmentation grid (270 m by default) as
block means or class fractions, and again to the province grid (810 m). The
stack is stored as one GeoTIFF per feature under ``data/interim/coarse`` (and
``.../province``) so every feature can be inspected in a GIS.

Feature definitions (names are what ``config.yaml`` weights refer to):

=============  ==============================================================
elev           mean elevation (m)
slope          mean slope (deg)
relief1k/5k    mean local relief in 1 km / 5 km windows (m)
tpi2k          mean TPI at 2 km (m)
hand           log1p of mean HAND/REM, clipped at 0 (m)
rem            mean REM/HAND clipped to [-5, 40] m (fine relief above drainage)
flood_frac     share of 90 m cells in the active-floodplain HAND class
meander_frac   share in the older meander-floodplain class
bar_frac       share in the bar-upland class
upland_frac    share in the upland class
valley_frac    share of geomorphons hollow / valley / pit
ridge_frac     share of geomorphons peak / ridge / shoulder / spur
dune_frac      share of hummocky aeolian cells (``dune_mask``)
basin          HydroBASINS id (centre sample) — one-hot within a province at clustering time
litho_<k>      share of lithology group k (geo8ag)
age            mean geological age rank
clay/sand/ph/soc  SoilGrids 0-30 cm means
bio1/bio4/bio12/bio15/gdd5/ai  CHELSA (bio12 and ai log-transformed)
=============  ==============================================================
"""
from __future__ import annotations

import gc
import json
import logging

import numpy as np

from .config import Paths
from .covariates import LITHO_GROUPS
from .grid import Grid, block_reduce, centre_sample, read, write

log = logging.getLogger(__name__)

LOG_FEATURES = {"bio12", "ai"}


def dune_mask(cfg, slope: np.ndarray, tpi450: np.ndarray, plains: np.ndarray, litho: np.ndarray) -> np.ndarray:
    """Hummocky aeolian texture: |TPI 450 m| and slope above thresholds, on the
    alluvial-plains mask or on mapped aeolian sand (geo8ag Qs)."""
    dc = cfg["terrain"]["dune"]
    return (np.abs(tpi450) >= dc["tpi_abs_m"]) & (slope >= dc["slope_min_deg"]) & (plains | (litho == 2))


def _coarsen(a: np.ndarray, f: int) -> np.ndarray:
    return block_reduce(a.astype("float32", copy=False), f)


def build(cfg, only: list[str] | None = None) -> None:
    paths = Paths(cfg)
    grid = Grid.load(paths.grid_json)
    f = cfg["grid"]["coarse_factor"]
    cg = grid.coarsen(f)
    out = paths.interim / "coarse"
    out.mkdir(exist_ok=True)
    names = []

    def want(name: str) -> bool:
        return only is None or name in only

    def put(name: str, arr90: np.ndarray, how: str = "mean"):
        if not want(name):
            return
        arr = _coarsen(arr90, f)
        write(out / f"{name}.tif", arr, cg, nodata=np.nan)
        names.append(name)
        log.info("feature %-13s %s", name, how)

    mask = read(paths.raster("mask"))
    study = (mask == 1).astype("float32")
    write(out / "study_frac.tif", _coarsen(study, f), cg, nodata=np.nan)
    sea = mask == 0
    del study

    simple = {"elev": "dem", "slope": "slope", "relief1k": "relief_1000", "relief5k": "relief_5000",
              "tpi2k": "tpi_2000", "clay": "soil_clay", "sand": "soil_sand", "ph": "soil_phh2o",
              "soc": "soil_soc", "bio1": "clim_bio1", "bio4": "clim_bio4", "bio12": "clim_bio12",
              "bio15": "clim_bio15", "gdd5": "clim_gdd5", "ai": "clim_ai", "age": "geo_age"}
    for name, src in simple.items():
        if not want(name):
            continue
        a = read(paths.raster(src)).astype("float32")
        a[sea] = np.nan
        if name in LOG_FEATURES:
            a = np.log10(np.clip(a, 1e-3, None))
        put(name, a)
        del a
        gc.collect()

    if want("hand") or want("rem"):
        h = read(paths.raster("hand"))
    else:
        h = np.zeros((1, 1), "float32")
    put("hand", np.log1p(np.clip(h, 0, None)))
    put("rem", np.clip(h, -5, 40))
    del h
    hc = read(paths.raster("hand_class")) if any(want(n) for n in ("flood_frac", "meander_frac", "bar_frac", "upland_frac")) else None
    for code, name in ((1, "flood_frac"), (2, "meander_frac"), (3, "bar_frac"), (4, "upland_frac")):
        if hc is not None and want(name):
            put(name, np.where(sea, np.nan, (hc == code).astype("float32")), "fraction")
    del hc
    if want("valley_frac") or want("ridge_frac"):
        g = read(paths.raster("geomorphons"))
        put("valley_frac", np.where(sea, np.nan, np.isin(g, [7, 9, 10]).astype("float32")), "fraction")
        put("ridge_frac", np.where(sea, np.nan, np.isin(g, [2, 3, 4, 5]).astype("float32")), "fraction")
        del g
    lit = read(paths.raster("geo_litho"))
    if want("dune_frac"):
        dune = dune_mask(cfg, read(paths.raster("slope")), read(paths.raster("tpi_450")),
                         read(paths.raster("plains")).astype(bool), lit)
        put("dune_frac", np.where(sea, np.nan, dune.astype("float32")), "fraction")
        del dune
    for k in LITHO_GROUPS:
        put(f"litho_{k}", np.where(sea, np.nan, (lit == k).astype("float32")), "fraction")
    del lit
    if want("basin"):
        lvl = cfg["segmentation"]["basin_level"]
        b = centre_sample(read(paths.raster(f"basin_l{lvl:02d}")), f, cg.shape)
        write(out / "basin.tif", b, cg, nodata=0)
        names.append("basin")
    gc.collect()

    # province grid (another integer aggregation)
    pf = cfg["grid"]["province_factor"]
    pg = cg.coarsen(pf)
    pout = paths.interim / "province"
    pout.mkdir(exist_ok=True)
    for name in names + (["study_frac"] if only is None else []):
        a = read(out / f"{name}.tif")
        if name == "basin":
            write(pout / "basin.tif", centre_sample(a, pf, pg.shape), pg, nodata=0)
        else:
            write(pout / f"{name}.tif", block_reduce(a, pf), pg, nodata=np.nan)
    if only is None:
        (paths.interim / "features.json").write_text(json.dumps(dict(features=names, coarse_res=cg.res,
                                                                     province_res=pg.res), indent=1))


def load_stack(paths: Paths, names: list[str], level: str = "coarse", window=None) -> dict[str, np.ndarray]:
    d = paths.interim / level
    return {n: read(d / f"{n}.tif", window=window) for n in names}


def run(cfg) -> None:
    build(cfg)
