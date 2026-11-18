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
nat_bare/green    Landsat 1984-93 bare-soil index / NDVI, converted ground filled from natural ground (``natural_fill``)
bio1/bio4/bio12/bio15/gdd5/ai  CHELSA (bio12 and ai log-transformed)
=============  ==============================================================
"""
from __future__ import annotations

import gc
import json
import logging

import numpy as np

from .config import Paths
from .covariates import litho_groups
from .grid import Grid, block_reduce, centre_sample, rasterize_geoms, read, write

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
    for k in litho_groups(cfg):
        put(f"litho_{k}", np.where(sea, np.nan, (lit == k).astype("float32")), "fraction")
    del lit
    if want("basin"):
        lvl = cfg["segmentation"]["basin_level"]
        b = centre_sample(read(paths.raster(f"basin_l{lvl:02d}")), f, cg.shape)
        write(out / "basin.tif", b, cg, nodata=0)
        names.append("basin")
    gc.collect()

    blend_seams(cfg, paths, grid, cg, out, names)

    wanted = [n for n, src in LANDSAT.items() if want(n) and paths.coarse(src).exists()]
    if wanted:  # one Landsat composite for the whole area, so no sheet seams to blend
        from rasterio.enums import Resampling

        from .fitcheck import WORLDCOVER_CONVERTED, layer
        conv = np.isin(layer(paths.raster("worldcover"), cg, Resampling.mode), WORLDCOVER_CONVERTED)
        for n in wanted:
            write(out / f"{n}.tif", natural_fill(read(paths.coarse(LANDSAT[n])), conv), cg, nodata=np.nan)
            names.append(n)
            log.info("feature %-13s Landsat 1984-93, converted ground filled from natural ground around it", n)

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


LANDSAT = {"nat_bare": "ls_bare", "nat_green": "ls_green"}  # feature: Landsat layer (stage landsat)


def natural_fill(a: np.ndarray, converted: np.ndarray, sigmas=(2, 8, 32, 128)) -> np.ndarray:
    """``a`` with converted cells (cropland, built-up) replaced by the mean of the natural ground around them, taken
    over widening Gaussian windows until every converted cell has natural neighbours: in 1984-93 irrigated land shows
    crops, not the sand or steppe a restoration would return to, and edges must not follow canal colonies."""
    from scipy import ndimage as ndi

    ok = np.isfinite(a) & ~converted
    out = np.where(ok, a, np.nan).astype("float32")
    todo = converted & np.isfinite(a)
    for s in sigmas:
        if not todo.any():
            break
        num = ndi.gaussian_filter(np.where(ok, a, 0).astype("float32"), s)
        den = ndi.gaussian_filter(ok.astype("float32"), s)
        got = todo & (den > 0.05)
        out[got] = num[got] / den[got]
        todo &= ~got
    return out


def blend(a: np.ndarray, seam: np.ndarray, res: float, band_m: float) -> np.ndarray:
    """``a`` passing smoothly across the cells in ``seam``: weight of the sharp layer rising from 0 at the seam to 1 at
    ``band_m``, the rest from a Gaussian-smoothed copy (sigma band/2, NaN-aware) that is continuous across it."""
    from scipy import ndimage as ndi

    if not seam.any():
        return a.astype("float32")
    # only cells within the band change, and their smoothed value reaches 4 sigma (= 2 bands) further: work in that window
    rr, cc = np.nonzero(seam)
    pad = int(np.ceil(3 * band_m / res)) + 2
    win = (slice(max(rr.min() - pad, 0), rr.max() + pad + 1), slice(max(cc.min() - pad, 0), cc.max() + pad + 1))
    if a[win].size < a.size:
        out = np.where(np.isfinite(a), a, np.nan).astype("float32")
        out[win] = blend(a[win], seam[win], res, band_m)
        return out
    w = np.clip(ndi.distance_transform_edt(~seam) * res / band_m, 0, 1).astype("float32")
    ok = np.isfinite(a)
    sig = band_m / 2 / res
    num = ndi.gaussian_filter(np.where(ok, a, 0).astype("float32"), sig)
    den = ndi.gaussian_filter(ok.astype("float32"), sig)
    smooth = np.where(den > 1e-3, num / np.maximum(den, 1e-3), a)
    return np.where(ok, w * a + (1 - w) * smooth, np.nan).astype("float32")


def blend_seams(cfg, paths: Paths, grid: Grid, cg: Grid, out, names: list[str]) -> None:
    """Seams between national source maps (``seams`` in config.yaml): geology layers across geology sheet edges,
    listed layers across listed international borders. Every input is kept; only the step at the seam goes."""
    sc = cfg.get("seams")
    if not sc:
        return
    band = sc["band_km"] * 1000
    f = cfg["grid"]["coarse_factor"]
    todo = []
    if paths.raster("geo_sheet").exists():
        sh = centre_sample(read(paths.raster("geo_sheet")), f, cg.shape)
        seam = np.zeros(cg.shape, bool)
        dx = (sh[:, 1:] != sh[:, :-1]) & (sh[:, 1:] > 0) & (sh[:, :-1] > 0)
        dy = (sh[1:, :] != sh[:-1, :]) & (sh[1:, :] > 0) & (sh[:-1, :] > 0)
        seam[:, 1:] |= dx
        seam[:, :-1] |= dx
        seam[1:, :] |= dy
        seam[:-1, :] |= dy
        if seam.any():
            todo.append(("geology sheet edges", seam, [n for n in names if n.startswith("litho_") or n in sc.get("geology", [])]))
    for what, seam, layers in todo:
        for n in layers:
            write(out / f"{n}.tif", blend(read(out / f"{n}.tif"), seam, cg.res, band), cg, nodata=np.nan)
        log.info("seams: %s blended over %.0f km across %s", ", ".join(layers), sc["band_km"], what)

    # national seams inside one sheet, and in other compiled layers (soils): found by ``bordercheck.input_seams``
    # (a layer changing at a border more than terrain and climate do), blended along that border, tested again with
    # a wider band if still found; geology layers go together, as one map
    from .bordercheck import input_seams

    geo = lambda n: n.startswith("litho_") or n in sc.get("geology", [])  # noqa: E731
    for wide in sc.get("border_bands_km", [30, 50, 80]):
        rows, lines = input_seams(cfg, names)
        found: dict[str, set] = {}
        for r in rows:
            if r["seam"]:
                found.setdefault(r["border"], set()).update([n for n in names if geo(n)] if geo(r["layer"]) else [r["layer"]])
        if not found:
            break
        for b, layers in found.items():
            seam = rasterize_geoms([(lines[b], 1)], cg, all_touched=True).astype(bool)
            for n in sorted(layers):
                write(out / f"{n}.tif", blend(read(out / f"{n}.tif"), seam, cg.res, wide * 1000), cg, nodata=np.nan)
            log.info("seams: %s blended over %.0f km across the %s border", ", ".join(sorted(layers)), wide, b)


def load_stack(paths: Paths, names: list[str], level: str = "coarse", window=None) -> dict[str, np.ndarray]:
    d = paths.interim / level
    return {n: read(d / f"{n}.tif", window=window) for n in names}


def run(cfg) -> None:
    build(cfg)
