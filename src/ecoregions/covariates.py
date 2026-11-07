"""Stage 5 — non-terrain covariates resampled onto the 90 m analysis grid.

* CHELSA v2.1 (1981-2010): bio1, bio4, bio12, bio15, gdd5, ai — bilinear.
* SoilGrids 2.0: clay, sand, pH(H2O), SOC — thickness-weighted 0-30 cm mean
  of the 0-5, 5-15 and 15-30 cm layers, bilinear, converted to conventional units.
* USGS geo8ag geology: rasterized unit codes plus derived lithology group and
  age rank; geological *contacts* are taken from boundaries between polygons of
  different units (the line file's symbol codes do not match its metadata).
* ESA WorldCover 2021 (context only — never used for delineation).
* RESOLVE Ecoregions2017 rasterized for comparison.
"""
from __future__ import annotations

import logging

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.merge import merge

from . import vectors
from .config import Paths
from .grid import Grid, is_current, rasterize_geoms, reproject_array, reproject_file, write
from .qa import quicklook

log = logging.getLogger(__name__)

# geo8ag unit -> (lithology group, age rank, label). Age rank: 0 Quaternary ... 8 Precambrian.
# Units absent from the FGDC domain list (Mi, Qi, Prim, CMi, Mz, Mzi) are interpreted
# from their symbols and the regional geology; they are flagged as such.
LITHO_GROUPS = {
    1: "Quaternary alluvium and fans",
    2: "Quaternary aeolian sand",
    3: "Neogene molasse (sandstone, clay, conglomerate)",
    4: "Paleogene sediments (limestone, shale)",
    5: "Mesozoic sediments (limestone, shale, flysch)",
    6: "Metasediments (slate, schist, marble)",
    7: "Intrusive and ultramafic rocks",
    8: "Volcanic rocks",
    9: "Paleozoic sediments",
    10: "Precambrian basement",
    11: "Water / ice",
}
GEO_UNITS = {
    "Q": (1, 0), "Qs": (2, 0), "Qv": (8, 0), "Qi": (8, 0),
    "N": (3, 1), "Ts": (3, 1.5), "Tv": (8, 1.5), "Ti": (7, 1.5),
    "Pg": (4, 2), "TKs": (4, 2.5), "TKv": (8, 2.5), "TKim": (7, 2.5),
    "Ks": (5, 3), "KJs": (5, 3.5), "Mi": (7, 3.5), "Mz": (5, 4), "Mzi": (7, 4),
    "Jms": (6, 4), "JTr": (5, 4.5), "Trms": (6, 5), "Tr": (5, 5), "TrPr": (9, 5.5), "TrCs": (9, 5.5),
    "PTr": (9, 5.5), "Pr": (9, 6), "Prim": (7, 6), "MzPz": (6, 5),
    "Cs": (9, 6.5), "D": (9, 6.5), "S": (9, 7), "Pz": (9, 6.5), "Pzl": (9, 7), "Pzi": (7, 6.5),
    "CMi": (7, 7.5), "Cmsm": (6, 7.5), "PzpC": (10, 7.5), "pC": (10, 8), "Gry": (7, 6),
    "Osm": (6, 7), "SOc": (9, 7),
    "I": (11, 0), "H2O": (11, 0),
    "H20": (0, float("nan")),  # "other regions": outside the map's coverage (neighbouring countries)
}
INTERPRETED_UNITS = {"Mi", "Qi", "Prim", "CMi", "Mz", "Mzi", "PTr", "Tr", "MzPz", "Pzl", "Ti", "Qs"}

WORLDCOVER = {10: "tree cover", 20: "shrubland", 30: "grassland", 40: "cropland", 50: "built-up",
              60: "bare / sparse vegetation", 70: "snow and ice", 80: "permanent water", 90: "herbaceous wetland",
              95: "mangroves", 100: "moss and lichen"}

CLIMATE_UNITS = {"bio1": "°C", "bio4": "°C×100 (sd)", "bio12": "mm/yr", "bio15": "CV %", "gdd5": "°C·days",
                 "ai": "P/PET"}


def climate(cfg, paths: Paths, grid: Grid) -> None:
    for var in cfg["sources"]["chelsa"]["variables"]:
        out = paths.raster(f"clim_{var}")
        if is_current(out, grid):
            continue
        src = paths.raw / "chelsa" / f"CHELSA_{var}_1981-2010_V.2.1_clip.tif"
        arr = reproject_file(src, grid, Resampling.bilinear)
        with rasterio.open(src) as s:
            scale = s.scales[0]
        # CHELSA stores the aridity index as an integer x 10000 in some releases
        if var == "ai" and np.nanmedian(arr[::50, ::50]) > 50:
            arr = arr / 10000.0
        if var == "ai":
            # P/PET explodes where PET -> 0 (glaciers, ~1e7); cap at 5 (UNEP "humid" starts at 0.65)
            arr = np.clip(arr, 0, 5)
        write(out, arr.astype("float32"), grid, nodata=np.nan, tags=dict(units=CLIMATE_UNITS[var], scale=scale))
        log.info("climate %s: %.2f..%.2f", var, np.nanmin(arr), np.nanmax(arr))
        quicklook(arr, paths.qa / f"05_clim_{var}.png", f"CHELSA {var}", cmap="RdYlBu_r" if var != "bio12" else "YlGnBu")


SOIL_CONVERT = {"clay": 0.1, "sand": 0.1, "phh2o": 0.1, "soc": 0.1}  # -> %, %, pH, g/kg
SOIL_UNITS = {"clay": "%", "sand": "%", "phh2o": "pH", "soc": "g/kg"}


def soils(cfg, paths: Paths, grid: Grid) -> None:
    sc = cfg["sources"]["soilgrids"]
    for prop in sc["properties"]:
        out = paths.raster(f"soil_{prop}")
        if is_current(out, grid):
            continue
        acc = None
        wsum = None
        for depth, thick in sc["depths"].items():
            src = paths.raw / "soilgrids" / f"{prop}_{depth}_mean_clip.tif"
            a = reproject_file(src, grid, Resampling.bilinear, apply_scale=False)
            a[a < 0] = np.nan  # SoilGrids nodata (-32768) and impossible negatives
            v = ~np.isnan(a)
            if acc is None:
                acc = np.zeros(grid.shape, "float32")
                wsum = np.zeros(grid.shape, "float32")
            acc[v] += a[v] * thick
            wsum[v] += thick
            del a, v
        with np.errstate(invalid="ignore", divide="ignore"):
            res = np.where(wsum > 0, acc / wsum * SOIL_CONVERT[prop], np.nan).astype("float32")
        write(out, res, grid, nodata=np.nan, tags=dict(units=SOIL_UNITS[prop], depth="0-30 cm weighted"))
        quicklook(res, paths.qa / f"05_soil_{prop}.png", f"SoilGrids {prop} 0-30 cm ({SOIL_UNITS[prop]})",
                  cmap="copper_r")
        del acc, wsum, res


def geology(cfg, paths: Paths, grid: Grid) -> None:
    g = vectors.geology(cfg, "polygons")
    n_null = int(g.GLG.isna().sum())
    if n_null:
        log.warning("geology: %d polygons without a unit code (left unassigned)", n_null)
    g = g[g.GLG.notna()]
    codes = sorted(g.GLG.unique())
    unknown = [c for c in codes if c not in GEO_UNITS]
    if unknown:
        raise ValueError(f"geology units without a lithology mapping: {unknown}")
    unit_id = {c: i + 1 for i, c in enumerate(codes)}
    lab = rasterize_geoms(((geom, unit_id[c]) for geom, c in zip(g.geometry, g.GLG)), grid, dtype="uint16")
    write(paths.raster("geo_unit"), lab, grid, nodata=0, tags={str(v): k for k, v in unit_id.items()})
    lut_l = np.zeros(max(unit_id.values()) + 1, "uint8")
    lut_a = np.full(max(unit_id.values()) + 1, np.nan, "float32")
    for c, i in unit_id.items():
        lut_l[i], lut_a[i] = GEO_UNITS[c]
    write(paths.raster("geo_litho"), lut_l[lab], grid, nodata=0, tags={str(k): v for k, v in LITHO_GROUPS.items()})
    write(paths.raster("geo_age"), lut_a[lab], grid, nodata=np.nan, tags=dict(units="age rank 0 Quaternary .. 8 Precambrian"))
    # contacts = boundaries between different units
    e = np.zeros(grid.shape, bool)
    e[:, 1:] |= (lab[:, 1:] != lab[:, :-1]) & (lab[:, 1:] > 0) & (lab[:, :-1] > 0)
    e[1:, :] |= (lab[1:, :] != lab[:-1, :]) & (lab[1:, :] > 0) & (lab[:-1, :] > 0)
    write(paths.raster("geo_contact"), e.astype("uint8"), grid)
    quicklook(lut_l[lab], paths.qa / "05_geology_litho.png", "lithology group (geo8ag)", cmap="tab20", categorical=True)


def worldcover(cfg, paths: Paths, grid: Grid) -> None:
    out = paths.raster("worldcover")
    if is_current(out, grid):
        return
    files = sorted((paths.raw / "worldcover").glob("*.tif"))
    srcs = [rasterio.open(f) for f in files]
    mosaic, tr = merge(srcs, nodata=0)
    crs = srcs[0].crs
    for s in srcs:
        s.close()
    arr = reproject_array(mosaic[0], tr, crs, grid, Resampling.mode, src_nodata=0, dst_nodata=0, dtype="uint8")
    write(out, arr, grid, nodata=0, tags={str(k): v for k, v in WORLDCOVER.items()})
    quicklook(arr, paths.qa / "05_worldcover.png", "ESA WorldCover 2021 (context only)", cmap="tab20", categorical=True)


def resolve(cfg, paths: Paths, grid: Grid) -> None:
    r = vectors.resolve(cfg)
    r = r[r.ECO_ID > 0]
    lab = rasterize_geoms(((geom, int(i)) for geom, i in zip(r.geometry, r.ECO_ID)), grid, dtype="uint16")
    write(paths.raster("resolve"), lab, grid, nodata=0)


def basins(cfg, paths: Paths, grid: Grid) -> None:
    """HydroBASINS polygons at ``segmentation.basin_level`` as sequential ids."""
    lvl = cfg["segmentation"]["basin_level"]
    b = vectors.hydrobasins(cfg, lvl)
    lab = rasterize_geoms(((g, i + 1) for i, g in enumerate(b.geometry)), grid, dtype="int32")
    write(paths.raster(f"basin_l{lvl:02d}"), lab, grid, nodata=0,
          tags=dict(source=f"HydroBASINS level {lvl}", note="value = row number + 1 in the cached layer"))


def run(cfg) -> None:
    paths = Paths(cfg)
    grid = Grid.load(paths.grid_json)
    basins(cfg, paths, grid)
    climate(cfg, paths, grid)
    soils(cfg, paths, grid)
    geology(cfg, paths, grid)
    resolve(cfg, paths, grid)
    worldcover(cfg, paths, grid)
