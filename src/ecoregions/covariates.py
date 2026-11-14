"""Stage 5 — non-terrain covariates resampled onto the 90 m analysis grid.

* CHELSA v2.1 (1981-2010): bio1, bio4, bio12, bio15, gdd5, ai; monthly rain gives ``summer_rain``, the share
  of the annual rain falling in ``summer_months`` (0 winter rain only .. 1 monsoon only); and relative humidity and vapour-pressure
  deficit (hurs_mean, vpd_mean; ecoregion information only) — bilinear.
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

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.merge import merge

from . import vectors
from .config import Paths
from .grid import Grid, is_current, rasterize_geoms, read, reproject_array, reproject_file, write
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
    12: "Evaporites (salt crusts and diapirs)",  # Iran (geo2cg): kavir salt crusts, Hormuz salt domes
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
    # Iran (USGS OFR 97-470-G, geo2cg): same series conventions -- age prefix, then i intrusive, v volcanic,
    # m metamorphic, o ophiolite; lowercase a/b/i + e/i = acid/basic/intermediate extrusive/intrusive (mostly Cenozoic)
    "K": (5, 3), "J": (5, 4), "KJ": (5, 3.5), "TK": (4, 2.5), "T": (3, 1.5), "QN": (3, 0.5), "Qe": (2, 0),
    "Qsk": (12, 0), "Salt": (12, 7.5), "P": (9, 6), "TrP": (9, 5.5), "PC": (9, 6), "C": (9, 6.5), "CD": (9, 6.5),
    "SO": (9, 7), "O": (9, 7), "Cm": (9, 7.5), "pCm": (10, 8), "pCmi": (10, 8), "pCmv": (10, 8), "PzpCm": (10, 7.5),
    "CmpCm": (10, 7.5), "Czv": (8, 1.5), "CzMzv": (8, 2.5), "Mzv": (8, 4), "Pzv": (8, 6.5), "vs": (8, 2),
    "ae": (8, 1.5), "be": (8, 1.5), "ie": (8, 1.5), "ai": (7, 1.5), "bi": (7, 1.5), "ii": (7, 1.5),
    "acid": (7, 2), "basic": (7, 2), "inter": (7, 2), "CzMzi": (7, 2.5), "MzPzi": (7, 5),
    "Io": (7, 3), "Mzo": (7, 3.5), "To": (7, 1.5),
    "U": (0, float("nan")), "SEA": (11, 0),
    # Former Soviet Union (USGS OFR 97-470-E, geo1ec), codes as in its metadata: X, Y, Z, A = Proterozoic and Archean
    "QT": (3, 0.5), "NPg": (3, 1.5), "PgK": (4, 2.5), "PZ": (9, 6.5), "DS": (9, 7), "OCm": (9, 7.5), "Pi": (7, 6.5),
    "TKi": (7, 2.5), "PtAi": (10, 8), "Pt": (10, 8), "Z": (10, 8), "Y": (10, 8), "X": (10, 8), "A": (10, 8),
    "CmPt": (10, 7.5), "Ice": (11, 0), "oth": (0, float("nan")),
    # Europe including Turkey (USGS geo4_2l): m = metamorphic fill on the source sheets, u = upper; i, m = age unknown
    "Cz": (3, 1.5), "Mzm": (6, 4), "MzPzm": (6, 5), "Pzm": (6, 6.5), "PzpCmm": (6, 7.5), "Pzu": (9, 6), "i": (7, 4),
    "m": (6, 6), "ice": (11, 0), "unk": (0, float("nan")),
    "Sea1": (11, 0), "Sea2": (11, 0), "Sea3": (11, 0), "Sea4": (11, 0), "Sea5": (11, 0),
}
OUTSIDE = {"H20", "U", "SEA", "oth", "unk", "Sea1", "Sea2", "Sea3", "Sea4", "Sea5"}  # outside a map's coverage or open sea: dropped, so overlapping maps never overwrite real units
# USGS State Geologic Map Compilation (SGMC, Horton 2017), conterminous US: groups by rock type, then age.
# Assigned from each unit's major lithology (MAJOR1), generalized lithology, name and age (Age table, Ma).
SGMC_GROUPS = {
    1: "Quaternary sediments (alluvium, coastal and cover deposits)",  # incl. the High Plains cover sands
    2: "Quaternary aeolian sand",
    3: "Cenozoic sand, clay and gravel",
    4: "Mesozoic limestone",
    5: "Mesozoic chalk, marl and clay",
    6: "Mesozoic sandstone",
    7: "Permian and Triassic red beds",
    8: "Evaporites",
    9: "Paleozoic sedimentary rocks",
    10: "Precambrian basement and intrusive rocks",
    11: "Water",
    12: "Volcanic rocks",
}
AEOLIAN = r"eolian|windblown|wind-blown|dune|sand sheet"
VOLCANIC = {"Basalt", "Rhyolite", "Trachyte", "Andesite", "Dacite", "Tuff", "Hypabyssal-felsic-alkaline", "Latite",
            "Phonolite", "Volcanic", "Pyroclastic"}
EVAPORITE = {"Gypsum", "Anhydrite", "Evaporite", "Salt", "Halite"}
CARBONATE = {"Limestone", "Dolostone", "Carbonate"}
COARSE = {"Sandstone", "Sand", "Conglomerate", "Coarse-detrital", "Gravel", "Arkose", "Conglomerate-sandstone", "Boulders",
          "Sandstone-mudstone"}
# upper age (Ma) of each period -> age rank (0 Quaternary ... 8 Precambrian, as for geo8ag)
AGE_RANKS = [(2.58, 0), (23.03, 1), (66.0, 2), (145.0, 3), (201.4, 4), (251.9, 5), (298.9, 6), (358.9, 6.5),
             (419.2, 6.5), (443.8, 7), (485.4, 7), (538.8, 7.5)]


def age_rank(ma: float) -> float:
    return next((r for top, r in AGE_RANKS if ma <= top), 8.0) if ma == ma else float("nan")


def sgmc_group(major1: str, glith: str, name: str, ma: float) -> int:
    """Lithology group (``SGMC_GROUPS``) of one SGMC unit; ``ma`` = middle of its age range (Ma)."""
    import re

    major1, glith, name = major1 or "", glith or "", name or ""
    if glith == "Water":
        return 11
    if glith.startswith("Igneous, volcanic") or major1 in VOLCANIC:
        return 12
    if glith.startswith(("Igneous", "Metamorphic")) or ma > 538.8:
        return 10
    if ma <= 2.58:
        return 2 if re.search(AEOLIAN, name, re.I) else 1
    if major1 in EVAPORITE or glith == "Sedimentary, evaporite":
        return 8
    if ma <= 66.0:
        return 3
    if ma <= 251.9:  # Mesozoic
        if ma > 201.4 and major1 not in CARBONATE:
            return 7  # Triassic red beds
        return 4 if major1 in CARBONATE else 6 if major1 in COARSE else 5
    if ma <= 298.9 and major1 not in CARBONATE:
        return 7  # Permian red beds
    return 9


def litho_groups(cfg) -> dict[int, str]:
    """The lithology groups of this region's geology map."""
    return SGMC_GROUPS if cfg["sources"]["geology"].get("format") == "sgmc" else LITHO_GROUPS


INTERPRETED_UNITS = {"Mi", "Qi", "Prim", "CMi", "Mz", "Mzi", "PTr", "Tr", "MzPz", "Pzl", "Ti", "Qs"}

WORLDCOVER = {10: "tree cover", 20: "shrubland", 30: "grassland", 40: "cropland", 50: "built-up",
              60: "bare / sparse vegetation", 70: "snow and ice", 80: "permanent water", 90: "herbaceous wetland",
              95: "mangroves", 100: "moss and lichen"}

CLIMATE_UNITS = {"bio1": "°C", "bio4": "°C×100 (sd)", "bio12": "mm/yr", "bio15": "CV %", "gdd5": "°C·days",
                 "ai": "P/PET", "hurs_mean": "%", "vpd_mean": "Pa"}


def climate(cfg, paths: Paths, grid: Grid) -> None:
    for var in cfg["sources"]["chelsa"]["variables"]:
        out = paths.raster(f"clim_{var}")
        if is_current(out, grid):
            continue
        src = paths.raw_region / "chelsa" / f"CHELSA_{var}_1981-2010_V.2.1_clip.tif"
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
    # rain season: share of the annual rain falling in the summer (monsoon) months, from monthly CHELSA rain
    out = paths.raster("clim_summer_rain")
    if not is_current(out, grid):
        months = cfg["sources"]["chelsa"]["summer_months"]
        summer = total = None
        for m in range(1, 13):
            with rasterio.open(paths.raw_region / "chelsa" / f"CHELSA_pr_{m:02d}_1981-2010_V.2.1_clip.tif") as s:
                a = s.read(1).astype("float64")  # same window for every month; scale cancels in the ratio
                a[a == s.nodata] = np.nan
                transform, crs = s.transform, s.crs
            total = a if total is None else total + a
            if m in months:
                summer = a if summer is None else summer + a
        share = np.where(total > 0, summer / np.where(total > 0, total, 1), np.nan)
        share = reproject_array(share.astype("float32"), transform, crs, grid, Resampling.bilinear, src_nodata=np.nan)
        write(out, share, grid, nodata=np.nan, tags=dict(units=f"share of annual rain in months {months}"))
        quicklook(share, paths.qa / "05_clim_summer_rain.png", "summer (May-Oct) share of annual rain", cmap="BrBG")


def phenology(cfg, paths: Paths, grid: Grid) -> None:
    """MODIS phenology medians (``download.fetch_mcd12q2``) on the grid as ``pheno_<name>``."""
    for src in sorted((paths.raw_region / "mcd12q2" / "median").glob("*.tif")):
        out = paths.raster(f"pheno_{src.stem}")
        if is_current(out, grid):
            continue
        arr = reproject_file(src, grid, Resampling.bilinear)
        with rasterio.open(src) as s:
            tags = s.tags()
        write(out, arr.astype("float32"), grid, nodata=np.nan, tags=tags)
        log.info("phenology %s: %.2f..%.2f", src.stem, np.nanmin(arr), np.nanmax(arr))
        quicklook(arr, paths.qa / f"05_pheno_{src.stem}.png", f"MODIS {src.stem}", cmap="YlGn")


def glaciers(cfg, paths: Paths, grid: Grid) -> None:
    """RGI 7.0 glacier outlines: clipped to the study area (``glaciers.gpkg``, an overlay for the maps) and
    rasterised to the grid (``glacier.tif``, 1 = glacier). Glaciers are not an ecoregion type: a glacier tongue
    belongs to the ecoregion of its valley, and outlines move as glaciers retreat."""
    out, vec = paths.raster("glacier"), paths.vector("glaciers")
    if is_current(out, grid) and vec.exists():
        return
    files = cfg["sources"].get("rgi", {}).get("files", [])
    if not files:  # a region without glaciers
        write(out, np.zeros(grid.shape, "uint8"), grid)
        return
    w, s, e, n = cfg["grid"]["bbox_lonlat"]
    g = pd.concat([gpd.read_file(f"zip://{paths.raw / 'rgi' / f}", bbox=(w, s, e, n), columns=[])
                   for f in cfg["sources"]["rgi"]["files"]]).to_crs(grid.crs)
    g = g.set_geometry(g.make_valid())  # a few RGI outlines self-intersect after reprojection
    study =gpd.read_file(paths.vector("boundaries"), layer="study_area").to_crs(grid.crs)
    gpd.clip(g, study).explode()[lambda d: d.geom_type == "Polygon"].to_file(vec, layer="glaciers")
    arr = rasterize_geoms(((geom, 1) for geom in g.geometry), grid)
    write(out, arr, grid, nodata=None, tags=dict(units="1 = glacier (RGI 7.0)"))
    log.info("glaciers: %d outlines, %.0f km2 on the grid", len(g), arr.sum() * grid.res ** 2 / 1e6)
    quicklook(arr, paths.qa / "05_glacier.png", "RGI 7.0 glaciers", cmap="Blues")


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
            src = paths.raw_region / "soilgrids" / f"{prop}_{depth}_mean_clip.tif"
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


def sgmc_units(cfg, paths: Paths) -> tuple[gpd.GeoDataFrame, dict]:
    """SGMC polygons with a short unit code (``GLG``) and each code's (lithology group, age rank)."""
    import pyogrio

    src = cfg["sources"]["geology"]
    g = vectors.geology(cfg, "polygons")
    age = pyogrio.read_dataframe(paths.raw / "geology" / src["gdb"], layer="Age", read_geometry=False)
    ma = age.drop_duplicates("UNIT_LINK").set_index("UNIT_LINK")[["MIN_MA", "MAX_MA"]].astype(float).mean(axis=1)
    g["GLG"] = g.UNIT_LINK.str.removesuffix(";0")
    u = g.drop_duplicates("GLG").set_index("GLG")
    units = {c: (sgmc_group(r.MAJOR1, r.GENERALIZED_LITH, r.UNIT_NAME, ma.get(r.UNIT_LINK, float("nan"))),
                 age_rank(ma.get(r.UNIT_LINK, float("nan")))) for c, r in u.iterrows()}
    return g, units


def geology(cfg, paths: Paths, grid: Grid) -> None:
    sgmc = cfg["sources"]["geology"].get("format") == "sgmc"
    g, units = sgmc_units(cfg, paths) if sgmc else (vectors.geology(cfg, "polygons"), GEO_UNITS)
    n_null = int(g.GLG.isna().sum())
    if n_null:
        log.warning("geology: %d polygons without a unit code (left unassigned)", n_null)
    g = g[g.GLG.notna() & ~g.GLG.isin(OUTSIDE)]
    codes = sorted(g.GLG.unique())
    unknown = [c for c in codes if c not in units]
    if unknown:
        raise ValueError(f"geology units without a lithology mapping: {unknown}")
    unit_id = {c: i + 1 for i, c in enumerate(codes)}
    lab = rasterize_geoms(((geom, unit_id[c]) for geom, c in zip(g.geometry, g.GLG)), grid, dtype="uint16")
    write(paths.raster("geo_unit"), lab, grid, nodata=0, tags={str(v): k for k, v in unit_id.items()})
    lut_l = np.zeros(max(unit_id.values()) + 1, "uint8")
    lut_a = np.full(max(unit_id.values()) + 1, np.nan, "float32")
    for c, i in unit_id.items():
        lut_l[i], lut_a[i] = units[c]
    write(paths.raster("geo_litho"), lut_l[lab], grid, nodata=0, tags={str(k): v for k, v in litho_groups(cfg).items()})
    write(paths.raster("geo_age"), lut_a[lab], grid, nodata=np.nan, tags=dict(units="age rank 0 Quaternary .. 8 Precambrian"))
    # contacts = boundaries between different units
    e = np.zeros(grid.shape, bool)
    e[:, 1:] |= (lab[:, 1:] != lab[:, :-1]) & (lab[:, 1:] > 0) & (lab[:, :-1] > 0)
    e[1:, :] |= (lab[1:, :] != lab[:-1, :]) & (lab[1:, :] > 0) & (lab[:-1, :] > 0)
    write(paths.raster("geo_contact"), e.astype("uint8"), grid)
    quicklook(lut_l[lab], paths.qa / "05_geology_litho.png", f"lithology group ({cfg['sources']['geology'].get('label', 'geo8ag')})", cmap="tab20", categorical=True)


def worldcover(cfg, paths: Paths, grid: Grid) -> None:
    out = paths.raster("worldcover")
    if is_current(out, grid):
        return
    files = sorted((paths.raw_region / "worldcover").glob("*.tif"))
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
    phenology(cfg, paths, grid)
    glaciers(cfg, paths, grid)
    soils(cfg, paths, grid)
    geology(cfg, paths, grid)
    resolve(cfg, paths, grid)
    worldcover(cfg, paths, grid)
