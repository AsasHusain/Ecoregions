"""Readers for the vector sources, clipped to the processing extent and
reprojected to the analysis CRS. Results are cached in ``data/interim``."""
from __future__ import annotations

import logging
from pathlib import PurePosixPath

import geopandas as gpd
import pandas as pd
import pyogrio
from shapely.geometry import box

from .config import Paths

log = logging.getLogger(__name__)


def _read_clip(cfg, src: str, layer_name: str, columns=None, where=None, layer=None) -> gpd.GeoDataFrame:
    paths = Paths(cfg)
    cache = paths.vector("sources")
    if cache.exists() and layer_name in [l[0] for l in pyogrio.list_layers(cache)]:
        return gpd.read_file(cache, layer=layer_name)
    w, s, e, n = cfg["grid"]["bbox_lonlat"]
    bbox = gpd.GeoSeries([box(w - 1, s - 1, e + 1, n + 1)], crs=4326)  # lon/lat box, reprojected to the source's CRS
    srcs = src if isinstance(src, list) else [src]  # several files: an area spanning two source regions
    gdf = pd.concat([gpd.read_file(f, layer=layer, bbox=bbox, columns=columns, where=where).to_crs(cfg["crs"]["proj4"])
                     for f in srcs], ignore_index=True)
    gdf.to_file(cache, layer=layer_name)
    log.info("cached %s: %d features", layer_name, len(gdf))
    return gdf


def hydrorivers(cfg) -> gpd.GeoDataFrame:
    s = cfg["sources"]["hydrorivers"]
    srcs = []
    for u in [s["url"], *s.get("also", [])]:
        stem = PurePosixPath(u).stem  # HydroRIVERS_v10_<continent>_shp
        p = Paths(cfg).raw / "hydrorivers" / f"{stem}.zip"
        srcs.append(f"/vsizip/{p}/{stem}/{stem.removesuffix('_shp')}.shp")
    return _read_clip(cfg, srcs, "hydrorivers",
                      columns=["HYRIV_ID", "NEXT_DOWN", "MAIN_RIV", "UPLAND_SKM", "DIS_AV_CMS",
                               "ORD_STRA", "ORD_CLAS", "ENDORHEIC"])


def hydrobasins(cfg, level: int) -> gpd.GeoDataFrame:
    s = cfg["sources"]["hydrobasins"]
    srcs = []
    for u in [s["url"], *s.get("also", [])]:
        name = PurePosixPath(u).name  # hybas_<continent>_lev01-12_v1c.zip
        p = Paths(cfg).raw / "hydrobasins" / name
        srcs.append(f"/vsizip/{p}/{name.replace('lev01-12', f'lev{level:02d}').removesuffix('.zip')}.shp")
    return _read_clip(cfg, srcs, f"hydrobasins_l{level:02d}",
                      columns=["HYBAS_ID", "NEXT_DOWN", "MAIN_BAS", "SUB_AREA", "UP_AREA", "ENDO", "ORDER"])


def geology(cfg, kind: str = "polygons") -> gpd.GeoDataFrame:
    src = cfg["sources"]["geology"]
    d = Paths(cfg).raw / "geology"
    if src.get("format") == "sgmc":  # File Geodatabase unpacked by the download stage
        # ESRI rings: outer clockwise, holes counter-clockwise (the default ring analysis takes minutes)
        pyogrio.set_gdal_config_options({"OGR_ORGANIZE_POLYGONS": "ONLY_CCW"})
        try:
            return _read_clip(cfg, str(d / src["gdb"]), f"geology_{kind}", layer=src[kind])
        finally:
            pyogrio.set_gdal_config_options({"OGR_ORGANIZE_POLYGONS": None})
    srcs = [f"/vsizip/{d / x['file']}/{x[kind]}" for x in [src, *src.get("also", [])] if x.get(kind)]
    return _read_clip(cfg, srcs, f"geology_{kind}")


def resolve(cfg) -> gpd.GeoDataFrame:
    p = Paths(cfg).raw / "resolve" / "Ecoregions2017.zip"
    return _read_clip(cfg, f"/vsizip/{p}/Ecoregions2017.shp", "resolve")
