"""Readers for the vector sources, clipped to the processing extent and
reprojected to the analysis CRS. Results are cached in ``data/interim``."""
from __future__ import annotations

import logging

import geopandas as gpd
import pyogrio

from .config import Paths

log = logging.getLogger(__name__)


def _read_clip(cfg, src: str, layer_name: str, columns=None, where=None) -> gpd.GeoDataFrame:
    paths = Paths(cfg)
    cache = paths.vector("sources")
    if cache.exists() and layer_name in [l[0] for l in pyogrio.list_layers(cache)]:
        return gpd.read_file(cache, layer=layer_name)
    w, s, e, n = cfg["grid"]["bbox_lonlat"]
    gdf = gpd.read_file(src, bbox=(w - 1, s - 1, e + 1, n + 1), columns=columns, where=where)
    gdf = gdf.to_crs(cfg["crs"]["proj4"])
    gdf.to_file(cache, layer=layer_name)
    log.info("cached %s: %d features", layer_name, len(gdf))
    return gdf


def hydrorivers(cfg) -> gpd.GeoDataFrame:
    p = Paths(cfg).raw / "hydrorivers" / "HydroRIVERS_v10_as_shp.zip"
    return _read_clip(cfg, f"/vsizip/{p}/HydroRIVERS_v10_as_shp/HydroRIVERS_v10_as.shp", "hydrorivers",
                      columns=["HYRIV_ID", "NEXT_DOWN", "MAIN_RIV", "UPLAND_SKM", "DIS_AV_CMS",
                               "ORD_STRA", "ORD_CLAS", "ENDORHEIC"])


def hydrobasins(cfg, level: int) -> gpd.GeoDataFrame:
    p = Paths(cfg).raw / "hydrobasins" / "hybas_as_lev01-12_v1c.zip"
    return _read_clip(cfg, f"/vsizip/{p}/hybas_as_lev{level:02d}_v1c.shp", f"hydrobasins_l{level:02d}",
                      columns=["HYBAS_ID", "NEXT_DOWN", "MAIN_BAS", "SUB_AREA", "UP_AREA", "ENDO", "ORDER"])


def geology(cfg, kind: str = "polygons") -> gpd.GeoDataFrame:
    p = Paths(cfg).raw / "geology" / "geo8ag.zip"
    inner = cfg["sources"]["geology"][kind]
    return _read_clip(cfg, f"/vsizip/{p}/{inner}", f"geology_{kind}")


def resolve(cfg) -> gpd.GeoDataFrame:
    p = Paths(cfg).raw / "resolve" / "Ecoregions2017.zip"
    return _read_clip(cfg, f"/vsizip/{p}/Ecoregions2017.shp", "resolve")
