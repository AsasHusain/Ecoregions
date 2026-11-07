"""Stage 1 — study area, disputed areas and the analysis grid.

Study area: Pakistan proper, Azad Kashmir and Gilgit-Baltistan (Natural Earth
10 m, eastern limit the Line of Control) and the Siachen Glacier area (Natural
Earth B45). Other areas within the map extent are carried as a separate layer
drawn flat gray; no ecoregions are drawn inside them. An empty label means gray
but unlabelled.
"""
from __future__ import annotations

import logging

import geopandas as gpd
import pandas as pd
from shapely.geometry import box

from .config import Paths
from .grid import Grid, rasterize_geoms, write

log = logging.getLogger(__name__)

# Natural Earth BRK_A3 code -> display label of areas drawn gray ("" = no label)
DISPUTED_LABELS = {
    "B05": 'Indian-administered "Jammu and Kashmir"',
    "B07": "Chinese-administered Aksai Chin",
    "B06": "",  # Shaksgam Valley (Chinese-administered)
}
# Disputed areas mapped as part of the study area (changing this changes the grid:
# rebuild every stage from `grid` on).
STUDY_EXTRA = ["B45"]  # Siachen Glacier area


def load_boundaries(paths: Paths):
    ne = paths.raw / "naturalearth"
    countries = gpd.read_file(ne / "ne_10m_admin_0_countries.shp")
    disputed = gpd.read_file(ne / "ne_10m_admin_0_disputed_areas.shp")
    return countries, disputed


def run(cfg) -> None:
    paths = Paths(cfg)
    crs = cfg["crs"]["proj4"]
    countries, disputed = load_boundaries(paths)
    bbox = box(*cfg["grid"]["bbox_lonlat"])

    study = pd.concat([countries.loc[countries.ADM0_A3 == "PAK", ["geometry"]],
                       disputed.loc[disputed.BRK_A3.isin(STUDY_EXTRA), ["geometry"]]])
    study["name"] = "Pakistan, Azad Kashmir, Gilgit-Baltistan and the Siachen Glacier area"
    study = study.dissolve().reset_index(drop=True)

    dis = disputed[disputed.BRK_A3.isin(DISPUTED_LABELS)].copy()
    dis["label"] = dis.BRK_A3.map(DISPUTED_LABELS)
    dis = dis[["BRK_A3", "label", "NOTE_BRK", "geometry"]]
    dis["geometry"] = dis.geometry.intersection(bbox)
    # remove any overlap with the study area (Natural Earth layers are topologically clean,
    # but enforce it so gray areas can never carry ecoregions)
    dis["geometry"] = dis.geometry.difference(study.union_all())
    dis = dis[~dis.is_empty]

    ctx = countries[countries.intersects(bbox)][["ADM0_A3", "NAME", "geometry"]].copy()
    ctx["geometry"] = ctx.geometry.intersection(bbox.buffer(2))

    # Natural Earth's Pakistan-view file: area logged only, not used for the extent.
    pov = paths.raw / "naturalearth" / "ne_10m_admin_0_countries_pak.shp"
    if pov.exists():
        p = gpd.read_file(pov)
        p = p.loc[p.ADM0_A3 == "PAK"].to_crs(crs)
        log.info("Pakistan POV polygon: %.0f km² vs de facto %.0f km² (not used for the extent)",
                 p.area.sum() / 1e6, study.to_crs(crs).area.sum() / 1e6)

    out = paths.vector("boundaries")
    if out.exists():
        out.unlink()
    study_p = study.to_crs(crs)
    study_p.to_file(out, layer="study_area")
    dis.to_crs(crs).to_file(out, layer="disputed_gray")
    ctx.to_crs(crs).to_file(out, layer="countries_context")
    log.info("study area %.0f km²", study_p.area.sum() / 1e6)

    # analysis grid: study area + buffer, snapped to the resolution
    buf = cfg["grid"]["study_buffer_m"]
    b = study_p.buffer(buf).total_bounds
    grid = Grid.from_bounds(crs, b, cfg["grid"]["resolution"])
    grid.save(paths.grid_json)
    log.info("grid %dx%d @ %sm (%.0f M cells)", grid.width, grid.height, grid.res,
             grid.width * grid.height / 1e6)

    # masks: 1 = study area, 2 = gray (disputed), 3 = other land, 0 = sea
    land = countries.to_crs(crs)
    land = land[land.intersects(box(*grid.bounds))]
    mask = rasterize_geoms([(g, 3) for g in land.geometry], grid)
    gray = rasterize_geoms([(g, 1) for g in dis.to_crs(crs).geometry], grid)
    mask[gray == 1] = 2
    sa = rasterize_geoms([(g, 1) for g in study_p.geometry], grid)
    mask[sa == 1] = 1
    write(paths.raster("mask"), mask, grid, nodata=None,
          tags=dict(values="0 sea, 1 study area, 2 disputed (gray), 3 other land"))
    from .qa import quicklook

    quicklook(mask, paths.qa / "01_mask.png", "analysis mask", cmap="tab10", categorical=True)
    pd.Series(dict(study_km2=float((mask == 1).sum() * grid.cell_km2),
                   gray_km2=float((mask == 2).sum() * grid.cell_km2))).to_json(paths.interim / "mask_stats.json")
