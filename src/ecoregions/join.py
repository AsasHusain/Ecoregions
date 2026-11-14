"""Stage ``join`` — a parcel's published layers clipped at the border of the area it maps.

A parcel is built with a strip into its mapped neighbours (``grid.overlap``) so its edge units see their context;
the strip belongs to the neighbours' maps. This stage clips every polygon layer of ``ecoregions.gpkg`` and
``physiographic.gpkg`` to the mapped area, drops units left wholly in the strip, recomputes ``area_km2``, and
rewrites ``ecoregions.csv`` and ``l3_ecoregions.csv`` to match. Other statistics stay those of the whole unit.
Run after ``package`` and before the maps (and before a freeze, so the hash is of the published file).
"""
from __future__ import annotations

import logging

import geopandas as gpd
import pandas as pd
import pyogrio

from .boundaries import load_boundaries, mapped_area
from .config import Paths

log = logging.getLogger(__name__)


def clip_layer(g: gpd.GeoDataFrame, area) -> gpd.GeoDataFrame:
    """``g`` clipped to the polygon ``area`` (same CRS); empty pieces dropped; ``area_km2`` recomputed if present."""
    g = g.assign(geometry=g.geometry.intersection(area))
    g = g[~g.geometry.is_empty].copy()
    if "area_km2" in g:
        g["area_km2"] = (g.area / 1e6).round(1)
    return g


def run(cfg) -> None:
    if not cfg["grid"].get("overlap"):
        log.info("join: not a parcel (no grid.overlap); nothing to clip")
        return
    paths = Paths(cfg)
    area = mapped_area(cfg, paths, *load_boundaries(paths)).to_crs(cfg["crs"]["proj4"]).union_all()
    for f in (paths.outputs / "ecoregions.gpkg", paths.outputs / "physiographic.gpkg"):
        layers = [(n, gpd.read_file(f, layer=n)) for n, _ in pyogrio.list_layers(f)]
        f.unlink()
        for n, g in layers:
            before = len(g)
            if n == "rivers":  # context just beyond the border, as package draws them
                g = g[g.intersects(area.buffer(20000))]
            elif n == "borders":  # national borders and the area's outline; the strip's outer edge goes
                g = g.assign(geometry=g.geometry.intersection(area.buffer(1000)))
                g = g[~g.geometry.is_empty]
            elif n == "study_area":
                g = g.iloc[:1].assign(geometry=[area])
            else:
                g = clip_layer(g, area)
            g.to_file(f, layer=n)
            log.info("join: %s %s: %d -> %d features", f.name, n, before, len(g))
    eco = gpd.read_file(paths.outputs / "ecoregions.gpkg", layer="ecoregions")
    csv = pd.read_csv(paths.outputs / "ecoregions.csv")
    csv = csv[csv.unit.isin(eco.unit)].drop(columns="area_km2").merge(eco[["unit", "area_km2"]], on="unit")
    csv.to_csv(paths.outputs / "ecoregions.csv", index=False)
    l3 = gpd.read_file(paths.outputs / "physiographic.gpkg", layer="l3_ecoregions")
    l3csv = pd.read_csv(paths.outputs / "l3_ecoregions.csv")
    l3csv = l3csv[l3csv.unit.isin(l3.unit)].drop(columns="area_km2").merge(l3[["unit", "area_km2"]], on="unit")
    l3csv.to_csv(paths.outputs / "l3_ecoregions.csv", index=False)
    small = eco[eco.area_km2 < cfg["assemble"]["mmu_km2"]]
    log.info("join: %d Floral Ecoregions, %.0f km2; %d below the minimum size after clipping: %s", len(eco),
             eco.area_km2.sum(), len(small), ", ".join(f"{r.code} {r.area_km2:.0f} km2" for r in small.itertuples()))


if __name__ == "__main__":  # self-check: a square half outside the area keeps its inside half and that area
    from shapely.geometry import box
    g = gpd.GeoDataFrame({"area_km2": [8.0]}, geometry=[box(0, 0, 4000, 2000)])
    out = clip_layer(g, box(0, 0, 2000, 2000))
    assert len(out) == 1 and out.area_km2.iloc[0] == 4.0
    assert clip_layer(g, box(5000, 0, 6000, 1000)).empty
    print("ok")
