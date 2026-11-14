"""Draw the world confirmation mask once (docs/preregistration.md, section 8): every 2-degree square with at least a
quarter of its area on land not yet mapped gets a rank in one seeded random order; the lowest 30 % are sealed.

    PYTHONPATH=src python scripts/draw_world_mask.py

Refuses to overwrite docs/confirmation/world_mask.json.
"""
import json
import sys
from pathlib import Path

import geopandas as gpd
import shapely

from ecoregions.confirm import draw_world_mask

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "confirmation" / "world_mask.json"
MAPPED = ["outputs/pak-afg-irn/ecoregions.gpkg", "outputs/texas/ecoregions.gpkg"]  # areas mapped when drawn

if OUT.exists():
    sys.exit(f"{OUT} exists: the world mask is drawn once and never redrawn")
countries = gpd.read_file(ROOT / "data/raw/naturalearth/ne_10m_admin_0_countries.shp").to_crs(4326)
mapped = shapely.unary_union([gpd.read_file(ROOT / f, layer="study_area").to_crs(4326).union_all() for f in MAPPED])
m = draw_world_mask(countries, mapped.buffer(0.02), block_deg=2.0, share=0.3, min_land=0.25, seed=20261114)
m["excluded_mapped_areas"] = MAPPED
OUT.write_text(json.dumps(m), encoding="utf-8")
print(f"{m['candidates']} squares, {m['sealed_below_rank']} sealed -> {OUT}")
