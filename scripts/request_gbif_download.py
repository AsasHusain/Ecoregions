"""Request a GBIF occurrence download (Darwin Core Archive) for an area: all taxa, occurrence status present, within
the area's countries plus a margin. Credentials from GBIF_USER, GBIF_PWD, GBIF_EMAIL (never printed). Prints the key.

    PYTHONPATH=src python scripts/request_gbif_download.py TKM UZB TJK KGZ --margin 0.5
"""
import argparse
import os
from pathlib import Path

import geopandas as gpd
import requests
import shapely
from shapely.geometry.polygon import orient

ROOT = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser()
p.add_argument("iso3", nargs="+")
p.add_argument("--margin", type=float, default=0.5, help="degrees around the countries")
a = p.parse_args()

ne = gpd.read_file(ROOT / "data/raw/naturalearth/ne_10m_admin_0_countries.shp").to_crs(4326)
area = ne[ne.ADM0_A3.isin(a.iso3)].union_all().buffer(a.margin).simplify(0.05)
area = max(getattr(area, "geoms", [area]), key=lambda g: g.area) if area.geom_type == "MultiPolygon" else area
area = orient(shapely.Polygon(area.exterior), 1.0)  # counter-clockwise, no holes: GBIF's WKT rule
assert len(area.exterior.coords) < 2000, "polygon too detailed for GBIF"
body = {"creator": os.environ["GBIF_USER"], "notificationAddress": [os.environ["GBIF_EMAIL"]], "sendNotification": True,
        "format": "DWCA",
        "predicate": {"type": "and", "predicates": [
            {"type": "within", "geometry": area.wkt},
            {"type": "equals", "key": "OCCURRENCE_STATUS", "value": "present"}]}}
r = requests.post("https://api.gbif.org/v1/occurrence/download/request", json=body,
                  auth=(os.environ["GBIF_USER"], os.environ["GBIF_PWD"]), timeout=120)
r.raise_for_status()
print("download key:", r.text.strip(), "| vertices:", len(area.exterior.coords))
