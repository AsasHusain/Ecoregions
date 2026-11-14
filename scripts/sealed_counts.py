"""Enough sealed data? (docs/preregistration.md, section 8) Record COUNTS only, per 0.25-degree cell inside an area's
sealed squares, for plants and birds; squares are added in rank order until each taxon has >= 20 eligible 1-degree
blocks and >= 40 eligible cells (>= 100 records), or half the area's squares are sealed. Writes the rank cut-off.

    PYTHONPATH=src python scripts/sealed_counts.py central-asia TKM UZB TJK KGZ
"""
import json
import sys
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import requests
import shapely

ROOT = Path(__file__).resolve().parents[1]
name, iso = sys.argv[1], sys.argv[2:]
OUT = ROOT / "docs" / "confirmation" / f"{name}_sealed.json"
if OUT.exists():
    sys.exit(f"{OUT} exists: the cut-off is set once")
m = json.loads((ROOT / "docs/confirmation/world_mask.json").read_text())
d, cut = m["block_deg"], m["sealed_below_rank"]
area = gpd.read_file(ROOT / "data/raw/naturalearth/ne_10m_admin_0_countries.shp").to_crs(4326)
area = area[area.ADM0_A3.isin(iso)].union_all()
squares = sorted([(r, lon, lat) for lon, lat, r in m["blocks"] if shapely.box(lon, lat, lon + d, lat + d).intersects(area)])
TAXA = {"plants": 6, "birds": 212}  # GBIF backbone keys: Plantae, Aves
S = requests.Session()
S.headers["User-Agent"] = "ecoregions-research/1.0"


def count(lon, lat, key):
    for t in range(5):
        r = S.get("https://api.gbif.org/v1/occurrence/search", timeout=60, params=dict(
            decimalLongitude=f"{lon},{lon + 0.25}", decimalLatitude=f"{lat},{lat + 0.25}", taxonKey=key,
            occurrenceStatus="PRESENT", hasCoordinate="true", limit=0))
        if r.status_code == 200:
            return r.json()["count"]
        time.sleep(2 ** t)
    r.raise_for_status()


cells = {}  # (lon, lat) -> {taxon: count}, cells wholly inside a sealed square with their centre in the area


def add_square(lon, lat):
    for x in np.arange(lon, lon + d, 0.25):
        for y in np.arange(lat, lat + d, 0.25):
            if area.contains(shapely.Point(x + 0.125, y + 0.125)):
                cells[(float(x), float(y))] = {t: count(x, y, k) for t, k in TAXA.items()}


def status():
    out = {}
    for t in TAXA:
        ok = [c for c, v in cells.items() if v[t] >= 100]
        out[t] = dict(cells=len(ok), blocks=len({(np.floor(x), np.floor(y)) for x, y in ok}))
    return out


sealed = [s for s in squares if s[0] < cut]
for _, lon, lat in sealed:
    add_square(lon, lat)
st = status()
print("global share:", len(sealed), "squares", st, flush=True)
rest = [s for s in squares if s[0] >= cut]
while any(v["blocks"] < 20 or v["cells"] < 40 for v in st.values()) and rest and len(sealed) < len(squares) / 2:
    s = rest.pop(0)
    sealed.append(s)
    add_square(s[1], s[2])
    st = status()
    print("added rank", s[0], st, flush=True)
short = [t for t, v in st.items() if v["blocks"] < 20 or v["cells"] < 40]
res = dict(area=name, countries=iso, squares_in_area=len(squares), squares_sealed=len(sealed),
           rank_cut=max(r for r, _, _ in sealed) + 1, eligible=st, descriptive_only=short,
           note="record counts only (GBIF occurrence search, limit=0); no names, dates or other fields read")
OUT.write_text(json.dumps(res, indent=1), encoding="utf-8")
print(json.dumps(res, indent=1))
