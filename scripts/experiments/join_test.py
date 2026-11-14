"""Parcel-join test (criteria declared in docs/ledger.md, 2026-11-14, before the parcel was built).

Iran built alone (regions/irn.yaml, 75 km strip into Afghanistan and Pakistan) against the combined v0.5.0 map:
  (1) ecoregion type agreement over Iran (sampled on a 270 m lattice);
  (2) border continuity: at points every ~1 km along the Iran-Afghanistan and Iran-Pakistan borders, the types 1.5 km
      inside Iran and 1.5 km outside; where v0.5.0 has one type across, does the joined map (parcel inside, v0.5.0
      outside) too? Pass: both at least 95 %.

    PYTHONPATH=src python scripts/experiments/join_test.py
"""
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import shapely
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.transform import from_bounds

from ecoregions.config import Paths, load_config

comb, parc = Paths(load_config(region="pak-afg-irn")), Paths(load_config(region="irn"))
crs = load_config(region="pak-afg-irn")["crs"]["proj4"]
types = {}


def type_lut(paths):
    df = pd.read_csv(paths.interim / "ecoregions.csv")
    lut = np.zeros(int(df.unit.max()) + 1, np.int32)
    lut[df.unit.to_numpy()] = [types.setdefault(t, len(types) + 1) for t in df.type]
    return lut


lut_c, lut_p = type_lut(comb), type_lut(parc)
ne = gpd.read_file(comb.raw / "naturalearth" / "ne_10m_admin_0_countries.shp").to_crs(crs)
iran = ne[ne.ADM0_A3 == "IRN"].union_all()
nbr = ne[ne.ADM0_A3.isin(["AFG", "PAK"])].union_all()

# (1) agreement over Iran, both rasters resampled (nearest) onto one 270 m lattice over Iran's bounds
w, s, e, n = iran.bounds
W, H = int((e - w) / 270), int((n - s) / 270)
tr = from_bounds(w, s, e, n, W, H)


def on_lattice(paths, lut):
    with rasterio.open(paths.raster("ecoregions")) as src:
        win = rasterio.windows.from_bounds(w, s, e, n, src.transform)
        a = src.read(1, window=win, out_shape=(H, W), resampling=Resampling.nearest, boundless=True, fill_value=0)
    return lut[np.minimum(a, len(lut) - 1)]


tc, tp = on_lattice(comb, lut_c), on_lattice(parc, lut_p)
inside = rasterize([iran], out_shape=(H, W), transform=tr).astype(bool) & (tc > 0) & (tp > 0)
agree = float((tc[inside] == tp[inside]).mean())

# (2) border continuity
border = iran.boundary.intersection(nbr.buffer(500))
lines = [g for g in getattr(border, "geoms", [border]) if g.length > 0]
pts_in, pts_out = [], []
for ln in lines:
    for d in np.arange(500, ln.length, 1000):
        p, q = ln.interpolate(d - 100), ln.interpolate(d + 100)
        nx, ny = -(q.y - p.y), q.x - p.x
        L = np.hypot(nx, ny) or 1
        m = ln.interpolate(d)
        a = shapely.Point(m.x + 1500 * nx / L, m.y + 1500 * ny / L)
        b = shapely.Point(m.x - 1500 * nx / L, m.y - 1500 * ny / L)
        if iran.contains(b):
            a, b = b, a
        if iran.contains(a) and nbr.contains(b):
            pts_in.append((a.x, a.y))
            pts_out.append((b.x, b.y))


def sample(paths, lut, pts):
    with rasterio.open(paths.raster("ecoregions")) as src:
        v = np.array([x[0] for x in src.sample(pts)])
    return lut[np.minimum(v, len(lut) - 1)]


ci, co, pi = sample(comb, lut_c, pts_in), sample(comb, lut_c, pts_out), sample(parc, lut_p, pts_in)
ok = (ci > 0) & (co > 0) & (pi > 0)
cont = ok & (ci == co)
kept = float((pi[cont] == co[cont]).mean())
res = pd.DataFrame([dict(iran_type_agreement=round(agree, 4), border_points=int(ok.sum()),
                         border_points_continuous_in_v050=int(cont.sum()), continuity_kept=round(kept, 4),
                         passed=bool(agree >= 0.95 and kept >= 0.95))])
res.to_csv(comb.outputs.parent / "irn_join_test.csv", index=False)
print(res.to_string(index=False))
