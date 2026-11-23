import geopandas as gpd
from shapely.geometry import LineString, box

from ecoregions.world import adopt_strip, combine, natural_seam, on_line, shares


def test_natural_seam_moves_change_to_landform_edge():
    # area q east of x=0, one polygon of type B; the neighbour r west of it, type A, whose strip calls q's first
    # landform unit (0..5) type A as well
    own = gpd.GeoDataFrame({"type": ["B"], "parcel": ["q"], "area_km2": [1.0]}, geometry=[box(0, 0, 10, 10)], crs=6933)
    other = gpd.GeoDataFrame({"type": ["A"], "parcel": ["r"], "area_km2": [1.0]}, geometry=[box(-10, 0, 0, 10)], crs=6933)
    strip = gpd.GeoDataFrame({"type": ["A"]}, geometry=[box(0, 0, 6, 10)], crs=6933)
    cells = gpd.GeoSeries([box(0, 0, 5, 10), box(5, 0, 10, 10)], crs=6933)
    out = natural_seam(own, strip, other, cells, LineString([(0, 0), (0, 10)]), "type")
    by = {t: g.union_all() for t, g in out.groupby("type").geometry}
    assert by["A"].equals(box(0, 0, 5, 10)) and by["B"].equals(box(5, 0, 10, 10))
    assert set(out.parcel) == {"q"}


def test_on_line_finds_an_edge_on_the_border():
    # two countries split at x = 0; type A west, B east: the edge lies on the border for 20 km; C/D edge 4 km off it
    cty = gpd.GeoDataFrame({"NAME": ["W", "E"]}, geometry=[box(-30000, 0, 0, 60000), box(0, 0, 30000, 60000)], crs=3857)
    u = gpd.GeoDataFrame({"type": ["A", "B", "C", "D"]}, geometry=[box(-30000, 0, 0, 30000), box(0, 0, 30000, 30000),
                         box(-30000, 30000, 4000, 60000), box(4000, 30000, 30000, 60000)], crs=3857)
    hits = on_line(u, cty, "type")
    assert len(hits) == 1 and {"A", "B"} <= set(hits[0]["types"].split(" | ")) and 20 <= hits[0]["km"] <= 30, hits


def test_shares_read_the_area_layers_bounds():
    # the area layers write "Pakistan >99%"; dropping it left "Kashmir <1%" as the only country (2026-11-22)
    assert shares("Pakistan >99%; Kashmir <1%") == {"Pakistan": 99.5, "Kashmir": 0.5}
    assert combine(["Pakistan >99%; Kashmir <1%"], [1]) == "Pakistan >99%; Kashmir <1%"
    assert combine(["Iran 60%; Iraq 40%", "Iran 100%"], [1, 1]) == "Iran 80%; Iraq 20%"


def test_adopt_strip_moves_the_edge_to_the_neighbours_own_edge():
    # the seam is x = 0; the neighbour (B) mapped 3 km past it before its own edge; this side said A up to the seam
    own = gpd.GeoDataFrame({"type": ["A"], "parcel": ["q"]}, geometry=[box(0, 0, 10000, 10000)], crs=3857)
    other = gpd.GeoDataFrame({"type": ["B"]}, geometry=[box(-10000, 0, 0, 10000)], crs=3857)
    seam = LineString([(0, 0), (0, 10000)])
    strip = gpd.GeoDataFrame({"type": ["B", "A"]}, geometry=[box(0, 0, 3000, 10000), box(3000, 0, 10000, 10000)], crs=3857)
    out = adopt_strip(own, strip, other, seam, "type")
    assert abs(out[out["type"] == "B"].area.sum() - 3e7) < 1e3 and set(out.parcel) == {"q"}
    # a patch reaching more than reach_km from the seam is a deep disagreement: left alone
    short = gpd.GeoDataFrame({"type": ["B"]}, geometry=[box(0, 0, 10000, 10000)], crs=3857)
    assert (adopt_strip(own, short, other, seam, "type", reach_km=5)["type"] == "A").all()


def test_naturalize_moves_an_edge_from_the_border_to_the_terrain(monkeypatch):
    import numpy as np
    from rasterio.transform import from_origin
    import ecoregions.world as w

    # border at x = 20 km; type A west of it, B east; the land steps up at x = 26 km
    def fake(bounds, res, dem_paths, url):
        x0, y0, x1, y1 = bounds
        wd, ht = int(np.ceil((x1 - x0) / res)), int(np.ceil((y1 - y0) / res))
        xs = x0 + (np.arange(wd) + 0.5) * res
        rng = np.random.default_rng(0)
        elev = np.where(xs < 26000, 200.0, 900.0)[None, :].repeat(ht, 0) + rng.normal(0, 20, (ht, wd))
        f = {k: rng.normal(0, 1, (ht, wd)).astype("float32") for k in ("slope", "relief", "tpi", "bio1", "bio12", "ai")}
        f["elev"] = elev.astype("float32")
        return dict(f=f, transform=from_origin(x0, y1, res, res), shape=(ht, wd))

    monkeypatch.setattr(w, "_window_features", fake)
    cty = gpd.GeoDataFrame({"NAME": ["W", "E"]}, geometry=[box(-40000, 0, 20000, 60000), box(20000, 0, 80000, 60000)], crs=3857)
    u = gpd.GeoDataFrame({"type": ["A", "B"]}, geometry=[box(-40000, 0, 20000, 60000), box(20000, 0, 80000, 60000)], crs=3857)
    runs = on_line(u, cty, "type", on_km=0.5, min_km=5)
    assert runs
    out, rows = w.naturalize(u, runs, [], "", band_km=8, train_km=30, res=500)
    a = out[out["type"] == "A"].union_all()
    assert 24000 < a.bounds[2] < 28500, a.bounds  # A now reaches the step in the land
    assert on_line(out, cty, "type", on_km=0.5, min_km=5) == []
