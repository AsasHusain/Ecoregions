import geopandas as gpd
from shapely.geometry import LineString, box

from ecoregions.world import combine, natural_seam, shares


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


def test_shares_read_the_area_layers_bounds():
    # the area layers write "Pakistan >99%"; dropping it left "Kashmir <1%" as the only country (2026-11-22)
    assert shares("Pakistan >99%; Kashmir <1%") == {"Pakistan": 99.5, "Kashmir": 0.5}
    assert combine(["Pakistan >99%; Kashmir <1%"], [1]) == "Pakistan >99%; Kashmir <1%"
    assert combine(["Iran 60%; Iraq 40%", "Iran 100%"], [1, 1]) == "Iran 80%; Iraq 20%"
