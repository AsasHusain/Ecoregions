import numpy as np

from ecoregions.biotic import cut, gdm_fit, gdm_predict, ramps


def test_cut_follows_the_merge_order():
    ch = np.array([[0, 1], [2, 3], [4, 5]])  # 4 leaves: (0,1) -> 4, (2,3) -> 5, then all
    two = cut(ch, 4, 2)
    assert two[0] == two[1] != two[2] == two[3]
    assert len(set(cut(ch, 4, 4))) == 4 and len(set(cut(ch, 4, 1))) == 1


def test_gdm_recovers_known_turnover():
    rng = np.random.default_rng(0)
    F = rng.random((2000, 3))
    b = np.array([0.1, 0.5, 0.0, 1.0])
    assert np.allclose(gdm_fit(F, gdm_predict(F, b)), b, atol=0.02)


def test_ramps_are_monotone_and_bounded():
    r = ramps(np.linspace(-1, 5, 50), np.array([0, 1, 2, 4.0]))
    assert r.shape == (50, 3) and (np.diff(r, axis=0) >= 0).all() and r.min() == 0 and r.max() == 1


def test_fill_gaps_uses_neighbours():
    import pandas as pd
    from ecoregions.biotic import fill_gaps

    df = pd.DataFrame({"row": [0, 0, 0, 5], "col": [0, 1, 2, 5]})
    E = pd.DataFrame({"t": [1.0, np.nan, 3.0, 100.0]})
    assert fill_gaps(E, df)["t"].tolist() == [1.0, 2.0, 3.0, 100.0]


def test_names_do_not_double_and_are_unique():
    import geopandas as gpd
    import pandas as pd
    from shapely.geometry import Point
    from ecoregions.biotic import disambiguate, unit_name

    assert unit_name("Karakum Desert", "Deserts") == "Karakum Desert"
    assert unit_name("Lower Indus Plain and Delta", "Deserts") == "Lower Indus Plain & Delta Deserts"
    names = pd.Series(["Zagros Forests"] * 4 + ["Alborz Forests"])
    c = gpd.GeoSeries([Point(0, 0), Point(10, 0), Point(-10, 0), Point(0, 10), Point(5, 5)])
    out = disambiguate(names, c)
    assert out.is_unique and out.iloc[4] == "Alborz Forests" and "Central Zagros Forests" in set(out)
