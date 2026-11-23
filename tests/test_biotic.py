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
