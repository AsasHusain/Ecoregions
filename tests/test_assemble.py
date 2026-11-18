import numpy as np

from ecoregions.assemble import belt_zones, block_share


def test_belt_patches_become_one_zone():
    # a belt type (5) as scattered 90 m patches covering about 40 % of the land: as a zone it takes the area
    rng = np.random.default_rng(0)
    cls90 = np.where(rng.random((90, 90)) < 0.4, 5, 1).astype("uint8")
    cls = np.ones((30, 30), "uint8")  # the 270 m mode is the surrounding vegetation (1) everywhere
    assert np.allclose(block_share(cls90, 5, 3).mean(), (cls90 == 5).mean(), atol=1e-6)
    out = belt_zones(cls, cls90, 3, [5], {"window_km": 2, "min_share": 0.3}, 270.0)
    assert (out == 5).mean() > 0.9
    assert (belt_zones(cls, cls90, 3, [5], {"window_km": 2, "min_share": 0.6}, 270.0) == 5).sum() == 0
    own = {"window_km": 2, "min_share": 0.6, "by_type": {5: {"min_share": 0.3}}}
    assert (belt_zones(cls, cls90, 3, [5], own, 270.0) == 5).mean() > 0.9


def test_natural_fill_rebuilds_converted_ground_from_its_surroundings():
    import numpy as np
    from ecoregions.features import natural_fill

    a = np.full((60, 60), 0.12, "float32")  # bare sand
    a[20:40, 20:40] = 0.6                   # an irrigated block: green, not bare
    conv = np.zeros_like(a, bool)
    conv[20:40, 20:40] = True
    a[0, 0] = np.nan
    out = natural_fill(a, conv)
    assert np.allclose(out[20:40, 20:40], 0.12, atol=1e-3)
    assert np.isnan(out[0, 0]) and out[50, 50] == np.float32(0.12)


def test_belts_compare_by_their_own_thresholds():
    # type 5 at 25 % (threshold 0.2) and type 6 at 35 % (threshold 0.3) in one area: 5 exceeds its bar more and wins
    rng = np.random.default_rng(1)
    r = rng.random((90, 90))
    cls90 = np.where(r < 0.25, 5, np.where(r < 0.6, 6, 1)).astype("uint8")
    cls = np.ones((30, 30), "uint8")
    bc = {"window_km": 2, "min_share": 0.3, "by_type": {5: {"min_share": 0.2}}}
    assert (belt_zones(cls, cls90, 3, [5, 6], bc, 270.0) == 5).mean() > 0.5  # by raw share, 6 would take nearly all


def test_a_winning_belt_holds_the_slopes_between_its_forests():
    rng = np.random.default_rng(1)
    r = rng.random((90, 90))
    cls90 = np.where(r < 0.22, 5, np.where(r < 0.8, 6, 1)).astype("uint8")  # juniper (6) far commoner than spruce (5)
    cls = np.ones((30, 30), "uint8")
    bc = {"window_km": 2, "min_share": 0.3, "by_type": {5: {"min_share": 0.2, "wins": True}}}
    assert (belt_zones(cls, cls90, 3, [5, 6], bc, 270.0) == 5).mean() > 0.6


def test_seam_blend_in_a_window_matches_the_whole_grid():
    from scipy import ndimage as ndi

    from ecoregions.features import blend

    rng = np.random.default_rng(2)
    a = ndi.gaussian_filter(rng.random((400, 300)), 3).astype("float32")
    a[:, 150:] += 0.5  # a step at a border
    a[rng.random(a.shape) < 0.02] = np.nan
    seam = np.zeros(a.shape, bool)
    seam[180:220, 150] = True
    res, band = 270.0, 3000.0
    w = np.clip(ndi.distance_transform_edt(~seam) * res / band, 0, 1).astype("float32")
    ok = np.isfinite(a)
    sig = band / 2 / res
    num = ndi.gaussian_filter(np.where(ok, a, 0).astype("float32"), sig)
    den = ndi.gaussian_filter(ok.astype("float32"), sig)
    smooth = np.where(den > 1e-3, num / np.maximum(den, 1e-3), a)
    full = np.where(ok, w * a + (1 - w) * smooth, np.nan).astype("float32")
    np.testing.assert_allclose(blend(a, seam, res, band), full, rtol=0, atol=1e-6)
