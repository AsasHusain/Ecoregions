import numpy as np

from ecoregions.hydro import (accumulate, classify_hand, downstream_index, edges_of_labels, hand_flowpath,
                              priority_flood_d8, strahler_and_links, topological_order)


def _walled(z, wall=1000.0, outlet=(None, 0)):
    """Surround a DEM with high walls except a single low outlet cell."""
    z = z.astype("float32").copy()
    z[0, :] = z[-1, :] = wall
    z[:, 0] = z[:, -1] = wall
    r = z.shape[0] // 2 if outlet[0] is None else outlet[0]
    z[r, outlet[1]] = -1.0
    return z


def _route(z, valid=None):
    valid = np.ones(z.shape, bool) if valid is None else valid
    dirs = priority_flood_d8(z.astype("float32"), valid)
    order = topological_order(dirs, z.shape[1])
    return dirs, order


def test_tilted_plane_drains_to_low_edge():
    H, W = 20, 30
    z = _walled(np.tile(np.arange(W, dtype="float32"), (H, 1)))  # rises eastward, outlet mid-west
    dirs, order = _route(z)
    acc = accumulate(dirs, order, W).reshape(H, W)
    assert order.size == H * W
    # every interior cell drains toward the west (E->W components only)
    inner = dirs.reshape(H, W)[1:-1, 1:-1]
    assert not np.isin(inner, [0, 1, 7]).any()  # nothing flows uphill (eastward)
    # the single outlet collects all interior cells
    assert acc[H // 2, 0] >= (H - 2) * (W - 2)


def test_depression_is_filled_and_routed():
    z = np.full((15, 15), 10.0, "float32")
    z[7, 7] = 0.0  # pit
    z[:, 0] = 5.0  # low western edge
    dirs, order = _route(z)
    assert order.size == 225  # every cell reached (no cycles, no orphans)
    W = 15
    i = 7 * W + 7
    # follow the pit downstream: must reach an outlet without looping
    seen = 0
    while dirs[i] < 8:
        i = downstream_index(i, dirs[i], W)
        seen += 1
        assert seen < 100
    assert dirs[i] == 254


def test_strahler_two_first_order_make_second():
    # V-shaped valley draining south with two tributaries
    H, W = 40, 41
    yy, xx = np.mgrid[0:H, 0:W]
    z = _walled((np.abs(xx - 20) * 2.0 + (H - yy) * 0.5) + np.abs(xx - 10) % 20 * 0.3, outlet=(H - 1, 20))
    dirs, order = _route(z)
    acc = accumulate(dirs, order, W)
    stream = acc >= 15
    so, link = strahler_and_links(dirs, order, stream, W)
    assert so.max() >= 2
    assert link.max() >= 1


def test_hand_zero_on_channel_positive_on_slopes():
    H, W = 30, 31
    yy, xx = np.mgrid[0:H, 0:W]
    z = _walled(np.abs(xx - 15) * 1.0 + (H - yy) * 0.1, outlet=(H - 1, 15))
    dirs, order = _route(z)
    acc = accumulate(dirs, order, W)
    stream = acc >= 20
    hand = hand_flowpath(dirs, order, stream, z.ravel(), W).reshape(H, W)
    assert np.allclose(hand.ravel()[stream], 0)
    assert hand[15, 2] > 10


def test_classify_and_edges():
    h = np.array([[0.5, 4.0], [10.0, 50.0]])
    cls = classify_hand(h, {"active_floodplain": [-50, 3], "meander_floodplain": [3, 7],
                            "bar_upland": [7, 35], "upland": [35, 1e5]})
    assert cls.tolist() == [[1, 2], [3, 4]]
    lab = np.array([[1, 1, 2], [1, 1, 2]])
    assert edges_of_labels(lab).sum() == 2


def test_harmonic_relax_is_smooth_and_bounded():
    from ecoregions.hydro import harmonic_relax

    u = np.zeros((30, 30))
    fixed = np.zeros((30, 30), bool)
    u[:, 0], fixed[:, 0] = 10.0, True      # "river" at 10 m on the west
    u[:, -1], fixed[:, -1] = 0.0, True     # and at 0 m on the east
    out = harmonic_relax(u.copy(), fixed, 3000, 1.9)
    row = out[15]
    assert np.all(np.diff(row) < 0)                  # monotone between the rivers
    assert np.allclose(row, np.linspace(10, 0, 30), atol=0.05)  # linear solution of Laplace in 1-D
