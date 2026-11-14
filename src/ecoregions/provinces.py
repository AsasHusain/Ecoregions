"""Stage 7 — L2 physiographic provinces.

L2 is *expert-seeded, data-bounded*: each province is anchored by a few seed
points in ``config.yaml`` (the expert input), and the boundaries between
provinces are found by the data. For every province we compute the geodesic
cost from its seeds over an 8-connected 810 m grid where stepping between two
cells costs ``length * (1 + alpha * |f_i - f_j|)``, with ``f`` the weighted,
standardized province features. Each cell joins the cheapest province, so
boundaries settle where the landscape changes most abruptly between seeds
(mountain fronts, plateau rims, desert margins) rather than at fixed distances.

The ratio between the best and second-best cost is kept as a per-cell
assignment margin and feeds the L3 confidence score.
"""
from __future__ import annotations

import json
import logging

import numpy as np
from numba import njit
from scipy import ndimage as ndi
from pyproj import Transformer

from .config import Paths
from .covariates import litho_groups
from . import vectors
from .grid import Grid, rasterize_geoms, read, write

log = logging.getLogger(__name__)


@njit(cache=True)
def geodesic_cost(feat, valid, seeds_r, seeds_c, res, alpha, barrier):
    """Multi-source Dijkstra on an 8-connected grid with feature-difference costs;
    ``barrier`` (km-equivalents) is added when stepping into a cell."""
    H, W, K = feat.shape
    INF = np.float32(3e38)
    dist = np.full((H, W), INF, np.float32)
    done = np.zeros((H, W), np.uint8)
    cap = 1 << 20
    hk = np.empty(cap, np.float32)
    hv = np.empty(cap, np.int64)
    hs = 0
    dr = (0, 1, 1, 1, 0, -1, -1, -1)
    dc = (1, 1, 0, -1, -1, -1, 0, 1)
    for s in range(seeds_r.size):
        r, c = seeds_r[s], seeds_c[s]
        dist[r, c] = 0.0
        if hs == cap:
            cap *= 2
            a = np.empty(cap, np.float32); a[:hs] = hk[:hs]; hk = a
            b = np.empty(cap, np.int64); b[:hs] = hv[:hs]; hv = b
        j = hs
        hk[j] = 0.0
        hv[j] = r * W + c
        hs += 1
    while hs > 0:
        # pop
        d0 = hk[0]
        v = hv[0]
        hs -= 1
        if hs > 0:
            hk[0] = hk[hs]; hv[0] = hv[hs]
            j = 0
            while True:
                l = 2 * j + 1
                if l >= hs:
                    break
                m = l
                if l + 1 < hs and hk[l + 1] < hk[l]:
                    m = l + 1
                if hk[m] < hk[j]:
                    hk[m], hk[j] = hk[j], hk[m]
                    hv[m], hv[j] = hv[j], hv[m]
                    j = m
                else:
                    break
        r = v // W
        c = v - r * W
        if done[r, c]:
            continue
        done[r, c] = 1
        for k in range(8):
            rn = r + dr[k]
            cn = c + dc[k]
            if rn < 0 or cn < 0 or rn >= H or cn >= W or not valid[rn, cn] or done[rn, cn]:
                continue
            diff = 0.0
            for q in range(K):
                t = feat[r, c, q] - feat[rn, cn, q]
                diff += t * t
            step = res * (1.4142135 if (dr[k] != 0 and dc[k] != 0) else 1.0)
            nd = d0 + step * (1.0 + alpha * np.sqrt(diff)) + barrier[rn, cn]
            if nd < dist[rn, cn]:
                dist[rn, cn] = nd
                if hs == cap:
                    cap *= 2
                    a = np.empty(cap, np.float32); a[:hs] = hk[:hs]; hk = a
                    b = np.empty(cap, np.int64); b[:hs] = hv[:hs]; hv = b
                j = hs
                hk[j] = nd
                hv[j] = rn * W + cn
                hs += 1
                while j > 0:
                    p = (j - 1) >> 1
                    if hk[p] > hk[j]:
                        hk[p], hk[j] = hk[j], hk[p]
                        hv[p], hv[j] = hv[j], hv[p]
                        j = p
                    else:
                        break
    return dist


def feature_matrix(paths: Paths, weights: dict, level: str, valid: np.ndarray, smooth: float = 0.0) -> np.ndarray:
    """Standardize (over valid cells), weight and stack features -> (H, W, K).

    ``smooth`` (cells) applies a Gaussian filter first, so that fine *texture*
    (dune fields, dissected badlands) is cheap to traverse while broad *fronts*
    (mountain fronts, desert margins) stay expensive.
    """
    from scipy import ndimage as ndi

    d = paths.interim / level
    cols = []
    for name, w in weights.items():
        if w == 0:
            continue
        if name == "litho":
            # one-hot fractions; a full change of group contributes distance w
            for k in litho_groups(paths.cfg):
                a = read(d / f"litho_{k}.tif")
                cols.append(np.nan_to_num(a, nan=0.0) * (w / np.sqrt(2)))
            continue
        a = read(d / f"{name}.tif")
        v = a[valid]
        mu, sd = np.nanmean(v), np.nanstd(v)
        z = (a - mu) / (sd if sd > 0 else 1.0)
        z = np.where(np.isnan(z), 0.0, z)
        cols.append(z * w)
    X = np.stack(cols, axis=-1).astype("float32")
    if smooth:
        for k in range(X.shape[-1]):
            X[..., k] = ndi.gaussian_filter(X[..., k], smooth)
    return X


def seed_cells(units, grid: Grid, valid: np.ndarray, drop_outside: bool = False):
    t = Transformer.from_crs("EPSG:4326", grid.crs, always_xy=True)
    inv = ~grid.transform
    out = []
    for u in units:
        rs, cs = [], []
        for lon, lat in u["seeds"]:
            x, y = t.transform(lon, lat)
            c, r = inv * (x, y)
            r, c = int(r), int(c)
            if not (0 <= r < grid.height and 0 <= c < grid.width) or not valid[r, c]:
                # snap to the nearest valid cell within ~12 km (border/coast slop); fail beyond that
                rr, cc = np.nonzero(valid[max(0, r - 15):r + 16, max(0, c - 15):c + 16])
                if rr.size == 0:
                    if drop_outside:  # a parcel uses the provinces of a larger area: seeds beyond it are skipped
                        continue
                    raise ValueError(f"seed {lon},{lat} of province {u['code']} {u['name']} is outside the study area")
                k = np.argmin((rr + max(0, r - 15) - r) ** 2 + (cc + max(0, c - 15) - c) ** 2)
                r, c = int(rr[k] + max(0, r - 15)), int(cc[k] + max(0, c - 15))
                log.warning("seed %s,%s of province %s snapped into the study area", lon, lat, u["code"])
            rs.append(r)
            cs.append(c)
        out.append((np.array(rs, np.int64), np.array(cs, np.int64)))
    return out


def run(cfg) -> None:
    paths = Paths(cfg)
    pc = cfg["provinces"]
    grid90 = Grid.load(paths.grid_json)
    cg = grid90.coarsen(cfg["grid"]["coarse_factor"])
    pg = cg.coarsen(cfg["grid"]["province_factor"])
    valid = read(paths.interim / "province" / "study_frac.tif") >= 0.5
    feat = feature_matrix(paths, pc["features"], "province", valid, pc.get("smooth_cells", 0))
    units = pc["units"]
    seeds = seed_cells(units, pg, valid, pc.get("drop_outside_seeds", False))
    units, seeds = [u for u, s in zip(units, seeds) if s[0].size], [s for s in seeds if s[0].size]  # parcels: provinces with no seed inside drop out
    rb = pc.get("river_barrier")
    barrier = np.zeros(pg.shape, "float32")
    if rb:
        riv = vectors.hydrorivers(cfg)
        trunk = riv[riv.UPLAND_SKM >= rb["min_upland_km2"]]
        line = rasterize_geoms([(g, 1) for g in trunk.geometry], pg, all_touched=True).astype(bool)
        line = ndi.binary_dilation(line)  # no diagonal leaks between rasterized channel cells
        barrier = line.astype("float32") * rb["cost_km"]
    dists = np.stack([geodesic_cost(feat, valid, r, c, np.float32(pg.res / 1000), np.float32(pc["alpha"]), barrier)
                      for r, c in seeds])
    order = np.argsort(dists, axis=0)
    best = order[0]
    d1 = np.take_along_axis(dists, order[:1], 0)[0]
    d2 = np.take_along_axis(dists, order[1:2], 0)[0]
    lab = np.where(valid & (d1 < 1e30), best + 1, 0).astype("uint8")
    with np.errstate(divide="ignore", invalid="ignore"):
        margin = np.where(lab > 0, (d2 - d1) / np.maximum(d1 + d2, 1e-6), np.nan).astype("float32")
    write(paths.interim / "province" / "l2.tif", lab, pg, nodata=0,
          tags={str(i + 1): u["code"] for i, u in enumerate(units)})
    write(paths.interim / "province" / "l2_margin.tif", margin, pg, nodata=np.nan)
    (paths.interim / "province" / "l2_codes.json").write_text(
        json.dumps({i + 1: dict(code=u["code"], name=u["name"], family=u["family"]) for i, u in enumerate(units)}, indent=1))
    # 270 m version for segmentation (exact nesting: each province cell = 3x3 coarse cells)
    f = cfg["grid"]["province_factor"]
    lab_c = np.repeat(np.repeat(lab, f, 0), f, 1)[: cg.height, : cg.width]
    write(paths.coarse("l2"), lab_c, cg, nodata=0)
    for i, u in enumerate(units):
        log.info("L2 %s %-70s %8.0f km²", u["code"], u["name"], (lab == i + 1).sum() * (pg.res / 1000) ** 2)
    _plot(lab, seeds, units, paths)


def _plot(lab, seeds, units, paths):
    import matplotlib.pyplot as plt

    rel = read(paths.interim / "province" / "relief5k.tif")
    edges = np.zeros(lab.shape, bool)
    edges[:, 1:] |= lab[:, 1:] != lab[:, :-1]
    edges[1:, :] |= lab[1:, :] != lab[:-1, :]
    fig, axs = plt.subplots(1, 2, figsize=(20, 10), dpi=110)
    axs[0].imshow(np.ma.masked_equal(lab, 0), cmap="tab20", interpolation="nearest")
    axs[1].imshow(np.log10(np.clip(rel, 5, None)), cmap="gray_r", interpolation="nearest")
    axs[1].imshow(np.ma.masked_where(~edges | (lab == 0), edges), cmap="autumn", interpolation="nearest")
    for ax in axs:
        for (r, c), u in zip(seeds, units):
            ax.scatter(c, r, s=10, c="k")
            ax.annotate(u["code"], (c[0], r[0]), fontsize=8, weight="bold", color="b")
        ax.set_axis_off()
    axs[0].set_title("L2 provinces and seeds")
    axs[1].set_title("L2 boundaries over log local relief")
    fig.savefig(paths.qa / "07_l2_provinces.png", bbox_inches="tight")
    plt.close(fig)
