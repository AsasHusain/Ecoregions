"""Quick-look figures for eyeballing each stage (seams, nodata gaps, artifacts)."""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

log = logging.getLogger(__name__)


def quicklook(arr: np.ndarray, path: Path, title: str = "", cmap="viridis", vmin=None, vmax=None,
              max_px: int = 1600, categorical: bool = False, outline: np.ndarray | None = None) -> Path:
    step = max(1, int(np.ceil(max(arr.shape) / max_px)))
    a = arr[::step, ::step].astype("float32") if not categorical else arr[::step, ::step]
    fig, ax = plt.subplots(figsize=(10, 9), dpi=110)
    if categorical:
        a = np.ma.masked_equal(a, 0)
        im = ax.imshow(a, cmap=cmap, interpolation="nearest")
    else:
        if vmin is None:
            vmin, vmax = np.nanpercentile(a, [2, 98])
        im = ax.imshow(a, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
        fig.colorbar(im, ax=ax, shrink=0.6)
    if outline is not None:
        o = outline[::step, ::step]
        ax.contour(o.astype("float32"), levels=[0.5], colors="k", linewidths=0.6)
    ax.set_title(title)
    ax.set_axis_off()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    log.info("QA figure %s", path)
    return path


def zoom(arr: np.ndarray, grid, lonlat_bbox, path: Path, title="", cmap="viridis", vmin=None,
         vmax=None, categorical=False, overlay=None):
    """Full-resolution crop around a lon/lat bbox (for checking fine detail)."""
    from pyproj import Transformer

    t = Transformer.from_crs("EPSG:4326", grid.crs, always_xy=True)
    w, s, e, n = lonlat_bbox
    xs, ys = t.transform([w, e, w, e], [s, s, n, n])
    inv = ~grid.transform
    cols, rows = zip(*[inv * (x, y) for x, y in zip(xs, ys)])
    r0, r1 = int(max(0, min(rows))), int(min(arr.shape[0], max(rows)))
    c0, c1 = int(max(0, min(cols))), int(min(arr.shape[1], max(cols)))
    sub = arr[r0:r1, c0:c1]
    if sub.size == 0 or (sub.dtype.kind == "f" and np.isnan(sub).all()):  # box outside this grid (e.g. a parcel)
        log.info("QA zoom %s: box outside the grid, skipped", path.name)
        return None
    o = overlay[r0:r1, c0:c1] if overlay is not None else None
    return quicklook(sub, path, title, cmap, vmin, vmax, max_px=2400, categorical=categorical, outline=o)


def label_map(lab: np.ndarray, path: Path, title: str = "", shade: np.ndarray | None = None,
              max_px: int = 2400, seed: int = 0) -> Path:
    """Units in random colours (with edges) over an optional hillshade."""
    step = max(1, int(np.ceil(max(lab.shape) / max_px)))
    l = lab[::step, ::step]
    rng = np.random.default_rng(seed)
    pal = rng.uniform(0.25, 0.95, size=(int(l.max()) + 1, 3))
    rgb = pal[l]
    edge = np.zeros(l.shape, bool)
    edge[:, 1:] |= l[:, 1:] != l[:, :-1]
    edge[1:, :] |= l[1:, :] != l[:-1, :]
    if shade is not None:
        s = shade[::step, ::step][: l.shape[0], : l.shape[1]]
        rgb = rgb * (0.55 + 0.45 * np.nan_to_num(s, nan=1.0))[..., None]
    rgb[edge] = 0.05
    rgb[l == 0] = 1.0
    fig, ax = plt.subplots(figsize=(13, 12), dpi=150)
    ax.imshow(rgb, interpolation="nearest")
    ax.set_title(title)
    ax.set_axis_off()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    log.info("QA figure %s", path)
    return path
