"""Analysis grid definition and raster I/O helpers.

All derived rasters share one grid (Albers equal-area, 90 m) defined by the
``grid`` stage and stored in ``data/interim/grid.json``. A coarse grid for
segmentation is an exact integer aggregation of it, so 90 m and coarse pixels
nest perfectly.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import rasterio
from affine import Affine
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.warp import reproject
from rasterio.windows import Window

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Grid:
    crs: str
    x0: float
    y0: float  # top edge
    res: float
    width: int
    height: int

    @property
    def transform(self) -> Affine:
        return Affine(self.res, 0, self.x0, 0, -self.res, self.y0)

    @property
    def shape(self) -> tuple[int, int]:
        return self.height, self.width

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        return self.x0, self.y0 - self.height * self.res, self.x0 + self.width * self.res, self.y0

    @property
    def rio_crs(self) -> CRS:
        return CRS.from_string(self.crs)

    @property
    def cell_km2(self) -> float:
        return (self.res / 1000) ** 2

    def coarsen(self, factor: int) -> "Grid":
        return Grid(self.crs, self.x0, self.y0, self.res * factor,
                    int(np.ceil(self.width / factor)), int(np.ceil(self.height / factor)))

    def at_resolution(self, res: float) -> "Grid":
        w = int(np.ceil(self.width * self.res / res))
        h = int(np.ceil(self.height * self.res / res))
        return Grid(self.crs, self.x0, self.y0, res, w, h)

    def profile(self, dtype="float32", nodata=None, **kw) -> dict:
        p = dict(driver="GTiff", dtype=dtype, count=1, width=self.width, height=self.height,
                 crs=self.rio_crs, transform=self.transform, nodata=nodata, compress="deflate",
                 tiled=True, blockxsize=512, blockysize=512, BIGTIFF="YES", num_threads="ALL_CPUS")
        p["predictor"] = 3 if np.dtype(dtype).kind == "f" else 2
        p.update(kw)
        return p

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.__dict__, indent=1))

    @classmethod
    def load(cls, path: Path) -> "Grid":
        return cls(**json.loads(Path(path).read_text()))

    @classmethod
    def from_bounds(cls, crs: str, bounds, res: float) -> "Grid":
        xmin, ymin, xmax, ymax = bounds
        x0 = np.floor(xmin / res) * res
        y0 = np.ceil(ymax / res) * res
        w = int(np.ceil((xmax - x0) / res))
        h = int(np.ceil((y0 - ymin) / res))
        return cls(crs, float(x0), float(y0), float(res), w, h)

    def tiles(self, size: int, overlap: int) -> Iterator[tuple[Window, Window, Window]]:
        """Yield (read_window, core_window, core_in_read) for overlapping tiles."""
        for r0 in range(0, self.height, size):
            for c0 in range(0, self.width, size):
                core = Window(c0, r0, min(size, self.width - c0), min(size, self.height - r0))
                rr0, cc0 = max(0, r0 - overlap), max(0, c0 - overlap)
                rr1 = min(self.height, r0 + core.height + overlap)
                cc1 = min(self.width, c0 + core.width + overlap)
                read = Window(cc0, rr0, cc1 - cc0, rr1 - rr0)
                inner = Window(c0 - cc0, r0 - rr0, core.width, core.height)
                yield read, core, inner


def write(path: Path, arr: np.ndarray, grid: Grid, nodata=None, tags: dict | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".part.tif")
    with rasterio.open(tmp, "w", **grid.profile(arr.dtype, nodata)) as dst:
        dst.write(arr, 1)
        if tags:
            dst.update_tags(**{str(k): str(v) for k, v in tags.items()})
    tmp.replace(path)  # replace, not rename: rename fails on Windows if path exists
    return path


def is_current(path: Path, grid: Grid) -> bool:
    """True if ``path`` exists *and* lies on ``grid``: resumable stages may reuse it.
    (A bare existence check silently kept rasters from an older grid.)"""
    if not path.exists():
        return False
    with rasterio.open(path) as s:
        return s.shape == grid.shape and s.transform.almost_equals(grid.transform)


def read(path: Path, window: Window | None = None, masked: bool = False) -> np.ndarray:
    with rasterio.open(path) as src:
        return src.read(1, window=window, masked=masked)


def reproject_array(src_arr, src_transform, src_crs, grid: Grid, resampling=Resampling.bilinear,
                    src_nodata=None, dst_nodata=np.nan, dtype="float32") -> np.ndarray:
    dst = np.full(grid.shape, dst_nodata, dtype=dtype)
    reproject(src_arr, dst, src_transform=src_transform, src_crs=src_crs, src_nodata=src_nodata,
              dst_transform=grid.transform, dst_crs=grid.rio_crs, dst_nodata=dst_nodata,
              resampling=resampling, num_threads=4)
    return dst


def reproject_file(path: Path, grid: Grid, resampling=Resampling.bilinear, dtype="float32",
                   dst_nodata=np.nan, apply_scale: bool = True) -> np.ndarray:
    """Warp a raster file onto ``grid``; applies GDAL scale/offset if present."""
    with rasterio.open(path) as src:
        nod = src.nodata
        dst = np.full(grid.shape, dst_nodata, dtype=dtype)
        reproject(rasterio.band(src, 1), dst, src_nodata=nod, dst_transform=grid.transform,
                  dst_crs=grid.rio_crs, dst_nodata=dst_nodata, resampling=resampling, num_threads=4)
        scale, offset = src.scales[0], src.offsets[0]
    if apply_scale and (scale != 1 or offset != 0):
        valid = ~np.isnan(dst) if np.dtype(dtype).kind == "f" else slice(None)
        dst[valid] = dst[valid] * scale + offset
    return dst


def rasterize_geoms(shapes, grid: Grid, fill=0, dtype="uint8", all_touched=False) -> np.ndarray:
    return rasterize(shapes, out_shape=grid.shape, transform=grid.transform, fill=fill,
                     dtype=dtype, all_touched=all_touched)


def block_reduce(arr: np.ndarray, factor: int, func=np.nanmean) -> np.ndarray:
    """Aggregate a 2-D array by an integer factor (pads with NaN)."""
    h, w = arr.shape
    H, W = int(np.ceil(h / factor)), int(np.ceil(w / factor))
    pad = np.full((H * factor, W * factor), np.nan, dtype="float32")
    pad[:h, :w] = arr
    blocks = pad.reshape(H, factor, W, factor)
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return func(blocks, axis=(1, 3)).astype("float32")


def block_mode(arr: np.ndarray, factor: int, nodata: int = 0) -> np.ndarray:
    """Majority of an integer array within factor x factor blocks (ignores nodata)."""
    h, w = arr.shape
    H, W = int(np.ceil(h / factor)), int(np.ceil(w / factor))
    pad = np.full((H * factor, W * factor), nodata, dtype=arr.dtype)
    pad[:h, :w] = arr
    blocks = pad.reshape(H, factor, W, factor).transpose(0, 2, 1, 3).reshape(H, W, factor * factor)
    out = np.full((H, W), nodata, dtype=arr.dtype)
    vals = np.unique(arr)
    best = np.zeros((H, W), dtype="int32")
    for v in vals:
        if v == nodata:
            continue
        c = (blocks == v).sum(axis=2)
        upd = c > best
        out[upd] = v
        best[upd] = c[upd]
    return out


def centre_sample(a: np.ndarray, factor: int, shape: tuple[int, int], fill=0) -> np.ndarray:
    """Centre pixel of every factor x factor block, padded to the coarse ``shape``
    (the fine grid need not be an exact multiple of the factor)."""
    s = a[factor // 2::factor, factor // 2::factor]
    out = np.full(shape, fill, dtype=a.dtype)
    h, w = min(shape[0], s.shape[0]), min(shape[1], s.shape[1])
    out[:h, :w] = s[:h, :w]
    return out
