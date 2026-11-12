"""Stage 3 — terrain derivatives on the 90 m grid.

Outputs (all on the analysis grid):

* ``slope``        slope in degrees (Horn 3x3)
* ``tri``          terrain ruggedness index (Riley et al. 1999), metres
* ``tpi_<r>``      topographic position index at several radii (square windows)
* ``relief_<w>``   local relief (max - min) in square windows
* ``geomorphons``  Jasiewicz & Stepinski (2013) via WhiteboxTools (10 classes)
* ``hammond``      Hammond (1964) landform classes after Karagulle et al. (2017)

Sea cells are treated as elevation 0 inside moving windows so coastal TPI and
relief are defined; they are masked again on output.
"""
from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

from .config import Paths
from .grid import Grid, is_current, read, write
from .qa import quicklook, zoom

log = logging.getLogger(__name__)

GEOMORPHON_CLASSES = {1: "flat", 2: "peak", 3: "ridge", 4: "shoulder", 5: "spur", 6: "slope",
                      7: "hollow", 8: "footslope", 9: "valley", 10: "pit"}
HAMMOND_CLASSES = {1: "flat plains", 2: "smooth or irregular plains", 3: "tablelands",
                   4: "plains with hills", 5: "plains with mountains", 6: "open hills",
                   7: "open mountains", 8: "hills", 9: "low mountains", 10: "high mountains"}


def _strips(h: int, size: int = 2048):
    for r0 in range(0, h, size):
        yield r0, min(h, r0 + size)


def slope_deg(dem: np.ndarray, res: float) -> np.ndarray:
    """Horn (1981) slope in degrees, computed in strips to bound memory."""
    out = np.empty(dem.shape, dtype="float32")
    H = dem.shape[0]
    for r0, r1 in _strips(H):
        a0, a1 = max(0, r0 - 1), min(H, r1 + 1)
        z = np.pad(dem[a0:a1], ((1 if r0 == 0 else 0, 1 if r1 == H else 0), (1, 1)), mode="edge")
        z = z.astype("float32")
        dzdx = ((z[:-2, 2:] + 2 * z[1:-1, 2:] + z[2:, 2:]) - (z[:-2, :-2] + 2 * z[1:-1, :-2] + z[2:, :-2])) / (8 * res)
        dzdy = ((z[2:, :-2] + 2 * z[2:, 1:-1] + z[2:, 2:]) - (z[:-2, :-2] + 2 * z[:-2, 1:-1] + z[:-2, 2:])) / (8 * res)
        out[r0:r1] = np.degrees(np.arctan(np.hypot(dzdx, dzdy)))
    return out


def tri_riley(dem: np.ndarray) -> np.ndarray:
    out = np.empty(dem.shape, dtype="float32")
    H = dem.shape[0]
    for r0, r1 in _strips(H):
        a0, a1 = max(0, r0 - 1), min(H, r1 + 1)
        z = np.pad(dem[a0:a1], ((1 if r0 == 0 else 0, 1 if r1 == H else 0), (1, 1)), mode="edge")
        c = z[1:-1, 1:-1]
        acc = np.zeros(c.shape, dtype="float32")
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == dc == 0:
                    continue
                n = z[1 + dr:z.shape[0] - 1 + dr, 1 + dc:z.shape[1] - 1 + dc]
                acc += (n - c) ** 2
        out[r0:r1] = np.sqrt(acc)
    return out


def box_mean(a: np.ndarray, size: int) -> np.ndarray:
    return ndi.uniform_filter(a, size=size, output=np.float32, mode="nearest")


def relief(a: np.ndarray, size: int) -> np.ndarray:
    mx = ndi.maximum_filter(a, size=size, mode="nearest")
    mx -= ndi.minimum_filter(a, size=size, mode="nearest")
    return mx


def hammond(dem: np.ndarray, slope: np.ndarray, size: int, gentle_pct: float) -> np.ndarray:
    """Hammond landform classes (see HAMMOND_CLASSES)."""
    gentle = (np.tan(np.radians(slope)) * 100 < gentle_pct).astype("float32")
    frac_gentle = box_mean(gentle, size)
    mx = ndi.maximum_filter(dem, size=size, mode="nearest")
    mn = ndi.minimum_filter(dem, size=size, mode="nearest")
    rel = mx - mn
    mid = (mx + mn) / 2
    del mx, mn
    gentle_low = box_mean(gentle * (dem < mid), size)
    del mid
    with np.errstate(invalid="ignore", divide="ignore"):
        low_share = np.where(frac_gentle > 0, gentle_low / frac_gentle, 0.5)
    del gentle_low, gentle
    A = np.select([frac_gentle > 0.8, frac_gentle > 0.5, frac_gentle > 0.2], [1, 2, 3], 4).astype("uint8")
    del frac_gentle
    B = np.digitize(rel, [30, 91, 152, 305, 914, 1524]).astype("uint8") + 1  # 1..7
    del rel
    upland = low_share < 0.5  # gentle land mostly in the upper half -> tableland profile
    out = np.zeros(dem.shape, dtype="uint8")
    plainish = A <= 2
    out[plainish & (B == 1)] = 1
    out[plainish & (B == 2)] = 2
    out[plainish & (B >= 3) & upland] = 3
    out[plainish & (B >= 3) & (B <= 4) & ~upland] = 4
    out[plainish & (B >= 5) & ~upland] = 5
    out[(A == 3) & (B <= 4)] = 6
    out[(A == 3) & (B >= 5)] = 7
    out[(A == 4) & (B <= 4)] = 8
    out[(A == 4) & (B == 5)] = 9
    out[(A == 4) & (B >= 6)] = 10
    return out


WBT_URLS = {  # official WhiteboxTools builds by platform
    "linux": "https://www.whiteboxgeo.com/WBT_Linux/WhiteboxTools_linux_amd64.zip",
    "win32": "https://www.whiteboxgeo.com/WBT_Windows/WhiteboxTools_win_amd64.zip",
    "darwin": "https://www.whiteboxgeo.com/WBT_Darwin/WhiteboxTools_darwin_amd64.zip",
}


def _wbt_binary(paths: Paths) -> Path:
    """Locate the WhiteboxTools binary, installing it into ``./whitebox_tools`` if needed.

    Order: $WBT_PATH, ``whitebox_tools`` on $PATH, the project-local copy, then a
    fresh download of the official Linux build from whiteboxgeo.com.
    """
    import os
    import stat
    import zipfile

    if os.environ.get("WBT_PATH"):
        return Path(os.environ["WBT_PATH"])
    import sys

    name = "whitebox_tools.exe" if sys.platform == "win32" else "whitebox_tools"
    exe = shutil.which(name)
    if exe:
        return Path(exe)
    local = paths.root / "whitebox_tools"
    hits = [h for h in local.rglob(name) if h.is_file()] if local.exists() else []
    if not hits:
        from .download import http_download

        url = WBT_URLS.get(sys.platform, WBT_URLS["linux"])
        z = http_download(url, local / Path(url).name)
        with zipfile.ZipFile(z) as zf:
            zf.extractall(local)
        hits = [h for h in local.rglob(name) if h.is_file()]
    exe = hits[0]
    exe.chmod(exe.stat().st_mode | stat.S_IXUSR)
    return exe


def geomorphons(dem_path: Path, out_path: Path, paths: Paths, search: int, skip: int, flat: float,
                threads: int = 4) -> None:
    exe = _wbt_binary(paths)
    import rasterio

    with tempfile.TemporaryDirectory(dir=paths.interim) as td:
        tmp_in = Path(td) / "dem_plain.tif"
        tmp_out = Path(td) / "geomorph.tif"
        with rasterio.open(dem_path) as src:  # WhiteboxTools wants an uncompressed GeoTIFF
            prof = src.profile.copy()
            prof.update(compress=None, predictor=None, tiled=False, nodata=-32768.0, BIGTIFF="YES")
            prof.pop("blockxsize", None), prof.pop("blockysize", None)
            data = src.read(1)
            data[np.isnan(data)] = -32768.0
            with rasterio.open(tmp_in, "w", **prof) as dst:
                dst.write(data, 1)
            del data
        cmd = [str(exe), "--run=Geomorphons", f"--dem={tmp_in}", f"--output={tmp_out}",
               f"--search={search}", f"--threshold={flat}", "--fdist=0", f"--skip={skip}",
               "--forms", f"--max_procs={threads}"]
        log.info("WhiteboxTools: %s", " ".join(cmd[1:3]))
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
        with rasterio.open(tmp_out) as src:
            g = src.read(1)
        g = np.where((g >= 1) & (g <= 10), g, 0).astype("uint8")
        grid = Grid.load(paths.grid_json)
        write(out_path, g, grid, nodata=0, tags={str(k): v for k, v in GEOMORPHON_CLASSES.items()})


def run(cfg) -> None:
    paths = Paths(cfg)
    grid = Grid.load(paths.grid_json)
    tc = cfg["terrain"]
    dem = read(paths.raster("dem"))
    sea = np.isnan(dem)
    dem[sea] = 0.0
    res = grid.res

    done = lambda name: is_current(paths.raster(name), grid)  # noqa: E731  (resume an interrupted run)

    if done("slope"):
        s = read(paths.raster("slope"))
    else:
        s = slope_deg(dem, res)
        s[sea] = np.nan
        write(paths.raster("slope"), s, grid, nodata=np.nan)
        quicklook(s, paths.qa / "03_slope.png", "slope (deg)", cmap="magma", vmin=0, vmax=35)

    if not done("tri"):
        t = tri_riley(dem)
        t[sea] = np.nan
        write(paths.raster("tri"), t, grid, nodata=np.nan)
        del t

    for r in tc["tpi_radii_m"]:
        if done(f"tpi_{r}"):
            continue
        size = 2 * int(round(r / res)) + 1
        tpi = dem - box_mean(dem, size)
        tpi[sea] = np.nan
        write(paths.raster(f"tpi_{r}"), tpi, grid, nodata=np.nan, tags=dict(window_px=size))
        quicklook(tpi, paths.qa / f"03_tpi_{r}.png", f"TPI {r} m", cmap="RdBu_r")
        del tpi

    for w in tc["relief_windows_m"]:
        if done(f"relief_{w}"):
            continue
        size = int(round(w / res)) | 1
        rel = relief(dem, size)
        rel[sea] = np.nan
        write(paths.raster(f"relief_{w}"), rel, grid, nodata=np.nan, tags=dict(window_px=size))
        quicklook(rel, paths.qa / f"03_relief_{w}.png", f"local relief {w} m window", cmap="cividis",
                  vmin=0, vmax=np.nanpercentile(rel[::7, ::7], 98))
        del rel

    hc = tc["hammond"]
    size = int(round(hc["window_m"] / res)) | 1
    if not done("hammond"):
        hm = hammond(dem, s, size, hc["gentle_slope_pct"])
        hm[sea] = 0
        write(paths.raster("hammond"), hm, grid, nodata=0, tags=HAMMOND_CLASSES)
        quicklook(hm, paths.qa / "03_hammond.png", "Hammond landforms", cmap="tab10", categorical=True)
        del hm
    del s, dem

    gc = tc["geomorphons"]
    geomorphons(paths.raster("dem"), paths.raster("geomorphons"), paths, gc["search_cells"],
                gc["skip_cells"], gc["flatness_deg"])
    g = read(paths.raster("geomorphons"))
    quicklook(g, paths.qa / "03_geomorphons.png", "geomorphons", cmap="tab10", categorical=True)
    for name, bb in cfg.get("qa_zooms", {}).get("terrain", {}).items():
        zoom(g, grid, bb, paths.qa / f"03_geomorphons_{name}.png", f"geomorphons — {name}", cmap="tab10", categorical=True)
