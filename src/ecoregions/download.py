"""Stage 0 — fetch and cache raw inputs in ``data/raw``.

Downloads are idempotent: an existing file is never fetched twice. Remote global
rasters (CHELSA, SoilGrids, WorldCover) are *window-read* over HTTP so only the
study extent is stored, in the source CRS and resolution (no resampling here).
Every file is recorded in ``data/raw/manifest.json`` with its URL, size, SHA-256
and retrieval date so a run can be audited and reproduced.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import rasterio
import requests
from rasterio.enums import Resampling
from rasterio.warp import transform_bounds
from rasterio.windows import from_bounds

from .config import Paths

log = logging.getLogger(__name__)

GDAL_HTTP_ENV = dict(
    GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
    CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif,.vrt,.TIF",
    GDAL_HTTP_MAX_RETRY="6",
    GDAL_HTTP_RETRY_DELAY="3",
    GDAL_HTTP_MULTIPLEX="YES",
    GDAL_HTTP_VERSION="2",
    VSI_CACHE="TRUE",
    VSI_CACHE_SIZE=str(256 * 2**20),
    GDAL_CACHEMAX=1024 * 2**20,
)


class SourceUnavailable(RuntimeError):
    """Raised when a configured source cannot be fetched. We fail loudly rather
    than silently substituting another dataset."""


# ---------------------------------------------------------------------------
# manifest
# ---------------------------------------------------------------------------
def _sha256(path: Path, chunk: int = 2**22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


class Manifest:
    def __init__(self, path: Path):
        self.path = path
        self.data = json.loads(path.read_text()) if path.exists() else {}

    def add(self, file: Path, url: str, note: str = "") -> None:
        rel = str(file.relative_to(self.path.parent))
        self.data[rel] = dict(url=url, bytes=file.stat().st_size, sha256=_sha256(file),
                              retrieved=dt.date.today().isoformat(), note=note)

    def save(self) -> None:
        self.path.write_text(json.dumps(self.data, indent=1, sort_keys=True))


# ---------------------------------------------------------------------------
# plain HTTP files
# ---------------------------------------------------------------------------
def http_download(url: str, dest: Path, retries: int = 5, timeout: int = 120) -> Path:
    """Stream ``url`` to ``dest`` (atomic rename, resumable, exponential back-off)."""
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(retries):
        try:
            headers = {"Range": f"bytes={part.stat().st_size}-"} if part.exists() else {}
            with requests.get(url, stream=True, timeout=timeout, headers=headers) as r:
                if r.status_code == 416:  # already complete
                    break
                if 400 <= r.status_code < 500:  # client errors are permanent: no retry
                    raise SourceUnavailable(f"{url}: HTTP {r.status_code}")
                r.raise_for_status()
                mode = "ab" if r.status_code == 206 else "wb"
                with open(part, mode) as f:
                    for chunk in r.iter_content(2**20):
                        f.write(chunk)
            break
        except (requests.RequestException, OSError) as e:
            if attempt == retries - 1:
                raise SourceUnavailable(f"{url}: {e}") from e
            wait = 2 ** (attempt + 1)
            log.warning("retry %s in %ss (%s)", url, wait, e)
            time.sleep(wait)
    part.replace(dest)  # replace, not rename: rename fails on Windows if dest exists
    return dest


def fetch_natural_earth(cfg, paths: Paths, man: Manifest) -> None:
    src = cfg["sources"]["natural_earth"]
    out = paths.raw / "naturalearth"
    for layer in src["layers"]:
        for ext in src["extensions"]:
            url = f"{src['base']}/{layer}.{ext}"
            dest = out / f"{layer}.{ext}"
            try:
                if not dest.exists():
                    http_download(url, dest)
                    man.add(dest, url)
            except SourceUnavailable:
                if ext == "cpg":  # optional sidecar (the Pakistan POV file has none)
                    continue
                raise


def dem_tile_names(cfg) -> list[str]:
    """1x1 degree Copernicus tiles intersecting the bbox (only those that exist)."""
    src = cfg["sources"]["copernicus_dem"]
    w, s, e, n = cfg["grid"]["bbox_lonlat"]
    available = set(requests.get(src["tile_list"], timeout=60).text.split())
    names = []
    for lat in range(int(np.floor(s)), int(np.ceil(n))):
        for lon in range(int(np.floor(w)), int(np.ceil(e))):
            name = src["name"].format(ns="N" if lat >= 0 else "S", lat=abs(lat),
                                      ew="E" if lon >= 0 else "W", lon=abs(lon))
            if name in available:
                names.append(name)
    return names


def fetch_dem(cfg, paths: Paths, man: Manifest, workers: int = 8) -> None:
    src = cfg["sources"]["copernicus_dem"]
    out = paths.raw / "copernicus_dem"
    names = dem_tile_names(cfg)
    log.info("Copernicus GLO-90: %d tiles", len(names))

    def one(name):
        url = src["url"].format(name=name)
        dest = out / f"{name}.tif"
        if not dest.exists():
            http_download(url, dest)
        return dest, url

    with ThreadPoolExecutor(workers) as ex:
        for dest, url in ex.map(one, names):
            if str(dest.relative_to(paths.raw)) not in man.data:
                man.add(dest, url)


def fetch_zip(cfg, paths: Paths, man: Manifest, key: str, filename: str) -> Path:
    url = cfg["sources"][key]["url"]
    dest = paths.raw / key / filename
    if not dest.exists():
        log.info("downloading %s", url)
        http_download(url, dest)
        man.add(dest, url)
    return dest


def fetch_files(cfg, paths: Paths, man: Manifest, key: str) -> None:
    """Every ``{filename: url}`` under ``sources.<key>.files``."""
    for name, url in cfg["sources"][key]["files"].items():
        fetch_zip({"sources": {key: {"url": url}}}, paths, man, key, name)


# ---------------------------------------------------------------------------
# GBIF occurrence records (paged search API)
# ---------------------------------------------------------------------------
GBIF_FIELDS = ["gbifID", "datasetKey", "eventID", "eventDate", "year", "basisOfRecord", "samplingProtocol",
               "kingdom", "family", "genus", "species", "speciesKey", "scientificName", "taxonRank",
               "decimalLatitude", "decimalLongitude", "coordinateUncertaintyInMeters", "elevation",
               "country", "stateProvince", "locality", "habitat"]
GBIF_PAGE, GBIF_MAX_OFFSET = 300, 100_000
GBIF_CHUNK = 9_000  # deep offsets are very slow (offset 30,000: ~5 min per page), so keep queries small


def gbif_pages(get, params: dict):
    """Yield all records of one search. ``get(params) -> dict`` is injected so the
    paging logic is testable without the network."""
    n = get({**params, "limit": 0})["count"]
    if n > GBIF_MAX_OFFSET:
        raise SourceUnavailable(f"GBIF query returns {n:,} records, above the 100,000 paging limit: split it further")
    for off in range(0, n, GBIF_PAGE):
        page = get({**params, "limit": GBIF_PAGE, "offset": off})
        yield from page["results"]
        if page.get("endOfRecords"):
            break


def gbif_chunks(get, params: dict, bbox: tuple, cap: int = GBIF_CHUNK, depth: int = 0):
    """Bisect ``bbox`` (W, S, E, N; alternately by latitude and longitude) until
    every piece holds at most ``cap`` records; yield (params, count). Every record
    has coordinates, so nothing is dropped; ranges are inclusive, so records on a
    shared edge may appear twice (dedupe by gbifID)."""
    w, s, e, n = bbox
    prm = {**params, "decimalLatitude": f"{s},{n}", "decimalLongitude": f"{w},{e}"}
    cnt = get({**prm, "limit": 0})["count"]
    if cnt == 0:
        return
    if cnt <= cap or depth > 24:
        yield prm, cnt
        return
    if depth % 2 == 0:
        m = (s + n) / 2
        halves = ((w, s, e, m), (w, m, e, n))
    else:
        m = (w + e) / 2
        halves = ((w, s, m, n), (m, s, e, n))
    for h in halves:
        yield from gbif_chunks(get, params, h, cap, depth + 1)


def fetch_gbif_records(cfg, paths: Paths, man: Manifest) -> None:
    import pandas as pd

    src, api = cfg["sources"]["gbif_records"], cfg["sources"]["gbif"]["api"]
    w, s, e, n = cfg["grid"]["bbox_lonlat"]
    sess = requests.Session()

    def get(params):
        for attempt in range(8):
            try:
                r = sess.get(f"{api}/occurrence/search", params=params, timeout=(15, 90))
                if r.status_code == 429 or r.status_code >= 500:
                    raise requests.HTTPError(f"HTTP {r.status_code}")
                if r.status_code >= 400:
                    raise SourceUnavailable(f"GBIF rejected {params}: HTTP {r.status_code}")
                return r.json()
            except requests.RequestException as ex:
                wait = min(2 ** (attempt + 1), 120)
                log.warning("GBIF retry in %ss: %s", wait, ex)
                time.sleep(wait)
        raise SourceUnavailable("GBIF API unavailable after retries")

    for key, title in src["datasets"].items():
        dest = paths.raw / "gbif_records" / f"{key}.csv"
        if dest.exists():
            continue
        base = dict(datasetKey=key, hasCoordinate="true")
        total = get({**base, "limit": 0, "decimalLatitude": f"{s},{n}", "decimalLongitude": f"{w},{e}"})["count"]
        chunks = list(gbif_chunks(get, base, (w, s, e, n)))
        log.info("GBIF %s: %d records in bbox, %d chunks", title, total, len(chunks))
        part_dir = dest.parent / "parts"
        part_dir.mkdir(parents=True, exist_ok=True)
        files = []
        for prm, cnt in chunks:  # one file per chunk, so an interrupted download resumes
            tag = f"{prm['decimalLongitude']}_{prm['decimalLatitude']}".replace(",", "-")
            pf = part_dir / f"{key}_{tag}.csv"
            files.append(pf)
            if pf.exists():
                continue
            rows = [{f: r.get(f) for f in GBIF_FIELDS} for r in gbif_pages(get, prm)]
            pd.DataFrame(rows, columns=GBIF_FIELDS).to_csv(pf.with_suffix(".part.csv"), index=False, encoding="utf-8")
            pf.with_suffix(".part.csv").replace(pf)
            log.info("GBIF %s %s: %d records", key[:8], tag, len(rows))
        df = pd.concat([pd.read_csv(f, dtype=str) for f in files]).drop_duplicates("gbifID")
        if len(df) != total:
            raise SourceUnavailable(f"GBIF {key}: got {len(df)} of {total} records")
        tmp = dest.with_suffix(".part.csv")
        df.to_csv(tmp, index=False, encoding="utf-8")
        tmp.replace(dest)
        man.add(dest, f"{api}/occurrence/search?datasetKey={key}&bbox", note=f"{title}; {total} records")


# ---------------------------------------------------------------------------
# window reads of remote global rasters
# ---------------------------------------------------------------------------
def window_read(url: str, dest: Path, bbox_lonlat, margin_deg: float = 0.25,
                decimate: int = 1, resampling=Resampling.nearest) -> Path:
    """Read the bbox window of a remote raster (``/vsicurl``) and save it as a
    compressed GeoTIFF in the *source* CRS."""
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    w, s, e, n = bbox_lonlat
    bb = (w - margin_deg, s - margin_deg, e + margin_deg, n + margin_deg)
    with rasterio.Env(**GDAL_HTTP_ENV):
        try:
            src = rasterio.open(f"/vsicurl/{url}")
        except rasterio.errors.RasterioIOError as e:
            raise SourceUnavailable(f"{url}: {e}") from e
        with src:
            nb = transform_bounds("EPSG:4326", src.crs, *bb, densify_pts=41)
            win = from_bounds(*nb, transform=src.transform).round_offsets().round_lengths()
            win = win.intersection(rasterio.windows.Window(0, 0, src.width, src.height))
            shape = (max(1, int(win.height // decimate)), max(1, int(win.width // decimate)))
            data = src.read(1, window=win, out_shape=shape, resampling=resampling)
            transform = src.window_transform(win) * rasterio.Affine.scale(win.width / shape[1],
                                                                          win.height / shape[0])
            prof = dict(driver="GTiff", dtype=data.dtype, count=1, height=shape[0], width=shape[1],
                        crs=src.crs, transform=transform, nodata=src.nodata, compress="deflate",
                        tiled=True, blockxsize=512, blockysize=512, predictor=2 if data.dtype.kind in "iu" else 3,
                        BIGTIFF="IF_SAFER")
            tmp = dest.with_suffix(".part.tif")
            with rasterio.open(tmp, "w", **prof) as dst:
                dst.write(data, 1)
                dst.update_tags(source_url=url, scale=src.scales[0], offset=src.offsets[0])
                dst.scales = src.scales
                dst.offsets = src.offsets
            tmp.replace(dest)
    return dest


def fetch_chelsa(cfg, paths: Paths, man: Manifest) -> None:
    src = cfg["sources"]["chelsa"]
    for var in src["variables"]:
        url = src["url"].format(var=var)
        dest = paths.raw / "chelsa" / f"CHELSA_{var}_1981-2010_V.2.1_clip.tif"
        if not dest.exists():
            log.info("CHELSA %s", var)
            window_read(url, dest, cfg["grid"]["bbox_lonlat"])
            man.add(dest, url, note="bbox window of global file, native 30 arc-sec grid")


def fetch_soilgrids(cfg, paths: Paths, man: Manifest) -> None:
    src = cfg["sources"]["soilgrids"]
    for prop in src["properties"]:
        for depth in src["depths"]:
            url = src["url"].format(prop=prop, depth=depth)
            dest = paths.raw / "soilgrids" / f"{prop}_{depth}_mean_clip.tif"
            if not dest.exists():
                log.info("SoilGrids %s %s", prop, depth)
                window_read(url, dest, cfg["grid"]["bbox_lonlat"])
                man.add(dest, url, note="bbox window, native 250 m Homolosine grid")


def fetch_worldcover(cfg, paths: Paths, man: Manifest) -> None:
    """WorldCover 10 m, read through COG overviews at ~80 m (context only)."""
    src = cfg["sources"]["worldcover"]
    w, s, e, n = cfg["grid"]["bbox_lonlat"]
    dec = max(1, int(src["read_resolution_m"] // 10))
    for lat in range(int(np.floor(s / 3) * 3), int(np.ceil(n / 3) * 3), 3):
        for lon in range(int(np.floor(w / 3) * 3), int(np.ceil(e / 3) * 3), 3):
            url = src["url"].format(ns="N" if lat >= 0 else "S", lat=abs(lat),
                                    ew="E" if lon >= 0 else "W", lon=abs(lon))
            dest = paths.raw / "worldcover" / Path(url).name.replace(".tif", f"_x{dec}.tif")
            if dest.exists():
                continue
            if requests.head(url, timeout=60).status_code != 200:
                continue  # open sea: tile does not exist
            log.info("WorldCover %s", Path(url).name)
            window_read(url, dest, (lon, lat, lon + 3, lat + 3), margin_deg=0, decimate=dec,
                        resampling=Resampling.mode)
            man.add(dest, url, note=f"decimated x{dec} via overviews (mode)")


def run(cfg) -> None:
    paths = Paths(cfg)
    man = Manifest(paths.raw / "manifest.json")
    steps = [
        ("natural_earth", lambda: fetch_natural_earth(cfg, paths, man)),
        ("geology", lambda: fetch_zip(cfg, paths, man, "geology", "geo8ag.zip")),
        ("resolve", lambda: fetch_zip(cfg, paths, man, "resolve", "Ecoregions2017.zip")),
        ("hydrorivers", lambda: fetch_zip(cfg, paths, man, "hydrorivers", "HydroRIVERS_v10_as_shp.zip")),
        ("hydrobasins", lambda: fetch_zip(cfg, paths, man, "hydrobasins", "hybas_as_lev01-12_v1c.zip")),
        ("chelsa", lambda: fetch_chelsa(cfg, paths, man)),
        ("dem", lambda: fetch_dem(cfg, paths, man)),
        ("soilgrids", lambda: fetch_soilgrids(cfg, paths, man)),
        ("worldcover", lambda: fetch_worldcover(cfg, paths, man)),
        # v0.2 validation sources
        ("gmw", lambda: fetch_files(cfg, paths, man, "gmw")),
        ("pnv", lambda: fetch_files(cfg, paths, man, "pnv")),
        ("jung_habitat", lambda: fetch_files(cfg, paths, man, "jung_habitat")),
        ("gbif_records", lambda: fetch_gbif_records(cfg, paths, man)),
    ]
    only = os.environ.get("ECOREGIONS_DOWNLOAD_ONLY")
    for name, fn in steps:
        if only and name not in only.split(","):
            continue
        log.info("== download: %s", name)
        fn()
        man.save()
    (paths.raw / ".complete").write_text(dt.datetime.now().isoformat())
