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
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import rasterio
import requests
from rasterio.enums import Resampling
from rasterio.warp import transform_bounds
from rasterio.windows import Window, from_bounds

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
    out = paths.raw_region / "copernicus_dem"
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


def fetch_zip(cfg, paths: Paths, man: Manifest, key: str, filename: str, root: Path | None = None, url: str | None = None) -> Path:
    url = url or cfg["sources"][key]["url"]
    dest = (root or paths.raw) / key / filename
    if not dest.exists():
        log.info("downloading %s", url)
        http_download(url, dest)
    if str(dest.relative_to(man.path.parent)) not in man.data:  # also when fetched while another run held the manifest
        man.add(dest, url)
    return dest


def hydro_urls(cfg, key: str) -> list[str]:
    """HydroSHEDS continent files of a source: ``url`` plus any ``also`` (an area spanning two continents)."""
    s = cfg["sources"][key]
    return [s["url"], *s.get("also", [])]


def fetch_geology(cfg, paths: Paths, man: Manifest) -> None:
    """The geology map's zip; a File Geodatabase (SGMC) is unpacked too (reading it inside the zip takes minutes)."""
    src = cfg["sources"]["geology"]
    for extra in src.get("also", []):  # further maps for areas the first one does not cover
        fetch_zip(cfg, paths, man, "geology", extra["file"], url=extra["url"])
    zp = fetch_zip(cfg, paths, man, "geology", src["file"])
    if src.get("format") == "sgmc" and not (zp.parent / src["gdb"]).exists():
        import zipfile

        with zipfile.ZipFile(zp) as z:
            z.extractall(zp.parent)


def fetch_files(cfg, paths: Paths, man: Manifest, key: str) -> None:
    """Every ``{filename: url}`` under ``sources.<key>.files``."""
    for name, url in cfg["sources"][key]["files"].items():
        fetch_zip({"sources": {key: {"url": url}}}, paths, man, key, name)


# ---------------------------------------------------------------------------
# GBIF occurrence download (Darwin Core Archive)
# ---------------------------------------------------------------------------
DWCA_FIELDS = {"gbifID", "datasetKey", "eventDate", "year", "basisOfRecord", "samplingProtocol", "kingdom", "family",
               "genus", "species", "speciesKey", "scientificName", "taxonRank", "decimalLatitude", "decimalLongitude",
               "coordinateUncertaintyInMeters", "elevation", "countryCode", "stateProvince", "locality", "habitat",
               "establishmentMeans", "occurrenceRemarks", "fieldNotes", "issue"}
CULTIVATED = r"cultivat|planted|plantation|garden|nursery|ornamental"


def cultivated(df):
    """Records of planted or cultivated plants: GBIF's establishmentMeans MANAGED, living
    collections, or a note saying so (old herbarium sheets rarely fill establishmentMeans)."""
    text = df[[c for c in ("occurrenceRemarks", "habitat", "locality", "fieldNotes") if c in df]].fillna("")
    managed = df["establishmentMeans"].fillna("").str.upper().eq("MANAGED") if "establishmentMeans" in df else False
    living = df["basisOfRecord"].eq("LIVING_SPECIMEN") if "basisOfRecord" in df else False
    return managed | living | text.apply(lambda s: s.str.contains(CULTIVATED, case=False, regex=True)).any(axis=1)


def fetch_gbif_dwca(cfg, paths: Paths, man: Manifest) -> None:
    for src in cfg["sources"]["gbif_dwca"]["downloads"]:
        fetch_one_dwca(src, paths, man)


def fetch_one_dwca(src: dict, paths: Paths, man: Manifest) -> None:
    """A GBIF occurrence download (Darwin Core Archive, cited by its DOI), unpacked
    to the columns we use; fails if the record count differs from GBIF's."""
    import csv
    import zipfile

    import pandas as pd

    zp = fetch_zip({"sources": {"gbif_dwca": src}}, paths, man, "gbif_dwca", f"{src['key']}.zip")
    if str(zp.relative_to(paths.raw)) not in man.data:
        man.add(zp, src["url"], note=f"DOI {src['doi']}")
    dest = zp.with_name(f"occurrences_{src['key']}.csv.gz")
    if dest.exists():
        return
    tmp, n, kept, cult = dest.with_suffix(".part.gz"), 0, 0, 0
    try:
        with zipfile.ZipFile(zp) as z, z.open("occurrence.txt") as f:
            for i, ch in enumerate(pd.read_csv(f, sep="\t", dtype=str, quoting=csv.QUOTE_NONE, chunksize=200_000,
                                               usecols=lambda c: c in DWCA_FIELDS, encoding="utf-8")):
                n += len(ch)
                ch = ch.dropna(subset=["decimalLatitude", "decimalLongitude"])
                ch["cultivated"] = cultivated(ch)
                kept, cult = kept + len(ch), cult + int(ch.cultivated.sum())
                ch.to_csv(tmp, mode="a" if i else "w", header=not i, index=False, compression="gzip")
    except zipfile.BadZipFile as ex:
        raise SourceUnavailable(f"{zp}: not a complete zip ({ex}); delete it and download again") from ex
    if n != src["records"]:
        raise SourceUnavailable(f"GBIF download {src['key']}: {n:,} records, expected {src['records']:,}")
    tmp.replace(dest)
    man.add(dest, src["url"], note=f"DOI {src['doi']}; {kept:,} of {n:,} records with coordinates; "
                                   f"{cult:,} flagged cultivated")
    log.info("GBIF download: %d records, %d with coordinates, %d flagged cultivated", n, kept, cult)


# ---------------------------------------------------------------------------
# window reads of remote global rasters
# ---------------------------------------------------------------------------
def window_read(url: str, dest: Path, bbox_lonlat, margin_deg: float = 0.25,
                decimate: int = 1, resampling=Resampling.nearest) -> Path:
    """Read the bbox window of a remote raster (``/vsicurl``) and save it as a
    compressed GeoTIFF in the *source* CRS. A cached window that does not cover the
    bbox (the extent grew) is read again."""
    if dest.exists():
        with rasterio.open(dest) as r:
            have = transform_bounds(r.crs, "EPSG:4326", *r.bounds, densify_pts=41)
        w, s, e, n = bbox_lonlat
        if have[0] <= w and have[1] <= s and have[2] >= e and have[3] >= n:
            return dest
        log.info("%s does not cover the extent: reading it again", dest.name)
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
        dest = paths.raw_region / "chelsa" / f"CHELSA_{var}_1981-2010_V.2.1_clip.tif"
        window_read(url, dest, cfg["grid"]["bbox_lonlat"])
        man.add(dest, url, note="bbox window of global file, native 30 arc-sec grid")
    for month in range(1, 13):
        url = src["monthly_url"].format(month=month)
        dest = paths.raw_region / "chelsa" / f"CHELSA_pr_{month:02d}_1981-2010_V.2.1_clip.tif"
        window_read(url, dest, cfg["grid"]["bbox_lonlat"])
        man.add(dest, url, note="bbox window of global file, native 30 arc-sec grid")


def fetch_soilgrids(cfg, paths: Paths, man: Manifest) -> None:
    src = cfg["sources"]["soilgrids"]
    for prop in src["properties"]:
        for depth in src["depths"]:
            url = src["url"].format(prop=prop, depth=depth)
            dest = paths.raw_region / "soilgrids" / f"{prop}_{depth}_mean_clip.tif"
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
            dest = paths.raw_region / "worldcover" / Path(url).name.replace(".tif", f"_x{dec}.tif")
            if dest.exists():
                continue
            if requests.head(url, timeout=60).status_code != 200:
                continue  # open sea: tile does not exist
            log.info("WorldCover %s", Path(url).name)
            window_read(url, dest, (lon, lat, lon + 3, lat + 3), margin_deg=0, decimate=dec,
                        resampling=Resampling.mode)
            man.add(dest, url, note=f"decimated x{dec} via overviews (mode)")


GHSL_TILES = (-18041000.0, 9000000.0, 1e6)  # GHSL Mollweide tile grid: left, top, tile size (m); R1_C1 at top left


def ghs_tiles(bbox_lonlat) -> list[tuple[int, int]]:
    """(row, col) of the GHSL tiles covering a lon/lat box."""
    from pyproj import Transformer

    x0, y0, size = GHSL_TILES
    w, s, e, n = bbox_lonlat
    x, y = Transformer.from_crs(4326, "ESRI:54009", always_xy=True).transform(*np.meshgrid(np.linspace(w, e, 60), np.linspace(s, n, 60)))
    return sorted({(int((y0 - b) // size) + 1, int((a - x0) // size) + 1) for a, b in zip(np.ravel(x), np.ravel(y))})


def fetch_ghs_pop(cfg, paths: Paths, man: Manifest) -> None:
    """GHS-POP 1 km tiles over the bbox (tiles of open sea do not exist)."""
    for row, col in ghs_tiles(cfg["grid"]["bbox_lonlat"]):
        url = cfg["sources"]["ghs_pop"]["url"].format(row=row, col=col)
        name = url.rsplit("/", 1)[1]
        if not (paths.raw_region / "ghs_pop" / name).exists() and requests.head(url, timeout=60).status_code == 404:
            continue
        fetch_zip({"sources": {"ghs_pop": {"url": url}}}, paths, man, "ghs_pop", name, paths.raw_region)


def fetch_mcd12q2(cfg, paths: Paths, man: Manifest) -> None:
    """MODIS land-surface phenology (MCD12Q2.061, 500 m, yearly) clipped by NASA AppEEARS to the study
    area, reduced to medians over the years; the yearly files are then deleted (only the medians are kept).
    Needs a NASA Earthdata login in EARTHDATA_USERNAME / EARTHDATA_PASSWORD."""
    src = cfg["sources"]["mcd12q2"]
    d = paths.raw_region / "mcd12q2"
    med = d / "median"
    if (med / "complete").exists():
        return
    api, user, pw = src["api"], os.environ.get("EARTHDATA_USERNAME"), os.environ.get("EARTHDATA_PASSWORD")
    if not (user and pw):
        raise SourceUnavailable("mcd12q2: set EARTHDATA_USERNAME and EARTHDATA_PASSWORD (NASA Earthdata login)")
    r = requests.post(f"{api}/login", auth=(user, pw), timeout=60)
    if r.status_code != 200:
        raise SourceUnavailable(f"mcd12q2: AppEEARS login failed (HTTP {r.status_code})")
    h = {"Authorization": f"Bearer {r.json()['token']}"}
    state = d / "task.json"
    if not state.exists():
        import geopandas as gpd

        study = gpd.read_file(paths.vector("boundaries"), layer="study_area").to_crs(4326)
        g = study.union_all().buffer(0.05).simplify(0.02)  # ~5 km margin so edge cells have data
        y0, y1 = src["years"]
        task = {"task_type": "area", "task_name": "ecoregions_mcd12q2",
                "params": {"dates": [{"startDate": f"01-01-{y0}", "endDate": f"12-31-{y1}"}],
                           "layers": [{"product": "MCD12Q2.061", "layer": x} for x in src["layers"]],
                           "geo": json.loads(gpd.GeoSeries([g], crs=4326).to_json()),
                           "output": {"format": {"type": "geotiff"}, "projection": "geographic"}}}
        r = requests.post(f"{api}/task", json=task, headers=h, timeout=120)
        if r.status_code not in (200, 202):
            raise SourceUnavailable(f"mcd12q2: AppEEARS task refused (HTTP {r.status_code}): {r.text[:200]}")
        d.mkdir(parents=True, exist_ok=True)
        state.write_text(json.dumps(r.json()))
    tid = json.loads(state.read_text())["task_id"]
    while (st := requests.get(f"{api}/task/{tid}", headers=h, timeout=60).json()["status"]) != "done":
        if st not in ("pending", "queued", "processing"):
            raise SourceUnavailable(f"mcd12q2: AppEEARS task {tid} ended as {st}")
        log.info("mcd12q2: AppEEARS task %s %s; waiting", tid, st)
        time.sleep(300)
    files = [f for f in requests.get(f"{api}/bundle/{tid}", headers=h, timeout=60).json()["files"]
             if f["file_name"].endswith(".tif")]
    tiles = d / "tiles"
    tiles.mkdir(parents=True, exist_ok=True)
    for f in files:
        dst = tiles / Path(f["file_name"]).name
        if not (dst.exists() and dst.stat().st_size == f["file_size"]):
            with requests.get(f"{api}/bundle/{tid}/{f['file_id']}", headers=h, stream=True, timeout=300) as r:
                r.raise_for_status()
                with open(dst, "wb") as fh:
                    for chunk in r.iter_content(1 << 20):
                        fh.write(chunk)
    phenology_medians(tiles, med, src)
    for p in med.glob("*.tif"):
        man.add(p, f"{api}/task/{tid}", note=f"median {src['years'][0]}-{src['years'][1]} of yearly MCD12Q2.061 (AppEEARS)")
    for p in tiles.glob("*.tif"):
        p.unlink()
    (med / "complete").write_text(dt.datetime.now().isoformat())


PHENOLOGY = {  # median layer -> units (first growing cycle, years with QA_Overall <= max_qa)
    "season_days": "days (dormancy minus green-up)", "greenup_doy": "day of year (negative: previous year)",
    "evi_amplitude": "EVI2", "evi_area": "EVI2 x days", "cycles_per_year": "mean detected cycles (0 = none)"}


def phenology_medians(tiles: Path, out: Path, src: dict, rows: int = 200) -> None:
    """Per-pixel medians over the years of the first growing cycle (``*_0`` layers), in row blocks."""
    def files(layer):
        return sorted(tiles.glob(f"MCD12Q2.061_{layer}_doy*.tif"))

    years = [int(p.name.split("_doy")[1][:4]) for p in files("NumCycles")]
    srcs = {k: [rasterio.open(p) for p in files(k)] for k in
            ["Greenup_0", "Dormancy_0", "EVI_Amplitude_0", "EVI_Area_0", "QA_Overall_0", "NumCycles"]}
    if not years or any(len(v) != len(years) for v in srcs.values()):
        raise SourceUnavailable(f"mcd12q2: incomplete yearly layers in {tiles}")
    ref = srcs["NumCycles"][0]
    jan1 = np.array([(dt.date(y, 1, 1) - dt.date(1970, 1, 1)).days for y in years], "float32")[:, None, None]
    out.mkdir(parents=True, exist_ok=True)
    prof = ref.profile | dict(dtype="float32", nodata=np.nan, compress="deflate", tiled=True, blockxsize=256, blockysize=256)
    dsts = {k: rasterio.open(out / f"{k}.tif", "w", **prof) for k in PHENOLOGY}
    for r0 in range(0, ref.height, rows):
        w = Window(0, r0, ref.width, min(rows, ref.height - r0))
        a = {k: np.stack([s.read(1, window=w) for s in v]).astype("float32") for k, v in srcs.items()}
        bad = a["QA_Overall_0"] > src["max_qa"]  # fill (32767) included
        for k in ["Greenup_0", "Dormancy_0", "EVI_Amplitude_0", "EVI_Area_0"]:
            a[k][bad | (a[k] == 32767)] = np.nan
        n = a["NumCycles"]
        n[n == 32767] = 0  # no cycle detected that year (bare ground, ice, or no seasonal greening)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN pixels -> NaN median
            res = {"season_days": np.nanmedian(a["Dormancy_0"] - a["Greenup_0"], 0),
                   "greenup_doy": np.nanmedian(a["Greenup_0"] - jan1, 0),
                   "evi_amplitude": np.nanmedian(a["EVI_Amplitude_0"], 0) * 1e-4,
                   "evi_area": np.nanmedian(a["EVI_Area_0"], 0) * 0.1,
                   "cycles_per_year": n.mean(0)}
        for k, v in res.items():
            dsts[k].write(v.astype("float32"), 1, window=w)
    for k, dst in dsts.items():
        dst.update_tags(units=PHENOLOGY[k], years=f"{years[0]}-{years[-1]}")
        dst.close()
    for v in srcs.values():
        for s in v:
            s.close()


def run(cfg) -> None:
    paths = Paths(cfg)
    man = Manifest(paths.raw / "manifest.json")
    steps = [
        ("natural_earth", lambda: fetch_natural_earth(cfg, paths, man)),
        ("geology", lambda: fetch_geology(cfg, paths, man)),
        ("resolve", lambda: fetch_zip(cfg, paths, man, "resolve", "Ecoregions2017.zip")),
        ("hydrorivers", lambda: [fetch_zip(cfg, paths, man, "hydrorivers", Path(u).name, url=u) for u in hydro_urls(cfg, "hydrorivers")]),
        ("hydrobasins", lambda: [fetch_zip(cfg, paths, man, "hydrobasins", Path(u).name, url=u) for u in hydro_urls(cfg, "hydrobasins")]),
        ("chelsa", lambda: fetch_chelsa(cfg, paths, man)),
        ("dem", lambda: fetch_dem(cfg, paths, man)),
        ("soilgrids", lambda: fetch_soilgrids(cfg, paths, man)),
        ("worldcover", lambda: fetch_worldcover(cfg, paths, man)),
        ("ghs_pop", lambda: fetch_ghs_pop(cfg, paths, man)),
        # v0.2 validation sources
        ("gmw", lambda: fetch_files(cfg, paths, man, "gmw")),
        ("pnv", lambda: fetch_files(cfg, paths, man, "pnv")),
        ("jung_habitat", lambda: fetch_files(cfg, paths, man, "jung_habitat")),
        ("gbif_dwca", lambda: fetch_gbif_dwca(cfg, paths, man)),
        ("mcd12q2", lambda: fetch_mcd12q2(cfg, paths, man)),
        # reference maps: validation only, compared with the finished map
        ("references", lambda: [fetch_zip({"sources": {"references": r}}, paths, man, "references", r["file"])
                                for r in cfg.get("validation", {}).get("references", [])]),
    ]
    only = os.environ.get("ECOREGIONS_DOWNLOAD_ONLY")
    for name, fn in steps:
        if only and name not in only.split(","):
            continue
        log.info("== download: %s", name)
        fn()
        man.save()
    (paths.raw / ".complete").write_text(dt.datetime.now().isoformat())
