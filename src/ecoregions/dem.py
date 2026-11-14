"""Stage 2 — Copernicus GLO-90 mosaic on the analysis grid.

Tiles are assembled into a single GDAL VRT in their native geographic grid and
warped *once* to the Albers grid, so bilinear interpolation never sees a tile
edge (no seams). Remaining voids, if any, are filled by inverse-distance
interpolation and flagged in ``dem_filled_mask``.

Coastline at 90 m: the grid stage's sea comes from Natural Earth 1:10M (``mask_ne``). Here the sea is
redrawn from the DEM itself: Copernicus sets the ocean to exactly 0 m, so sea = cells at 0 m or with no
tile, connected to the open sea (inland flats at 0 m stay land). Land the DEM adds joins the nearest
Natural Earth land class (study area or other land). Only a window around the Natural Earth coast is
examined; the result is ``mask`` (and ``coast_changes`` in the boundaries file, for the web land layer).
"""
from __future__ import annotations

import logging
from pathlib import Path
from xml.sax.saxutils import escape

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.features import shapes
from rasterio.fill import fillnodata
from scipy import ndimage as ndi
from shapely.geometry import shape

from .config import Paths
from .grid import Grid, reproject_file, write
from .qa import quicklook, zoom

log = logging.getLogger(__name__)


def build_vrt(tiles: list[Path], dest: Path) -> Path:
    """Write a mosaic VRT for same-resolution geographic tiles."""
    metas = []
    for t in tiles:
        with rasterio.open(t) as s:
            metas.append((t, s.transform, s.width, s.height, s.nodata, s.crs))
    res_x = metas[0][1].a
    res_y = -metas[0][1].e
    if any(abs(m[1].a - res_x) > 1e-12 or abs(-m[1].e - res_y) > 1e-12 for m in metas):
        raise ValueError("tiles with differing resolution; cannot mosaic as a simple VRT")
    west = min(m[1].c for m in metas)
    north = max(m[1].f for m in metas)
    east = max(m[1].c + m[2] * res_x for m in metas)
    south = min(m[1].f - m[3] * res_y for m in metas)
    W = int(round((east - west) / res_x))
    H = int(round((north - south) / res_y))
    crs_wkt = escape(metas[0][5].to_wkt())
    srcs = []
    for t, tr, w, h, nod, _ in metas:
        xoff = int(round((tr.c - west) / res_x))
        yoff = int(round((north - tr.f) / res_y))
        srcs.append(f"""    <SimpleSource>
      <SourceFilename relativeToVRT="0">{escape(str(t.resolve()))}</SourceFilename>
      <SourceBand>1</SourceBand>
      <SrcRect xOff="0" yOff="0" xSize="{w}" ySize="{h}"/>
      <DstRect xOff="{xoff}" yOff="{yoff}" xSize="{w}" ySize="{h}"/>
    </SimpleSource>""")
    xml = f"""<VRTDataset rasterXSize="{W}" rasterYSize="{H}">
  <SRS>{crs_wkt}</SRS>
  <GeoTransform>{west!r}, {res_x!r}, 0, {north!r}, 0, {-res_y!r}</GeoTransform>
  <VRTRasterBand dataType="Float32" band="1">
    <NoDataValue>-32767</NoDataValue>
{chr(10).join(srcs)}
  </VRTRasterBand>
</VRTDataset>
"""
    dest.write_text(xml)
    return dest


def coast_window(ne: np.ndarray, margin: int, block: int = 64) -> tuple[slice, slice] | None:
    """Rows and columns around the Natural Earth coast (land next to sea), padded by ``margin`` cells."""
    h, w = ne.shape
    H, W = -(-h // block), -(-w // block)
    pad = np.zeros((H * block, W * block), bool)
    pad[:h, :w] = ne == 0
    sea_any = pad.reshape(H, block, W, block).any(axis=(1, 3))
    pad[:h, :w] = ne > 0
    land_any = pad.reshape(H, block, W, block).any(axis=(1, 3))
    rows, cols = np.nonzero(sea_any & land_any)
    if not len(rows):
        return None
    r0, r1 = max(0, rows.min() * block - margin), min(h, (rows.max() + 1) * block + margin)
    c0, c1 = max(0, cols.min() * block - margin), min(w, (cols.max() + 1) * block + margin)
    return slice(r0, r1), slice(c0, c1)


def coast_from_dem(ne: np.ndarray, sea_dem: np.ndarray, open_sea_cells: int, max_inland_cells: int) -> np.ndarray:
    """New mask for one window: ``ne`` (0 sea, 1 study, 3 other land) with the sea taken from ``sea_dem``
    (DEM ocean or no data) where it connects to Natural Earth sea at least ``open_sea_cells`` from land.
    Natural Earth land turns to sea only within ``max_inland_cells`` of its coast: tidal creeks do, salt flats
    that Copernicus also flattens to 0 m (the Rann of Kutch) beyond that stay land."""
    lab, _ = ndi.label(sea_dem)
    open_sea = ndi.distance_transform_edt(ne == 0) >= open_sea_cells
    keep = np.unique(lab[open_sea & sea_dem])
    sea = np.isin(lab, keep[keep > 0]) & (ndi.distance_transform_edt(ne > 0) <= max_inland_cells)
    near = ndi.distance_transform_edt(ne == 0, return_distances=False, return_indices=True)
    out = ne[near[0], near[1]].astype("uint8")  # every cell: its nearest Natural Earth land class
    out[sea] = 0
    return out


def inland_sea(dem: np.ndarray, level: float, min_cells: int) -> np.ndarray:
    """Cells of connected patches where the DEM sits exactly at ``level`` (Copernicus flattens water surfaces), at
    least ``min_cells`` large: the open water of an inland sea. Lagoons and shores at other heights stay land."""
    lab, _ = ndi.label(np.abs(dem - level) < 0.01)
    big = np.flatnonzero(np.bincount(lab.ravel()) >= min_cells)
    return np.isin(lab, big[big > 0])


def sea_window(dem: np.ndarray, grid, sea: dict) -> np.ndarray:
    """Full-grid boolean of one ``grid.inland_seas`` entry (name, level_m, bbox_lonlat, min_km2)."""
    from pyproj import Transformer

    t = Transformer.from_crs("EPSG:4326", grid.crs, always_xy=True)
    w, s, e, n = sea["bbox_lonlat"]
    xs, ys = t.transform([w, w, e, e], [s, n, s, n])
    inv = ~grid.transform
    cols, rows = zip(*(inv * (x, y) for x, y in zip(xs, ys)))
    r0, r1 = max(0, int(min(rows))), min(dem.shape[0], int(max(rows)) + 1)
    c0, c1 = max(0, int(min(cols))), min(dem.shape[1], int(max(cols)) + 1)
    out = np.zeros(dem.shape, bool)
    out[r0:r1, c0:c1] = inland_sea(dem[r0:r1, c0:c1], sea["level_m"], round(sea["min_km2"] / grid.cell_km2))
    log.info("inland sea %s: %.0f km² set to sea", sea["name"], out.sum() * grid.cell_km2)
    return out


def run(cfg) -> None:
    paths = Paths(cfg)
    grid = Grid.load(paths.grid_json)
    tiles = sorted((paths.raw_region / "copernicus_dem").glob("*.tif"))
    if not tiles:
        raise FileNotFoundError("no DEM tiles; run the download stage first")
    vrt = build_vrt(tiles, paths.interim / "copernicus_mosaic.vrt")
    log.info("mosaic VRT of %d tiles", len(tiles))
    dem = reproject_file(vrt, grid, Resampling.bilinear, apply_scale=False)
    ne = rasterio.open(paths.raster("mask_ne")).read(1)
    mask = ne.copy()
    win = coast_window(ne, margin=round(cfg["grid"]["coast_window_m"] / grid.res))
    if win is not None:
        near = reproject_file(vrt, grid, Resampling.nearest, apply_scale=False)[win]  # no blur across the shore
        mask[win] = coast_from_dem(ne[win], (near == 0) | np.isnan(near), round(cfg["grid"]["open_sea_m"] / grid.res),
                                     round(cfg["grid"]["coast_max_inland_m"] / grid.res))
        del near
        sub = ne[win]
        changed = {"sea_added": (sub > 0) & (mask[win] == 0), "land_added": (sub == 0) & (mask[win] > 0)}
        tr = grid.transform * grid.transform.translation(win[1].start, win[0].start)
        rows = [(k, shape(g)) for k, m in changed.items() if m.any()
                for g, _ in shapes(m.astype("uint8"), mask=m, transform=tr)]
        gpd.GeoDataFrame({"change": [k for k, _ in rows]}, geometry=[g for _, g in rows], crs=grid.crs).to_file(
            paths.vector("boundaries"), layer="coast_changes")
        log.info("coast from the DEM: %.0f km² of Natural Earth land is sea, %.0f km² of its sea is land "
                 "(study area %.0f -> %.0f km²)", changed["sea_added"].sum() * grid.cell_km2,
                 changed["land_added"].sum() * grid.cell_km2, (ne == 1).sum() * grid.cell_km2, (mask == 1).sum() * grid.cell_km2)
    for sea in cfg["grid"].get("inland_seas", []):  # e.g. the Caspian (surface at -28 m, so not sea by the 0 m test above)
        mask[sea_window(dem, grid, sea)] = 0
    added = ((ne == 0) & (mask == 1)).sum() / max(1, (mask == 1).sum())
    log.info("study area the DEM adds beyond Natural Earth's coast: %.2f%%", 100 * added)
    if added > 0.01:  # Texas 0.2 %; the Caspian (at -28 m, so not 0 m sea) once added 4 %
        raise SystemExit(f"dem: the DEM turns {100 * added:.1f}% of the study area from Natural Earth sea into land; "
                         "an inland sea below 0 m? (grid.inland_seas)")
    write(paths.raster("mask"), mask, grid, nodata=None, tags=dict(values="0 sea, 1 study area, 3 other land",
                                                                   coast="Copernicus GLO-90 (0 m ocean)"))
    from .boundaries import raster_borders
    b = raster_borders(cfg, paths, mask, grid)  # border lines on the same 90 m grid as the regions
    b.to_file(paths.vector("boundaries"), layer="borders")
    log.info("borders on the 90 m grid: %.0f km solid, %.0f km dotted",
             b.length[b["style"] == "solid"].sum() / 1e3, b.length[b["style"] == "dotted"].sum() / 1e3)
    land = mask > 0
    void = land & np.isnan(dem)
    log.info("voids on land before fill: %d cells", int(void.sum()))
    study_void = (void & (mask == 1)).sum() / (mask == 1).sum()  # outside the study area the grid may reach past the tiles
    log.info("study area without DEM before fill: %.3f%%", 100 * study_void)
    if study_void > 0.005:  # whole tiles missing, not GLO-90's small voids
        raise SystemExit(f"dem: {100 * study_void:.1f}% of the study area has no DEM; missing tiles? "
                         f"(compare {paths.raw_region / 'copernicus_dem'} with the download stage's tile list)")
    if void.any():
        valid = (~np.isnan(dem)).astype("uint8")
        dem = fillnodata(np.nan_to_num(dem, nan=0), mask=valid, max_search_distance=200)
    dem[~land] = np.nan  # sea
    write(paths.raster("dem"), dem.astype("float32"), grid, nodata=np.nan,
          tags=dict(source="Copernicus GLO-90 DSM (EGM2008 heights)", resampling="bilinear"))
    write(paths.raster("dem_filled_mask"), void.astype("uint8"), grid, nodata=None)
    quicklook(dem, paths.qa / "02_dem.png", "Copernicus GLO-90 on Albers 90 m grid", cmap="terrain",
              vmin=0, vmax=6000, outline=(mask == 1))
    # tile-corner close-ups in the plains, where seams would be most visible
    for name, bb in cfg.get("qa_zooms", {}).get("dem", {}).items():
        zoom(dem, grid, bb, paths.qa / f"02_dem_{name}_zoom.png", f"{name} (seam check)", cmap="terrain")
