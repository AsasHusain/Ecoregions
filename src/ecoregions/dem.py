"""Stage 2 — Copernicus GLO-90 mosaic on the analysis grid.

Tiles are assembled into a single GDAL VRT in their native geographic grid and
warped *once* to the Albers grid, so bilinear interpolation never sees a tile
edge (no seams). Remaining voids, if any, are filled by inverse-distance
interpolation and flagged in ``dem_filled_mask``.
"""
from __future__ import annotations

import logging
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.fill import fillnodata

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


def run(cfg) -> None:
    paths = Paths(cfg)
    grid = Grid.load(paths.grid_json)
    tiles = sorted((paths.raw / "copernicus_dem").glob("*.tif"))
    if not tiles:
        raise FileNotFoundError("no DEM tiles; run the download stage first")
    vrt = build_vrt(tiles, paths.interim / "copernicus_mosaic.vrt")
    log.info("mosaic VRT of %d tiles", len(tiles))
    dem = reproject_file(vrt, grid, Resampling.bilinear, apply_scale=False)
    mask = rasterio.open(paths.raster("mask")).read(1)
    land = mask > 0
    void = land & np.isnan(dem)
    log.info("voids on land before fill: %d cells", int(void.sum()))
    if void.any():
        valid = (~np.isnan(dem)).astype("uint8")
        dem = fillnodata(np.nan_to_num(dem, nan=0), mask=valid, max_search_distance=200)
    dem[~land] = np.nan  # sea
    write(paths.raster("dem"), dem.astype("float32"), grid, nodata=np.nan,
          tags=dict(source="Copernicus GLO-90 DSM (EGM2008 heights)", resampling="bilinear"))
    write(paths.raster("dem_filled_mask"), void.astype("uint8"), grid, nodata=None)
    quicklook(dem, paths.qa / "02_dem.png", "Copernicus GLO-90 on Albers 90 m grid", cmap="terrain",
              vmin=0, vmax=6000, outline=(mask == 1))
    # tile-corner close-up in the plains, where seams would be most visible
    zoom(dem, grid, (71.8, 30.8, 72.2, 31.2), paths.qa / "02_dem_tilecorner_zoom.png",
         "tile corner 72E/31N (seam check)", cmap="terrain")
