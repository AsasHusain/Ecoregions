"""Stage 14 — interactive HTML map (MapLibre GL JS).

A single self-contained file, ``outputs/ecoregions_map.html``: all layers and a
hillshade image are embedded, so it opens offline from disk. Geometries are
simplified for display only (``maps.web_simplify_m``, shared borders kept
identical); the GeoPackage holds the full-resolution boundaries.
"""
from __future__ import annotations

import base64
import io
import json
import logging

import geopandas as gpd
import numpy as np
import shapely
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform, reproject, transform_bounds

from .colors import l3_colors
from .config import Paths
from .grid import Grid, read

log = logging.getLogger(__name__)

MAPLIBRE = "5.6.0"
POPUP_FIELDS = ["code", "name", "l2_name", "l1_name", "area_km2", "elev_p10", "elev_median", "elev_p90", "slope_mean",
                "landform_hammond", "litho_dominant", "geo_unit_dominant", "soil_summary", "climate_summary",
                "hand_mix", "plains_share", "gbif_records", "gbif_species", "gbif_records_per_100km2", "gbif_sampling",
                "confidence", "name_basis", "review_flag", "also_contains", "description"]


def hillshade(z: np.ndarray, res: float, az: float = 315, alt: float = 45, zf: float = 2.0) -> np.ndarray:
    z = np.nan_to_num(z, nan=0.0) * zf
    gy, gx = np.gradient(z, res)
    slope = np.arctan(np.hypot(gx, gy))
    aspect = np.arctan2(-gx, gy)
    a, b = np.radians(360 - az + 90), np.radians(alt)
    hs = np.sin(b) * np.cos(slope) + np.cos(b) * np.sin(slope) * np.cos(a - aspect)
    return np.clip(hs, 0, 1)


def hillshade_png(paths: Paths, cfg, grid: Grid) -> tuple[str, list]:
    cg = grid.coarsen(cfg["grid"]["coarse_factor"])
    elev = read(paths.interim / "coarse" / "elev.tif")
    hs = hillshade(elev, cg.res)
    hs[np.isnan(elev)] = np.nan
    dst_crs = "EPSG:3857"
    res = cfg["maps"]["hillshade_res_m"]
    tr, w, h = calculate_default_transform(cg.rio_crs, dst_crs, cg.width, cg.height, *cg.bounds, resolution=res)
    out = np.full((h, w), np.nan, "float32")
    reproject(hs.astype("float32"), out, src_transform=cg.transform, src_crs=cg.rio_crs, dst_transform=tr,
              dst_crs=dst_crs, resampling=Resampling.bilinear, src_nodata=np.nan, dst_nodata=np.nan)
    # grayscale with alpha: shadows dark, flat ground transparent
    shadow = np.nan_to_num(1 - out / max(np.nanpercentile(out, 99), 1e-6), nan=0)
    shadow = np.clip(shadow, 0, 1)
    # palette PNG: one near-black colour at 32 transparency levels (~10x smaller than RGBA)
    levels = 24
    idx = np.clip(np.round(shadow ** 0.9 * (levels - 1)), 0, levels - 1).astype("uint8")
    from PIL import Image

    img = Image.fromarray(idx, "P")
    img.putpalette([20, 20, 20] * levels)
    alpha = bytes(int(i / (levels - 1) * 210) for i in range(levels))
    buf = io.BytesIO()
    img.save(buf, "PNG", optimize=True, transparency=alpha)
    w_, s_, e_, n_ = transform_bounds(dst_crs, "EPSG:4326", tr.c, tr.f + h * tr.e, tr.c + w * tr.a, tr.f)
    coords = [[w_, n_], [e_, n_], [e_, s_], [w_, s_]]
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode(), coords


def _geojson(gdf: gpd.GeoDataFrame, tol: float, cols: list[str]) -> dict:
    g = gdf.copy()
    if tol:
        g["geometry"] = shapely.coverage_simplify(g.geometry.values, tolerance=tol)
    g = g.to_crs(4326)
    g["geometry"] = shapely.set_precision(g.geometry.values, 1e-3)  # ~100 m: below the display simplification
    g = g[~g.is_empty]
    return json.loads(g[cols + ["geometry"]].to_json(drop_id=True))


def run(cfg) -> None:
    paths = Paths(cfg)
    grid = Grid.load(paths.grid_json)
    gpkg = paths.outputs / "ecoregions.gpkg"
    mc = cfg["maps"]
    l3 = gpd.read_file(gpkg, layer="l3_ecoregions")
    l3["color"] = l3_colors(l3, mc["family_colors"])
    l2 = gpd.read_file(gpkg, layer="l2_provinces")
    l1 = gpd.read_file(gpkg, layer="l1_resolve")
    gray = gpd.read_file(gpkg, layer="disputed_gray")
    riv = gpd.read_file(gpkg, layer="rivers")
    ctx = gpd.read_file(paths.vector("boundaries"), layer="countries_context")
    riv = riv[riv.UPLAND_SKM >= 3000]
    tol = mc["web_simplify_m"]
    cols = [c for c in POPUP_FIELDS if c in l3.columns] + ["color", "l2_family"]
    for c in l3.select_dtypes("float").columns:
        l3[c] = l3[c].round(2)
    data = dict(
        l3=_geojson(l3, tol, cols),
        l2=_geojson(l2, tol, ["code", "name", "family", "area_km2", "n_l3_units"]),
        l1=_geojson(l1, tol, ["ECO_ID", "ECO_NAME", "BIOME_NAME"]),
        gray=_geojson(gray, tol, ["label"]),
        land=_geojson(ctx, 2000, ["NAME"]),
        rivers=_geojson(riv.assign(geometry=riv.simplify(tol)), 0, ["UPLAND_SKM"]),
    )
    labels = []
    for _, r in l2.to_crs(4326).iterrows():
        p = r.geometry.representative_point() if r.geometry else None
        labels.append(dict(code=r.code, name=r["name"], lon=p.x, lat=p.y))
    gl = []
    for _, r in gray[gray.label.fillna("") != ""].to_crs(4326).iterrows():
        p = r.geometry.representative_point()
        gl.append(dict(label=r.label, lon=p.x, lat=p.y))
    img, coords = hillshade_png(paths, cfg, grid)
    legend = []
    for _, r in l2.sort_values("code").iterrows():
        legend.append(dict(code=r.code, name=r["name"], color=mc["family_colors"][r.family], family=r.family))
    bounds = list(gpd.read_file(gpkg, layer="study_area").to_crs(4326).total_bounds)
    html = TEMPLATE
    for k, v in {"__DATA__": json.dumps(data, separators=(",", ":")), "__LABELS__": json.dumps(labels),
                 "__GRAYLABELS__": json.dumps(gl), "__HILLSHADE__": img, "__HSCOORDS__": json.dumps(coords),
                 "__LEGEND__": json.dumps(legend), "__BOUNDS__": json.dumps(bounds), "__GRAY__": mc["gray"],
                 "__RIVER__": mc["river_color"], "__MAPLIBRE__": MAPLIBRE, "__N__": str(len(l3)), "__VERSION__": cfg["project"]["version"],
                 "__NL2__": str(len(l2))}.items():
        html = html.replace(k, v)
    out = paths.outputs / "ecoregions_map.html"
    out.write_text(html, encoding="utf-8")  # Windows defaults to cp1252
    log.info("web map %s (%.1f MB)", out, out.stat().st_size / 1e6)


TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Revised Ecoregions of Pakistan — v__VERSION__</title>
<link rel="stylesheet" href="https://unpkg.com/maplibre-gl@__MAPLIBRE__/dist/maplibre-gl.css">
<script src="https://unpkg.com/maplibre-gl@__MAPLIBRE__/dist/maplibre-gl.js"></script>
<style>
:root {
  color-scheme: light;
  --surface: #fcfcfb; --panel: #ffffffee; --ink: #1b1b1a; --ink-2: #52514e; --ink-3: #7a7974;
  --rule: #dcdbd6; --sea: #dfe8ee; --land: #f1efe9; --accent: #2a78d6;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    color-scheme: dark;
    --surface: #1a1a19; --panel: #232322ee; --ink: #f0efec; --ink-2: #c3c2b7; --ink-3: #9a998f;
    --rule: #3a3a37; --sea: #1f2a31; --land: #2b2a27; --accent: #3987e5;
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --surface: #1a1a19; --panel: #232322ee; --ink: #f0efec; --ink-2: #c3c2b7; --ink-3: #9a998f;
  --rule: #3a3a37; --sea: #1f2a31; --land: #2b2a27; --accent: #3987e5;
}
* { box-sizing: border-box; }
html, body { margin: 0; height: 100%; background: var(--surface); color: var(--ink);
  font: 13px/1.45 system-ui, -apple-system, "Segoe UI", sans-serif; }
#map { position: absolute; inset: 0; }
.panel { position: absolute; background: var(--panel); border: 1px solid var(--rule); border-radius: 8px;
  padding: 10px 12px; backdrop-filter: blur(4px); box-shadow: 0 2px 10px #0002; }
#ctl { top: 10px; left: 10px; width: 270px; max-height: calc(100% - 20px); overflow: auto; }
#info { top: 10px; right: 10px; width: 340px; max-height: calc(100% - 20px); overflow: auto; }
h1 { font-size: 15px; margin: 0 0 2px; }
.sub { color: var(--ink-2); font-size: 12px; margin-bottom: 8px; }
label { display: flex; gap: 6px; align-items: center; margin: 3px 0; cursor: pointer; }
select { width: 100%; margin: 4px 0 8px; padding: 3px; background: var(--surface); color: var(--ink);
  border: 1px solid var(--rule); border-radius: 4px; }
.legend div { display: flex; gap: 6px; align-items: flex-start; margin: 2px 0; font-size: 12px; }
.sw { flex: 0 0 14px; height: 14px; border-radius: 3px; margin-top: 1px; border: 1px solid #0003; }
.k { color: var(--ink-3); font-size: 11px; text-transform: uppercase; letter-spacing: .03em; margin-top: 6px; }
.v { color: var(--ink); }
#info h2 { font-size: 15px; margin: 0; }
#info .code { color: var(--ink-2); font-size: 12px; }
.flag { display: inline-block; font-size: 11px; padding: 1px 6px; border-radius: 10px; border: 1px solid var(--rule);
  color: var(--ink-2); margin-top: 4px; }
.desc { margin-top: 8px; color: var(--ink); }
.ramp { height: 10px; border-radius: 3px; margin: 4px 0 2px; }
.ramp-l { display: flex; justify-content: space-between; color: var(--ink-2); font-size: 11px; }
.lbl { font: 600 10px system-ui, sans-serif; color: #1b1b1a; text-shadow: 0 0 3px #fff, 0 0 3px #fff, 0 0 2px #fff;
  white-space: nowrap; pointer-events: none; }
.glbl { font: italic 11px system-ui, sans-serif; color: #3d3d3a; text-shadow: 0 0 3px #eee, 0 0 3px #eee;
  max-width: 160px; white-space: normal; text-align: center; pointer-events: none; }
button.theme { float: right; font-size: 11px; background: none; border: 1px solid var(--rule); color: var(--ink-2);
  border-radius: 4px; cursor: pointer; }
@media (max-width: 720px) {
  #ctl { width: calc(100% - 20px); max-height: 40%; }
  #info { top: auto; bottom: 10px; left: 10px; right: 10px; width: auto; max-height: 40%; }
}
</style>
</head>
<body>
<div id="map"></div>
<div id="ctl" class="panel">
  <button class="theme" id="theme">◐</button>
  <h1>Revised Ecoregions of Pakistan — v__VERSION__</h1>
  <div class="sub">__N__ Level III units within __NL2__ Level II provinces. Hover for details.</div>
  <div class="k">Colour by</div>
  <select id="colorby">
    <option value="family">L2 province family</option>
    <option value="confidence">Confidence score</option>
    <option value="gbif">GBIF records per 100 km²</option>
    <option value="review">Name status</option>
  </select>
  <div class="k">Layers</div>
  <label><input type="checkbox" id="t-l3" checked> L3 units (fill)</label>
  <label><input type="checkbox" id="t-hs" checked> Hillshade</label>
  <label><input type="checkbox" id="t-riv" checked> Major rivers</label>
  <label><input type="checkbox" id="t-l2" checked> L2 province outlines</label>
  <label><input type="checkbox" id="t-l1"> L1 RESOLVE ecoregion outlines</label>
  <label><input type="checkbox" id="t-lbl" checked> L2 labels</label>
  <div class="k">Legend</div>
  <div class="legend" id="legend"></div>
  <div class="sub" style="margin-top:8px">Gray: areas outside the study area (administered by India or China);
  no ecoregions are drawn there. Boundaries simplified for display; see the
  GeoPackage for 90 m lines.</div>
</div>
<div id="info" class="panel"><div class="sub">Hover over a unit.</div></div>
<script>
const DATA = __DATA__;
const LABELS = __LABELS__;
const GRAYLABELS = __GRAYLABELS__;
const LEGEND = __LEGEND__;
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();

const map = new maplibregl.Map({
  container: 'map',
  style: { version: 8, sources: {}, layers: [{ id: 'bg', type: 'background', paint: { 'background-color': css('--sea') } }] },
  bounds: __BOUNDS__, fitBoundsOptions: { padding: 30 }, attributionControl: false
});
map.addControl(new maplibregl.NavigationControl({ showCompass: false }), 'bottom-right');
map.addControl(new maplibregl.ScaleControl({ unit: 'metric' }), 'bottom-left');
map.addControl(new maplibregl.AttributionControl({ compact: true, customAttribution:
  'Copernicus GLO-90 · HydroSHEDS · USGS geo8ag · SoilGrids · CHELSA · ESA WorldCover · RESOLVE · GBIF · Natural Earth' }));

const SEQ = ['#cde2fb', '#86b6ef', '#3987e5', '#1c5cab', '#0d366b'];
function fillColor(mode) {
  if (mode === 'confidence') return ['interpolate', ['linear'], ['get', 'confidence'], 0.2, SEQ[0], 0.4, SEQ[1], 0.55, SEQ[2], 0.7, SEQ[3], 0.85, SEQ[4]];
  if (mode === 'gbif') return ['interpolate', ['linear'], ['log10', ['max', ['coalesce', ['get', 'gbif_records_per_100km2'], 0.01], 0.01]],
                                -1, SEQ[0], 0, SEQ[1], 1, SEQ[2], 2, SEQ[3], 3, SEQ[4]];
  if (mode === 'review') return ['case', ['==', ['get', 'name_basis'], 'gazetteer'], '#3987e5', '#eb6834'];
  return ['get', 'color'];
}
function legendHTML(mode) {
  if (mode === 'confidence') return `<div class="ramp" style="background:linear-gradient(90deg,${SEQ.join(',')})"></div><div class="ramp-l"><span>0.2 low</span><span>0.85 high</span></div>`;
  if (mode === 'gbif') return `<div class="ramp" style="background:linear-gradient(90deg,${SEQ.join(',')})"></div><div class="ramp-l"><span>&lt;0.1</span><span>1</span><span>10</span><span>100</span><span>1000+</span></div><div class="sub">GBIF occurrence records per 100 km² (log scale). Pale = under-sampled.</div>`;
  if (mode === 'review') return `<div><span class="sw" style="background:#3987e5"></span>Named from a known landform / place</div><div><span class="sw" style="background:#eb6834"></span>Provisional name — needs review</div>`;
  return LEGEND.map(l => `<div><span class="sw" style="background:${l.color}"></span><span><b>${l.code}</b> ${l.name}</span></div>`).join('');
}

map.on('load', () => {
  map.addSource('l3', { type: 'geojson', data: DATA.l3, generateId: true });
  map.addSource('l2', { type: 'geojson', data: DATA.l2 });
  map.addSource('l1', { type: 'geojson', data: DATA.l1 });
  map.addSource('gray', { type: 'geojson', data: DATA.gray });
  map.addSource('land', { type: 'geojson', data: DATA.land });
  map.addSource('rivers', { type: 'geojson', data: DATA.rivers });
  map.addSource('hs', { type: 'image', url: '__HILLSHADE__', coordinates: __HSCOORDS__ });
  map.addLayer({ id: 'land', type: 'fill', source: 'land', paint: { 'fill-color': css('--land') } });
  map.addLayer({ id: 'land-line', type: 'line', source: 'land', paint: { 'line-color': '#a9a8a2', 'line-width': 0.6 } });
  map.addLayer({ id: 'gray', type: 'fill', source: 'gray', paint: { 'fill-color': '__GRAY__', 'fill-opacity': 1 } });
  map.addLayer({ id: 'l3-fill', type: 'fill', source: 'l3', paint: { 'fill-color': fillColor('family'), 'fill-opacity': 0.82 } });
  map.addLayer({ id: 'hs', type: 'raster', source: 'hs', paint: { 'raster-opacity': 0.75, 'raster-fade-duration': 0 } });
  map.addLayer({ id: 'l3-line', type: 'line', source: 'l3', paint: { 'line-color': '#ffffff', 'line-width': 0.6, 'line-opacity': 0.8 } });
  map.addLayer({ id: 'rivers', type: 'line', source: 'rivers', paint: { 'line-color': '__RIVER__',
     'line-width': ['interpolate', ['linear'], ['get', 'UPLAND_SKM'], 3000, 0.6, 100000, 1.6, 500000, 2.6] } });
  map.addLayer({ id: 'l1', type: 'line', source: 'l1', layout: { visibility: 'none' },
     paint: { 'line-color': '#c0392b', 'line-width': 1.4, 'line-dasharray': [3, 2] } });
  map.addLayer({ id: 'l2', type: 'line', source: 'l2', paint: { 'line-color': '#1b1b1a', 'line-width': 1.6 } });
  map.addLayer({ id: 'gray-line', type: 'line', source: 'gray', paint: { 'line-color': '#8d8d88', 'line-width': 0.8 } });
  map.addLayer({ id: 'l3-hover', type: 'line', source: 'l3', paint: { 'line-color': '#000', 'line-width':
     ['case', ['boolean', ['feature-state', 'hover'], false], 2.6, 0] } });

  const l2markers = LABELS.map(l => { const el = document.createElement('div'); el.className = 'lbl';
    el.textContent = l.code + ' ' + l.name.replace(/ \(.*\)/, ''); return new maplibregl.Marker({ element: el }).setLngLat([l.lon, l.lat]).addTo(map); });
  GRAYLABELS.forEach(l => { const el = document.createElement('div'); el.className = 'glbl'; el.textContent = l.label;
    new maplibregl.Marker({ element: el }).setLngLat([l.lon, l.lat]).addTo(map); });

  const vis = (id, on) => map.setLayoutProperty(id, 'visibility', on ? 'visible' : 'none');
  document.getElementById('t-l3').onchange = e => vis('l3-fill', e.target.checked);
  document.getElementById('t-hs').onchange = e => vis('hs', e.target.checked);
  document.getElementById('t-riv').onchange = e => vis('rivers', e.target.checked);
  document.getElementById('t-l2').onchange = e => vis('l2', e.target.checked);
  document.getElementById('t-l1').onchange = e => vis('l1', e.target.checked);
  document.getElementById('t-lbl').onchange = e => l2markers.forEach(m => m.getElement().style.display = e.target.checked ? '' : 'none');
  const sel = document.getElementById('colorby');
  const setMode = m => { map.setPaintProperty('l3-fill', 'fill-color', fillColor(m)); document.getElementById('legend').innerHTML = legendHTML(m); };
  sel.onchange = () => setMode(sel.value);
  setMode('family');

  let hovered = null;
  const fmt = (v, d = 0) => (v === null || v === undefined || Number.isNaN(v)) ? '–' : Number(v).toLocaleString(undefined, { maximumFractionDigits: d, minimumFractionDigits: d });
  map.on('mousemove', 'l3-fill', e => {
    const f = e.features[0];
    if (hovered !== null) map.setFeatureState({ source: 'l3', id: hovered }, { hover: false });
    hovered = f.id; map.setFeatureState({ source: 'l3', id: hovered }, { hover: true });
    map.getCanvas().style.cursor = 'pointer';
    const p = f.properties;
    const row = (k, v) => `<div class="k">${k}</div><div class="v">${v}</div>`;
    document.getElementById('info').innerHTML =
      `<h2>${p.name}</h2><div class="code">L3 ${p.code} · ${p.l2_name}</div>` +
      (p.review_flag === true || p.review_flag === 'true' ? '<span class="flag">provisional name — review</span>' : '') +
      `<div class="desc">${p.description}</div>` +
      row('Area', fmt(p.area_km2) + ' km²') +
      row('Elevation', `${fmt(p.elev_p10)}–${fmt(p.elev_p90)} m (median ${fmt(p.elev_median)})`) +
      row('Landform', `${p.landform_hammond}; mean slope ${fmt(p.slope_mean, 1)}°`) +
      row('Geology', `${p.litho_dominant} (geo8ag ${p.geo_unit_dominant})`) +
      row('Soil', p.soil_summary) + row('Climate', p.climate_summary) +
      row('Height above drainage', p.hand_mix) +
      row('GBIF', `${fmt(p.gbif_records)} records, ${fmt(p.gbif_species)} species (${fmt(p.gbif_records_per_100km2, 1)} per 100 km²) — ${p.gbif_sampling}`) +
      row('L1 parent (RESOLVE)', p.l1_name) +
      row('Confidence', fmt(p.confidence, 2)) +
      (p.also_contains ? row('Also contains', p.also_contains) : '');
  });
  map.on('mouseleave', 'l3-fill', () => {
    if (hovered !== null) map.setFeatureState({ source: 'l3', id: hovered }, { hover: false });
    hovered = null; map.getCanvas().style.cursor = '';
  });
});

document.getElementById('theme').onclick = () => {
  const r = document.documentElement;
  const dark = r.getAttribute('data-theme') ? r.getAttribute('data-theme') === 'dark' : matchMedia('(prefers-color-scheme: dark)').matches;
  r.setAttribute('data-theme', dark ? 'light' : 'dark');
  if (map.getLayer('bg')) map.setPaintProperty('bg', 'background-color', css('--sea'));
  if (map.getLayer('land')) map.setPaintProperty('land', 'fill-color', css('--land'));
};
</script>
</body>
</html>
"""
