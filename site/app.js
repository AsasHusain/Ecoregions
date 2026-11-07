// Revised Ecoregions of the World: local globe viewer.
// Data come from `python -m ecoregions globe` (site/data/manifest.json + per-country GeoJSON).
const OCEAN = '#0f2a3d', LAND = '#e9e4d8';

const map = new maplibregl.Map({
  container: 'map',
  style: {
    version: 8,
    projection: { type: 'globe' },
    sources: {},
    layers: [{ id: 'ocean', type: 'background', paint: { 'background-color': OCEAN } }],
  },
  center: [69, 28], zoom: 1.7, attributionControl: false,
});
map.addControl(new maplibregl.NavigationControl({ visualizePitch: true }), 'top-right');
map.addControl(new maplibregl.ScaleControl({ unit: 'metric' }), 'bottom-right');

const get = (url) => fetch(url).then((r) => { if (!r.ok) throw new Error(`${url}: HTTP ${r.status}`); return r.json(); });
const unitLayers = [], provLayers = [], markers = [];

map.on('load', async () => {
  let manifest = { countries: {} };
  try { manifest = await get('data/manifest.json'); } catch (e) { console.warn('no manifest yet', e); }

  map.addSource('world', { type: 'geojson', data: 'data/world.geojson' });
  map.addSource('resolve', { type: 'geojson', data: 'data/resolve.geojson' });
  map.addLayer({ id: 'land', type: 'fill', source: 'world', paint: { 'fill-color': LAND } });
  map.addLayer({ id: 'resolve-fill', type: 'fill', source: 'resolve',
    paint: { 'fill-color': ['coalesce', ['get', 'COLOR'], '#cccccc'], 'fill-opacity': 0.42 } });
  map.addLayer({ id: 'resolve-line', type: 'line', source: 'resolve',
    paint: { 'line-color': '#ffffff', 'line-width': 0.4, 'line-opacity': 0.5 } });

  for (const [iso, c] of Object.entries(manifest.countries)) {
    const f = c.files;
    map.addSource(`${iso}-units`, { type: 'geojson', data: `data/${f.units}` });
    map.addSource(`${iso}-prov`, { type: 'geojson', data: `data/${f.provinces}` });
    map.addSource(`${iso}-gray`, { type: 'geojson', data: `data/${f.gray}` });
    map.addSource(`${iso}-outline`, { type: 'geojson', data: `data/${f.outline}` });
    map.addLayer({ id: `${iso}-units-fill`, type: 'fill', source: `${iso}-units`,
      paint: { 'fill-color': ['get', 'color'], 'fill-opacity': 0.88 } });
    map.addLayer({ id: `${iso}-units-line`, type: 'line', source: `${iso}-units`,
      paint: { 'line-color': '#ffffff', 'line-width': ['interpolate', ['linear'], ['zoom'], 3, 0.15, 8, 0.9] } });
    map.addLayer({ id: `${iso}-gray-fill`, type: 'fill', source: `${iso}-gray`, paint: { 'fill-color': '#b9b9b4' } });
    map.addLayer({ id: `${iso}-prov-line`, type: 'line', source: `${iso}-prov`,
      paint: { 'line-color': '#1b1b1a', 'line-width': ['interpolate', ['linear'], ['zoom'], 3, 0.6, 8, 1.8] } });
    map.addLayer({ id: `${iso}-outline-line`, type: 'line', source: `${iso}-outline`,
      paint: { 'line-color': '#1b1b1a', 'line-width': 1.6 } });
    unitLayers.push(`${iso}-units-fill`, `${iso}-units-line`);
    provLayers.push(`${iso}-prov-line`);
    if (f.labels) {
      for (const l of (await get(`data/${f.labels}`)).labels) {
        const el = document.createElement('div');
        el.className = `lbl ${l.kind}`;
        el.textContent = l.text;
        markers.push(new maplibregl.Marker({ element: el }).setLngLat([l.lon, l.lat]).addTo(map));
      }
    }
  }
  map.addLayer({ id: 'borders', type: 'line', source: 'world',
    paint: { 'line-color': '#5b5b55', 'line-width': 0.6, 'line-opacity': 0.8 } });

  // country list
  const ul = document.getElementById('countries');
  const entries = Object.entries(manifest.countries);
  if (!entries.length) ul.innerHTML = '<li class="empty">None exported yet</li>';
  for (const [iso, c] of entries) {
    const li = document.createElement('li');
    li.innerHTML = `<span class="name">${c.name}</span><span class="st">${c.status}<br>${c.units} units · ${c.updated}</span>`;
    li.onclick = () => map.fitBounds(c.bbox, { padding: 60, duration: 2200 });
    ul.appendChild(li);
  }
  if (entries.length === 1) map.fitBounds(entries[0][1].bbox, { padding: 80, duration: 0 });

  wireToggles();
  wireTooltip();
  map.on('moveend', declutter);
  declutter();
});

// Hide labels that would overlap one already shown (first come first served); re-run after every move.
function declutter() {
  const shown = [];
  for (const m of markers) {
    const el = m.getElement();
    el.style.visibility = 'visible';
    const r = el.getBoundingClientRect();
    if (!r.width) continue; // labels switched off
    if (shown.some((s) => r.left < s.right && r.right > s.left && r.top < s.bottom && r.bottom > s.top)) {
      el.style.visibility = 'hidden';
    } else shown.push(r);
  }
}

function setVis(ids, on) { for (const id of ids) if (map.getLayer(id)) map.setLayoutProperty(id, 'visibility', on ? 'visible' : 'none'); }
function wireToggles() {
  const t = (id, fn) => document.getElementById(id).addEventListener('change', (e) => fn(e.target.checked));
  t('t-units', (on) => setVis(unitLayers, on));
  t('t-prov', (on) => setVis(provLayers, on));
  t('t-labels', (on) => markers.forEach((m) => { m.getElement().style.display = on ? '' : 'none'; }));
  t('t-resolve', (on) => setVis(['resolve-fill', 'resolve-line'], on));
  t('t-borders', (on) => setVis(['borders'], on));
}

const esc = (s) => String(s ?? '').replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
function wireTooltip() {
  const tip = document.getElementById('tip');
  map.on('mousemove', (e) => {
    const layers = [...unitLayers.filter((l) => l.endsWith('-fill')), 'resolve-fill'].filter((l) => map.getLayer(l));
    const f = map.queryRenderedFeatures(e.point, { layers })[0];
    if (!f) { tip.hidden = true; return; }
    const p = f.properties;
    tip.innerHTML = f.layer.id === 'resolve-fill'
      ? `<h2>${esc(p.ECO_NAME)}</h2><div class="meta">RESOLVE 2017 baseline · ${esc(p.BIOME_NAME)}</div>`
      : `<h2>${esc(p.name)}</h2><div class="meta">${esc(p.code)} · ${esc(p.l2_name)} · ${Math.round(p.area_km2).toLocaleString()} km²</div>` +
        (p.description ? `<div>${esc(p.description)}</div>` : '');
    tip.hidden = false;
    tip.style.left = `${Math.min(e.point.x + 14, window.innerWidth - 360)}px`;
    tip.style.top = `${e.point.y + 14}px`;
  });
  map.on('mouseout', () => { tip.hidden = true; });
}

// ?debug: log rendered feature counts per layer once the map is idle (for headless checks)
if (location.search.includes('debug')) {
  map.on('idle', () => console.log('IDLE ' + JSON.stringify(Object.fromEntries(
    map.getStyle().layers.filter((l) => l.type !== 'background')
      .map((l) => [l.id, map.queryRenderedFeatures({ layers: [l.id] }).length])))));
}
