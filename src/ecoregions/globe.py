"""Export to the local globe site (``site/``): the world, RESOLVE 2017 as the
baseline for countries not yet revised, and every finished country's layers.

Countries are added one at a time: each run exports the country named in
``globe.country`` from ``outputs/ecoregions.gpkg`` to ``site/data/<ISO3>/`` and
records it in ``site/data/manifest.json``; entries for other countries are kept.
World and RESOLVE layers are built once (delete them to rebuild). Open the site
with ``python -m http.server 8000 --directory site``. Only derived products are
written (no raw licensed data).
"""
from __future__ import annotations

import datetime as dt
import json
import logging

import geopandas as gpd
import shapely

from .colors import l3_colors
from .config import Paths
from .webmap import POPUP_FIELDS, _geojson

log = logging.getLogger(__name__)


def _write(obj: dict, path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, separators=(",", ":")), encoding="utf-8")
    log.info("globe: %s (%.1f MB)", path.name, path.stat().st_size / 1e6)


def _world_layer(gdf: gpd.GeoDataFrame, cols: list[str], tol_deg: float) -> dict:
    """Global layer in lon/lat, simplified per polygon (display only).
    shortcut: GeoJSON for the whole world; switch to PMTiles once many countries are finished."""
    g = gdf[cols + ["geometry"]].to_crs(4326)
    g["geometry"] = shapely.set_precision(shapely.make_valid(g.geometry.simplify(tol_deg, preserve_topology=True).values), 1e-3)
    g = g[~g.is_empty]
    return json.loads(g.to_json(drop_id=True))


def run(cfg) -> None:
    paths = Paths(cfg)
    gc = cfg["globe"]
    data = paths.root / gc["site_dir"] / "data"

    if not (data / "world.geojson").exists():
        ne = gpd.read_file(paths.raw / "naturalearth" / "ne_10m_admin_0_countries.shp")
        _write(_world_layer(ne, ["ADM0_A3", "NAME"], gc["world_simplify_deg"]), data / "world.geojson")
    if not (data / "resolve.geojson").exists():
        r = gpd.read_file(f"/vsizip/{paths.raw / 'resolve' / 'Ecoregions2017.zip'}/Ecoregions2017.shp")
        r = r[r.ECO_ID > 0]
        _write(_world_layer(r, ["ECO_NAME", "BIOME_NAME", "REALM", "COLOR"], gc["world_simplify_deg"]),
               data / "resolve.geojson")

    c = gc["country"]
    iso = c["iso3"]
    gpkg = paths.outputs / "ecoregions.gpkg"
    tol = cfg["maps"]["web_simplify_m"]
    import pyogrio

    if "ecoregions" in [n for n, _ in pyogrio.list_layers(gpkg)]:  # v0.2: community ecoregions in faunal regions
        from .communities import load_rules

        types = load_rules(paths.root / cfg["communities"]["rules"])["ecoregion_types"]
        l3 = gpd.read_file(gpkg, layer="ecoregions")
        l3["color"] = l3["type"].map(lambda t: types[t]["color"])
        l3["description"] = l3.biome + " · " + l3.community_mix
        l3["l2_name"] = l3.faunal_name
        cols = ["code", "name", "type_name", "biome", "faunal_name", "l2_name", "area_km2", "natural_share",
                "confidence", "community_mix", "description", "color"]
        prov = gpd.read_file(gpkg, layer="faunal_regions").rename(columns={"faunal_name": "name"})
        prov["code"] = prov.faunal_region.astype(str)
    else:
        l3 = gpd.read_file(gpkg, layer="l3_ecoregions")
        l3["color"] = l3_colors(l3, cfg["maps"]["family_colors"])
        cols = [x for x in POPUP_FIELDS if x in l3.columns] + ["color"]
        prov = gpd.read_file(gpkg, layer="l2_provinces")
    layers = {
        "units": (l3, tol, cols),
        "provinces": (prov, tol, ["code", "name"]),
        "gray": (gpd.read_file(gpkg, layer="disputed_gray"), tol, ["label"]),
        "outline": (gpd.read_file(gpkg, layer="study_area"), tol, []),
    }
    files = {}
    for name, (g, t, cl) in layers.items():
        _write(_geojson(g, t, cl), data / iso / f"{name}.geojson")
        files[name] = f"{iso}/{name}.geojson"
    # label anchor points (the site draws labels as HTML markers: no font server needed)
    pts = []
    for kind, g, col in (("province", layers["provinces"][0], "name"), ("gray", layers["gray"][0], "label")):
        g = g[g[col].fillna("") != ""]
        p4 = g.geometry.representative_point().to_crs(4326)
        pts += [dict(kind=kind, text=t.split(" (")[0] if kind == "province" else t,
                     code=(r.get("code") or ""), lon=round(pt.x, 4), lat=round(pt.y, 4))
                for (_, r), t, pt in zip(g.iterrows(), g[col], p4)]
    _write({"labels": pts}, data / iso / "labels.json")
    files["labels"] = f"{iso}/labels.json"

    mpath = data / "manifest.json"
    manifest = json.loads(mpath.read_text(encoding="utf-8")) if mpath.exists() else {"countries": {}}
    manifest["countries"][iso] = dict(
        name=c["name"], status=c["status"], version=cfg["project"]["version"], units=len(l3),
        bbox=[round(v, 3) for v in l3.to_crs(4326).total_bounds], files=files,
        updated=dt.date.today().isoformat())
    _write(manifest, mpath)
    log.info("globe: %s exported (%d units); serve with: python -m http.server 8000 --directory %s",
             c["name"], len(l3), gc["site_dir"])
