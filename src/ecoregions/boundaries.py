"""Stage 1 — study area, country shapes for display and the analysis grid.

Study area: the countries in ``grid.countries`` and the states or provinces in ``grid.states``
(ISO 3166-2 codes), both Natural Earth 10 m. A region with Pakistan (incl. Azad Kashmir and
Gilgit-Baltistan) also maps the Siachen Glacier area, Jammu and Kashmir (incl. Ladakh), Demchok,
Aksai Chin and the Shaksgam Valley.

Display: Pakistan's border is drawn solid around Pakistan, the Siachen Glacier area,
Jammu and Kashmir and Demchok; the line where Jammu and Kashmir
meets the rest of Pakistan is dotted. Aksai Chin and the Shaksgam Valley are drawn
as part of China. The country shapes of Pakistan, India and China are adjusted to
match (``display_countries``) wherever countries are drawn.
"""
from __future__ import annotations

import logging

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from shapely.geometry import box

from .config import Paths
from .grid import Grid, rasterize_geoms, write

log = logging.getLogger(__name__)

# Natural Earth disputed-area codes (BRK_A3)
WITH_PAKISTAN = ["B45", "B05", "B03"]  # Siachen Glacier, Jammu and Kashmir, Demchok
DOTTED = ["B05", "B03"]                # their line with the rest of Pakistan is dotted
WITH_CHINA = ["B07", "B06"]                   # Aksai Chin, Shaksgam Valley
# changing these changes the grid: rebuild every stage from `grid` on


def load_boundaries(paths: Paths):
    ne = paths.raw / "naturalearth"
    countries = gpd.read_file(ne / "ne_10m_admin_0_countries.shp")
    disputed = gpd.read_file(ne / "ne_10m_admin_0_disputed_areas.shp")
    return countries, disputed


def mapped_states(cfg, paths: Paths) -> gpd.GeoDataFrame:
    """The states or provinces in ``grid.states`` (Natural Earth admin-1, by ISO 3166-2 code; lon/lat)."""
    codes = cfg["grid"].get("states", [])
    if not codes:
        return gpd.GeoDataFrame({"admin": []}, geometry=[], crs=4326)
    st = gpd.read_file(paths.raw / "naturalearth" / "ne_10m_admin_1_states_provinces.shp")
    st = st[st.iso_3166_2.isin(codes)]
    if missing := set(codes) - set(st.iso_3166_2):
        raise ValueError(f"grid.states: no Natural Earth admin-1 unit {sorted(missing)}")
    return st


def mapped_area(cfg, paths: Paths, countries: gpd.GeoDataFrame, disputed: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """The area this region maps (its countries, states and, with Pakistan, the areas drawn with it), without a
    parcel's overlap strip (lon/lat, one row)."""
    mapped = cfg["grid"].get("countries", [])
    states = mapped_states(cfg, paths)
    study = pd.concat([countries.loc[countries.ADM0_A3.isin(mapped), ["geometry"]], states[["geometry"]]]
                      + ([disputed.loc[disputed.BRK_A3.isin(WITH_PAKISTAN + WITH_CHINA), ["geometry"]]] if "PAK" in mapped else []))
    study["name"] = cfg["project"]["area"]
    return study.dissolve().reset_index(drop=True)


def display_countries(countries: gpd.GeoDataFrame, disputed: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Country shapes as drawn: Pakistan with the areas in ``WITH_PAKISTAN``, China with
    ``WITH_CHINA``, India without either (lon/lat, same columns as ``countries``)."""
    out = countries.copy()
    geo = lambda codes: disputed.loc[disputed.BRK_A3.isin(codes)].union_all()  # noqa: E731
    pak, chn = geo(WITH_PAKISTAN), geo(WITH_CHINA)
    for a3, add in (("PAK", pak), ("CHN", chn)):
        i = out.index[out.ADM0_A3 == a3]
        out.loc[i, "geometry"] = out.loc[i].geometry.union(add).make_valid()
    i = out.index[out.ADM0_A3 == "IND"]
    out.loc[i, "geometry"] = out.loc[i].geometry.difference(pak.union(chn)).make_valid()
    return out


def country_parts(cfg, paths: Paths) -> gpd.GeoDataFrame:
    """Country of each part of the study area, for the country shares of ecoregions (grid CRS):
    the mapped countries; mapped states as their country; with Pakistan, Jammu and Kashmir with Demchok
    as "Kashmir", the Siachen Glacier area as Pakistan, Aksai Chin and the Shaksgam Valley as China."""
    countries, disputed = load_boundaries(paths)
    geo = lambda codes: disputed.loc[disputed.BRK_A3.isin(codes)].union_all()  # noqa: E731
    codes = cfg["grid"].get("countries", [])
    mapped = countries[countries.ADM0_A3.isin(codes)]
    st = mapped_states(cfg, paths)
    parts = list(zip(mapped.NAME, mapped.geometry)) + list(zip(st.admin, st.geometry))
    if "PAK" in codes:
        parts += [("Pakistan", geo(["B45"])), ("Kashmir", geo(DOTTED)), ("China", geo(WITH_CHINA))]
    return gpd.GeoDataFrame({"country": [c for c, _ in parts]}, geometry=[g for _, g in parts],
                            crs=countries.crs).to_crs(cfg["crs"]["proj4"])


def border_lines(countries: gpd.GeoDataFrame, disputed: gpd.GeoDataFrame, crs, mapped: list[str],
                 states: gpd.GeoDataFrame | None = None) -> gpd.GeoDataFrame:
    """Borders of the mapped countries and states as drawn: ``solid`` around each (Pakistan extended),
    ``dotted`` where Jammu and Kashmir (with Demchok) meets the rest of Pakistan."""
    pak = "PAK" in mapped
    core = pd.concat([countries.loc[countries.ADM0_A3 == "PAK", ["geometry"]],
                      disputed.loc[disputed.BRK_A3 == "B45", ["geometry"]]]).to_crs(crs).union_all() if pak else shapely.Polygon()
    jk = disputed.loc[disputed.BRK_A3.isin(DOTTED)].to_crs(crs).union_all() if pak else shapely.Polygon()
    others = countries.loc[countries.ADM0_A3.isin([c for c in mapped if c != "PAK"]), ["geometry"]]
    if states is not None and len(states):
        others = pd.concat([others, states[["geometry"]]])
    others = others.to_crs(crs)
    solid = core.union(jk).boundary.union(others.boundary.union_all()) if len(others) else core.union(jk).boundary
    # coasts are not borders: drop the outline where it meets the sea (the maps draw their own 90 m shore)
    near = (core.union(jk) if pak else others.union_all()).envelope.buffer(200_000)
    land = pd.concat([countries[["geometry"]], disputed[["geometry"]]]).to_crs(crs)
    sea = near.difference(land[land.intersects(near)].union_all().buffer(50))  # 50 m closes slivers between neighbours
    solid = shapely.line_merge(solid.difference(sea.buffer(100)))  # union splits outlines where they meet
    dotted = core.boundary.intersection(jk.buffer(50))  # 50 m: tolerance for vertex mismatches
    lines = gpd.GeoDataFrame({"style": ["solid", "dotted"]}, geometry=[solid, dotted], crs=crs)
    return lines[~lines.is_empty].explode(index_parts=False).reset_index(drop=True)


def run(cfg) -> None:
    paths = Paths(cfg)
    crs = cfg["crs"]["proj4"]
    countries, disputed = load_boundaries(paths)
    bbox = box(*cfg["grid"]["bbox_lonlat"])

    study = mapped_area(cfg, paths, countries, disputed)
    ov = cfg["grid"].get("overlap")  # a parcel: a strip into its neighbours, built for context and clipped at the join
    if ov:
        sp = study.to_crs(crs).union_all()
        nb = countries.loc[countries.ADM0_A3.isin(ov["into"])].to_crs(crs).union_all()
        study = gpd.GeoDataFrame({"name": [cfg["project"]["area"]]},
                                 geometry=[sp.union(sp.buffer(ov["km"] * 1000).intersection(nb))], crs=crs).to_crs(4326)
        log.info("parcel overlap: %g km into %s", ov["km"], ", ".join(ov["into"]))
    if not study.within(bbox).all():
        raise ValueError(f"grid.bbox_lonlat {cfg['grid']['bbox_lonlat']} does not contain the study area "
                         f"{[round(v, 2) for v in study.total_bounds]}")

    shown = display_countries(countries, disputed)
    ctx = shown[shown.intersects(bbox)][["ADM0_A3", "NAME", "geometry"]].copy()
    ctx["geometry"] = ctx.geometry.intersection(bbox.buffer(2))

    out = paths.vector("boundaries")
    if out.exists():
        out.unlink()
    study_p = study.to_crs(crs)
    study_p.to_file(out, layer="study_area")
    ctx.to_crs(crs).to_file(out, layer="countries_context")
    ne_lines = border_lines(countries, disputed, crs, mapped, states)
    ne_lines.to_file(out, layer="borders_ne")  # Natural Earth lines, matching RESOLVE's land
    ne_lines.to_file(out, layer="borders")     # replaced by the 90 m lines in the dem stage
    log.info("study area %.0f km²", study_p.area.sum() / 1e6)

    # analysis grid: study area + buffer, snapped to the resolution
    buf = cfg["grid"]["study_buffer_m"]
    b = study_p.buffer(buf).total_bounds
    grid = Grid.from_bounds(crs, b, cfg["grid"]["resolution"])
    grid.save(paths.grid_json)
    log.info("grid %dx%d @ %sm (%.0f M cells)", grid.width, grid.height, grid.res,
             grid.width * grid.height / 1e6)

    # masks: 1 = study area, 3 = other land, 0 = sea
    land = countries.to_crs(crs)
    land = land[land.intersects(box(*grid.bounds))]
    mask = rasterize_geoms([(g, 3) for g in land.geometry], grid)
    sa = rasterize_geoms([(g, 1) for g in study_p.geometry], grid)
    mask[sa == 1] = 1
    # Natural Earth coast; the dem stage redraws the coast from the DEM into ``mask``
    for name in ("mask_ne", "mask"):
        write(paths.raster(name), mask, grid, nodata=None, tags=dict(values="0 sea, 1 study area, 3 other land"))
    from .qa import quicklook

    quicklook(mask, paths.qa / "01_mask.png", "analysis mask", cmap="tab10", categorical=True)
    pd.Series(dict(study_km2=float((mask == 1).sum() * grid.cell_km2))).to_json(paths.interim / "mask_stats.json")


SEA, OTHER = 0, 255  # classes of the border raster besides the country codes 1..n


def edge_styles(cls: np.ndarray, groups: list[set], dotted: set) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Cell edges where a border is drawn: ``solid`` where one side is inside a group (greater Pakistan;
    Afghanistan) and the other is not, ``dotted`` between the two codes in ``dotted``; never at the sea.
    Returns per style the (vertical, horizontal) edge masks: vertical[r, c] is the edge between
    (r, c) and (r, c + 1), horizontal[r, c] between (r, c) and (r + 1, c)."""
    out = {}
    for axis, (a, b) in enumerate([(cls[:, :-1], cls[:, 1:]), (cls[:-1, :], cls[1:, :])]):
        land = (a != SEA) & (b != SEA) & (a != b)
        solid = np.zeros(a.shape, bool)
        for g in groups:
            solid |= np.isin(a, list(g)) != np.isin(b, list(g))
        dot = (np.isin(a, list(dotted)) & np.isin(b, list(dotted)))
        for k, m in (("solid", land & solid & ~dot), ("dotted", land & dot)):
            out.setdefault(k, [None, None])[axis] = m
    return {k: tuple(v) for k, v in out.items()}


def edge_lines(vert: np.ndarray, horiz: np.ndarray, transform, r0: int = 0):
    """Merge cell edges into lines (grid CRS); r0 offsets the rows of a chunk."""
    x0, res, y0 = transform.c, transform.a, transform.f
    r, c = np.nonzero(vert)
    v = np.stack([np.stack([x0 + (c + 1) * res, y0 - (r + r0) * res], 1),
                  np.stack([x0 + (c + 1) * res, y0 - (r + r0 + 1) * res], 1)], 1)
    r, c = np.nonzero(horiz)
    h = np.stack([np.stack([x0 + c * res, y0 - (r + r0 + 1) * res], 1),
                  np.stack([x0 + (c + 1) * res, y0 - (r + r0 + 1) * res], 1)], 1)
    return shapely.linestrings(np.concatenate([v, h])) if len(v) + len(h) else np.array([])


def fill_nearest_country(cls: np.ndarray, miss: np.ndarray) -> None:
    """Give cells in ``miss`` the code of the nearest country cell, in place. Each cluster of missing
    cells is solved in its own window, widened until it holds a country cell (offshore islets the 90 m
    coast adds are reached across the sea)."""
    from scipy import ndimage as ndi

    lab, _ = ndi.label(miss)
    for i, sl in enumerate(ndi.find_objects(lab), 1):
        pad = 4
        while True:
            win = tuple(slice(max(s.start - pad, 0), min(s.stop + pad, n)) for s, n in zip(sl, cls.shape))
            src = (cls[win] != SEA) & (cls[win] != OTHER) & ~miss[win]
            if src.any():
                break
            if all(w.start == 0 and w.stop == n for w, n in zip(win, cls.shape)):
                raise ValueError("study-area cells with no country anywhere")
            pad *= 4
        near = ndi.distance_transform_edt(~src, return_distances=False, return_indices=True)
        c, m = cls[win], lab[win] == i
        c[m] = c[near[0][m], near[1][m]]


def raster_borders(cfg, paths: Paths, mask: np.ndarray, grid) -> gpd.GeoDataFrame:
    """Border lines on the 90 m grid, so they follow the region edges exactly: country parts rasterized
    like the country shares (``country_parts``); study-area cells a polygon misses (land the 90 m
    coast adds, slivers) take the nearest country; edges between cells drawn as in ``border_lines``."""
    from .grid import rasterize_geoms

    parts = country_parts(cfg, paths)
    names = list(dict.fromkeys(parts.country))
    cls = rasterize_geoms(((g, names.index(c) + 1) for g, c in zip(parts.geometry, parts.country)), grid)
    cls[mask == 0] = SEA
    cls[mask == 3] = OTHER
    fill_nearest_country(cls, (mask == 1) & ((cls == SEA) | (cls == OTHER)))
    pak = {names.index(n) + 1 for n in ("Pakistan", "Kashmir") if n in names}
    groups = [pak] + [{names.index(n) + 1} for n in names if n not in ("Pakistan", "Kashmir", "China")]
    lines = {"solid": [], "dotted": []}
    step = 2000
    for r in range(0, cls.shape[0], step):  # row chunks with one row of overlap for the horizontal edges
        sub = cls[r:r + step + 1]
        e = edge_styles(sub, groups, pak)
        for k, (v, h) in e.items():
            lines[k].append(edge_lines(v[:step], h[:step], grid.transform, r))
    geoms = {k: shapely.line_merge(shapely.multilinestrings(np.concatenate([x for x in v if len(x)])))
             for k, v in lines.items() if any(len(x) for x in v)}
    g = gpd.GeoDataFrame({"style": list(geoms)}, geometry=list(geoms.values()), crs=grid.crs)
    return g[~g.is_empty].explode(index_parts=False).reset_index(drop=True)
