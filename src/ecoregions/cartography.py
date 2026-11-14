"""Cartography for the v0.2 maps, floral (ecoregions) and faunal (faunal regions),
drawn the same way: fills over relief, layered boundaries and a grouped colour
key. No names are set on the map (not even on the areas outside the study area);
the key identifies every colour. Floral key: ecoregion types by biome (types with
ecoregions flagged by the audit carry an asterisk). Faunal key: faunal regions by
group (the coarser clustering that contains them).
"""
from __future__ import annotations

import logging

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Patch
from shapely.geometry import box

from .colors import SHADES, shade

log = logging.getLogger(__name__)


def faunal_colors(fr: gpd.GeoDataFrame, cfg) -> list[str]:
    """Group hue + a lightness step per region (by area rank within the group), so every
    region of a group has its own shade: the key must tell them apart, not only touching ones."""
    rank = fr.groupby("group").area_km2.rank(ascending=False, method="first").astype(int) - 1
    pal = cfg["maps"]["faunal_group_colors"]
    return [shade(pal[g], SHADES[r % len(SHADES)]) for g, r in zip(fr.group, rank)]


def region_map(cfg, paths, out, mode: str = "floral") -> list[str]:
    """Draw the floral or faunal map to ``out``; returns the key colours (kept exact when quantizing)."""
    from .communities import load_rules
    from .grid import Grid, read
    from .webmap import hillshade

    mc = cfg["maps"]
    v = cfg["project"]["version"]
    gpkg = paths.outputs / "ecoregions.gpkg"
    borders = gpd.read_file(gpkg, layer="borders")
    riv = gpd.read_file(gpkg, layer="rivers")
    ctx = gpd.read_file(paths.vector("boundaries"), layer="countries_context")

    notes = []
    if mode == "floral":
        from .biomes import with_key_biomes

        types = with_key_biomes(cfg, paths, load_rules(paths.root / cfg["communities"]["rules"])["ecoregion_types"])
        g = gpd.read_file(gpkg, layer="ecoregions")
        g["color"] = g["type"].map(lambda t: types[t]["color"])  # .type is the geometry type in geopandas
        audit = pd.read_csv(paths.outputs / "ecoregion_audit.csv")
        flagged = set(audit[audit.review].type)
        order = g.groupby("type").area_km2.sum().sort_values(ascending=False).index
        groups = pd_groupby_keep_order([(types[t]["biome"], (types[t]["name"] + (" *" if t in flagged else ""),
                                                             types[t]["color"])) for t in order])
        if flagged:
            notes.append("* includes Floral Ecoregions flagged for review")
        key_title = "FLORAL ECOREGIONS"
        title = (f"Floral Ecoregions of {cfg['project']['area']} — v{v}\n"
                 f"{len(g)} Floral Ecoregions of {g['type'].nunique()} types in {g.biome.nunique()} biomes")
        credit = ("Floral Ecoregions: potential natural vegetation communities (rules on Copernicus GLO-90, CHELSA v2.1, "
                  f"SoilGrids, {cfg['sources']['geology']['label']}, HydroSHEDS), boundaries snapped to landforms at 90 m.\n"
                  "Checked against Global Mangrove Watch v3, Jung et al. 2020, Hengl et al. 2018, GBIF, RESOLVE 2017. ")
    else:
        g = gpd.read_file(gpkg, layer="faunal_regions")
        g["color"] = faunal_colors(g, cfg)
        gsize = g.groupby("group_name").area_km2.sum()
        g = g.assign(gsize=g.group_name.map(gsize)).sort_values(["gsize", "area_km2"], ascending=False)
        groups = pd_groupby_keep_order([(r.group_name, (r.faunal_name, r.color)) for r in g.itertuples()])
        key_title = "FAUNAL ECOREGIONS"
        title = (f"Faunal Ecoregions of {cfg['project']['area']} — v{v}\n"
                 f"{len(g)} Faunal Ecoregions in {g.group.nunique()} groups")
        nsp = int((paths.interim / "faunal" / "n_species.txt").read_text())
        credit = (f"Faunal Ecoregions: Simpson beta diversity of {nsp:,} mammals, birds, reptiles and amphibians (IUCN Red List "
                  "and BirdLife ranges trimmed to elevation limits) on 10 km cells, UPGMA clustering; boundaries follow "
                  "landforms.\n")
    credit += "Albers equal-area."

    grid = Grid.load(paths.grid_json)
    cg = grid.coarsen(cfg["grid"]["coarse_factor"])
    elev = read(paths.interim / "coarse" / "elev.tif")
    hs = hillshade(elev, cg.res)
    xmin, _, _, ymax = cg.bounds
    ext = (xmin, xmin + cg.width * cg.res, ymax - cg.height * cg.res, ymax)

    xmn, ymn, xmx, ymx = g.total_bounds
    pad = 40000
    aspect = (ymx - ymn + 2 * pad) / (xmx - xmn + 2 * pad)
    fig, ax = plt.subplots(figsize=(18, 18 * aspect + 1.4), dpi=mc["dpi"])
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.04, top=0.95)
    ax.set_facecolor(mc["sea_color"])
    ax.set_xlim(xmn - pad, xmx + pad)
    ax.set_ylim(ymn - pad, ymx + pad)
    ax.set_aspect("equal")
    ax.set_axis_off()

    # no relief where the DEM is exactly 0 m: land beyond the downloaded tiles is filled with 0 (a flat wedge otherwise)
    relief = np.ma.masked_invalid(np.where(np.isnan(elev) | (elev == 0), np.nan, hs))
    ctx.plot(ax=ax, color="#efede6", edgecolor="#b4b2aa", linewidth=0.5, zorder=1)
    ax.imshow(relief, extent=ext, cmap="gray", vmin=0, vmax=1, alpha=0.35, zorder=2, interpolation="bilinear")
    g.plot(ax=ax, color=g.color, linewidth=0, alpha=0.92, zorder=3)
    ax.imshow(relief, extent=ext, cmap="gray", vmin=0, vmax=1, alpha=0.22, zorder=4, interpolation="bilinear")
    glaciers = paths.vector("glaciers")
    if glaciers.exists():  # overlay (RGI 7.0): glaciers lie within ecoregions, they are not one
        gl = gpd.read_file(glaciers).to_crs(g.crs)
        gl.plot(ax=ax, color="#cfe5f7", edgecolor="#86afd6", linewidth=0.05, alpha=0.95, zorder=4.5)
    g.boundary.plot(ax=ax, color="white", linewidth=2.0, alpha=0.55, zorder=5)
    g.boundary.plot(ax=ax, color="#2d2a26", linewidth=0.75, alpha=0.85, zorder=5)
    lw = 0.25 + 1.4 * np.clip(np.log10(riv.UPLAND_SKM / 1000) / 2.7, 0, 1)
    riv.plot(ax=ax, color=mc["river_color"], linewidth=lw, alpha=0.75, zorder=6)
    draw_borders(ax, borders, z=10)

    handles = []
    for heading, items in groups:
        handles.append(Patch(facecolor="none", edgecolor="none", label=heading.upper()))
        handles += [Patch(facecolor=c, edgecolor="#555", linewidth=0.4, label=name) for name, c in items]
    if glaciers.exists():
        handles += [Patch(facecolor="none", edgecolor="none", label=" "),
                    Patch(facecolor="#cfe5f7", edgecolor="#3a78b5", linewidth=0.6, label="Glaciers (c. 2000)")]
        credit = credit.replace("Albers equal-area.", "Glaciers: RGI 7.0 Consortium 2023. Albers equal-area.")
    if notes:
        handles.append(Patch(facecolor="none", edgecolor="none", label=" "))
        handles += [Patch(facecolor="none", edgecolor="none", label=n) for n in notes]
    font = mc["fonts"]["units"]
    land = g.union_all()
    def key(loc, anchor):
        return ax.legend(handles=handles, loc=loc, bbox_to_anchor=anchor, prop=FontProperties(family=font, size=8.2),
                         frameon=True, framealpha=0.96, title=key_title,
                         title_fontproperties=FontProperties(family=font, size=9.5, weight="bold"),
                         borderpad=1.1, labelspacing=0.5, handlelength=2.2, handleheight=1.15, handletextpad=0.8,
                         alignment="left")
    def cover(leg):  # mapped area under the key, in m²
        fig.canvas.draw()
        (x0_, y0_), (x1_, y1_) = ax.transData.inverted().transform(leg.get_window_extent().get_points())
        return land.intersection(box(x0_, y0_, x1_, y1_)).area
    corners = {"upper left": (0, 1), "upper right": (1, 1), "lower left": (0, 0), "lower right": (1, 0.07)}
    loc = min(corners, key=lambda c: (cover(key(c, corners[c])), c != "upper left"))  # least cover; ties keep upper left
    anchor = corners[loc]
    if cover(key(loc, anchor)) > 0:  # every corner hides mapped land: put the key beside the map
        loc, anchor = "upper left", (1.005, 1)
    leg = key(loc, anchor)
    leg.get_frame().set_edgecolor("#c9c6bd")
    leg.get_frame().set_linewidth(0.6)
    for t, h in zip(leg.get_texts(), handles):
        if t.get_text().isupper():  # group headings: small bold capitals
            t.set_fontweight("bold")
            t.set_fontsize(7.0)
            t.set_color("#5a574f")
        elif h.get_facecolor()[3] == 0:  # notes
            t.set_fontsize(7.2)
            t.set_color("#5a574f")

    x0, y0 = xmx - 420000, ymn - pad + 30000
    for i, c in enumerate(["#1b1b1a", "white", "#1b1b1a", "white"]):
        ax.add_patch(plt.Rectangle((x0 + i * 50000, y0), 50000, 9000, facecolor=c, edgecolor="#1b1b1a", lw=0.6, zorder=23))
    for d in (0, 100, 200):
        ax.text(x0 + d * 1000, y0 + 15000, f"{d}", fontsize=7, ha="center", zorder=23)
    ax.text(x0 + 220000, y0 + 15000, "km", fontsize=7, zorder=23)
    nx = x0 - 60000 if loc == "lower right" else xmx - 30000  # keep the arrow clear of a lower-right key
    ny = y0 if loc == "lower right" else ymn + 240000
    ax.annotate("N", xy=(nx, ny + 90000), xytext=(nx, ny), ha="center", fontsize=11,
                weight="bold", arrowprops=dict(arrowstyle="-|>", color="#1b1b1a", lw=1.4), zorder=23)
    ax.set_title(title, fontsize=13, loc="left")
    ax.text(0.0, -0.012, credit, fontsize=6.3, color="#52514e", transform=ax.transAxes, va="top")
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log.info("%s map: %d areas, %d key entries", mode, len(g), sum(len(i) for _, i in groups))
    return [c for _, items in groups for _, c in items]


def draw_borders(ax, borders: gpd.GeoDataFrame, z: int) -> None:
    """Borders of the mapped countries: solid, dotted where Jammu and Kashmir meets the rest of Pakistan."""
    borders[borders["style"] == "solid"].plot(ax=ax, color="#1b1b1a", linewidth=1.6, zorder=z)
    borders[borders["style"] == "dotted"].plot(ax=ax, color="#1b1b1a", linewidth=1.5, linestyle=(0, (0.4, 2.2)),
                                               zorder=z, capstyle="round")


def pd_groupby_keep_order(pairs):
    """[(group, item), ...] -> [(group, [items]), ...] in first-seen group order."""
    out = {}
    for g, i in pairs:
        out.setdefault(g, []).append(i)
    return list(out.items())
