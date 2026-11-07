"""Stage 15 — static high-resolution map (``outputs/ecoregions_map_v<version>.png`` + a README preview)."""
from __future__ import annotations

import logging

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from .colors import l3_colors
from .config import Paths
from .grid import Grid, read
from .webmap import hillshade

log = logging.getLogger(__name__)


def run(cfg) -> None:
    paths = Paths(cfg)
    grid = Grid.load(paths.grid_json)
    mc = cfg["maps"]
    gpkg = paths.outputs / "ecoregions.gpkg"
    l3 = gpd.read_file(gpkg, layer="l3_ecoregions")
    l3["color"] = l3_colors(l3, mc["family_colors"])
    l2 = gpd.read_file(gpkg, layer="l2_provinces")
    gray = gpd.read_file(gpkg, layer="disputed_gray")
    riv = gpd.read_file(gpkg, layer="rivers")
    ctx = gpd.read_file(paths.vector("boundaries"), layer="countries_context")

    cg = grid.coarsen(cfg["grid"]["coarse_factor"])
    elev = read(paths.interim / "coarse" / "elev.tif")
    hs = hillshade(elev, cg.res)
    xmin, ymin, xmax, ymax = cg.bounds
    ext = (xmin, xmin + cg.width * cg.res, ymax - cg.height * cg.res, ymax)

    xmn, ymn, xmx, ymx = l3.total_bounds
    pad = 40000
    aspect = (ymx - ymn + 2 * pad) / (xmx - xmn + 2 * pad)
    fig, ax = plt.subplots(figsize=(18, 18 * aspect + 1.2), dpi=360)
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.04, top=0.95)
    ax.set_facecolor("#dfe8ee")
    ctx.plot(ax=ax, color="#f1efe9", edgecolor="#a9a8a2", linewidth=0.5, zorder=1)
    ax.imshow(np.ma.masked_invalid(np.where(np.isnan(elev), np.nan, hs)), extent=ext, cmap="gray", vmin=0, vmax=1,
              alpha=0.55, zorder=2, interpolation="bilinear")
    l3.plot(ax=ax, color=l3.color, alpha=0.78, linewidth=0, zorder=3)
    l3.boundary.plot(ax=ax, color="white", linewidth=0.25, zorder=4)
    gray.plot(ax=ax, color=mc["gray"], edgecolor="#8d8d88", linewidth=0.5, zorder=5)
    lw = 0.25 + 1.6 * np.clip(np.log10(riv.UPLAND_SKM / 1000) / 2.7, 0, 1)
    riv.plot(ax=ax, color=mc["river_color"], linewidth=lw, zorder=6)
    l2.boundary.plot(ax=ax, color="#1b1b1a", linewidth=1.1, zorder=7)

    halo = [pe.withStroke(linewidth=2.2, foreground="white")]
    for _, r in l3.iterrows():
        if r.area_km2 < 600:
            continue
        p = r.geometry.representative_point()
        ax.text(p.x, p.y, r.code, fontsize=3.6, ha="center", va="center", color="#1b1b1a", zorder=8,
                path_effects=[pe.withStroke(linewidth=1.0, foreground="white")], clip_on=True)
    for _, r in l2.iterrows():
        p = r.geometry.representative_point()
        ax.text(p.x, p.y + 25000, f"{r.code}  {r['name'].split(' (')[0]}", fontsize=8.5, weight="bold", ha="center",
                color="#1b1b1a", zorder=9, path_effects=halo)
    for _, r in gray[gray.label.fillna("") != ""].iterrows():
        p = r.geometry.representative_point()
        if not (xmn - pad < p.x < xmx + pad and ymn - pad < p.y < ymx + pad):
            continue  # polygon mostly outside the frame (e.g. Aksai Chin)
        ax.text(p.x, p.y, r.label, fontsize=6.5, style="italic",
                ha="center", color="#3d3d3a", zorder=9, path_effects=halo, clip_on=True)

    ax.set_xlim(xmn - pad, xmx + pad)
    ax.set_ylim(ymn - pad, ymx + pad)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")

    fams = {}
    for _, r in l2.sort_values("code").iterrows():
        fams.setdefault(r.family, []).append(f"{r.code} {r['name']}")
    handles = []
    for fam, names in fams.items():
        handles.append(Patch(facecolor=mc["family_colors"][fam], edgecolor="none",
                             label=fam.replace("_", " ").capitalize() + ":\n  " + "\n  ".join(names)))
    handles.append(Patch(facecolor=mc["gray"], edgecolor="#8d8d88",
                         label="Not mapped: administered by India or China"))
    leg = ax.legend(handles=handles, loc="upper left", fontsize=6.6, frameon=True, framealpha=0.92,
                    title="Level II provinces (L3 units shaded within each family)", title_fontsize=7.5,
                    borderpad=0.8, labelspacing=0.7)
    leg.get_frame().set_edgecolor("#bbbbbb")

    # scale bar (Albers metres)
    x0, y0 = xmx - 420000, ymn - pad + 30000
    for i, c in enumerate(["#1b1b1a", "white", "#1b1b1a", "white"]):
        ax.add_patch(plt.Rectangle((x0 + i * 50000, y0), 50000, 9000, facecolor=c, edgecolor="#1b1b1a", lw=0.6, zorder=10))
    for d in (0, 100, 200):
        ax.text(x0 + d * 1000, y0 + 15000, f"{d}", fontsize=7, ha="center", zorder=10)
    ax.text(x0 + 220000, y0 + 15000, "km", fontsize=7, zorder=10)
    ax.annotate("N", xy=(xmx - 30000, ymn + 330000), xytext=(xmx - 30000, ymn + 240000), ha="center", fontsize=11,
                weight="bold", arrowprops=dict(arrowstyle="-|>", color="#1b1b1a", lw=1.4))

    ax.set_title(f"Revised Ecoregions of Pakistan — v{cfg['project']['version']}\n"
                 f"{len(l3)} Level III units within {len(l2)} Level II physiographic provinces (Omernik-style weight-of-evidence delineation)",
                 fontsize=13, loc="left")
    credit = ("Data: Copernicus GLO-90 DEM (ESA/Airbus); HydroSHEDS/HydroRIVERS (WWF, Lehner & Grill 2013); USGS OFR 97-470C geo8ag; "
              "ISRIC SoilGrids 2.0; CHELSA v2.1; ESA WorldCover 2021 (context); RESOLVE Ecoregions2017; GBIF.org; Natural Earth.\n"
              "Albers equal-area (lat1 25.8°, lat2 34.9°, lon0 69.3°). Eastern limit in Kashmir: Line of Control, plus the Siachen Glacier area. Boundaries derived at 90 m. "
              "Unit names marked provisional in the data need review.")
    ax.text(0.0, -0.012, credit, fontsize=6.5, color="#52514e", transform=ax.transAxes, va="top")
    # version in the file name: a new URL per release, so cached copies of an older map never show
    v = cfg["project"]["version"]
    phys = paths.outputs / f"physiographic_map_v{v}.png"
    fig.savefig(phys, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    _compress(phys)
    log.info("physiographic map %s", phys)

    import pyogrio

    main = phys
    if "ecoregions" in [n for n, _ in pyogrio.list_layers(paths.outputs / "ecoregions.gpkg")]:
        from .cartography import ecoregion_map

        main = paths.outputs / f"ecoregions_map_v{v}.png"
        ecoregion_map(cfg, paths, main)
        _compress(main)
    prev = _preview(main, mc["preview_px"])
    log.info("main map %s (preview %s)", main, prev.name)

    # keep the README pointing at this version's files (a stale link showed an old map)
    import re

    readme = paths.root / "README.md"
    txt = readme.read_text(encoding="utf-8")
    new = re.sub(r"outputs/(?:ecoregions|physiographic)_map(?:_v[0-9][0-9.]*)?(_preview)\.png",
                 lambda m: f"outputs/{main.stem}_preview.png", txt)
    new = re.sub(r"outputs/ecoregions_map(?:_v[0-9][0-9.]*)?\.png", f"outputs/{main.stem}.png", new)
    new = re.sub(r"outputs/physiographic_map(?:_v[0-9][0-9.]*)?\.png", f"outputs/{phys.stem}.png", new)
    if new != txt:
        readme.write_text(new, encoding="utf-8")
        log.info("README image links -> %s", main.name)


def _compress(path) -> None:
    """256-colour adaptive palette: ~3x smaller, visually indistinguishable at this scale."""
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = None
    Image.open(path).convert("RGB").quantize(colors=256, method=Image.Quantize.MEDIANCUT,
                                             dither=Image.Dither.FLOYDSTEINBERG).save(path, optimize=True)


def _preview(path, px: int):
    """Light copy for the README (the full map is ~13 MB)."""
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = None
    im = Image.open(path).convert("RGB")
    im.thumbnail((px, px), Image.Resampling.LANCZOS)
    prev = path.with_name(path.stem + "_preview.png")
    im.quantize(colors=256, method=Image.Quantize.MEDIANCUT).save(prev, optimize=True)
    return prev
