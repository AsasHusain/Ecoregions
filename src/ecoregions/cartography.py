"""Cartography for the v0.2 ecoregion map: labels that follow the shape of each
region, layered boundaries and a biome-grouped legend.

Labels: each region gets a smoothed centre line (slices across its principal
axis; midpoint of the interior in each slice). A name is set letter by letter
along that line with typographic advance widths if it fits and the line
bends gently; otherwise it is set straight (one or two lines) at the region's
representative point if it fits there; otherwise it is left out. Placed labels
reserve their footprint, so later (smaller) labels never overlap earlier ones.
"""
from __future__ import annotations

import logging

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import shapely
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Patch
from matplotlib.textpath import TextToPath
from shapely.geometry import LineString, MultiPolygon, Point, box
from shapely.ops import substring

log = logging.getLogger(__name__)
_T2P = TextToPath()


def largest(geom):
    return max(geom.geoms, key=lambda g: g.area) if isinstance(geom, MultiPolygon) else geom


def centerline(geom, n: int = 48, smooth: int = 7) -> LineString | None:
    """Smoothed line through the middle of ``geom`` along its principal axis."""
    g = largest(geom)
    xmin, ymin, xmax, ymax = g.bounds
    rng = np.random.default_rng(0)
    pts = rng.uniform([xmin, ymin], [xmax, ymax], size=(4000, 2))
    pts = pts[shapely.contains_xy(g, pts[:, 0], pts[:, 1])]
    if len(pts) < 50:
        return None
    mu = pts.mean(axis=0)
    _, _, vt = np.linalg.svd(pts - mu, full_matrices=False)
    ax_, perp = vt[0], vt[1]
    t = (pts - mu) @ ax_
    lo, hi = np.percentile(t, [4, 96])
    span = np.ptp((pts - mu) @ perp) * 4 + 1
    mids, prev = [], None
    for x in np.linspace(lo, hi, n):
        c = mu + x * ax_
        cut = LineString([c - perp * span, c + perp * span]).intersection(g)
        parts = [p for p in getattr(cut, "geoms", [cut]) if p.length > 0]
        if not parts:
            continue
        if prev is None:
            seg = max(parts, key=lambda p: p.length)
        else:
            seg = min(parts, key=lambda p: Point(p.interpolate(0.5, normalized=True)).distance(prev))
        m = seg.interpolate(0.5, normalized=True)
        mids.append((x, (np.array([m.x, m.y]) - mu) @ perp))
        prev = m
    if len(mids) < 8:
        return None
    xs, ys = np.array(mids).T
    k = max(1, min(smooth, len(ys) // 3))
    ys = np.convolve(np.pad(ys, k, mode="edge"), np.ones(2 * k + 1) / (2 * k + 1), mode="same")[k:-k]
    line = LineString([tuple(mu + a * ax_ + b * perp) for a, b in zip(xs, ys)])
    if line.coords[0][0] > line.coords[-1][0]:  # read left to right
        line = LineString(list(line.coords)[::-1])
    return line


class Labeler:
    """Places labels on an equal-aspect axis, avoiding collisions."""

    def __init__(self, ax, fig, simplify_m: float = 250.0):
        self.ax, self.fig, self.simplify_m = ax, fig, simplify_m
        self.taken = []
        ax.apply_aspect()  # equal aspect shrinks the axes box: measure after it
        x0, x1 = ax.get_xlim()
        w_in = ax.get_position().width * fig.get_figwidth()
        self.unit_per_pt = (x1 - x0) / (w_in * 72)  # data units per typographic point

    def _widths(self, text: str, fp: FontProperties, size: float, tracking: float) -> np.ndarray:
        """Advance positions (data units) of each character's start, plus total width.
        Typographic advances, not ink extents: those drop spaces and side bearings."""
        f = fp.copy()
        f.set_size(size)
        pre = [_T2P.get_text_width_height_descent(text[:i], f, ismath=False)[0] if i else 0.0
               for i in range(len(text) + 1)]
        pre = np.array(pre) + np.arange(len(text) + 1) * tracking * size
        return pre * self.unit_per_pt

    def _free(self, foot) -> bool:
        return not any(foot.intersects(t) for t in self.taken)

    def curved(self, text, line: LineString, size, fp, color, tracking=0.0, halo=2.0, max_bend=24, z=20) -> bool:
        if line is None:
            return False
        adv = self._widths(text, fp, size, tracking)
        W, L = adv[-1], line.length
        if W > 0.92 * L:
            return False
        s0 = (L - W) / 2
        h = size * self.unit_per_pt
        d = h * 0.6
        placed, angs = [], []
        for i, ch in enumerate(text):
            s = s0 + (adv[i] + adv[i + 1]) / 2
            p, a, b = line.interpolate(s), line.interpolate(max(0, s - d)), line.interpolate(min(L, s + d))
            ang = np.degrees(np.arctan2(b.y - a.y, b.x - a.x))
            angs.append(ang)
            placed.append((ch, p.x, p.y, ang))
        if len(angs) > 1 and np.max(np.abs(np.diff(angs))) > max_bend:
            return False
        sub = substring(line, s0, s0 + W)
        foot = sub.buffer(h * 0.75)
        if not self._free(foot):
            return False
        for ch, x, y, ang in placed:
            if ch == " ":
                continue
            self.ax.text(x, y, ch, fontproperties=fp, fontsize=size, color=color, rotation=ang,
                         rotation_mode="anchor", ha="center", va="center", zorder=z, clip_on=True,
                         path_effects=[pe.withStroke(linewidth=halo, foreground="white", alpha=0.9)])
        self.taken.append(foot)
        return True

    def straight(self, text, geom, size, fp, color, halo=2.0, z=20, tracking=0.0) -> bool:
        g = largest(geom)
        p = g.representative_point()
        h = size * self.unit_per_pt
        for lines in ([text], self._wrap(text)):
            if lines is None:
                continue
            w = max(self._widths(t, fp, size, tracking)[-1] for t in lines)
            ht = h * 1.25 * len(lines)
            foot = box(p.x - w / 2, p.y - ht / 2, p.x + w / 2, p.y + ht / 2)
            if g.buffer(h * 0.3).contains(foot) and self._free(foot):
                self.ax.text(p.x, p.y, "\n".join(lines), fontproperties=fp, fontsize=size, color=color,
                             ha="center", va="center", linespacing=1.05, zorder=z, clip_on=True,
                             path_effects=[pe.withStroke(linewidth=halo, foreground="white", alpha=0.9)])
                self.taken.append(foot)
                return True
        return False

    @staticmethod
    def _wrap(text):
        words = text.split()
        if len(words) < 2:
            return None
        best = min(range(1, len(words)), key=lambda i: abs(len(" ".join(words[:i])) - len(" ".join(words[i:]))))
        return [" ".join(words[:best]), " ".join(words[best:])]

    def label(self, text, geom, size, fp, color, tracking=0.0, halo=2.0, z=20, min_size=None) -> bool:
        """Curved if possible, else straight; shrink once before giving up."""
        geom = geom.simplify(self.simplify_m)  # fitting tests on 90 m outlines were very slow
        line = centerline(geom)
        for sz in (size, max(min_size or size * 0.8, size * 0.8)):
            if self.curved(text, line, sz, fp, color, tracking, halo, z=z):
                return True
            if self.straight(text, geom, sz, fp, color, halo, z=z, tracking=tracking):
                return True
        return False


def ecoregion_map(cfg, paths, out) -> None:
    """The v0.2 map: community ecoregions (fill by type), faunal regions (bold
    outlines and spaced capitals), biome-grouped legend, hillshade relief."""
    from .communities import load_rules
    from .grid import Grid, read
    from .webmap import hillshade

    mc = cfg["maps"]
    types = load_rules(paths.root / cfg["communities"]["rules"])["ecoregion_types"]
    gpkg = paths.outputs / "ecoregions.gpkg"
    eco = gpd.read_file(gpkg, layer="ecoregions")
    fr = gpd.read_file(gpkg, layer="faunal_regions")
    gray = gpd.read_file(gpkg, layer="disputed_gray")
    riv = gpd.read_file(gpkg, layer="rivers")
    study = gpd.read_file(gpkg, layer="study_area")
    ctx = gpd.read_file(paths.vector("boundaries"), layer="countries_context")
    eco["color"] = eco["type"].map(lambda t: types[t]["color"])  # .type is the geometry type in geopandas

    grid = Grid.load(paths.grid_json)
    cg = grid.coarsen(cfg["grid"]["coarse_factor"])
    elev = read(paths.interim / "coarse" / "elev.tif")
    hs = hillshade(elev, cg.res)
    xmin, _, _, ymax = cg.bounds
    ext = (xmin, xmin + cg.width * cg.res, ymax - cg.height * cg.res, ymax)

    xmn, ymn, xmx, ymx = eco.total_bounds
    pad = 40000
    aspect = (ymx - ymn + 2 * pad) / (xmx - xmn + 2 * pad)
    fig, ax = plt.subplots(figsize=(18, 18 * aspect + 1.4), dpi=mc["dpi"])
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.04, top=0.95)
    ax.set_facecolor(mc["sea_color"])
    ax.set_xlim(xmn - pad, xmx + pad)
    ax.set_ylim(ymn - pad, ymx + pad)
    ax.set_aspect("equal")
    ax.set_axis_off()

    ctx.plot(ax=ax, color="#efede6", edgecolor="#b4b2aa", linewidth=0.5, zorder=1)
    ax.imshow(np.ma.masked_invalid(np.where(np.isnan(elev), np.nan, hs)), extent=ext, cmap="gray", vmin=0, vmax=1,
              alpha=0.35, zorder=2, interpolation="bilinear")
    eco.plot(ax=ax, color=eco.color, linewidth=0, alpha=0.92, zorder=3)
    ax.imshow(np.ma.masked_invalid(np.where(np.isnan(elev), np.nan, hs)), extent=ext, cmap="gray", vmin=0, vmax=1,
              alpha=0.22, zorder=4, interpolation="bilinear")  # relief over the fills
    eco.boundary.plot(ax=ax, color="white", linewidth=0.6, alpha=0.9, zorder=5)
    lw = 0.25 + 1.4 * np.clip(np.log10(riv.UPLAND_SKM / 1000) / 2.7, 0, 1)
    riv.plot(ax=ax, color=mc["river_color"], linewidth=lw, alpha=0.75, zorder=6)
    gray.plot(ax=ax, color=mc["gray"], edgecolor="#8d8d88", linewidth=0.5, zorder=7)
    fr.boundary.plot(ax=ax, color="white", linewidth=3.2, alpha=0.85, zorder=8)
    fr.boundary.plot(ax=ax, color="#2d2a26", linewidth=1.3, zorder=9)
    study.boundary.plot(ax=ax, color="#1b1b1a", linewidth=1.6, zorder=10)

    lab = Labeler(ax, fig)
    serif = FontProperties(family=mc["fonts"]["regions"], style="italic")
    sans = FontProperties(family=mc["fonts"]["units"], weight="semibold")
    # faunal regions first: spaced capitals along the region
    for _, r in fr.sort_values("area_km2", ascending=False).iterrows():
        size = float(np.clip(7 + 2.2 * np.log10(r.area_km2 / 5000), 7, 13))
        lab.label(str(r.faunal_name).upper(), r.geometry, size, serif, "#3b3630", tracking=0.18, halo=3.0, z=22)
    # then ecoregions, largest first
    for _, r in eco.sort_values("area_km2", ascending=False).iterrows():
        size = float(np.clip(4.6 + 1.5 * np.log10(r.area_km2 / 1000), 4.6, 9))
        dark = sum(int(r.color[i:i + 2], 16) for i in (1, 3, 5)) < 330
        lab.label(r.type_name, r.geometry, size, sans, "#ffffff" if dark else "#1d1b18",
                  halo=0.0 if dark else 2.0, z=21, min_size=4.2)
    for _, r in gray[gray.label.fillna("") != ""].iterrows():
        p = r.geometry.representative_point()
        if xmn - pad < p.x < xmx + pad and ymn - pad < p.y < ymx + pad:
            ax.text(p.x, p.y, r.label, fontsize=7, style="italic", ha="center", color="#3d3d3a", zorder=21,
                    path_effects=[pe.withStroke(linewidth=2.2, foreground="white")], clip_on=True)

    # legend grouped by biome
    handles = []
    order = eco.groupby("type").area_km2.sum().sort_values(ascending=False).index
    for biome, keys in pd_groupby_keep_order([(types[t]["biome"], t) for t in order]):
        handles.append(Patch(facecolor="none", edgecolor="none", label=biome.upper()))
        handles += [Patch(facecolor=types[t]["color"], edgecolor="#777", linewidth=0.3, label="   " + types[t]["name"])
                    for t in keys]
    handles.append(Patch(facecolor=mc["gray"], edgecolor="#8d8d88", label="Not mapped: administered by India or China"))
    leg = ax.legend(handles=handles, loc="upper left", fontsize=6.4, frameon=True, framealpha=0.94,
                    title="Ecoregion types by biome   ·   bold lines: faunal regions", title_fontsize=7.2,
                    borderpad=0.9, labelspacing=0.42, handlelength=1.6)
    leg.get_frame().set_edgecolor("#bbbbbb")
    for t in leg.get_texts():
        if t.get_text().isupper():
            t.set_fontweight("bold")
            t.set_fontsize(6.0)

    x0, y0 = xmx - 420000, ymn - pad + 30000
    for i, c in enumerate(["#1b1b1a", "white", "#1b1b1a", "white"]):
        ax.add_patch(plt.Rectangle((x0 + i * 50000, y0), 50000, 9000, facecolor=c, edgecolor="#1b1b1a", lw=0.6, zorder=23))
    for d in (0, 100, 200):
        ax.text(x0 + d * 1000, y0 + 15000, f"{d}", fontsize=7, ha="center", zorder=23)
    ax.text(x0 + 220000, y0 + 15000, "km", fontsize=7, zorder=23)
    ax.annotate("N", xy=(xmx - 30000, ymn + 330000), xytext=(xmx - 30000, ymn + 240000), ha="center", fontsize=11,
                weight="bold", arrowprops=dict(arrowstyle="-|>", color="#1b1b1a", lw=1.4), zorder=23)

    ax.set_title(f"Revised Ecoregions of Pakistan — v{cfg['project']['version']}\n"
                 f"{len(eco)} community ecoregions in {len(fr)} faunal regions and {eco.biome.nunique()} biomes",
                 fontsize=13, loc="left")
    credit = ("Ecoregions: potential natural vegetation communities (rules on Copernicus GLO-90, CHELSA v2.1, SoilGrids, "
              "USGS geo8ag, HydroSHEDS) grouped within faunal regions from IUCN Red List and BirdLife range maps "
              "(Simpson beta diversity); boundaries snapped to landforms at 90 m.\n"
              "Checked against Global Mangrove Watch v3, Jung et al. 2020, Hengl et al. 2018, GBIF. Albers equal-area. "
              "Eastern limit in Kashmir: Line of Control, plus the Siachen Glacier area.")
    ax.text(0.0, -0.012, credit, fontsize=6.3, color="#52514e", transform=ax.transAxes, va="top")
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log.info("ecoregion map: %d labels placed", len(lab.taken))


def pd_groupby_keep_order(pairs):
    """[(group, item), ...] -> [(group, [items]), ...] in first-seen group order."""
    out = {}
    for g, i in pairs:
        out.setdefault(g, []).append(i)
    return list(out.items())
