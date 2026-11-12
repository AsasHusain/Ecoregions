"""Colour assignment shared by the web and static maps.

L3 units take their L2 family hue (validated palette in ``config.yaml``) and a
lightness step chosen by greedy graph colouring so that adjacent units of the
same family never share a shade.
"""
from __future__ import annotations

import geopandas as gpd
import numpy as np
from matplotlib.colors import to_hex, to_rgb

SHADES = [0.0, 0.32, -0.28, 0.55, -0.45, 0.16]  # >0 toward white, <0 toward black


def shade(hex_color: str, t: float) -> str:
    rgb = np.array(to_rgb(hex_color))
    out = rgb + (1 - rgb) * t if t >= 0 else rgb * (1 + t)
    return to_hex(np.clip(out, 0, 1))


def l3_colors(l3: gpd.GeoDataFrame, family_colors: dict) -> list[str]:
    """Family hue + adjacency-aware lightness step for each L3 unit (row order)."""
    g = l3.reset_index(drop=True)
    sidx = g.sindex
    step = np.full(len(g), -1)
    order = np.argsort(-g.area.values)  # big units first
    for i in order:
        cand = sidx.query(g.geometry.iloc[i], predicate="touches")
        used = {step[j] for j in cand if g.l2_family.iloc[j] == g.l2_family.iloc[i] and step[j] >= 0}
        step[i] = next(k for k in range(len(SHADES) + 10) if k not in used) % len(SHADES)
    return [shade(family_colors[f], SHADES[s]) for f, s in zip(g.l2_family, step)]
