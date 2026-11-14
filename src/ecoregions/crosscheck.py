"""Stage 12 — do the L3 units resolve known physiographic units?

For each gazetteer feature marked ``check: true``:

* **distinct** — the unit(s) holding its anchor points hold no other checked
  feature (except those listed in ``may_share``);
* **named** — at least one unit carries the feature's name;
* **character** — the unit's statistics meet the feature's ``expect`` test
  (e.g. a bar upland must be dominantly in the bar-upland HAND class).
  A feature that fails its test does not count as occupying the unit.

Status: *resolved* (all three); *not resolved* (fails its character test and
names no unit — typically a feature below the minimum mapping unit absorbed into
a unit of different character); *partly resolved* (named or distinct, but not
both, or the character test fails); *merged* (shares a unit with another
checked feature and is not named).

Doabs are checked as regions: the alluvial plain is cut along the major-river
channels (HydroRIVERS, >= ``rem.river_min_upland_km2``); the piece containing
the doab point is the doab. *Interfluve purity* = share of the doab's
interfluve units (excluding river-corridor units, which correctly span both
banks) that lie >= 80% inside it.
"""
from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd
import yaml
from pyproj import Transformer
from scipy import ndimage as ndi

from . import vectors
from .config import Paths
from .grid import Grid, centre_sample, rasterize_geoms, read

log = logging.getLogger(__name__)


def expectation(r: pd.Series, exp: dict) -> tuple[bool, str]:
    msgs, ok = [], True

    def need(cond, msg):
        nonlocal ok
        if not cond:
            ok = False
            msgs.append(msg)

    if "hand_class" in exp:
        v = r[f"hand_{exp['hand_class']}"]
        need(v >= exp.get("min_frac", 0.35), f"{exp['hand_class']} share {v:.0%} < {exp.get('min_frac', 0.35):.0%}")
    if "hand_min" in exp:
        need(r.hand_median >= exp["hand_min"], f"median height above drainage {r.hand_median:.1f} m < {exp['hand_min']}")
    if "dune_min" in exp:
        need(r.dune_share >= exp["dune_min"], f"dune share {r.dune_share:.0%} < {exp['dune_min']:.0%}")
    if "relief_min" in exp:
        need(r.relief5k_mean >= exp["relief_min"], f"relief {r.relief5k_mean:.0f} m < {exp['relief_min']}")
    if "relief_max" in exp:
        need(r.relief5k_mean <= exp["relief_max"], f"relief {r.relief5k_mean:.0f} m > {exp['relief_max']}")
    if "elev_min" in exp:
        need(r.elev_median >= exp["elev_min"], f"median elevation {r.elev_median:.0f} m < {exp['elev_min']}")
    if "elev_max" in exp:
        need(r.elev_median <= exp["elev_max"], f"median elevation {r.elev_median:.0f} m > {exp['elev_max']}")
    if "slope_min" in exp:
        need(r.slope_mean >= exp["slope_min"], f"slope {r.slope_mean:.1f}° < {exp['slope_min']}")
    if "slope_max" in exp:
        need(r.slope_mean <= exp["slope_max"], f"slope {r.slope_mean:.1f}° > {exp['slope_max']}")
    if "litho" in exp:
        need(exp["litho"].lower() in r.litho_dominant.lower(), f"dominant substrate is {r.litho_dominant}")
    return ok, "; ".join(msgs)


def doab_regions(cfg, paths: Paths, grid: Grid, cg: Grid) -> np.ndarray:
    f = cfg["grid"]["coarse_factor"]
    study = read(paths.interim / "coarse" / "study_frac.tif") >= 0.5
    plains = centre_sample(read(paths.raster("plains")), f, cg.shape).astype(bool)
    riv = vectors.hydrorivers(cfg)
    big = riv[riv.UPLAND_SKM >= cfg["hydrology"]["rem"]["river_min_upland_km2"]]
    r = rasterize_geoms([(g, 1) for g in big.geometry], cg, all_touched=True).astype(bool)
    r = ndi.binary_dilation(r, iterations=2)  # ~ ±540 m channel belt
    lab, _ = ndi.label(study & plains & ~r)
    return lab


def run(cfg) -> None:
    paths = Paths(cfg)
    grid = Grid.load(paths.grid_json)
    cg = grid.coarsen(cfg["grid"]["coarse_factor"])
    gaz = paths.load_gazetteer()
    att = pd.read_csv(paths.interim / "l3_attributes.csv", dtype={"code": str, "l2_code": str}).set_index("unit")
    hits = {k: v for k, v in json.loads((paths.interim / "gazetteer_hits.json").read_text()).items()}
    checked = [f for f in gaz["features"] if f.get("check")]

    def passes(f, u):
        return not f.get("expect") or expectation(att.loc[u], f["expect"])[0]

    # a feature "occupies" a unit only if the unit passes its character test:
    # an inselberg absorbed into a plain does not make the plain non-distinct
    unit_feats = {}
    for f in checked:
        for u in hits.get(f["name"], []):
            if passes(f, u):
                unit_feats.setdefault(u, []).append(f["name"])
    rows = []
    for f in checked:
        us = hits.get(f["name"], [])
        if not us:
            rows.append(dict(feature=f["name"], kind=f["kind"], status="not mapped",
                             notes="anchor point(s) fall outside the study area or on no unit"))
            continue
        others = sorted({o for u in us for o in unit_feats.get(u, []) if o != f["name"] and o not in f.get("may_share", [])})
        named_units = [u for u in us if att.loc[u, "name"].startswith(f["name"])]
        ok_char, why = True, ""
        if f.get("expect"):
            res = [expectation(att.loc[u], f["expect"]) for u in us]
            ok_char = any(r[0] for r in res)
            why = " | ".join(r[1] for r in res if r[1])
        distinct = not others
        if distinct and named_units and ok_char:
            status = "resolved"
        elif not ok_char and not named_units:
            status = "not resolved"  # absorbed into a unit of different character (e.g. below the MMU)
        elif named_units or distinct:
            status = "partly resolved"
        else:
            status = "merged"
        rows.append(dict(feature=f["name"], kind=f["kind"], status=status,
                         l3_codes=", ".join(att.loc[u, "code"] for u in us),
                         l3_names=" | ".join(att.loc[u, "name"] for u in us),
                         shares_unit_with=", ".join(others), character_test="pass" if ok_char else "fail",
                         notes=why))
    # doabs
    lab_c = doab_regions(cfg, paths, grid, cg)
    f = cfg["grid"]["coarse_factor"]
    l3c = centre_sample(read(paths.raster("l3")), f, cg.shape)
    t = Transformer.from_crs("EPSG:4326", grid.crs, always_xy=True)
    inv = ~cg.transform
    for d in gaz["doabs"]:
        x, y = t.transform(*d["point"])
        c, r = inv * (x, y)
        if not (0 <= r < lab_c.shape[0] and 0 <= c < lab_c.shape[1]):
            continue  # a shared gazetteer's doab outside this parcel
        reg = lab_c[int(r), int(c)]
        if reg == 0:
            rows.append(dict(feature=d["name"], kind="doab", status="not mapped", notes="point not on the plains mask"))
            continue
        D = lab_c == reg
        ids, a_in = np.unique(l3c[D & (l3c > 0)], return_counts=True)
        tot = np.bincount(l3c.ravel())[ids]
        inside = a_in / tot >= 0.8
        # river-corridor units (mostly active + meander floodplain) legitimately span both
        # banks; purity is judged on the interfluve (non-corridor) units only
        corridor = np.array([(att.loc[i, "hand_active_floodplain"] + att.loc[i, "hand_meander_floodplain"]) > 0.6
                             if i in att.index else False for i in ids])
        inter = ~corridor
        purity = a_in[inside & inter].sum() / max(a_in[inter].sum(), 1)
        status = "resolved" if purity >= 0.8 else ("partly resolved" if purity >= 0.5 else "merged")
        rows.append(dict(feature=d["name"], kind="doab", status=status,
                         l3_codes=", ".join(att.loc[i, "code"] for i in ids[inside & inter] if i in att.index),
                         notes=f"doab area {D.sum() * cg.cell_km2:,.0f} km²; interfluve purity {purity:.0%} "
                               f"({(inside & inter).sum()} interfluve units inside, {(~inside & inter).sum()} straddle a "
                               f"bounding river; {corridor.sum()} river-corridor units excluded)"))
    out = pd.DataFrame(rows)
    out.to_csv(paths.outputs / "crosscheck.csv", index=False)
    log.info("cross-check: %s", out.status.value_counts().to_dict())
