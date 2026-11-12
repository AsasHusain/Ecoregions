"""Biomes from one global key (stage ``biomes``, before ``assemble``).

A biome is a class of vegetation structure and function, not a species assemblage (docs/review, DEFINITIONS). Each
ecoregion type's biome follows from (a) the structure mix of its communities (``structure`` in the rule files, set
from vegetation descriptions) and (b) its climate (CHELSA 1981-2010 means over the type's area), through the key in
``config.yaml`` -> ``biomes.key``. The key is identical everywhere and was fixed before any biome was tested.

The key is applied to every ~900 m pixel (its own community's structure, its own climate); a type takes the biome
covering most of its area, and the share agreeing is its biome purity. (The first version applied the key once per
type, to type-average climate and structure shares; kept as ``biome_type_level`` for comparison.)

Key, in order (first match wins; for a single pixel the shares are 0 or 1):
 1. ice share >= ``ice``                               -> Rock & Ice
 2. mangrove share >= ``mangrove``                     -> Mangroves
 3. herbaceous wetland share >= ``wetland``            -> Flooded Grasslands & Savannas
 4. closed forest share >= ``forest``                  -> a forest biome (below)
 5. alpine share >= ``alpine``, or the estimated mean temperature of the coldest month below ``montane_tcold``
    (cold, high ground without closed forest)          -> Montane Grasslands & Shrublands
 6. aridity index < ``desert_aridity`` or sparse share >= ``sparse``, or shrubland + sparse share >= ``xeric.share``
    in a semi-arid or drier climate (aridity < ``xeric.aridity``)  -> Deserts & Xeric Shrublands
 7. mild-winter (coldest month >= ``mediterranean.tcold_min``) winter rain (share of rain May-Oct below ``mediterranean.summer_rain_max``), mean temperature and aridity in
    the ``mediterranean`` ranges, and woody + shrub share >= ``woody``  -> Mediterranean Forests, Woodlands & Scrub
 8. forest + woodland share >= ``woody``               -> a forest biome (below)
 9. otherwise (savanna, shrubland, grassland)          -> Tropical & Subtropical, or Temperate, Grasslands,
                                                          Savannas & Shrublands
Forest biomes: conifer (forest + woodland) share >= broadleaf -> Tropical & Subtropical Coniferous or Temperate
Conifer Forests; else Tropical & Subtropical Moist (aridity >= ``moist_aridity``) or Dry Broadleaf Forests, or
Temperate Broadleaf & Mixed Forests. "Tropical & subtropical" = estimated coldest-month mean >= ``subtropical_tcold``
(coldest month estimated as bio1 - sqrt(2) x bio4 / 100, the low point of a sinusoidal year).
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import rasterio

from .communities import load_rules
from .config import Paths

log = logging.getLogger(__name__)
CLASSES = ["forest_conifer", "forest_broadleaf", "woodland_conifer", "woodland_broadleaf", "savanna", "shrubland",
           "grassland", "sparse", "wetland", "mangrove", "alpine", "ice"]


def coarse(paths: Paths, name: str, shape=None, step: int = 10, how="nearest"):
    with rasterio.open(paths.raster(name)) as r:
        shape = shape or (r.height // step, r.width // step)
        return r.read(1, out_shape=shape, resampling=getattr(rasterio.enums.Resampling, how)), shape


def biome_of(s: dict, c: dict, k: dict) -> str:
    """The key above, for one type: structure shares ``s`` (class -> share) and climate ``c`` (temp_c, tcold, aridity,
    summer_rain)."""
    forest = s["forest_conifer"] + s["forest_broadleaf"]
    woody = forest + s["woodland_conifer"] + s["woodland_broadleaf"]
    sub = c["tcold"] >= k["subtropical_tcold"]

    def forest_biome():
        conifer = s["forest_conifer"] + s["woodland_conifer"]
        broad = s["forest_broadleaf"] + s["woodland_broadleaf"]
        if conifer >= broad:
            return "Tropical & Subtropical Coniferous Forests" if sub else "Temperate Conifer Forests"
        if sub:
            return ("Tropical & Subtropical Moist Broadleaf Forests" if c["aridity"] >= k["moist_aridity"]
                    else "Tropical & Subtropical Dry Broadleaf Forests")
        return "Temperate Broadleaf & Mixed Forests"

    if s["ice"] >= k["ice"]:
        return "Rock & Ice"
    if s["mangrove"] >= k["mangrove"]:
        return "Mangroves"
    if s["wetland"] >= k["wetland"]:
        return "Flooded Grasslands & Savannas"
    if forest >= k["forest"]:
        return forest_biome()
    if s["alpine"] >= k["alpine"] or c["tcold"] < k["montane_tcold"]:
        return "Montane Grasslands & Shrublands"
    if c["aridity"] < k["desert_aridity"] or s["sparse"] >= k["sparse"]:
        return "Deserts & Xeric Shrublands"
    if s["shrubland"] + s["sparse"] >= k["xeric"]["share"] and c["aridity"] < k["xeric"]["aridity"]:
        return "Deserts & Xeric Shrublands"  # xeric shrubland: shrub-dominated in a semi-arid or drier climate
    m = k["mediterranean"]
    if (c["summer_rain"] <= m["summer_rain_max"] and m["temp_c"][0] <= c["temp_c"] <= m["temp_c"][1]
            and c["tcold"] >= m["tcold_min"] and m["aridity"][0] <= c["aridity"] < m["aridity"][1]
            and woody + s["shrubland"] >= k["woody"]):
        return "Mediterranean Forests, Woodlands & Scrub"
    if woody >= k["woody"]:
        return forest_biome()
    return ("Tropical & Subtropical Grasslands, Savannas & Shrublands" if sub
            else "Temperate Grasslands, Savannas & Shrublands")


def pixel_inputs(cfg, paths: Paths):
    """On a ~900 m grid: each pixel's ecoregion type (after rainfall zones), structure class and climate."""
    from .assemble import zone_types

    rules = load_rules(paths.root / cfg["communities"]["rules"])
    comms, types = rules["communities"], rules["ecoregion_types"]
    tkeys = list(types)
    tidx = {t: i + 1 for i, t in enumerate(tkeys)}
    lut_t = np.zeros(256, "int32")
    lut_s = np.full(256, -1, "int32")
    for j, c in enumerate(comms):
        lut_t[j + 1] = tidx[c["ecoregion"]]
        lut_s[j + 1] = CLASSES.index(c["structure"])
    com, shape = coarse(paths, "communities")
    typ, struct = lut_t[com], lut_s[com]
    for z in cfg["assemble"].get("zones", []):
        v, _ = coarse(paths, z["raster"], shape, how="average")
        zone_types(typ, v.astype(float), [tidx[t] for t in z["types"]], z["breaks"], [tidx[t] for t in z["into"]])
    clim = {k: coarse(paths, f, shape, how="average")[0].astype(float)
            for k, f in (("temp_c", "clim_bio1"), ("bio4", "clim_bio4"), ("aridity", "clim_ai"), ("summer_rain", "clim_summer_rain"))}
    clim["tcold"] = clim["temp_c"] - np.sqrt(2) * clim.pop("bio4") / 100
    return types, tkeys, tidx, typ, struct, clim


def pixel_biomes(struct: np.ndarray, c: dict, k: dict) -> np.ndarray:
    """The key applied to every pixel: its own community's structure class and its own climate (vectorised biome_of
    with one-hot shares)."""
    cls = np.array(CLASSES + ["none"])[np.where(struct >= 0, struct, len(CLASSES))]
    isin = lambda *n: np.isin(cls, n)  # noqa: E731
    sub = c["tcold"] >= k["subtropical_tcold"]
    conifer = isin("forest_conifer", "woodland_conifer")
    forest_b = np.where(conifer, np.where(sub, "Tropical & Subtropical Coniferous Forests", "Temperate Conifer Forests"),
                        np.where(sub, np.where(c["aridity"] >= k["moist_aridity"], "Tropical & Subtropical Moist Broadleaf Forests",
                                               "Tropical & Subtropical Dry Broadleaf Forests"), "Temperate Broadleaf & Mixed Forests"))
    m = k["mediterranean"]
    med = ((c["summer_rain"] <= m["summer_rain_max"]) & (c["temp_c"] >= m["temp_c"][0]) & (c["temp_c"] <= m["temp_c"][1])
           & (c["tcold"] >= m["tcold_min"]) & (c["aridity"] >= m["aridity"][0]) & (c["aridity"] < m["aridity"][1])
           & isin("forest_conifer", "forest_broadleaf", "woodland_conifer", "woodland_broadleaf", "shrubland"))
    grass = np.where(sub, "Tropical & Subtropical Grasslands, Savannas & Shrublands", "Temperate Grasslands, Savannas & Shrublands")
    conds = [isin("none"), isin("ice"), isin("mangrove"), isin("wetland"), isin("forest_conifer", "forest_broadleaf"),
             isin("alpine") | (c["tcold"] < k["montane_tcold"]),
             isin("sparse") | (c["aridity"] < k["desert_aridity"]) | (isin("shrubland") & (c["aridity"] < k["xeric"]["aridity"])),
             med, isin("woodland_conifer", "woodland_broadleaf")]
    choices = ["", "Rock & Ice", "Mangroves", "Flooded Grasslands & Savannas", forest_b, "Montane Grasslands & Shrublands",
               "Deserts & Xeric Shrublands", "Mediterranean Forests, Woodlands & Scrub", forest_b]
    return np.select(conds, choices, default=grass)


def type_profiles(cfg, paths: Paths) -> pd.DataFrame:
    """Per ecoregion type: structure shares, climate means, the type-level key (the first version) and the pixel-level
    key: each pixel's biome from its own structure and climate, the type taking the biome covering most of it."""
    types, tkeys, tidx, typ, struct, clim = pixel_inputs(cfg, paths)
    k = cfg["biomes"]["key"]
    pb = pixel_biomes(struct, clim, k)
    rows = []
    for t in tkeys:
        m = typ == tidx[t]
        if not m.any():
            continue
        sh = np.bincount(struct[m & (struct >= 0)], minlength=len(CLASSES)) / max((m & (struct >= 0)).sum(), 1)
        cm = {kk: float(np.nanmean(v[m])) for kk, v in clim.items()}
        b, n = np.unique(pb[m & (pb != "")], return_counts=True)
        rows.append(dict(type=t, name=types[t]["name"], biome_hand=types[t].get("biome", ""), pixels=int(m.sum()),
                         biome_type_level=biome_of(dict(zip(CLASSES, sh)), cm, k),
                         biome=b[n.argmax()] if len(n) else "", biome_purity=round(float(n.max() / n.sum()), 3) if len(n) else np.nan,
                         **{f"share_{c}": round(float(v), 3) for c, v in zip(CLASSES, sh)}, **{kk: round(v, 3) for kk, v in cm.items()}))
    return pd.DataFrame(rows)


def key_biomes(paths: Paths) -> dict:
    """type -> biome from the key (interim/type_biomes.csv, written by the ``biomes`` stage); {} before it has run."""
    f = paths.interim / "type_biomes.csv"
    return pd.read_csv(f).set_index("type").biome.to_dict() if f.exists() else {}


def with_key_biomes(cfg, paths: Paths, types: dict) -> dict:
    """The rule file's types with their biome replaced by the key's, when ``biomes.use_key`` is set."""
    if not cfg.get("biomes", {}).get("use_key"):
        return types
    kb = key_biomes(paths)
    missing = [t for t in types if t not in kb]
    if kb and missing:
        log.info("biomes: %d types without area keep their hand-set biome: %s", len(missing), ", ".join(missing))
    return {t: {**v, "biome": kb.get(t, v.get("biome"))} for t, v in types.items()}


def run(cfg) -> None:
    paths = Paths(cfg)
    df = type_profiles(cfg, paths)  # biome: the pixel-level key (majority); biome_type_level kept for comparison
    df.to_csv(paths.interim / "type_biomes.csv", index=False)
    (paths.outputs / "type_biomes.csv").write_text(df.to_csv(index=False), encoding="utf-8")
    changed = df[df.biome != df.biome_hand]
    log.info("biomes: %d types in %d biomes by the key; %d differ from the hand-set biome:\n%s", len(df), df.biome.nunique(),
             len(changed), changed[["name", "biome_hand", "biome"]].to_string(index=False))
