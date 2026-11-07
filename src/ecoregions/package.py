"""Stage 13 — assemble ``outputs/ecoregions.gpkg``.

Layers (all in the project's Albers equal-area CRS):

* ``l3_ecoregions`` — the fine units with full attributes (main deliverable)
* ``l2_provinces``  — dissolved from L3, so L3 nests exactly in L2
* ``l1_resolve``    — RESOLVE Ecoregions2017 clipped to the study area
* ``disputed_gray`` — areas shown flat gray, neutral labels, no ecoregions
* ``study_area``    — Pakistan, Azad Kashmir and Gilgit-Baltistan (to the LoC)
* ``rivers``        — HydroRIVERS reaches with upstream area >= 1000 km² (context)
* ``ecoregions``, ``faunal_regions``, ``biomes`` — v0.2 community ecoregions (``assemble``), when built
"""
from __future__ import annotations

import json
import logging

import geopandas as gpd
import pandas as pd

from . import vectors
from .config import Paths

log = logging.getLogger(__name__)

HAND_LABELS = {"active_floodplain": "active floodplain", "meander_floodplain": "meander floodplain",
               "bar_upland": "bar upland", "upland": "upland"}

L3_COLUMNS = [
    "code", "name", "name_basis", "review_flag", "also_contains", "l2_code", "l2_name", "l2_family",
    "l1_eco_id", "l1_name", "l1_share", "area_km2",
    "elev_min", "elev_p10", "elev_median", "elev_mean", "elev_p90", "elev_max", "slope_mean", "slope_median",
    "relief5k_mean", "landform_hammond", "plains_share", "dune_share",
    "litho_dominant", "litho_share", "geo_unit_dominant", "geo_age_rank",
    "soil_texture", "soil_clay", "soil_sand", "soil_ph", "soil_soc", "soil_summary",
    "bio1", "bio4", "bio12", "bio15", "gdd5", "ai", "aridity_class", "climate_summary",
    "hand_median", "hand_active_floodplain", "hand_meander_floodplain", "hand_bar_upland", "hand_upland", "hand_mix",
    "gbif_records", "gbif_species", "gbif_records_per_100km2", "gbif_species_per_100km2", "gbif_sampling",
    "confidence", "conf_boundary", "conf_homogeneity", "conf_l2_margin", "conf_named",
    "description", "unit",
]


def run(cfg) -> None:
    paths = Paths(cfg)
    l3 = gpd.read_file(paths.vector("l3_polygons"), layer="l3")
    gb = paths.outputs / "gbif_unit_stats.csv"
    if gb.exists():
        l3 = l3.merge(pd.read_csv(gb), on="unit", how="left")
    else:
        log.warning("GBIF statistics missing — run the validate stage")
    l3["l2_code"] = l3.l2_code.astype(str).str.zfill(2)
    l3["soil_summary"] = l3.apply(lambda r: f"{r.soil_texture}; clay {r.soil_clay:.0f}%, sand {r.soil_sand:.0f}%, "
                                            f"pH {r.soil_ph:.1f}, SOC {r.soil_soc:.0f} g/kg (SoilGrids 0–30 cm)", axis=1)
    l3["climate_summary"] = l3.apply(lambda r: f"{r.aridity_class}; MAT {r.bio1:.1f} °C; MAP {r.bio12:,.0f} mm; "
                                               f"precip. seasonality CV {r.bio15:.0f}%; GDD5 {r.gdd5:,.0f}; AI {r.ai:.2f}", axis=1)
    l3["hand_mix"] = l3.apply(lambda r: " · ".join(f"{v} {r[f'hand_{k}'] * 100:.0f}%" for k, v in HAND_LABELS.items()), axis=1)
    cols = [c for c in L3_COLUMNS if c in l3.columns]
    l3 = l3[cols + ["geometry"]].sort_values("code")
    num = l3.select_dtypes("float").columns
    l3[num] = l3[num].round(3)

    out = paths.outputs / "ecoregions.gpkg"
    if out.exists():
        out.unlink()
    l3.to_file(out, layer="l3_ecoregions")

    codes = json.loads((paths.interim / "province" / "l2_codes.json").read_text())
    l2 = l3.dissolve("l2_code", aggfunc={"area_km2": "sum", "code": "count"}).reset_index()
    l2 = l2.rename(columns={"code": "n_l3_units", "l2_code": "code"})
    meta = {v["code"]: v for v in codes.values()}
    l2["name"] = l2.code.map(lambda c: meta[c]["name"])
    l2["family"] = l2.code.map(lambda c: meta[c]["family"])
    l2[["code", "name", "family", "area_km2", "n_l3_units", "geometry"]].to_file(out, layer="l2_provinces")

    b = paths.vector("boundaries")
    study = gpd.read_file(b, layer="study_area")
    study.to_file(out, layer="study_area")
    gpd.read_file(b, layer="disputed_gray").to_file(out, layer="disputed_gray")
    res = vectors.resolve(cfg)
    l1 = gpd.clip(res, study)
    l1 = l1[["ECO_ID", "ECO_NAME", "BIOME_NAME", "REALM", "geometry"]].dissolve("ECO_ID", aggfunc="first").reset_index()
    l1["area_km2"] = l1.area / 1e6
    l1.to_file(out, layer="l1_resolve")
    riv = vectors.hydrorivers(cfg)
    riv = gpd.clip(riv[riv.UPLAND_SKM >= 1000], study.buffer(20000))
    riv[["HYRIV_ID", "UPLAND_SKM", "ORD_STRA", "DIS_AV_CMS", "geometry"]].to_file(out, layer="rivers")
    l3.drop(columns="geometry").to_csv(paths.outputs / "l3_ecoregions.csv", index=False)
    ev = paths.vector("ecoregions")  # v0.2 community ecoregions (assemble stage), when built
    if ev.exists():
        for layer in ("ecoregions", "faunal_regions", "biomes"):
            gpd.read_file(ev, layer=layer).to_file(out, layer=layer)
        pd.read_csv(paths.interim / "ecoregions.csv").to_csv(paths.outputs / "ecoregions.csv", index=False)
    log.info("wrote %s: %d L3, %d L2, %d L1", out, len(l3), len(l2), len(l1))
