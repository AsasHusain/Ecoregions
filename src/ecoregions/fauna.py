"""v0.2 stage — species presence on an equal-area grid (faunal regions, step 1).

Ranges: IUCN Red List range polygons (``data/raw/iucn/ranges/*/spatial_data.gpkg``)
and BirdLife BOTW (``data/raw/birdlife/BOTW_2025.gpkg``), licensed non-commercial:
read here, never redistributed; only per-cell presence is derived.

1. Species: the classes in ``fauna.classes``, terrestrial systems only (IUCN
   assessment ``systems``), without the marine-dependent ones (``marine_dependent``:
   no suitable land habitat, or no freshwater system and more marine than land
   habitats). Seabirds and sea turtles carry the Terrestrial system for their
   nesting sites but describe the sea, not a land assemblage (as in Kreft & Jetz
   2010, Holt et al. 2013); coastal-wintering land birds and inland-water birds
   stay. Range polygons with the configured presence /
   origin / seasonal codes (default: extant or probably extant; native or
   reintroduced; resident or breeding). IUCN and BirdLife share species ids
   (``id_no`` = ``sisid``); duplicates are merged.
2. Each range is rasterized on a fine grid (``fine_m``) nested in the analysis
   cells (``cell_m``); range cover per cell = share of fine cells inside.
3. Three presence versions, each "present if it covers >= ``min_share`` of a
   cell" (a species that never reaches that share keeps its best cells, so
   small-range endemics are not lost):
   * ``range`` - the range polygons as drawn;
   * ``elev``  - range within the species' elevation limits (default for the
     faunal regions);
   * ``aoh``   - Area of Habitat (Brooks et al. 2019): ``elev`` and a habitat
     class (Jung et al. 2020 lvl2 = IUCN classes) coded *suitable*. Artificial
     cells (14.x: crops, pasture, plantations, urban) take the habitat of their
     potential natural vegetation (Hengl et al. 2018 PNV biome, crosswalk in
     ``fauna.pnv_to_habitat``), because the target is potential distribution.
   Treating artificial cells as suitable for every species, as a first draft
   did, inflated richness on the irrigated plains; and Jung's land-cover-based
   classes (desert, rocky areas) under-match species coded for dry shrubland,
   deflating the drylands - hence ``elev`` as the default and ``aoh`` as a
   sensitivity check. Species with no habitat or elevation data are not
   filtered on that criterion.

Outputs in ``interim/fauna/``: ``presence_{range,elev,aoh}.npz`` (cells x species,
sparse), ``species.csv``, ``cells.csv`` and QA richness maps.
"""
from __future__ import annotations

import logging
import zipfile

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio
import scipy.sparse as sp
from rasterio.enums import Resampling
from rasterio.features import rasterize
from shapely.geometry import box

from .config import Paths
from .grid import Grid, block_reduce, read, reproject_array, reproject_file, write

log = logging.getLogger(__name__)


def habitat_code(code: str) -> int | None:
    """IUCN habitat code '1.4' -> Jung lvl2 104; deeper codes ('11.1.1') -> their lvl2 parent."""
    parts = str(code).split(".")
    try:
        return int(parts[0]) * 100 + (int(parts[1]) if len(parts) > 1 else 0)
    except ValueError:
        return None


def habitat_kind(code: str) -> str:
    """IUCN habitat code -> 'marine' (9-12 neritic ... intertidal, 13.1 sea cliffs and offshore islands,
    15.10-15.13 marine artificial), 'unknown' (17 other, 18 unknown) or 'land' (all else, coastal dunes,
    caves and lagoons 13.2-13.5 included)."""
    p = [int(x) for x in str(code).split(".") if x.isdigit()]
    if not p or p[0] in (17, 18):
        return "unknown"
    if p[0] in (9, 10, 11, 12) or p[:2] == [13, 1] or (p[0] == 15 and len(p) > 1 and p[1] >= 10):
        return "marine"
    return "land"


def marine_dependent(codes, systems: str) -> bool:
    """No suitable land habitat at all; or, without the Freshwater system (which marks inland-water
    birds such as grebes, mergansers and lake-breeding terns), more marine than land habitats."""
    kinds = [habitat_kind(c) for c in set(codes)]
    land, sea = kinds.count("land"), kinds.count("marine")
    if not sea:
        return False
    return land == 0 or ("Marine" in str(systems) and "Freshwater" not in str(systems) and sea > land)


def species_traits(assess_zip, classes: list[str]) -> pd.DataFrame:
    """id, name, class, elevation limits and suitable lvl2 habitat codes of the
    terrestrial species in ``classes``, marine-dependent ones left out (``marine_dependent``)."""
    z = zipfile.ZipFile(assess_zip)
    rd = lambda n, **k: pd.read_csv(z.open(n), dtype=str, **k)  # noqa: E731
    tax = rd("taxonomy.csv", usecols=["internalTaxonId", "scientificName", "className"]).drop_duplicates("internalTaxonId")
    sysm = rd("assessments.csv", usecols=["internalTaxonId", "systems"]).drop_duplicates("internalTaxonId")
    elev = rd("all_other_fields.csv", usecols=["internalTaxonId", "ElevationLower.limit", "ElevationUpper.limit"]
              ).drop_duplicates("internalTaxonId")
    hab = rd("habitats.csv", usecols=["internalTaxonId", "code", "suitability"])
    hab = hab[hab.suitability == "Suitable"]
    codes = hab.groupby("internalTaxonId").code.agg(set)
    hab = hab.assign(lvl2=lambda d: d.code.map(habitat_code)).dropna(subset=["lvl2"])
    hab = hab.groupby("internalTaxonId").lvl2.agg(lambda s: sorted({int(v) for v in s}))
    t = tax[tax.className.isin(classes)].merge(sysm, on="internalTaxonId", how="left")
    t = t[t.systems.fillna("").str.contains("Terrestrial")]
    t = t.merge(elev, on="internalTaxonId", how="left")
    t["elev_lo"] = pd.to_numeric(t["ElevationLower.limit"], errors="coerce")
    t["elev_hi"] = pd.to_numeric(t["ElevationUpper.limit"], errors="coerce")
    t["habitats"] = t.internalTaxonId.map(hab)
    marine = pd.Series([marine_dependent(codes.get(i, ()), sy) for i, sy in zip(t.internalTaxonId, t.systems)], index=t.index)
    log.info("fauna: %d marine-dependent species left out (``marine_dependent``), e.g. %s",
             marine.sum(), ", ".join(sorted(t.scientificName[marine])[:40]))
    t = t[~marine].copy()
    t["id"] = t.internalTaxonId.astype(int)
    return t[["id", "scientificName", "className", "elev_lo", "elev_hi", "habitats"]].set_index("id")


def read_ranges(path, layer: str, key: str, ids: set, bbox_ll, fc: dict) -> gpd.GeoDataFrame:
    """Range polygons of ``ids`` intersecting ``bbox_ll`` with the configured codes."""
    where = (f"presence IN ({','.join(map(str, fc['presence']))}) AND origin IN ({','.join(map(str, fc['origin']))})"
             f" AND seasonal IN ({','.join(map(str, fc['seasonal']))})")
    g = pyogrio.read_dataframe(path, layer=layer, columns=[key], bbox=tuple(bbox_ll), where=where)
    g = g.rename(columns={key: "id"})
    g["id"] = g["id"].astype(int)
    g = g[g.id.isin(ids)]
    return g.clip(box(*bbox_ll))  # many ranges are continent-sized: clip before reprojecting


def run(cfg) -> None:
    paths = Paths(cfg)
    fc = cfg["fauna"]
    out = paths.interim / "fauna"
    out.mkdir(parents=True, exist_ok=True)
    crs = cfg["crs"]["proj4"]

    # grids: analysis cells and the nested fine grid, over the study area + buffer
    study = gpd.read_file(paths.vector("boundaries"), layer="study_area")
    b = study.buffer(fc["buffer_m"]).total_bounds
    k = fc["cell_m"] // fc["fine_m"]
    cell = Grid.from_bounds(crs, b, fc["cell_m"])
    fine = Grid(cell.crs, cell.x0, cell.y0, fc["fine_m"], cell.width * k, cell.height * k)
    bbox_ll = gpd.GeoSeries([study.buffer(fc["buffer_m"]).union_all()], crs=crs).to_crs(4326).total_bounds

    # fine-grid environment for AOH
    g90 = Grid.load(paths.grid_json)
    dem = reproject_array(read(paths.raster("dem")), g90.transform, g90.rio_crs, fine, Resampling.average)
    hab = reproject_file(paths.interim / "vegcheck" / "jung_lvl2.tif", fine, Resampling.mode, dtype="uint16", dst_nodata=0)
    pnv = reproject_file(paths.interim / "vegcheck" / "pnv_biome.tif", fine, Resampling.mode, dtype="uint8", dst_nodata=255)
    lut = np.zeros(256, "uint16")
    for biome, code in fc["pnv_to_habitat"].items():
        lut[int(biome)] = code
    art = (hab >= 1400) & (hab < 1500)
    hab[art] = lut[pnv[art]]  # artificial -> potential natural habitat (0 = unknown, not filtered)
    frac_study = block_reduce(rasterize([(g, 1) for g in study.geometry], out_shape=fine.shape,
                                        transform=fine.transform, dtype="uint8").astype("float32"), k)

    traits = species_traits(paths.raw / "iucn" / "redlist_assessments_2026-11-06.zip", fc["classes"])
    ids = set(traits.index)
    srcs = [(p, "exported", "id_no") for p in sorted((paths.raw / "iucn" / "ranges").glob("*/spatial_data.gpkg"))]
    srcs.append((paths.raw / "birdlife" / "BOTW_2025.gpkg", "all_species", "sisid"))
    ranges = pd.concat([read_ranges(p, lay, key, ids, bbox_ll, fc) for p, lay, key in srcs]).to_crs(crs)
    log.info("fauna: %d range polygons for %d species", len(ranges), ranges.id.nunique())

    n_cells = cell.width * cell.height
    kinds = ("range", "elev", "aoh")
    cols = {kk: [] for kk in kinds}
    rows = []
    for sid, g in ranges.groupby("id"):
        r = rasterize(((geom, 1) for geom in g.geometry), out_shape=fine.shape, transform=fine.transform,
                      dtype="uint8").astype(bool)
        if not r.any():
            continue
        t = traits.loc[sid]
        e = r.copy()
        if not np.isnan(t.elev_lo):
            e &= dem >= t.elev_lo
        if not np.isnan(t.elev_hi):
            e &= dem <= t.elev_hi
        a = e & (np.isin(hab, t.habitats) | (hab == 0)) if isinstance(t.habitats, list) and t.habitats else e
        for kk, m in zip(kinds, (r, e, a)):
            cov = block_reduce(m.astype("float32"), k).ravel()
            pres = cov >= fc["min_share"]
            if not pres.any() and cov.max() > 0:  # small-range species: keep its best cells
                pres = cov >= cov.max() * 0.999
            cols[kk].append(np.flatnonzero(pres))
        rows.append(dict(id=sid, name=t.scientificName, cls=t.className,
                         **{f"{kk}_cells": len(cols[kk][-1]) for kk in kinds},
                         has_elev=not np.isnan(t.elev_lo) or not np.isnan(t.elev_hi),
                         has_habitat=isinstance(t.habitats, list)))

    def matrix(cl):
        data = [np.full(len(c), True) for c in cl]
        return sp.csc_matrix((np.concatenate(data), (np.concatenate(cl), np.repeat(np.arange(len(cl)), [len(c) for c in cl]))),
                             shape=(n_cells, len(cl)), dtype=bool)

    mats = {kk: matrix(cols[kk]) for kk in kinds}
    for kk, m in mats.items():
        sp.save_npz(out / f"presence_{kk}.npz", m)
    spc = pd.DataFrame(rows)
    spc.to_csv(out / "species.csv", index=False)
    rr, cc = np.divmod(np.arange(n_cells), cell.width)
    x = cell.x0 + (cc + 0.5) * cell.res
    y = cell.y0 - (rr + 0.5) * cell.res
    pd.DataFrame(dict(cell=np.arange(n_cells), row=rr, col=cc, x=x, y=y, study_share=frac_study.ravel())
                 ).to_csv(out / "cells.csv", index=False)
    cell.save(out / "grid.json")
    log.info("fauna: %d species rasterized\n%s", len(spc),
             spc.groupby("cls")[[f"{kk}_cells" for kk in kinds]].agg(["count", "median"]).to_string())

    from .qa import quicklook

    for kk, m in mats.items():
        for cls in ["ALL"] + (sorted(spc.cls.unique()) if kk == cfg["faunal"]["presence"] else []):
            sel = np.ones(len(spc), bool) if cls == "ALL" else (spc.cls == cls).to_numpy()
            rich = np.asarray(m[:, sel].sum(axis=1)).reshape(cell.shape).astype("float32")
            rich[frac_study.reshape(cell.shape) < 0.5] = np.nan
            write(out / f"richness_{kk}_{cls.lower()}.tif", rich, cell, nodata=np.nan)
            quicklook(rich, paths.qa / f"30_richness_{kk}_{cls.lower()}.png", f"species richness ({kk}) - {cls.lower()}",
                      cmap="viridis")
