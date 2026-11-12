"""v0.2 stage — potential natural vegetation communities at 90 m (LANDFIRE-BpS-like).

Each community in ``communities.yaml`` carries explicit rules on the v0.1 rasters
(``where``: every condition must hold; numeric ``[min, max)`` with ``null`` open,
categorical lists, booleans). A cell takes the matching community with the
highest ``priority`` (ties: file order); cells no rule claims take their
province's ``fallback`` community and are flagged, so the confidence step can
treat them as weakly supported.

Derived inputs computed here, all on the 270 m grid unless noted: distance to
the sea (from the analysis mask), distance to a major river channel (flow
accumulation >= ``river_min_upland_km2``), northness = cos(aspect) (from the
DEM, 90 m) and dune terrain = the share of dune-textured cells
(``features.dune_mask``, as in v0.1) within ``dune_window_m`` reaching
``dune_min_share``, so isolated textured cells do not speckle the map; and
creek_m = distance (90 m, capped at ``creek_max_m``) to open or tidal water:
ESA WorldCover class 80 plus the sea. Only WorldCover's *water* class is used
(a physical feature locating tidal creeks); its vegetation and cropland classes
never enter the rules. The L2 province of each cell comes
from the snapped 90 m L3 units, so community and physiographic layers share
boundaries.

Outputs: ``communities.tif`` (uint8 ids, 0 = outside study area),
``communities_fallback.tif`` (1 = no rule matched), ``communities_area.csv`` and
QA maps. Validation layers (GMW, Jung, PNV, GBIF) are never used here.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import ndimage

from .config import Paths
from .features import dune_mask
from .grid import Grid, block_reduce, read, write
from .hydro import HAND_CLASS_CODES

log = logging.getLogger(__name__)

NUMERIC = {"elev": "dem", "slope": "slope", "bio1": "clim_bio1", "bio12": "clim_bio12", "ai": "clim_ai",
           "gdd5": "clim_gdd5", "summer_rain": "clim_summer_rain", "coast_km": None, "river_km": None, "creek_m": None, "northness": None,
           "tpi": "tpi_450"}
CATEGORICAL = {"hand": "hand_class", "litho": "geo_litho", "province": None}
BOOLEAN = {"dune": None}


def load_rules(path: Path) -> dict:
    """Read and validate the rules file; fail on anything we would silently ignore."""
    doc = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    comms, ids = doc["communities"], [c["id"] for c in doc["communities"]]
    if len(set(ids)) != len(ids):
        raise ValueError(f"duplicate community ids in {path}")
    if len(ids) > 254:
        raise ValueError("at most 254 communities fit the uint8 output")
    for c in comms:
        for k, v in c["where"].items():
            if k not in NUMERIC | CATEGORICAL | BOOLEAN:
                raise ValueError(f"{c['id']}: unknown rule key {k!r}")
            if k in NUMERIC and not (isinstance(v, list) and len(v) == 2):
                raise ValueError(f"{c['id']}: {k} must be [min, max]")
            if k == "hand" and not set(v) <= set(HAND_CLASS_CODES):
                raise ValueError(f"{c['id']}: unknown HAND class in {v}")
    missing = set(doc["fallback"].values()) - set(ids)
    if missing:
        raise ValueError(f"fallback names unknown communities: {missing}")
    types = doc.get("ecoregion_types", {})
    bad = {c["id"]: c.get("ecoregion") for c in comms if c.get("ecoregion") not in types}
    if bad:
        raise ValueError(f"communities without a defined ecoregion type: {bad}")
    return doc


def classify(layers: dict, rules: dict) -> tuple[np.ndarray, np.ndarray]:
    """Apply the rules to arrays of equal shape. ``layers`` holds every rule key
    (``province`` as the integer L2 code, ``hand`` as HAND class codes). Returns
    community index 1..n (fallback applied; 0 where no rule and no fallback) and
    a boolean 'matched by a rule'."""
    comms = rules["communities"]
    shape = next(iter(layers.values())).shape
    out = np.zeros(shape, "uint8")
    order = sorted(range(len(comms)), key=lambda i: -comms[i]["priority"])  # stable: ties keep file order
    for i in order:
        m = out == 0
        for k, v in comms[i]["where"].items():
            a = layers[k]
            if k in NUMERIC:
                lo, hi = v
                if lo is not None:
                    m &= a >= lo
                if hi is not None:
                    m &= a < hi
            elif k == "hand":
                m &= np.isin(a, [HAND_CLASS_CODES[x] for x in v])
            elif k == "province":
                m &= np.isin(a, [int(x) for x in v])
            elif k == "litho":
                m &= np.isin(a, v)
            else:  # boolean
                m &= a == bool(v)
        out[m] = i + 1
    matched = out > 0
    idx = {c["id"]: j + 1 for j, c in enumerate(comms)}
    for code, cid in rules["fallback"].items():
        out[~matched & (layers["province"] == int(code))] = idx[cid]
    return out, matched


def northness(dem: np.ndarray) -> np.ndarray:
    """cos(aspect): +1 north-facing, -1 south-facing, 0 flat (rows run north->south)."""
    gy, gx = np.gradient(dem.astype("float32"))
    mag = np.hypot(gx, gy)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(mag > 1e-6, gy / mag, 0.0).astype("float32")


def distance_km(target: np.ndarray, grid: Grid, factor: int, share: float = 0.5) -> np.ndarray:
    """Distance (km) to cells where ``target`` covers more than ``share`` of a
    ``factor``-coarser block, on that coarse grid.
    shortcut: +-1 coarse cell (270 m) accuracy; compute at 90 m if a rule ever needs < 1 km."""
    t = block_reduce(target.astype("float32"), factor) > share
    return (ndimage.distance_transform_edt(~t) * grid.res * factor / 1000).astype("float32")


def dune_share(dune: np.ndarray, window: int) -> np.ndarray:
    """Share of dune-textured cells in a ``window`` x ``window`` neighbourhood."""
    return ndimage.uniform_filter(dune.astype("float32"), size=window, mode="nearest")


def run(cfg) -> None:
    paths = Paths(cfg)
    grid = Grid.load(paths.grid_json)
    ccfg = cfg["communities"]
    rules = load_rules(paths.root / ccfg["rules"])
    comms = rules["communities"]
    f = cfg["grid"]["coarse_factor"]

    coast = distance_km(read(paths.raster("mask")) == 0, grid, f)
    # any channel cell in the block counts (share 0): channels are one cell wide
    river = distance_km(read(paths.raster("flowacc_km2")) >= ccfg["river_min_upland_km2"], grid, f, share=0.0)
    win = max(1, round(ccfg["dune_window_m"] / grid.res)) | 1  # odd window
    creek_cells = int(np.ceil(ccfg["creek_max_m"] / grid.res))
    for c in comms:  # creek distance is exact only up to the tile overlap
        lo_hi = c["where"].get("creek_m")
        if lo_hi and lo_hi[1] is not None and lo_hi[1] > ccfg["creek_max_m"]:
            raise ValueError(f"{c['id']}: creek_m above communities.creek_max_m ({ccfg['creek_max_m']} m)")
    att = pd.read_csv(paths.interim / "l3_attributes.csv", dtype={"l2_code": str})
    l2_of_unit = np.zeros(int(att.unit.max()) + 1, "uint8")
    l2_of_unit[att.unit.to_numpy()] = att.l2_code.astype(int).to_numpy()

    H, W = grid.shape
    out = np.zeros((H, W), "uint8")
    fallback = np.zeros((H, W), "uint8")
    files = {k: paths.raster(v) for k, v in (NUMERIC | CATEGORICAL).items() if v}
    for rd, core, inner in grid.tiles(ccfg["tile_size"], max(win // 2, creek_cells) + 1):
        sl = (slice(inner.row_off, inner.row_off + inner.height), slice(inner.col_off, inner.col_off + inner.width))
        tile = {k: read(p, rd) for k, p in files.items()}
        l3 = read(paths.raster("l3"), rd)
        study = l3 > 0
        tile["province"] = np.where(study, l2_of_unit[np.minimum(l3, len(l2_of_unit) - 1)], 0)
        tile["northness"] = northness(tile["elev"])
        tex = dune_mask(cfg, tile["slope"], read(paths.raster("tpi_450"), rd),
                        read(paths.raster("plains"), rd) > 0, tile["litho"])
        tile["dune"] = dune_share(tex, win) >= ccfg["dune_min_share"]
        r = np.arange(rd.row_off, rd.row_off + rd.height) // f
        c = np.arange(rd.col_off, rd.col_off + rd.width) // f
        rr, cc = np.minimum(r, coast.shape[0] - 1), np.minimum(c, coast.shape[1] - 1)
        tile["coast_km"] = coast[rr][:, cc]
        tile["river_km"] = river[rr][:, cc]
        water = (read(paths.raster("worldcover"), rd) == 80) | (read(paths.raster("mask"), rd) == 0)
        tile["creek_m"] = np.minimum(ndimage.distance_transform_edt(~water) * grid.res,
                                     ccfg["creek_max_m"] + grid.res).astype("float32")
        ids, matched = classify(tile, rules)
        ids[~study] = 0
        dst = (slice(core.row_off, core.row_off + core.height), slice(core.col_off, core.col_off + core.width))
        out[dst] = ids[sl]
        fallback[dst] = (study & ~matched)[sl]
        log.info("tile r%d c%d done", core.row_off, core.col_off)

    names = {str(j + 1): c["name"] for j, c in enumerate(comms)}
    write(paths.raster("communities"), out, grid, nodata=0, tags=names)
    write(paths.raster("communities_fallback"), fallback, grid, nodata=None)

    # areas, overall and by province
    km2 = grid.cell_km2
    n = np.bincount(out.ravel(), minlength=len(comms) + 1)
    nf = np.bincount(out[fallback == 1], minlength=len(comms) + 1)
    df = pd.DataFrame(dict(id=[c["id"] for c in comms], name=[c["name"] for c in comms],
                           group=[c["group"] for c in comms], area_km2=n[1:] * km2,
                           fallback_km2=nf[1:] * km2))
    df["share_pct"] = 100 * df.area_km2 / df.area_km2.sum()
    df.to_csv(paths.interim / "communities_area.csv", index=False)
    log.info("communities: %d of %d types mapped; %.1f%% of the study area by fallback",
             (df.area_km2 > 0).sum(), len(df), 100 * df.fallback_km2.sum() / df.area_km2.sum())
    plot(out, comms, paths.qa / "20_communities.png")


def plot(ids: np.ndarray, comms: list, path: Path, max_px: int = 2400) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    step = max(1, int(np.ceil(max(ids.shape) / max_px)))
    a = ids[::step, ::step]
    cols = [plt.get_cmap(m)(i) for m in ("tab20", "tab20b", "tab20c") for i in range(20)][: len(comms)]
    fig, ax = plt.subplots(figsize=(16, 13), dpi=130)
    ax.imshow(np.ma.masked_equal(a, 0), cmap=ListedColormap(cols), vmin=0.5, vmax=len(comms) + 0.5,
              interpolation="nearest")
    present = set(np.unique(a)) - {0}
    ax.legend(handles=[Patch(color=cols[j], label=f"{j + 1} {c['name']}") for j, c in enumerate(comms)
                       if j + 1 in present], loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=7, frameon=False)
    ax.set_title("Potential natural vegetation communities (draft rules)")
    ax.set_axis_off()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    log.info("QA figure %s", path)
