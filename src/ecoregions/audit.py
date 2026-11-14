"""v0.2 stage — per-ecoregion audit: is each ecoregion where its type belongs?

The rules allow a community wherever its province list and climate bands match,
so a type can turn up where nobody has recorded it (chir pine in Waziristan in
v0.2.2). This stage checks every ecoregion and writes a review list; it never
changes the map.

1. Evidence: wild GBIF plant records (calibration cells only, see
   ``vegcheck.holdout``) inside the ecoregion of indicator species distinctive
   of its type (listed for no other type), against those of every other type.
   *contradicted*: another type's indicators outnumber its own ``rival_ratio``
   times (at least ``min_rival`` records); *supported*: at least ``min_own``
   records of its own and not contradicted; otherwise *unverified*, split into
   "sparse data" (fewer than ``min_records_adequate`` plant records: absence of
   indicators says little) and "adequate data but no indicators". Record density
   (per 1,000 km² of the ecoregion inside calibration cells) is reported too. A type
   whose distinctive indicators have no record in the study area (or that has
   none) is *untestable*: GBIF can neither support nor contradict it.
2. Reference: share of the area in RESOLVE 2017 ecoregions listed as compatible
   with the type (``ecoregion_types.<type>.resolve``). A flag, never a verdict.
3. Coherence: Bray-Curtis similarity of the ecoregion's community mix to the
   rest of its type; a low value means one name covers different vegetation.
4. Stability: from the ``evaluate`` sensitivity test.

An ecoregion needs review if it is contradicted or incoherent, or if it is
RESOLVE-incompatible or unstable without supporting evidence.

Rule gaps: cells no rule claims take their province's fallback community. Where
those cells lie far outside the fallback's own elevation band (lowland scrub on
a 3,000 m summit), a rule is missing; each such province is reported.
Outputs: ``outputs/ecoregion_audit.csv``, ``outputs/rule_gaps.csv``.
"""
from __future__ import annotations

import logging

import geopandas as gpd
import numpy as np
import pandas as pd

from . import vectors
from .assemble import pair_counts
from .communities import load_rules
from .config import Paths
from .grid import Grid, read
from .vegcheck import binomial, holdout, to_cells, wild_plants

log = logging.getLogger(__name__)


def distinctive_indicators(rules: dict) -> dict[str, str]:
    """Indicator species (binomial) -> the one ecoregion type whose communities list it."""
    types = {}
    for c in rules["communities"]:
        for s in c["indicators"]:
            types.setdefault(binomial(s), set()).add(c["ecoregion"])
    return {s: next(iter(t)) for s, t in types.items() if len(t) == 1}


def evidence(own: int, rival: int, min_own: int, min_rival: int, ratio: float, testable: bool = True,
             records: int | None = None, min_records: int = 0) -> str:
    """supported / contradicted / untestable, or unverified: "sparse data" when the ecoregion has fewer than
    ``min_records`` plant records (finding no indicator says little), else "adequate data but no indicators"."""
    if not testable:
        return "untestable"
    if rival >= min_rival and rival >= ratio * max(own, 1):
        return "contradicted"
    if own >= min_own:
        return "supported"
    if records is None:
        return "unverified"
    return "unverified, sparse data" if records < min_records else "unverified, adequate data but no indicators"


def calibration_area_km2(eco: np.ndarray, grid: Grid, n: int, deg: float, step: int = 10) -> np.ndarray:
    """Area of each label inside calibration cells (the half of the checkerboard the audit uses), from every
    ``step``-th cell."""
    from pyproj import Transformer

    sub = eco[::step, ::step]
    r, c = np.nonzero(sub > 0)
    x = grid.bounds[0] + (c * step + 0.5) * grid.res
    y = grid.bounds[3] - (r * step + 0.5) * grid.res
    lon, lat = Transformer.from_crs(grid.rio_crs, "EPSG:4326", always_xy=True).transform(x, y)
    cal = ~holdout(lon, lat, deg)
    return np.bincount(sub[r[cal], c[cal]], minlength=n) * grid.cell_km2 * step * step


def coherence(counts: np.ndarray, etype: np.ndarray) -> np.ndarray:
    """Bray-Curtis similarity (sum of minimum shares) of each row's mix to the pooled
    mix of the other rows of its type; NaN for a type with one row."""
    out = np.full(len(counts), np.nan)
    for t in np.unique(etype):
        rows = np.flatnonzero(etype == t)
        if len(rows) < 2:
            continue
        tot = counts[rows].sum(axis=0)
        for r in rows:
            p, q = counts[r], tot - counts[r]
            out[r] = np.minimum(p / max(p.sum(), 1), q / max(q.sum(), 1)).sum()
    return out


def review_flags(row, ac) -> list[str]:
    f = []
    if row.evidence == "contradicted":
        f.append(f"GBIF: {row.rival_records} records of {row.top_rival} indicators, {row.own_records} of its own")
    if row.coherence < ac["min_coherence"]:
        f.append(f"vegetation differs from the rest of its type (similarity {row.coherence:.2f})")
    if row.evidence != "supported":
        if row.resolve_compatible < ac["min_resolve_share"]:
            f.append(f"{100 * row.resolve_compatible:.0f} % in compatible RESOLVE units (main: {row.resolve_main})")
        if row.stability < ac["min_stability"]:
            f.append(f"unstable under rule changes ({100 * row.stability:.0f} %)")
    return f


def rule_gaps(fb_elev: pd.DataFrame, rules: dict, tol_m: float) -> pd.DataFrame:
    """Per province: fallback area and elevation (10th-90th percentile) against the
    fallback community's own elevation band; ``gap`` where they differ by more than ``tol_m``."""
    by_id = {c["id"]: c for c in rules["communities"]}
    rows = []
    for l2, g in fb_elev.groupby("l2"):
        cid = rules["fallback"][f"{l2:02d}"]
        lo, hi = by_id[cid]["where"].get("elev", [None, None])
        p10, p90 = g.elev.quantile([0.1, 0.9])
        gap = (hi is not None and p90 > hi + tol_m) or (lo is not None and p10 < lo - tol_m)
        rows.append(dict(l2=f"{l2:02d}", fallback=cid, fallback_km2=round(g.km2.sum()), elev_p10=round(p10),
                         elev_p90=round(p90), rule_elev=f"{lo}-{hi}", gap=bool(gap)))
    return pd.DataFrame(rows)


def run(cfg) -> None:
    paths = Paths(cfg)
    grid = Grid.load(paths.grid_json)
    ac = cfg["audit"]
    rules = load_rules(paths.root / cfg["communities"]["rules"])
    types, tkeys = rules["ecoregion_types"], list(rules["ecoregion_types"])
    e = pd.read_csv(paths.interim / "ecoregions.csv", dtype={"code": str}).set_index("unit")
    eco = read(paths.raster("ecoregions"))
    n = int(eco.max()) + 1

    # 1. evidence
    ind = distinctive_indicators(rules)
    rec = to_cells(wild_plants(cfg, paths), grid)
    rec = rec.assign(unit=eco[rec.row, rec.col], sp=rec.species.map(binomial),
                     site=rec.decimalLatitude.round(2).astype(str) + rec.decimalLongitude.round(2).astype(str))
    rec = rec[rec.unit > 0]
    # a species counts once per ~1 km site: many sheets from one collecting trip are one observation
    rec = rec.drop_duplicates(["sp", "site"])
    effort = rec.groupby("unit").size()
    found = set(rec.sp)
    absent = sorted({binomial(s) for c in rules["communities"] for s in c["indicators"]} - found)
    log.warning("audit: %d indicators have no record in the study area (rare, or named differently in GBIF): %s",
                len(absent), ", ".join(absent))
    testable = {ind[s] for s in found if s in ind}
    # rainfall (or other) belts (assemble.zones): a belt's own indicators are those of the types it regroups
    own_of = {t: [t] for t in tkeys}
    for z in cfg["assemble"].get("zones", []):
        for t in z["into"]:
            own_of[t] = [t, *z["types"]]
            if testable & set(z["types"]):
                testable.add(t)
    rec = rec.assign(type=rec.sp.map(ind)).dropna(subset=["type"])
    hits = rec.groupby(["unit", "type"]).size().unstack(fill_value=0).reindex(index=e.index, columns=tkeys, fill_value=0)

    # 2. RESOLVE compatibility
    res_tab = vectors.resolve(cfg)[["ECO_ID", "ECO_NAME"]].drop_duplicates("ECO_ID").set_index("ECO_ID").ECO_NAME
    # names are checked against the whole world: shared rule files list RESOLVE regions of other parcels
    world = gpd.read_file(f"/vsizip/{paths.raw / 'resolve' / 'Ecoregions2017.zip'}/Ecoregions2017.shp",
                          columns=["ECO_NAME"], ignore_geometry=True).ECO_NAME
    unknown = {s for t in types.values() for s in t.get("resolve", [])} - set(world)
    missing = [t for t in tkeys if not types[t].get("resolve")]
    if unknown or missing:
        raise ValueError(f"ecoregion_types: unknown RESOLVE names {unknown}; types without a resolve list {missing}")
    res = read(paths.raster("resolve"))
    rid = np.unique(res)
    pos = np.zeros(int(rid.max()) + 1, np.int64)
    pos[rid] = np.arange(len(rid))
    rc = pair_counts(eco, pos[res], n, len(rid)).astype(float)
    del res
    rc[:, rid == 0] = 0
    rnames = np.array([res_tab.get(int(r), "") for r in rid])

    # 3. coherence of community mixes within each type
    cc = pair_counts(eco, read(paths.raster("communities")), n, len(rules["communities"]) + 1)[:, 1:]
    coh = coherence(cc[e.index], e.type.to_numpy())
    stab = pd.read_csv(paths.outputs / "validation_stability.csv").set_index("unit").stability
    cal_km2 = calibration_area_km2(eco, grid, n, cfg["sources"]["gbif"]["cell_deg"])

    rows = []
    for i, (u, r) in enumerate(e.iterrows()):
        own = int(hits.loc[u, own_of[r.type]].sum())
        rivals = hits.loc[u].drop(own_of[r.type])
        ok = np.isin(rnames, types[r.type]["resolve"])
        rows.append(dict(code=r.code, name=r["name"], type=r.type, area_km2=r.area_km2,
                         plant_records=int(effort.get(u, 0)),
                         records_per_1000km2=round(1000 * effort.get(u, 0) / max(cal_km2[u], 1), 1), own_records=own,
                         top_rival=rivals.idxmax() if rivals.max() > 0 else "", rival_records=int(rivals.max()),
                         evidence=evidence(own, int(rivals.max()), ac["min_own"], ac["min_rival"], ac["rival_ratio"],
                                           r.type in testable, int(effort.get(u, 0)), ac["min_records_adequate"]),
                         resolve_compatible=round(rc[u, ok].sum() / max(rc[u].sum(), 1), 3),
                         resolve_main=rnames[rc[u].argmax()], coherence=round(coh[i], 2),
                         stability=stab.get(u, np.nan)))
    df = pd.DataFrame(rows)
    df["flags"] = ["; ".join(review_flags(r, ac)) for r in df.itertuples()]
    df["review"] = df["flags"] != ""  # not df.flags: that is a pandas DataFrame attribute
    df = df.sort_values("code", key=lambda s: s.map(lambda c: tuple(map(int, c.split(".")))))
    df.to_csv(paths.outputs / "ecoregion_audit.csv", index=False)

    # rule gaps (every 3rd cell is plenty for percentiles)
    sub = (slice(None, None, 3), slice(None, None, 3))
    fb = read(paths.raster("communities_fallback"))[sub] == 1
    l3 = read(paths.raster("l3"))[sub][fb].astype(np.int64)
    att = pd.read_csv(paths.interim / "l3_attributes.csv", dtype={"l2_code": str})
    l2_of = np.zeros(int(att.unit.max()) + 1, np.int64)
    l2_of[att.unit.to_numpy()] = att.l2_code.astype(int).to_numpy()
    fbe = pd.DataFrame(dict(l2=l2_of[np.minimum(l3, len(l2_of) - 1)], elev=read(paths.raster("dem"))[sub][fb],
                            km2=grid.cell_km2 * 9))
    gaps = rule_gaps(fbe[fbe.l2 > 0], rules, ac["gap_tolerance_m"])
    gaps.to_csv(paths.outputs / "rule_gaps.csv", index=False)
    if not gaps.empty and gaps.gap.any():  # no fallback cells (Texas) gives an empty table without columns
        log.warning("audit: possible rule gaps (fallback far outside its elevation band):\n%s",
                    gaps[gaps.gap].to_string(index=False))
    log.info("audit: evidence %s; %d of %d ecoregions need review:\n%s", df.evidence.value_counts().to_dict(),
             df.review.sum(), len(df), df[df.review][["code", "name", "flags"]].to_string(index=False))
