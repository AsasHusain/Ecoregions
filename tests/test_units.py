import numpy as np
import pandas as pd

from ecoregions.attributes import aridity_class, compass, texture_class
from ecoregions.assemble import check_invariants, country_mix, label_sums, disambiguate, place_name, join_same_type, majority, merge_small_nested, zone_types
from ecoregions.audit import coherence, evidence, rule_gaps
from ecoregions.colors import shade
from ecoregions.communities import classify, load_rules, northness
from ecoregions.crosscheck import expectation
from ecoregions.download import SourceUnavailable, cultivated, phenology_medians
from ecoregions.fauna import habitat_code
from ecoregions.faunal import beta_sim, clean_fragments, explained_turnover
from ecoregions.vegcheck import binomial, holdout, precision_recall
from ecoregions.grid import Grid, block_mode, block_reduce, is_current, write
from ecoregions.segment import adjacency, label_means, merge_small
from ecoregions.snap import internal_edges, make_contiguous, merge_small_units, nearest_unit
from ecoregions.validate import _wkt, anosim_r, jaccard_matrix, ranks_of


def test_grid_nesting_and_tiles():
    g = Grid("EPSG:3857", 0.0, 900.0, 90.0, 10, 10)
    c = g.coarsen(3)
    assert c.res == 270 and c.width == 4 and c.bounds[0] == g.bounds[0] and c.bounds[3] == g.bounds[3]
    cores = [core for _, core, _ in g.tiles(4, 1)]
    assert sum(w.width * w.height for w in cores) == 100


def test_block_reduce_and_mode():
    a = np.arange(36, dtype="float32").reshape(6, 6)
    assert block_reduce(a, 3)[0, 0] == np.mean(a[:3, :3])
    m = np.array([[1, 1, 2], [1, 2, 2], [1, 1, 0]], dtype="uint8")
    assert block_mode(m, 3)[0, 0] == 1


def test_adjacency_and_merge_small():
    lab = np.array([[1, 1, 2, 2], [1, 1, 2, 2], [3, 3, 3, 4]])
    A = adjacency(lab, 4).toarray()
    assert A[0, 1] and A[1, 3] and A[2, 3] and not A[0, 3]  # 1-2, 2-4 (diagonal), 3-4 touch; 1-4 do not
    X = np.zeros(lab.shape + (1,), "float32")
    X[..., 0] = np.where(lab == 4, 0.9, np.where(lab == 2, 1.0, 0.0))
    merged = merge_small(lab, X, min_cells=2)
    assert 4 not in merged and merged[2, 3] == 2  # unit 4 joins its most similar neighbour (2)
    M = label_means(lab, X, 4)
    assert np.isclose(M[1, 0], 1.0)


def test_internal_edges_and_contiguity():
    lab = np.array([[1, 1, 2, 2], [1, 2, 2, 2], [1, 1, 1, 2], [2, 1, 1, 1]], dtype=np.int32)
    assert internal_edges(lab).any()
    out, _ = make_contiguous(lab.copy(), min_new_px=100)
    # the detached single 2-pixel at bottom-left is absorbed by unit 1
    assert out[3, 0] == 1


def test_language_helpers():
    assert texture_class(10, 80) in {"loamy sand", "sandy loam"}
    assert texture_class(45, 20) in {"clay", "silty clay"}
    assert texture_class(15, 20) == "silt loam"
    assert aridity_class(0.03) == "hyper-arid" and aridity_class(0.3) == "semi-arid" and aridity_class(1.0) == "humid"
    assert compass(1, 0) == "east" and compass(0, 1) == "north" and compass(-1, -1) == "southwest"
    assert shade("#000000", 0.5) == "#808080"


def test_expectation():
    r = pd.Series(dict(hand_bar_upland=0.5, dune_share=0.0, relief5k_mean=20, elev_median=180, slope_mean=0.5,
                       litho_dominant="Quaternary alluvium and fans"))
    ok, _ = expectation(r, {"hand_class": "bar_upland", "min_frac": 0.35})
    assert ok
    ok, why = expectation(r, {"dune_min": 0.2, "litho": "Precambrian"})
    assert not ok and "dune" in why and "substrate" in why


def test_anosim_separates_clear_groups():
    rng = np.random.default_rng(0)
    a = rng.random((10, 40)) < 0.6
    a[:, 20:] = False
    b = rng.random((10, 40)) < 0.6
    b[:, :20] = False
    P = np.vstack([a, b])
    rk = ranks_of(jaccard_matrix(P))
    g = np.array([0] * 10 + [1] * 10)
    assert anosim_r(rk, g) > 0.8
    assert abs(anosim_r(rk, rng.permutation(g))) < 0.5


def test_anosim_bounded_after_subsetting():
    from ecoregions.validate import compare_partitions

    rng = np.random.default_rng(1)
    P = rng.random((40, 60)) < 0.3
    g = np.repeat(np.arange(8), 5)
    g[0] = 99  # a singleton group, dropped before ranking
    xy = rng.random((40, 2))
    df = compare_partitions(P, {"x": g}, xy, n_null=5, n_perm=9, seed=0)
    assert -1 <= df.R.iloc[0] <= 1


def test_community_rules_priority_ranges_and_fallback():
    rules = {"communities": [
        {"id": "low", "name": "low", "group": "g", "priority": 10, "where": {"elev": [None, 1000]}},
        {"id": "wet_low", "name": "wet low", "group": "g", "priority": 50, "where": {"elev": [None, 1000], "hand": ["active_floodplain"]}},
        {"id": "high_p8", "name": "high", "group": "g", "priority": 20, "where": {"elev": [1000, None], "province": ["08"]}},
    ], "fallback": {"09": "low"}}
    layers = {"elev": np.array([[500, 500, 1500, 1500]], "float32"),
              "hand": np.array([[1, 3, 4, 4]]),
              "province": np.array([[8, 8, 8, 9]])}
    ids, matched = classify(layers, rules)
    # floodplain beats the lower-priority 'low'; 1000 m is the open upper bound of 'low'; province 9 falls back
    assert ids.tolist() == [[2, 1, 3, 1]] and matched.tolist() == [[True, True, True, False]]


def test_northness_sign():
    rows = np.arange(5, dtype="float32")[:, None] * np.ones((1, 5), "float32")  # rising southward
    assert np.allclose(northness(rows)[2, 2], 1.0)  # downslope faces north
    assert np.allclose(northness(-rows)[2, 2], -1.0)


def test_shipped_community_rules_are_valid():
    doc = load_rules("communities.yaml")
    assert len(doc["communities"]) > 30


def test_is_current_rejects_rasters_from_another_grid(tmp_path):
    g = Grid("EPSG:3857", 0.0, 900.0, 90.0, 10, 10)
    f = tmp_path / "a.tif"
    assert not is_current(f, g)
    write(f, np.zeros(g.shape, "uint8"), g)
    assert is_current(f, g)
    wider = Grid("EPSG:3857", 0.0, 900.0, 90.0, 12, 10)
    assert not is_current(f, wider)


def test_habitat_codes_and_names():
    assert habitat_code("1.4") == 104 and habitat_code("14.1") == 1401 and habitat_code("11.1.1") == 1101
    assert habitat_code("8") == 800 and habitat_code("x") is None
    assert binomial("Juniperus excelsa subsp. polycarpos") == "juniperus excelsa"


def test_precision_recall_with_tolerance():
    truth = np.zeros((5, 5), bool); truth[2, 2] = True
    pred = np.zeros((5, 5), bool); pred[2, 3] = True
    assert precision_recall(pred, truth) == (0.0, 0.0)
    assert precision_recall(pred, truth, tol_cells=1) == (1.0, 1.0)


def test_beta_sim_ignores_richness_differences():
    import scipy.sparse as sp
    X = sp.csr_matrix(np.array([[1, 1, 1, 1], [1, 1, 0, 0], [0, 0, 1, 1], [0, 0, 0, 0]], bool))
    d = beta_sim(X)
    assert d[0, 1] == 0.0          # a nested, poorer cell is not 'different' (Simpson)
    assert d[1, 2] == 1.0          # no shared species
    assert d[3, 0] == 1.0 and d[3, 3] == 0.0
    i, j = np.triu_indices(3, 1)
    dv = d[:3, :3][i, j]
    assert explained_turnover(dv, i, j, np.array([1, 1, 2])) == 1.0  # only the dissimilar pair is split
    assert explained_turnover(dv, i, j, np.array([1, 2, 2])) < 0     # the dissimilar pair grouped together


def test_clean_fragments_absorbs_specks():
    lab = np.ones((6, 6), "uint8"); lab[3:, :] = 2; lab[1, 1] = 2
    assert clean_fragments(lab, 2)[1, 1] == 1


def test_majority_filter_removes_specks():
    lab = np.ones((7, 7), "uint8"); lab[3, 3] = 2; lab[0, :] = 0
    out = majority(lab, 2, 3)
    assert out[3, 3] == 1 and (out[0] == 0).all()


def test_merge_small_nested_prefers_same_parent():
    lab = np.array([[1, 1, 2, 3, 3]] * 3, np.int32)
    parent = np.array([[1, 1, 1, 2, 2]] * 3, np.int32)
    out = merge_small_nested(lab.copy(), parent, min_px=4)
    assert (out[:, 2] == 1).all()  # the small part joins its own region's neighbour, not the larger region-2 part


def test_disambiguate_names():
    names = pd.Series(["A", "A", "B", "C", "C"])
    x = pd.Series([0.0, 10.0, 5.0, 3.0, 3.0]); y = pd.Series([0.0, 0.0, 5.0, 1.0, 1.0])
    out = disambiguate(names, x, y)
    assert out.is_unique and out[0] == "A (western)" and out[1] == "A (eastern)" and out[2] == "B"


def test_join_same_type_merges_touching_parts_of_one_type():
    # 1 and 3 share type A and touch; 2 is type B; 4 is type A but touches only 2
    lab = np.array([[1, 1, 3, 3, 2, 4],
                    [1, 1, 3, 3, 2, 4],
                    [0, 0, 0, 0, 0, 0]])
    typ = np.array([0, 5, 6, 5, 5])
    out = join_same_type(lab, typ, rows=1)
    assert out[0, 0] == out[0, 2] != out[0, 4]
    assert out[0, 5] not in (out[0, 0], out[0, 4])
    assert (out[2] == 0).all() and (out[:2] > 0).all()


def test_cultivated_flags_managed_and_notes():
    df = pd.DataFrame({"establishmentMeans": ["MANAGED", None, None, "NATIVE"],
                       "occurrenceRemarks": [None, "Cultivated in lawn", None, "on rocky slope"],
                       "locality": [None, None, "Kohat, roadside plantation", "Parachinar"]})
    assert cultivated(df).tolist() == [True, True, True, False]
    assert cultivated(df.assign(basisOfRecord="LIVING_SPECIMEN")).all()
    assert cultivated(df.drop(columns="establishmentMeans")).tolist() == [False, True, True, False]


def test_merge_small_units_keeps_a_unit_that_grew_past_its_mmu():
    # 1 (4 cells) merges into 2 (6 cells); 2 then has 10 >= 7 and must stay whole, not leave pieces behind
    row = [0, 1, 1, 2, 2, 2] + [3] * 6
    lab = np.array([row, row])
    out = merge_small_units(lab.copy(), {1: 7, 2: 7, 3: 7})
    assert set(np.unique(out)) == {0, 2, 3} and (out == 2).sum() == 10


def test_check_invariants_catches_split_and_same_type_neighbours():
    eco = np.array([[1, 1, 2, 2, 1],
                    [1, 1, 2, 2, 0],
                    [3, 3, 3, 0, 4]])
    etype = np.array([0, 1, 2, 2, 5])  # 2 and 3 share a type; 4 is an island
    area = np.array([0, 5.0, 4, 3, 1])
    codes, names = pd.Series(["a", "b", "c", "d"]), pd.Series(["A", "B", "C", "D"])
    try:
        check_invariants(eco, etype, area, 1.0, codes, names)
        raise AssertionError("expected a failure")
    except RuntimeError as err:
        msg = str(err)
    assert "ecoregions 2 and 3 touch" in msg and "ecoregion 1 has a detached piece of 1 cells" in msg
    assert "ecoregion 4" not in msg  # islands are allowed
    eco[0, 4], etype[3] = 2, 3
    check_invariants(eco, etype, area, 1.0, codes, names)


def test_holdout_is_a_checkerboard():
    assert list(holdout([0.1, 0.3, 0.1, 0.3], [0.1, 0.1, 0.3, 0.3], 0.25)) == [False, True, True, False]


def test_audit_evidence_coherence_and_rule_gaps():
    assert evidence(0, 10, 3, 5, 5) == "contradicted" and evidence(4, 10, 3, 5, 5) == "supported"
    assert evidence(1, 4, 3, 5, 5) == "unverified"
    assert evidence(0, 40, 3, 5, 5, testable=False) == "untestable"
    c = coherence(np.array([[10, 0], [9, 1], [0, 10], [5, 5]]), np.array(["a", "a", "b", "c"]))
    assert c[0] == c[1] == 0.9 and np.isnan(c[2]) and np.isnan(c[3])
    rules = {"communities": [{"id": "low", "where": {"elev": [None, 900]}}], "fallback": {"08": "low"}}
    fb = pd.DataFrame(dict(l2=8, elev=[300, 800, 2500, 3000], km2=1.0))
    assert rule_gaps(fb, rules, 300).gap.all() and not rule_gaps(fb[:2], rules, 300).gap.any()


def test_make_contiguous_leaves_no_piece_behind():
    # the 3 inside the ring of 1s joins that ring (a detached part of 1); the ring then joins 3's main
    # part, which in a single pass left the centre cell as a detached piece of 1
    lab = np.array([[1, 1, 1, 0, 1, 1, 1],
                    [1, 1, 1, 0, 1, 3, 1],
                    [1, 1, 1, 0, 1, 1, 1],
                    [3, 3, 3, 3, 3, 3, 3]])
    out, _ = make_contiguous(lab.copy(), 100)
    from scipy import ndimage as ndi
    assert all(ndi.label(out == v, structure=np.ones((3, 3)))[1] == 1 for v in (1, 3))


def test_place_name_is_one_place():
    names = {1: "Upper Indus Plain", 2: "Potohar Plateau and Salt Range", 3: "Thal Desert"}
    assert place_name(np.array([0, 40, 60, 0]), names) == "Potohar Plateau & Salt Range"
    assert place_name(np.array([500, 90, 5, 5]), names) == "Upper Indus Plain"  # no-province cells never name
    # no province holds half: the enclosing region covering most of the area names it
    regions = {1: "Indus Plains", 3: "Indus Plains"}
    assert place_name(np.array([0, 30, 40, 30]), names, regions) == "Indus Plains"
    assert place_name(np.array([0, 30, 40, 30]), names) == "Potohar Plateau & Salt Range"


def test_country_mix():
    names = ["Pakistan", "Afghanistan", "Kashmir"]
    assert country_mix(np.array([5, 620, 379, 1]), names) == "Pakistan 62%; Afghanistan 38%; Kashmir <1%"
    assert country_mix(np.array([0, 0, 1000, 0]), names) == "Afghanistan 100%"
    assert country_mix(np.array([0, 10000, 0, 1]), names) == "Pakistan >99%; Kashmir <1%"  # any area counts
    assert country_mix(np.zeros(4), names) == ""


def test_label_sums():
    lab = np.array([[0, 1, 1], [2, 2, 1]])
    val = np.array([[5.0, 1.0, np.nan], [2.0, 4.0, 3.0]])
    s, c = label_sums(lab, val, 3, rows=1)  # NaN skipped; row blocks add up
    assert s.tolist() == [5.0, 4.0, 6.0] and c.tolist() == [1, 2, 2]


def test_rarefy_draws_without_replacement():
    from ecoregions.evaluate import presence, rarefy

    rng = np.random.default_rng(0)
    counts = {"a": 98, "b": 1, "c": 1}
    assert set(rarefy(counts, 100, rng)) == {"a", "b", "c"}  # all records drawn: every species
    assert rarefy({"a": 5}, 3, rng) == ["a"]
    draws = [len(rarefy(counts, 2, rng)) for _ in range(200)]
    assert max(draws) <= 2 and min(draws) >= 1
    P = presence([["a", "b"], ["b"]])
    assert P.tolist() == [[True, True], [False, True]]


def test_evidence_split_by_data():
    from ecoregions.audit import evidence

    assert evidence(0, 0, 3, 5, 5, True, 120, 500) == "unverified, sparse data"
    assert evidence(1, 2, 3, 5, 5, True, 900, 500) == "unverified, adequate data but no indicators"
    assert evidence(4, 0, 3, 5, 5, True, 10, 500) == "supported"
    assert evidence(0, 8, 3, 5, 5, True, 10, 500) == "contradicted"
    assert evidence(0, 0, 3, 5, 5) == "unverified"


def test_weighted_quantiles():
    from ecoregions.vegcheck import weighted_quantiles

    v = np.array([0.0, 1.0, 2.0, 3.0, np.nan])
    assert weighted_quantiles(v, np.ones(5), [50]) == [1.5]
    # one heavily sampled value no longer dominates once weights even it out
    v = np.array([0.0] * 9 + [1.0])
    w = np.array([1 / 9] * 9 + [1.0])
    assert weighted_quantiles(v, np.ones(10), [50])[0] == 0.0
    assert 0.0 < weighted_quantiles(v, w, [50])[0] < 1.0


def test_phenology_medians(tmp_path):
    import rasterio
    from rasterio.transform import from_origin

    tiles = tmp_path / "tiles"
    tiles.mkdir()
    prof = dict(driver="GTiff", width=2, height=1, count=1, dtype="int16", nodata=32767, crs="EPSG:4326",
                transform=from_origin(60, 30, 0.01, 0.01))
    F = 32767
    jan = {2001: 11323, 2002: 11688, 2003: 12053}  # days since 1970-01-01
    # pixel 0: season 100, 120, 140 days; 2003 poor quality (QA 3) -> median of 100, 120 = 110
    # pixel 1: no cycle in any year (fill everywhere)
    for y, d in [(2001, 100), (2002, 120), (2003, 140)]:
        layers = {"Greenup_0": [jan[y] + 90, F], "Dormancy_0": [jan[y] + 90 + d, F], "EVI_Amplitude_0": [3000, F],
                  "EVI_Area_0": [200, F], "QA_Overall_0": [3 if y == 2003 else 1, F], "NumCycles": [1 if y < 2003 else 2, F]}
        for k, v in layers.items():
            with rasterio.open(tiles / f"MCD12Q2.061_{k}_doy{y}001000000_aid0001.tif", "w", **prof) as s:
                s.write(np.array([v], "int16"), 1)
    phenology_medians(tiles, tmp_path / "med", {"max_qa": 2})
    r = {k: rasterio.open(tmp_path / "med" / f"{k}.tif").read(1)[0] for k in
         ["season_days", "greenup_doy", "evi_amplitude", "cycles_per_year"]}
    assert r["season_days"][0] == 110 and np.isnan(r["season_days"][1])
    assert r["greenup_doy"][0] == 90 and abs(r["evi_amplitude"][0] - 0.3) < 1e-6
    assert abs(r["cycles_per_year"][0] - 4 / 3) < 1e-6 and r["cycles_per_year"][1] == 0


def test_coast_from_dem():
    from ecoregions.dem import coast_from_dem, coast_window

    ne = np.zeros((12, 12), "uint8")
    ne[:6, :6], ne[:6, 6:] = 1, 3  # study (west) and other land (east) north of a straight coast; sea south
    sea_dem = ne == 0
    sea_dem[4:6, 1:3] = True   # Natural Earth land that the DEM shows as sea, open to the sea: becomes sea
    sea_dem[1, 9] = True       # an inland 0 m flat, not connected: stays land
    sea_dem[6, 3:5] = False    # Natural Earth sea that the DEM shows as land: joins the nearest land (study)
    out = coast_from_dem(ne, sea_dem, open_sea_cells=3, max_inland_cells=3)
    assert (out[4:6, 1:3] == 0).all() and out[1, 9] == 3 and (out[6, 3:5] == 1).all()
    assert (out[9:, :] == 0).all() and out[0, 0] == 1
    sea_dem[0:6, 0] = True  # a 0 m flat reaching far inland from the coast: only its seaward 3 cells turn to sea
    out = coast_from_dem(ne, sea_dem, open_sea_cells=3, max_inland_cells=3)
    assert (out[3:6, 0] == 0).all() and (out[0:2, 0] == 1).all()
    r, c = coast_window(ne, margin=1, block=4)
    assert r.start == 3 and c.start == 0


def test_merge_small_units_island_joins_nearest():
    lab = np.zeros((10, 12), np.int32)
    lab[:, :4] = 1       # mainland unit 1 (west)
    lab[:, 8:] = 2       # mainland unit 2 (east)
    lab[4:6, 6:7] = 3    # a 2-cell island, nearer unit 2, touching nothing
    assert nearest_unit(lab, 3, (slice(4, 6), slice(6, 7))) == 2
    out = merge_small_units(lab.copy(), {1: 5, 2: 5, 3: 5})
    assert (out[4:6, 6] == 2).all() and set(np.unique(out)) == {0, 1, 2}


def test_wkt_many_islets_fits_url():
    from urllib.parse import quote_plus
    from shapely.geometry import MultiPolygon, box
    islets = MultiPolygon([box(67 + i * 0.02, 24, 67 + i * 0.02 + 0.01, 24.01) for i in range(80)])
    w = _wkt(islets)
    assert len(quote_plus(w)) <= 2800 and w.startswith("MULTIPOLYGON")


def test_raster_border_edges():
    import numpy as np
    from ecoregions.boundaries import edge_lines, edge_styles
    from rasterio.transform import Affine
    # 1 Pakistan, 3 Kashmir, 2 Afghanistan, 255 other land, 0 sea
    cls = np.array([[2, 1, 3, 255],
                    [2, 1, 3, 255],
                    [0, 0, 0, 0]], "uint8")
    e = edge_styles(cls, [{1, 3}, {2}], {1, 3})
    v, h = e["solid"]
    assert v.tolist() == [[True, False, True], [True, False, True], [False] * 3]  # AFG|PAK, KAS|other; none at the sea
    assert not h.any()
    assert e["dotted"][0].tolist() == [[False, True, False], [False, True, False], [False] * 3]
    lines = edge_lines(*e["dotted"], Affine(90, 0, 0, 0, -90, 0))
    assert len(lines) == 2 and all(l.length == 90 for l in lines)


def test_fill_nearest_country_reaches_islets():
    import numpy as np
    from ecoregions.boundaries import fill_nearest_country
    cls = np.array([[1, 1, 0, 0, 0, 0],
                    [1, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0]], "uint8")
    miss = np.zeros(cls.shape, bool)
    miss[1, 1] = miss[2, 5] = True   # a gap next to land, and an islet four cells offshore
    fill_nearest_country(cls, miss)
    assert cls[1, 1] == 1 and cls[2, 5] == 1 and cls[0, 3] == 0   # sea stays sea


def test_marine_dependent_keeps_land_and_inland_water_species():
    from ecoregions.fauna import marine_dependent
    assert marine_dependent({"10.1", "9.1", "13.1"}, "Marine, Terrestrial")              # shearwater
    assert marine_dependent({"9.1", "10.1", "12.2", "13.3"}, "Marine, Terrestrial")      # sea turtle, one beach code
    assert not marine_dependent({"13.3"}, "Terrestrial")                                 # jerboa of coastal dunes
    assert not marine_dependent({"18"}, "Terrestrial")                                   # unknown habitat
    assert not marine_dependent({"4.4", "3.8", "13.3", "12.1"}, "Marine, Terrestrial")   # lark wintering on coasts
    assert not marine_dependent({"5.5", "9.1", "12.1", "10.1"}, "Freshwater (=Inland waters), Marine, Terrestrial")


def test_region_config_merges_over_shared(tmp_path):
    import pytest
    from ecoregions.config import load_config
    (tmp_path / "regions").mkdir()
    (tmp_path / "config.yaml").write_text("grid: {resolution: 90, countries: [PAK]}\npaths: {raw: data/raw}\n")
    (tmp_path / "regions" / "tx.yaml").write_text("grid: {countries: [USA]}\npaths: {interim: data/tx}\n")
    c = load_config(tmp_path / "config.yaml", "tx")
    assert c["grid"] == {"resolution": 90, "countries": ["USA"]}  # lists replace, dicts merge
    assert c["paths"] == {"raw": "data/raw", "interim": "data/tx"} and c["_region"] == "tx"
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "config.yaml", "nowhere")


def test_border_lines_around_a_state_skip_the_coast():
    import geopandas as gpd
    from shapely.geometry import box

    from ecoregions.boundaries import border_lines

    crs = "+proj=eqc +units=m"  # metres: the 50/100 m tolerances apply as in the real CRS
    land = gpd.GeoDataFrame({"ADM0_A3": ["USA"]}, geometry=[box(0, 0, 1e6, 1e6)], crs=crs)
    disputed = gpd.GeoDataFrame({"BRK_A3": []}, geometry=[], crs=crs)
    state = gpd.GeoDataFrame(geometry=[box(5e5, 0, 7e5, 2e5)], crs=crs)  # south edge on the coast
    ln = border_lines(land, disputed, crs, [], state)
    assert set(ln["style"]) == {"solid"}
    assert abs(ln.length.sum() - 6e5) < 1e3  # west, north and east sides only


def test_sgmc_groups_by_rock_type_then_age():
    from ecoregions.covariates import age_rank, sgmc_group

    assert sgmc_group("Sand", "Unconsolidated, undifferentiated", "Sand sheet deposits", 0.01) == 2
    assert sgmc_group("Fine-detrital", "Unconsolidated, undifferentiated", "Alluvium", 0.01) == 1
    assert sgmc_group("Limestone", "Sedimentary, carbonate", "Edwards Limestone", 105) == 4
    assert sgmc_group("Marlstone", "Sedimentary, carbonate", "Austin Chalk", 86) == 5
    assert sgmc_group("Sandstone-mudstone", "Sedimentary, clastic", "Woodbine Formation", 96) == 6
    assert sgmc_group("Sandstone", "Sedimentary, clastic", "Dockum Group", 220) == 7
    assert sgmc_group("Anhydrite", "Sedimentary, evaporite", "Salado and Castile Formations", 255) == 8
    assert sgmc_group("Granite", "Igneous, intrusive", "Town Mountain Granite", 1100) == 10
    assert sgmc_group("Rhyolite", "Igneous, volcanic", "Davis Mountains", 35) == 12
    assert sgmc_group("", "Water", "Water", float("nan")) == 11
    assert age_rank(0.5) == 0 and age_rank(100) == 3 and age_rank(1000) == 8


def test_v_measure_same_and_unrelated_partitions():
    from ecoregions.evaluate import v_measure

    assert v_measure(np.array([[5, 0], [0, 7]]))[2] == 1.0
    assert v_measure(np.array([[5, 5], [5, 5]]))[2] < 1e-9
    hom, com, _ = v_measure(np.array([[4, 0], [3, 0], [0, 6]]))  # ours finer than the reference
    assert hom == 1.0 and com < 1.0


def test_zone_types_regroups_only_the_named_types():
    cls = np.array([[1, 1, 2], [1, 3, 1]], "uint8")
    rain = np.array([[700, 900, 900], [1100, 1100, np.nan]], float)
    zone_types(cls, rain, src=[1], breaks=[800, 1000], into=[4, 5, 6])
    # type 1 cells go to their belt; types 2 and 3 stay; a cell without a value stays
    assert cls.tolist() == [[4, 5, 2], [6, 3, 1]]


def test_scorecard_endpoints_and_controls():
    from ecoregions.scorecard import (Ranked, boundary_contrast, coarsening_tree, graph_partition, jaccard_pairs,
                                      level_map, prediction_loss, renamed)
    from ecoregions.validate import jaccard_matrix

    rng = np.random.default_rng(0)
    # two groups of cells with different species pools
    A = rng.random((40, 40)) < np.r_[np.full(20, 0.6), np.full(20, 0.05)]
    B = rng.random((40, 40)) < np.r_[np.full(20, 0.05), np.full(20, 0.6)]
    P = np.vstack([A, B])
    g = np.r_[np.zeros(40, int), np.ones(40, int)]
    dev, test = np.arange(0, 80, 2), np.arange(1, 80, 2)
    prior = P[dev].mean(0)
    real = prediction_loss(P[dev].astype(float), g[dev], P[test].astype(float), g[test], prior, 20).mean()
    shuffled = rng.permutation(g)
    fake = prediction_loss(P[dev].astype(float), shuffled[dev], P[test].astype(float), shuffled[test], prior, 20).mean()
    assert real < fake
    # renaming units changes nothing
    r = renamed(g, rng)
    same = prediction_loss(P[dev].astype(float), r[dev], P[test].astype(float), r[test], prior, 20).mean()
    assert np.isclose(real, same) and Ranked(jaccard_matrix(P)).R(g) == Ranked(jaccard_matrix(P)).R(r)
    # boundary contrast: pairs across the real boundary differ more than pairs inside
    pr = np.array([(i, j) for i in range(80) for j in range(i + 1, 80)])
    D = jaccard_pairs(P, pr)
    bins = np.zeros(len(pr), int)
    assert boundary_contrast(D, g[pr[:, 0]] != g[pr[:, 1]], bins) > 0.3
    assert abs(boundary_contrast(D, shuffled[pr[:, 0]] != shuffled[pr[:, 1]], bins)) < 0.1
    # random maps on the land graph are contiguous and cover every cell
    ij = np.array([(i, j) for i in range(10) for j in range(10)])
    pos = {tuple(c): n for n, c in enumerate(ij)}
    lab = graph_partition(ij, pos, 4, rng)
    assert (lab >= 0).all() and len(np.unique(lab)) == 4
    import scipy.ndimage as ndi
    grid = lab.reshape(10, 10)
    assert all(ndi.label(grid == k)[1] == 1 for k in np.unique(lab))
    # coarsening tree: the most similar neighbours merge first, using community mixes only
    eco = pd.DataFrame(dict(unit=[1, 2, 3], area_km2=[1000.0, 1000.0, 1000.0],
                            community_mix=["Oak 90%; Grass 10%", "Oak 85%; Grass 15%", "Desert 100%"]))
    tree = coarsening_tree(eco, {(1, 2), (2, 1), (2, 3), (3, 2)})
    assert (tree.kept[0], tree.joined[0]) == (1, 2) and level_map(tree, [1, 2, 3], 2) == {1: 1, 2: 1, 3: 3}


def test_pixel_biome_key_follows_structure_then_climate():
    from ecoregions.biomes import CLASSES, pixel_biomes

    k = dict(ice=0.5, mangrove=0.3, wetland=0.5, forest=0.5, alpine=0.5, montane_tcold=-6.0, desert_aridity=0.2, sparse=0.5,
             xeric=dict(share=0.5, aridity=0.5), woody=0.5, moist_aridity=0.65, subtropical_tcold=15.0,
             mediterranean=dict(summer_rain_max=0.35, temp_c=[10, 20], tcold_min=0.0, aridity=[0.2, 0.65]))
    s = np.array([CLASSES.index(c) for c in ["forest_conifer", "grassland", "shrubland", "grassland", "woodland_broadleaf", "ice"]] + [-1])
    c = dict(temp_c=np.array([8, 18, 22, 25, 15, -5, 10.0]), tcold=np.array([-2, 5, 12, 18, 3, -15, 0.0]),
             aridity=np.array([0.8, 0.5, 0.3, 0.6, 0.4, 0.5, 0.5]), summer_rain=np.array([0.6, 0.6, 0.6, 0.6, 0.2, 0.5, 0.5]))
    assert pixel_biomes(s, c, k).tolist() == ["Temperate Conifer Forests", "Temperate Grasslands, Savannas & Shrublands",
                                              "Deserts & Xeric Shrublands", "Tropical & Subtropical Grasslands, Savannas & Shrublands",
                                              "Mediterranean Forests, Woodlands & Scrub", "Rock & Ice", ""]
