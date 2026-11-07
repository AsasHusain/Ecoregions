import numpy as np
import pandas as pd

from ecoregions.attributes import aridity_class, compass, texture_class
from ecoregions.assemble import disambiguate, join_same_type, majority, merge_small_nested
from ecoregions.colors import shade
from ecoregions.communities import classify, load_rules, northness
from ecoregions.crosscheck import expectation
from ecoregions.download import SourceUnavailable, gbif_chunks, gbif_pages
from ecoregions.fauna import habitat_code
from ecoregions.faunal import beta_sim, clean_fragments, explained_turnover
from ecoregions.vegcheck import binomial, precision_recall
from ecoregions.grid import Grid, block_mode, block_reduce, is_current, write
from ecoregions.segment import adjacency, label_means, merge_small
from ecoregions.snap import internal_edges, make_contiguous
from ecoregions.validate import anosim_r, jaccard_matrix, ranks_of


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


def test_gbif_pages_reads_all_and_refuses_oversized_queries():
    recs = list(range(650))

    def get(p):
        if p["limit"] == 0:
            return {"count": len(recs)}
        o = p["offset"]
        return {"results": recs[o:o + p["limit"]], "endOfRecords": o + p["limit"] >= len(recs)}

    assert list(gbif_pages(get, {})) == recs
    try:
        list(gbif_pages(lambda p: {"count": 100_001}, {}))
        assert False, "should refuse queries above the paging limit"
    except SourceUnavailable:
        pass


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


def test_gbif_chunks_bisect_until_small_without_losing_records():
    rng = np.random.default_rng(0)
    pts = rng.uniform([60, 23], [78, 37], size=(500, 2))
    pts[:40] = [69.0, 30.0]  # a dense cluster (a herbarium locality)

    def get(p):
        w, e = map(float, p["decimalLongitude"].split(","))
        s, n = map(float, p["decimalLatitude"].split(","))
        inside = (pts[:, 0] >= w) & (pts[:, 0] <= e) & (pts[:, 1] >= s) & (pts[:, 1] <= n)
        return {"count": int(inside.sum()), "idx": np.flatnonzero(inside)}

    ch = list(gbif_chunks(get, {}, (60, 23, 78, 37), cap=60))
    assert all(n <= 60 for _, n in ch)
    got = set().union(*(set(get(p)["idx"]) for p, _ in ch))
    assert got == set(range(500))  # nothing dropped (edge duplicates are removed later)


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
