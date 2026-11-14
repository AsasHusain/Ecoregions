from ecoregions.config import load_config
from ecoregions.preflight import check_seeds, seed_groups


def test_far_cluster_is_a_second_group():
    near = [[61.0, 31.0], [61.5, 31.2], [62.0, 31.4]]
    assert seed_groups(near, 400) == 1
    assert seed_groups(near + [[58.4, 26.4], [59.0, 27.3]], 400) == 2


def test_current_regions_pass():
    for region in ("pak-afg-irn", "texas", "pak-afg"):
        assert check_seeds(load_config(region=region)["provinces"]["units"], 400) == []
