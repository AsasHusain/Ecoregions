"""Command-line entry point: ``python -m ecoregions <stage> [--region pak-afg] [--config config.yaml]``.

Stages are thin wrappers around module-level ``run(cfg)`` functions so the
Makefile (and notebooks) can call any step in isolation.
"""
from __future__ import annotations

import argparse
import importlib
import logging
import time

from .config import load_config, setup_logging

STAGES = {
    "download": "ecoregions.download",
    "preflight": "ecoregions.preflight",
    "grid": "ecoregions.boundaries",
    "dem": "ecoregions.dem",
    "terrain": "ecoregions.terrain",
    "hydro": "ecoregions.hydro",
    "covariates": "ecoregions.covariates",
    "features": "ecoregions.features",
    "provinces": "ecoregions.provinces",
    "segment": "ecoregions.segment",
    "snap": "ecoregions.snap",
    "attributes": "ecoregions.attributes",
    "communities": "ecoregions.communities",
    "vegcheck": "ecoregions.vegcheck",
    "fauna": "ecoregions.fauna",
    "faunal": "ecoregions.faunal",
    "global_fauna": "ecoregions.global_fauna",
    "biomes": "ecoregions.biomes",
    "assemble": "ecoregions.assemble",
    "evaluate": "ecoregions.evaluate",
    "scorecard": "ecoregions.scorecard",
    "merges": "ecoregions.merges",
    "confirm": "ecoregions.confirm",
    "plots": "ecoregions.plots",
    "audit": "ecoregions.audit",
    "names": "ecoregions.names",
    "validate": "ecoregions.validate",
    "crosscheck": "ecoregions.crosscheck",
    "package": "ecoregions.package",
    "webmap": "ecoregions.webmap",
    "staticmap": "ecoregions.staticmap",
}


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="ecoregions")
    p.add_argument("stage", choices=list(STAGES))
    p.add_argument("--config", default=None)
    p.add_argument("--region", default=None, help="mapped region (regions/<region>.yaml); default pak-afg")
    p.add_argument("-v", "--verbose", action="store_true")
    a = p.parse_args(argv)
    setup_logging(logging.DEBUG if a.verbose else logging.INFO)
    cfg = load_config(a.config, a.region)
    t0 = time.time()
    importlib.import_module(STAGES[a.stage]).run(cfg)
    logging.getLogger("ecoregions").info("stage %s finished in %.1f min", a.stage, (time.time() - t0) / 60)


if __name__ == "__main__":
    main()
