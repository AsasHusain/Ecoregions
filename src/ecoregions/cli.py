"""Command-line entry point: ``python -m ecoregions <stage> [--config config.yaml]``.

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
    "assemble": "ecoregions.assemble",
    "names": "ecoregions.names",
    "validate": "ecoregions.validate",
    "crosscheck": "ecoregions.crosscheck",
    "package": "ecoregions.package",
    "webmap": "ecoregions.webmap",
    "staticmap": "ecoregions.staticmap",
    "globe": "ecoregions.globe",
}


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="ecoregions")
    p.add_argument("stage", choices=list(STAGES))
    p.add_argument("--config", default=None)
    p.add_argument("-v", "--verbose", action="store_true")
    a = p.parse_args(argv)
    setup_logging(logging.DEBUG if a.verbose else logging.INFO)
    cfg = load_config(a.config)
    t0 = time.time()
    importlib.import_module(STAGES[a.stage]).run(cfg)
    logging.getLogger("ecoregions").info("stage %s finished in %.1f min", a.stage, (time.time() - t0) / 60)


if __name__ == "__main__":
    main()
