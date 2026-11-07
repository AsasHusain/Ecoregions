"""Stage 10b — (re)name units, score confidence and write descriptions.

Fast: reads the statistics written by ``attributes``. Run it after editing
``gazetteer.yaml`` (``make names``) without re-vectorizing.
"""
from .attributes import finalize


def run(cfg) -> None:
    finalize(cfg)
