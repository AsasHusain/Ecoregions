#!/usr/bin/env bash
# `make all` for machines without make (e.g. Git Bash on Windows): same stages, same sentinels.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python}
REGION=${REGION:-pak-afg}
DONE=$(PYTHONPATH=src "$PY" -c "from ecoregions.config import load_config; print(load_config(None, '$REGION')['paths']['interim'])")/.done  # each region's own interim folder
mkdir -p "$DONE"
for s in download preflight grid dem terrain hydro covariates features provinces segment snap attributes communities vegcheck fauna faunal assemble evaluate validate crosscheck package join webmap staticmap; do
  [ -f "$DONE/$s" ] && { echo "skip $s"; continue; }
  echo "=== $s $(date +%T)"
  "$PY" -m ecoregions --config config.yaml --region "$REGION" "$s"
  touch "$DONE/$s"
done
echo "=== ALL DONE $(date +%T)"
