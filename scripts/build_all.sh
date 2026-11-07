#!/usr/bin/env bash
# `make all` for machines without make (e.g. Git Bash on Windows): same stages, same sentinels.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python}
DONE=data/interim/.done
mkdir -p "$DONE"
for s in download grid dem terrain hydro covariates features provinces segment snap attributes communities vegcheck fauna faunal assemble validate crosscheck package webmap staticmap globe; do
  [ -f "$DONE/$s" ] && { echo "skip $s"; continue; }
  echo "=== $s $(date +%T)"
  "$PY" -m ecoregions --config config.yaml "$s"
  touch "$DONE/$s"
done
echo "=== ALL DONE $(date +%T)"
