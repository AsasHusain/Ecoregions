# Pre-registration: confirmatory tests for Texas (vegetation plots) and Iran

Registered 2026-11-12, before any of the data below were opened. The commit that adds this file, and its hash in
[`docs/ledger.md`](ledger.md), fix the plan. Anything decided after the data are opened is reported as a deviation.
Thresholds and rules follow the independent review's protocol (see the ledger), adapted where stated.

## 1. What has and has not been seen

* **sPlotOpen v2.0** (Sabatini et al. 2021, CC BY 4.0): only the header table's plot locations and counts per area
  were read (Texas box about 1,100 plots; Iran 277; Pakistan & Afghanistan none). No species composition, structure,
  cover or naturalness values have been looked at.
* **GBIF download for Iran** (0001335-261110092616502, doi:10.15468/dl.u2kux5): on disk, never read.
* **GBIF API cells for Iran**: never queried.
* **eBird Basic Dataset**: requested, not received.

## 2. Data roles

* **Texas.** The map under test is the published v0.2.0 (rainfall belts), frozen: the SHA-256 of
  `outputs/texas/ecoregions.gpkg` is recorded in the ledger before the plots are opened. All sPlotOpen plots in Texas
  are confirmation data. The GBIF calibration (development) cells, already used to build the map, supply the fitted
  species frequencies.
* **Iran.** The confirmation mask [`docs/confirmation/iran_mask.json`](confirmation/iran_mask.json) was drawn from
  geography alone (`ecoregions.confirm`): Iran's land in 2-degree squares; 14 of the 47 squares with at least a quarter
  of their area in Iran, drawn with seed 20261111; a 50 km buffer around them; both clipped to Iran. While the combined
  Pakistan, Afghanistan & Iran map is built:
  * development = Iranian records outside the confirmation squares and the buffer (rule calibration, audits);
  * buffer = used for nothing;
  * confirmation = Iranian records inside the squares (GBIF download, GBIF API cells, later eBird), plus **all**
    sPlotOpen plots in Iran wherever they lie (plots are never used in construction).
  The Pakistan and Afghanistan parts of the combined map stay exploratory: their records were used in development.

## 3. Endpoints

All endpoints are computed for each candidate and comparator on the same observations, with 2,000 bootstrap
resamples of 1-degree blocks (blocks of plots, or of cells) for every paired difference.

**A. Prediction of species lists (plants; birds in Iran).** Unit species frequencies are fitted on development cells
(GBIF, 0.125-degree cells, as `scorecard`), shrunk toward the overall frequency with 20 pseudo-cells, and predict each
confirmation observation's presence list. Brier loss averaged over species, then observations within blocks, then
blocks. Reported as the relative improvement over each comparator.
* *Plots:* species universe = species (binomials) present both in development cells and in at least one eligible plot.
* *Iran GBIF confirmation cells:* as `scorecard` (0.25 and 0.125 degrees, rarefied to 100 records, 100 draws).

**B. Boundary contrast.** Mean Jaccard dissimilarity of confirmation-observation pairs 1–50 km apart (plots) or
10–50 km (cells) lying across a unit boundary, minus that of pairs inside one unit, matched by 10 km distance bins,
bins weighted equally. Reported as the difference between candidate and comparator.

**C. Biome structure (plots).** Each eligible plot's formations are its true flags among Forest (or is_forest),
Shrubland, Grassland, Wetland and Sparse_vegetation. A plot agrees with a biome when one of its formations is in that
biome's defining set:

| Biome | Formations |
|---|---|
| any Forests biome (temperate, tropical, coniferous, broadleaf, moist, dry); Boreal Forests | Forest |
| Mediterranean Forests, Woodlands & Scrub | Forest, Shrubland |
| Temperate, or Tropical & Subtropical, Grasslands, Savannas & Shrublands | Grassland, Shrubland |
| Deserts & Xeric Shrublands | Shrubland, Sparse_vegetation |
| Montane Grasslands & Shrublands; Tundra | Grassland, Shrubland, Sparse_vegetation |
| Flooded Grasslands & Savannas | Wetland, Grassland |
| Mangroves | Forest, Wetland |
| Rock & Ice | Sparse_vegetation |

Score = agreement share, block-weighted. The same table scores our biomes and RESOLVE's.

## 4. Eligible observations

* Plots: inside the area's study polygon; `Location_uncertainty` known and at most 1,000 m; `Naturalness` not 3
  (missing kept); recorded 1950 or later; at least 5 species (endpoints A, B); `Plant_recorded` "All vascular plants"
  or "Not specified" (A, B; any for C). For C, plots with no true formation flag are left out.
* GBIF Iran cells: as `scorecard` (at least 100 records and the minimum species count; at least half the cell's area
  in Iran and on the map); a cell crossing a role boundary is left out.

## 5. Candidates and comparators

* Candidates: our floral ecoregions (full detail); our floral ecoregions coarsened to RESOLVE's count along the
  construction-only tree (`coarsening_tree`, archived before opening); for birds, our faunal ecoregions; for C, our
  biomes from the pixel-level key (`biomes`, frozen in commit with this file; `biomes.use_key` decides only what the
  published maps show).
* Comparators: RESOLVE 2017 ecoregions (C: RESOLVE biomes); EPA Level III and IV (Texas only); our hand-set biomes
  (C); climate-only k-means at our and RESOLVE's counts; 99 random contiguous maps on the land-cell graph; the
  spatial-only predictor (10 nearest development cells; A only).

## 6. Decision rules (per area x taxon x endpoint)

* **Improvement at equal detail:** the 95 % interval's lower bound on relative prediction improvement over RESOLVE
  > +0.02 **and** on the boundary-contrast difference > +0.02.
* **Non-inferiority at full detail:** both lower bounds > -0.01.
* **Biomes (C):** improvement over RESOLVE if the agreement difference's lower bound > 0; non-inferior if > -0.02.
* Anything else is "not demonstrated", including when too few observations qualify (fewer than 20 blocks with
  eligible observations: reported, but descriptive only). No threshold is changed after opening; Texas and Iran, and
  plants and birds, are reported separately and not averaged.

## 7. Order of operations

1. Commit this file, the mask, the frozen biome key and the coarsening-tree code; record the commit hash.
2. Texas: record the v0.2.0 map hash; then open the plots once and run A, B and C.
3. Iran: build the combined map using development records only; archive its coarsening tree; freeze it (hash); then
   open the confirmation records once and run A, B and C.
4. Report every result, including failures, in the README and the ledger.

## 8. Amendment of 2026-11-14: every area mapped from now on

Added after the Iran tests (sections 1–7 unchanged). It applies to every area whose construction starts after this
section is committed, and adds a second, temporal test for every area, including those already mapped.

**Sealed squares.** [`docs/confirmation/world_mask.json`](confirmation/world_mask.json) was drawn from geography alone
(`scripts/draw_world_mask.py`): every 2-degree square with at least a quarter of its area on land not yet mapped
(Pakistan, Afghanistan & Iran and Texas excluded), each given a rank in one random order (seed 20261114); squares
ranked in the lowest 30 % are sealed. Around them, a buffer at least 50 km wide in every direction. For an area being
built, the sealed squares and buffer inside it play the roles of section 2: records there (GBIF downloads and API
cells, eBird if available, any other occurrence data) are not used for rules, calibration, audits or any choice made
while building, until the area's frozen map hash is committed.

**Enough sealed data, settled before building, from counts only.** Before an area's construction starts, GBIF record
*counts* per 0.25-degree cell inside its sealed squares are read for plants and for birds (counts only: no names,
dates or other fields). A cell is eligible with at least 100 records; a 1-degree block is eligible with at least one
eligible cell. If a taxon has fewer than 20 eligible blocks or fewer than 40 eligible cells (the scorecard needs 30
after its own filters), the area seals its next squares in rank order, re-counting after each, until both are met or
half the area's squares are sealed. A taxon still short is declared descriptive-only in the ledger before building.
The counts, the final rank cut-off and any declaration are recorded in the ledger before construction starts.

**Temporal test (all areas).** Observations made on or after 2026-11-14 (GBIF `eventDate`) are never used in
construction. For each area, the count of such records is checked (counts only) at most every six months; once a
taxon reaches 20 eligible blocks and 40 eligible cells, the area's current map hash is committed and the test is run
once on those records, with the endpoints and decision rules of sections 3 and 6. These records are new in time but
not in space (they fall among the development cells), so the result is reported as a temporal test, separately from
the spatial confirmation.

**Order of operations for each new area.** (1) Count the sealed data and set the rank cut-off; (2) build using
development records only; (3) freeze (hash in the ledger, committed); (4) open the sealed records once and run the
tests; (5) report every result, including failures.
