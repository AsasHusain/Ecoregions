# Analysis ledger

Every analysis whose held-out results were looked at, and what was decided after. Kept because the held-out GBIF
cells of Pakistan & Afghanistan and Texas have been viewed many times while the maps were developed: all their
results are **exploratory**, and an independent confirmation needs observations no one has looked at (see
`docs/review/`). New entries go at the bottom, dated, before the result is used.

| Date | Area | Analysis (code) | Held-out results viewed | Decision taken afterwards |
|---|---|---|---|---|
| 2026-11-06 | Pakistan | Assemblage test v0.2.0 (`evaluate`) | yes (all cells, then the checkerboard from v0.2.3) | Rules revised in v0.2.1–v0.2.3 (calibration cells used for thresholds) |
| 2026-11-08 | Pakistan | Checkerboard holdout introduced; audit on calibration cells only (`vegcheck`, `audit`) | yes | Rule fixes v0.2.3 |
| 2026-11-10 | Pakistan & Afghanistan | Rain-season rule variants under the rarefied test (`outputs/validation_rain_season_variants.csv`) | yes, for each variant | Not adopted |
| 2026-11-11 | Texas | Full evaluation, rarefied (`evaluate`) | yes | Central Texas flagged |
| 2026-11-11 | both | Equal-detail merges, three merge rules (`scripts/experiments/equal_detail.py`) | yes, for each rule | Third rule (calibration R) preferred because the first two snowballed |
| 2026-11-11 | Texas | 0.1-degree plant test, central band (`scripts/experiments/fine_test.py`) | yes | Central band redrawn |
| 2026-11-11 | Texas | Rainfall belts chosen on calibration cells only (`scripts/experiments/choose_belts.py`) | no (calibration) | Breaks at 1,100 mm; a 10 % minimum belt share added after seeing the calibration ranking |
| 2026-11-11 | Texas | Belts scored on held-out cells at 0.25 and 0.1 degrees (`equal_detail.py`, `fine_test.py`) | yes | Belts kept (not yet published) |
| 2026-11-11 | both | Scorecard v1 (rarefied 100/300, full Simpson; z-scores, accuracy, IndVal; controls) | yes | Full lists + Simpson + z chosen as primary because it alone passed both controls (a data-dependent choice) |
| 2026-11-11 | both | Independent review (Codex, `docs/review/`) | n/a | Scorecard rebuilt to the review's structure (v2); five scorecard bugs fixed |
| 2026-11-11 | both | Scorecard v2: prediction loss and boundary contrast, land-graph random maps, construction-only coarsening, block bootstrap (`scorecard`) | yes | Plants: on par with RESOLVE (Pakistan & Afghanistan +1.4 % at 0.125 deg on 2 % of the land); birds (floral map): RESOLVE ahead by ~1.4 %; no difference reaches the protocol's +2 % |
| 2026-11-11 | both | Biome key: first run's type assignments read (no test) | no | Key revised from biome definitions: xeric-shrubland rule, mild-winter condition for Mediterranean, subtropical at a 15 C coldest month; then frozen before the structure test |
| 2026-11-11 | both | Biome structure test (WorldCover natural classes; `scorecard.biome_structure`) | yes | Key biomes beat RESOLVE in Pakistan & Afghanistan (+20 %), lose in Texas (-46 %); key not adopted (`biomes.use_key: false`) |
| 2026-11-11 | both | Type merges (`merges`): construction-only tree, 5-fold block cross-validation on calibration cells, one-SE rule fixed in advance | no (calibration) | The rule collapsed Texas to 7 types and Pakistan & Afghanistan to 1: the cross-validated loss is flat (best level equals the full map within a tenth of a standard error). No merges applied; GBIF calibration data cannot resolve type distinctions |
| 2026-11-12 | both | Pre-registration committed (30572c9): Texas plot tests and the Iran confirmation design; Iran mask drawn (14 of 47 squares) | no | Biome key revised to pixel level before registration (disclosed: after the WorldCover test, which is therefore exploratory) |
| 2026-11-12 | Texas | v0.2.0 published (rainfall belts); map frozen for the plot tests: outputs/texas/ecoregions.gpkg SHA-256 92c7d6901afd66889a3b177dedad45f08b2557bf1e9464bce430f3e58d6e63da | — | Plots to be opened once, after this entry is committed |
| 2026-11-12 | Texas | Pre-registered sPlotOpen tests, run once on the frozen v0.2.0 (`plots`; outputs/texas/plots_*.csv) | yes (first opening) | Inconclusive by the registered rule: 168 plots with species lists in 5 one-degree blocks (20 required), so descriptive only. A: ours vs RESOLVE -1.7 % (-3.0 to +0.2), all maps barely above the overall prior; B: no plot pairs cross our, RESOLVE's or EPA III's boundaries within 50 km (not computable); C: 52 % of plots agree with our biomes, 65 % with RESOLVE's (6 blocks). Not re-run with other filters. Lesson: sPlotOpen is too sparse and clustered in Texas to judge a regional map |
| 2026-11-12 | both | Public history squashed into one commit, to remove references to the private website | n/a | Commits cited above (30572c9 pre-registration, 54fec92 Texas v0.2.0 freeze, c7d6e48 plot results) are kept in a local archive. Fingerprints at the squash: docs/preregistration.md SHA-256 31118227d0885fe6dcccb92065e9c09b2c01de5eaa45bae1e4d0c3d4b035db56; docs/confirmation/iran_mask.json SHA-256 d235a4770f04d208473c437263d7fd7fe29c365505ea51de60ed0281d9283c00. The Iran confirmation records were still unopened at this commit, so its public timestamp predates them |
