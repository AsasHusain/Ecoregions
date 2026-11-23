# World Atlas of Biotic Regions: method plan

This plan is fixed before the first map of the atlas is built. Changes to it are recorded in
[`ledger.md`](ledger.md), and the test rules are registered in [`preregistration.md`](preregistration.md) before any
result is seen.

## Purpose

The atlas shows the natural communities of every place, including land that is now farmed or built on. For any
site it gives the reference to restore toward: the natural vegetation expected there, the Biotic Region it belongs
to, and the native plants and animals of that region, with the evidence behind each.

## Definition

A Biotic Region is a connected area whose native plants and animals, taken together, form a community distinct from
those of its neighbours. Each region holds a characteristic mix of habitats. Size neither qualifies nor disqualifies
a region: a small, coherent area with a distinctive native biota is eligible.

The target is the native biota of the recent historical period, before modern land conversion:

- **Plants:** records of species native where they were made, checked against the World Checklist of Vascular
  Plants. A species native in a neighbouring botanical area within 100 km also counts as native, so status does not
  change at a national border. Planted and cultivated records are excluded.
- **Animals:** native ranges of mammals, birds, reptiles and amphibians, including ranges where a species is recently
  extinct (reported with and without them).
- **Converted land:** the reference community is inferred from surviving natural analogues, environment and records,
  and each cell states whether its community is observed or reconstructed.

## Evidence

All evidence sits on one 10 km equal-area grid covering the whole mapped area in one run, so the map has no seams at
national borders.

| Evidence | Source |
|---|---|
| Plant composition | A dissimilarity model of native plant turnover, fitted on plant records and vegetation plots |
| Animal composition | Range maps, with birds, mammals, and reptiles with amphibians weighted equally |
| Vegetation mix | The share of each natural vegetation type around each cell, from the 90 m vegetation map |
| Environment | Climate, terrain, soils and geology |
| Land-surface seasonality | Multi-year satellite measures of green-up, peak and senescence, from natural ground only |
| Canopy structure | Satellite lidar measures of canopy height and cover, as a check on forest, woodland and open types |

The kinds of evidence overlap: the vegetation map and the plant model both draw on climate. They are therefore added
in steps, starting from plants and animals alone, and each addition is kept only if it improves predictions on places
not used to build the map.

Satellite measures describe today's land surface, so they are read only from unconverted ground and never stand in
for the reference community on converted land.

Regions are grown on the grid's neighbour graph, so similarity is computed between neighbouring cells and to region
centres, never between every pair of cells.

## Boundaries at every scale

Species evidence decides which region each place belongs to; it is reliable at 10 km and no finer. The landscape
decides where each boundary runs. Boundaries are therefore drawn in nested steps, each finer than the last:

| Level | Grid | Boundary may move within | Placed on |
|---|---|---|---|
| Regions | 10 km | — | species composition and the other evidence above |
| 1 | 1 km | 15 km either side of the level above | terrain, climate, soils, geology (global 1 km layers) |
| 2 | 90 m | 2 km either side | terrain, land cover on natural ground, geology |
| 3 | 30 m | 200 m either side | terrain and land cover on natural ground |

At each level the boundary between two regions moves to the line in its band where the two regions' own ground
differs most: the band's cells are scored on the measures that best separate the cores of the two regions (cells
deep inside each, on that level's grid), and the boundary follows the sharpest change in that score. Where nothing
in the band separates them, the boundary stays where the level above put it. A level never adds a region, removes
one, or moves ground between regions outside its band, so a place belongs to the same region at every zoom.

Finer levels are computed only inside the bands, a few percent of the land, so their cost follows the length of the
boundaries, not the area mapped, and fine data are downloaded only where a band needs them.

The band's own width is kept: a boundary on a crest narrows to a few hundred metres, one across a plain may stay
kilometres wide. Each boundary carries this width as its transition zone.

**Viewing.** Each scale of view has its own level: the 10 km regions, smoothed, for the world; level 1 for a
country; levels 2 and 3 for a valley or a slope.

## Continuity across borders

Every input is checked for changes at national borders that the landscape does not show, and each is repaired at its
source:

| Source | Repair |
|---|---|
| Geology compiled country by country | One rock-type legend for all sheets; in border strips where sheets disagree, rock type is predicted from terrain and soils |
| Native status by botanical area | Native in any botanical area within 100 km counts as native |
| Range maps that stop at a border | Extended across the border within the species' elevation and habitat limits, checked against records on the far side |
| Vegetation layer built in parts | Its mix of types per 10 km cell is checked at every border like any other input; a step there that the landscape does not show is repaired before the layer is used |

The map itself is built in one run over everything mapped, so no boundary can come from joining separate builds.
Every boundary, at every level, is then tested against the borders it lies near (pre-registration, section 10).

## Evidence support

Each cell carries three measures:

- how far its conditions lie from those where plant data exist;
- how many records and plots lie nearby;
- how much the plant model's prediction varies when it is refitted.

Where plant support is weak, the map marks the cell provisional and says that its plant distinctions are modelled,
not observed.

## Testing

**Development.** Choices of method, weights, neighbourhood size, number of regions and boundary refinement are made
with spatial cross-validation. Each test fold's records are removed before the plant model is fitted and the map is
built. Choices are tuned inside the training folds only.

**Level of detail.** A region is split only where its parts differ in held-out species composition repeatedly across
folds, and the split loses no accuracy for any well-sampled group. Splits that cannot be tested for lack of data are
kept and marked provisional.

**Comparisons**, at equal numbers of regions:

| Comparison | Purpose |
|---|---|
| Regions from location alone | Null: nearness only |
| Regions from climate alone, and from climate and terrain | Nulls: environment only |
| The vegetation map and the animal map separately | The project's other designs |
| RESOLVE *Ecoregions 2017* | An established map, as an outside benchmark |

**Confirmation.** One chosen map is frozen, with its test plan, before sealed records are opened. They are opened
once. Before freezing, a script counts how many sealed test areas are usable, without revealing any species. The
test rules, thresholds and data roles are registered in [`preregistration.md`](preregistration.md), section 10.

## Claims

Results are reported per group (plants, birds, mammals, reptiles) and support three separate claims:

1. **Useful:** the map captures biological structure that holds at new places, shown by beating the null comparisons.
2. **Best of the project's designs:** at least as good as each alternative for every well-sampled group.
3. **An improvement on established maps:** stated only if the map beats RESOLVE and other outside benchmarks.

If the map does not beat a null comparison with adequate precision, added value has not been shown under that test.
