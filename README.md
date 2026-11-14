# Revised Ecoregions of the World

A new ecoregion map of the whole planet, to replace the coarse WWF/RESOLVE *Ecoregions 2017*.
The world is mapped one area at a time with one method, and each area is published when it is
done.

## Progress

| Area | Version | Land area | Ecoregions | Types | Faunal ecoregions | Maps |
|---|---|---|---|---|---|---|
| Pakistan, Afghanistan & Iran (with Kashmir, Aksai Chin and the Shaksgam Valley) | 0.5.0 | 3,281,449 km² | 165 | 44 | 20 | [ecoregions](outputs/pak-afg-irn/ecoregions_map_v0.5.0.png) · [faunal](outputs/pak-afg-irn/faunal_map_v0.5.0.png) |
| Texas | 0.2.0 | 685,641 km² | 31 | 19 | 12 | [ecoregions](outputs/texas/ecoregions_map_v0.2.0.png) · [faunal](outputs/texas/faunal_map_v0.2.0.png) |
| **Mapped so far** | | **3,967,090 km²** | **196** | | | **2.7 % of the world's land** |

World land: 146.7 million km² (Natural Earth, equal-area, Antarctica included). Next: Central Asia, built as its
own area and joined to its neighbours at the border.

| Date | Done |
|---|---|
| 2026-11-06 | Pakistan, Azad Kashmir and Gilgit-Baltistan (v0.2.0) |
| 2026-11-08 | Jammu and Kashmir, Ladakh, Demchok, Aksai Chin and the Shaksgam Valley (v0.3.0) |
| 2026-11-09 | Afghanistan (v0.4.0) |
| 2026-11-11 | Pakistan & Afghanistan v0.4.2; Texas v0.1.0 |
| 2026-11-12 | Texas v0.2.0 (central band regrouped by rainfall); held-out scorecard and pre-registered confirmation tests |
| 2026-11-13 | Iran: Pakistan, Afghanistan & Iran v0.5.0; first confirmation test on sealed records |

## How the maps are made

Each area gets three nested levels and one separate map:

- **Ecoregions**: the main map. Potential natural vegetation communities are mapped at 90 m by
  explicit rules on terrain, climate, soils, geology and hydrology (one rule file per area, e.g.
  [`communities.yaml`](communities.yaml)). Communities are grouped into **ecoregion types**, and
  each ecoregion is one continuous area of one type, at least 1,000 km², with boundaries snapped
  to landforms.
- **Biomes**: the RESOLVE biome of each ecoregion type.
- **Faunal ecoregions** (separate map): areas with distinct mammals, birds, reptiles and amphibians,
  clustered from IUCN and BirdLife ranges on 10 km cells. They never cut the ecoregions, and the
  ecoregions never cut them.

Each map is tested on GBIF species records: half the sample cells (a checkerboard) are kept out of rule
calibration and used only for testing, against RESOLVE, reference maps (comparison only), a climate-only
clustering and random maps of the same size.

> **The held-out results are exploratory.** The held-out cells were looked at many times while the
> maps were developed (every such analysis is listed in [`docs/ledger.md`](docs/ledger.md)), so they cannot
> confirm an improvement independently. Iran's sealed records are the first confirmation test (below). An independent methodological review set out the
> protocol a confirmation must follow: fresh, spatially separated observations, two pre-declared measures
> (prediction of held-out species lists, and species turnover across boundaries) and fixed decision rules.

## Pakistan, Afghanistan & Iran (v0.5.0)

[![Ecoregions of Pakistan, Afghanistan & Iran](outputs/pak-afg-irn/ecoregions_map_v0.5.0_preview.png)](outputs/pak-afg-irn/ecoregions_map_v0.5.0.png)

[![Faunal ecoregions of Pakistan, Afghanistan & Iran](outputs/pak-afg-irn/faunal_map_v0.5.0_preview.png)](outputs/pak-afg-irn/faunal_map_v0.5.0.png)

> Borders: Afghanistan's and Iran's borders are drawn solid. Pakistan's border is drawn solid around Pakistan, the Siachen Glacier area, Jammu and Kashmir and Demchok; the line where Jammu and Kashmir meets the rest of Pakistan is dotted. Aksai Chin and the Shaksgam Valley are shown as part of China. All of these areas are mapped.

Exploratory, held-out GBIF cells (`scorecard`): how much better each map predicts held-out species lists than
RESOLVE 2017 (positive: ours better; 95 % interval from resampling 1-degree blocks):

| | Ours vs RESOLVE | Land tested |
|---|---|---|
| Plants, floral ecoregions, 0.25 / 0.125 degrees | +0.2 % (-0.3 to +0.6) / +1.2 % (+0.4 to +1.9) | 3.8 % / 1.1 % |
| Birds, faunal ecoregions, 0.25 degrees | +1.3 % (+0.4 to +2.3) | 5 % |

**Confirmation (pre-registered, run once on the frozen map).** Iranian records in 14 sealed two-degree squares, and
all sPlotOpen vegetation plots in Iran, were opened only after the map's hash was committed:

| Test | Result | Verdict by the registered rules |
|---|---|---|
| Birds, sealed GBIF cells (26 blocks) | ours -2.9 % (-4.6 to -1.6); faunal map -0.9 % (-3.4 to +1.0) | **Not demonstrated**: RESOLVE predicts the sealed bird lists better |
| Plants, sealed GBIF cells | 29 cells, below the minimum of 30 | Not testable |
| Vegetation plots (12 blocks; 20 required) | species lists -1.0 % (-2.5 to +0.9); biome agreement 94 % vs 54 % | Descriptive only |

So the improvement seen on the held-out cells did not hold up for birds on fresh, spatially separate records, and
plants could not be tested. Details in [`docs/ledger.md`](docs/ledger.md).

## Texas (v0.2.0)

v0.2.0 regroups the central band (Blackland, Post Oak, Cross Timbers, Grand Prairie) by rainfall rather than soil:
plants there turn over east to west with rain, across the old north-south soil bands.

[![Ecoregions of Texas](outputs/texas/ecoregions_map_v0.2.0_preview.png)](outputs/texas/ecoregions_map_v0.2.0.png)

[![Faunal ecoregions of Texas](outputs/texas/faunal_map_v0.2.0_preview.png)](outputs/texas/faunal_map_v0.2.0.png)

Exploratory, as above:

| | Ours vs RESOLVE | Land tested |
|---|---|---|
| Plants, floral ecoregions, 0.25 / 0.125 degrees | -0.4 % (-0.8 to 0.0) / 0.0 % (-0.5 to +0.5) | 44 % / 26 % |
| Birds, faunal ecoregions, 0.25 degrees | 0.0 % (-1.0 to +0.9) | 48 % |

At two to three times RESOLVE's detail, the plant map predicts plant assemblages as well as RESOLVE (the protocol's
non-inferiority margin is -1 %). Agreement with the EPA ecoregions: V-measure 0.66 (Level III), 0.64 (Level IV).
A confirmatory test on independent vegetation plots (sPlotOpen), pre-registered in [`docs/preregistration.md`](docs/preregistration.md)
and run once, was inconclusive: the eligible plots fall in only 5 one-degree blocks (20 required), so it is reported as
descriptive in [`docs/ledger.md`](docs/ledger.md).

## Outputs

Pakistan & Afghanistan v0.4.2 in `outputs/`, every other area in `outputs/<area>/` (Pakistan, Afghanistan & Iran in
`outputs/pak-afg-irn/`, its GeoPackages zipped to stay under GitHub's file limit):

| File | Contents |
|---|---|
| `ecoregions.gpkg` | ecoregions, ecoregion types, faunal ecoregions, biomes, study area, borders, rivers (Albers equal-area) |
| `ecoregions.csv` | ecoregion attributes: type, biome, faunal ecoregion, area, elevation, climate, population, land converted, countries |
| `ecoregions_map_v*.png`, `faunal_map_v*.png` | static maps |
| `physiographic.gpkg`, `physiographic_map_v*.png` | the physiographic layer the ecoregions are snapped to |
| `validation_*.csv`, `ecoregion_audit.csv` | validation tests and the per-ecoregion audit |

## How to run

```bash
pip install -r requirements.txt && pip install -e .
make all                     # Pakistan & Afghanistan end to end
make all REGION=texas        # another area (settings in regions/<area>.yaml)
python -m ecoregions --region texas <stage>   # one stage
make test
```

On Windows without `make`: `bash scripts/build_all.sh` from Git Bash. Raw downloads are cached in
`data/raw/` (never committed: several are licensed for non-commercial use only).

## Data sources

| Data | Use | Citation | Licence |
|---|---|---|---|
| Copernicus GLO-90 DEM | terrain, hydrology | © DLR e.V. 2010–2014 and © Airbus Defence and Space GmbH 2014–2018, provided under COPERNICUS by the European Union and ESA | Copernicus DEM licence |
| HydroRIVERS v1.0, HydroBASINS v1c | rivers, basins | Lehner & Grill 2013, *Hydrol. Process.* 27: 2171–2186 | CC BY 4.0 |
| CHELSA v2.1 (1981–2010) | climate | Karger et al. 2017, *Sci. Data* 4: 170122 | EnviDat open terms |
| SoilGrids 2.0 | soils | Poggio et al. 2021, *SOIL* 7: 217–240 | CC BY 4.0 |
| Geologic map of South Asia (geo8ag) | geology (Pakistan & Afghanistan) | Wandrey & Law 1998, USGS OFR 97-470-C | public domain |
| USGS State Geologic Map Compilation | geology (Texas) | Horton 2017, doi:10.5066/F7WH2N65 | public domain |
| ESA WorldCover 2021 | land converted | Zanaga et al. 2022, doi:10.5281/zenodo.7254221 | CC BY 4.0 |
| GHS-POP R2023A (2020) | population | Schiavina et al. 2023, JRC | CC BY 4.0 |
| IUCN Red List range maps and assessments | faunal ecoregions, Red List status | IUCN 2026, *The IUCN Red List of Threatened Species* | IUCN terms (non-commercial) |
| BirdLife bird distribution maps 2025 | faunal ecoregions | BirdLife International and Handbook of the Birds of the World 2025 | BirdLife terms (non-commercial) |
| GBIF occurrence downloads | validation, audit | GBIF.org 2026, doi:10.15468/dl.22b9a4, doi:10.15468/dl.6t43nn, doi:10.15468/dl.gv4z7m | per record; only aggregates stored |
| RESOLVE Ecoregions 2017 | biomes, comparison | Dinerstein et al. 2017, *BioScience* 67: 534–545 | CC BY 4.0 |
| EPA Level III and IV ecoregions | comparison only (Texas) | Griffith et al. 2007, *Ecoregions of Texas*, U.S. EPA | public domain |
| Jung et al. 2020 habitat map; Hengl et al. 2018 potential natural vegetation | checks of the community rules | Jung et al. 2020, *Sci. Data* 7: 220; Hengl et al. 2018, *PeerJ* 6: e5457 | CC BY 4.0 |
| WhiteboxTools | geomorphons | Lindsay 2016, *Comput. Geosci.* 95: 75–84 | MIT |
| Mapzen Terrain Tiles | site relief and 3D terrain | AWS Open Data, Terrain Tiles | per source (attribution) |
| Global Mangrove Watch v3 | check of the mangrove rules | Bunting et al. 2022, *Remote Sens.* 14: 3657 | CC BY 4.0 |
| RGI 7.0 | glaciers | RGI 7.0 Consortium 2023, doi:10.5067/f6jmovy5navz | public |
| Natural Earth 10 m | boundaries | naturalearthdata.com | public domain |

## Licence and ownership

Copyright © 2026 Asas Husain. All rights not granted below are reserved.

* **Code** (`src/`, `tests/`, the rule files and configuration): [PolyForm Noncommercial
  1.0.0](LICENSE). Anyone may use, copy and change it for non-commercial purposes (research,
  teaching, personal study, public-interest organisations) and must pass on the line beginning
  `Required Notice:` in `LICENSE`. Any commercial use needs a separate written licence from the author.
* **Maps and derived data** (`outputs/`, the static maps): [CC BY-NC
  4.0](https://creativecommons.org/licenses/by-nc/4.0/). Credit: "Husain, A. (2026) Revised
  Ecoregions of the World: <area>, vX.Y.Z. https://github.com/AsasHusain/Ecoregions", plus
  the source attributions above. No commercial use without written permission.
* **Third-party material keeps its own terms.** The source datasets are cited above and their raw files are never redistributed. Several (IUCN and BirdLife ranges,
  some GBIF records) are licensed for non-commercial use only, which the derived products inherit.
* **Commercial licensing and permissions:** contact the author through GitHub (@AsasHusain).
* Versions released before this change (up to commit c5ab452, November 2026) were published
  under MIT and remain so for anyone who obtained them then; this licence applies from v0.4.1.
