# Revised Ecoregions of the World

Revised Ecoregions of the World maps the world's ecoregions from open data on native life. An ecoregion is a
connected area whose native plants and animals together form a community distinct from those of its neighbours.
Each region holds its own mix of habitats; a riparian forest or a juniper belt is a habitat inside a region, not a
region of its own.

The map shows natural communities everywhere, including land that is now farmed or built on. For any site it gives
the reference to restore toward: the natural vegetation, the ecoregion and its native plants and animals.

The map stands on its own evidence. Each map is tested against established maps, such as RESOLVE
*Ecoregions 2017*, and against maps built from climate or location alone.

## Status

The first map, now in development, covers Pakistan, Afghanistan, Iran, Central Asia, the Caucasus and Anatolia as one
continuous area of about 5.5 million km². It combines six kinds of evidence on a 10 km grid:

- native plant composition, modelled from plant records and vegetation plots;
- native mammals, birds, reptiles and amphibians, from range maps;
- the mix of natural vegetation types, from the vegetation layer;
- climate, terrain, soils and geology;
- satellite measures of seasonal greenness, from natural ground;
- satellite lidar measures of canopy height and cover, as a check on woodland and open vegetation.

Each kind of evidence stays in the model only if it improves predictions of species at places not used to build the
map, and every cell reports the plant evidence behind it. The method is in [`docs/plan.md`](docs/plan.md).

## Earlier maps

The project's earlier maps, of natural vegetation and of vertebrate regions (v0.2 to v0.7), are kept with their test
results in `outputs/<area>/`. Their vegetation rules, landform units and range-map analysis carry forward as layers
of the ecoregion map.

## How to run

```bash
pip install -r requirements.txt && pip install -e .
make all REGION=pak-afg-irn  # one area end to end
python -m ecoregions --region texas <stage>   # one stage (settings in regions/<area>.yaml)
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
| Geologic maps of South Asia (geo8ag), Iran (geo2cg) and the former Soviet Union (geo1ec) | geology | Wandrey & Law 1998, USGS OFR 97-470-C; USGS OFR 97-470-G; USGS OFR 97-470-E, doi:10.5066/P9GUL0PQ | public domain |
| USGS State Geologic Map Compilation | geology (Texas) | Horton 2017, doi:10.5066/F7WH2N65 | public domain |
| ESA WorldCover 2021 | land converted | Zanaga et al. 2022, doi:10.5281/zenodo.7254221 | CC BY 4.0 |
| GHS-POP R2023A (2020) | population | Schiavina et al. 2023, JRC | CC BY 4.0 |
| IUCN Red List range maps and assessments | animal layer | IUCN 2026, *The IUCN Red List of Threatened Species* | IUCN terms (non-commercial) |
| BirdLife bird distribution maps 2025 | animal layer | BirdLife International and Handbook of the Birds of the World 2025 | BirdLife terms (non-commercial) |
| EltonTraits 1.0 | body mass for the megafauna lists | Wilman et al. 2014, *Ecology* 95: 2027, doi:10.6084/m9.figshare.3559887 | CC0 |
| GBIF occurrence downloads | testing, audit | GBIF.org 2026, doi:10.15468/dl.22b9a4, doi:10.15468/dl.6t43nn, doi:10.15468/dl.gv4z7m, doi:10.15468/dl.u2kux5, doi:10.15468/dl.77g6hc | per record; only aggregates stored |
| RESOLVE Ecoregions 2017 | comparison and crosswalk | Dinerstein et al. 2017, *BioScience* 67: 534–545 | CC BY 4.0 |
| EPA Level III and IV ecoregions | comparison only (Texas) | Griffith et al. 2007, *Ecoregions of Texas*, U.S. EPA | public domain |
| Jung et al. 2020 habitat map; Hengl et al. 2018 potential natural vegetation | checks of the vegetation rules | Jung et al. 2020, *Sci. Data* 7: 220; Hengl et al. 2018, *PeerJ* 6: e5457 | CC BY 4.0 |
| WhiteboxTools | geomorphons | Lindsay 2016, *Comput. Geosci.* 95: 75–84 | MIT |
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
  4.0](https://creativecommons.org/licenses/by-nc/4.0/). Credit for the earlier maps: "Husain, A.
  (2026) Revised Ecoregions of the World: <area>, vX.Y.Z. https://github.com/AsasHusain/Ecoregions",
  plus the source attributions above. No commercial use without written permission.
* **Third-party material keeps its own terms.** The source datasets are cited above and their raw files are never redistributed. Several (IUCN and BirdLife ranges,
  some GBIF records) are licensed for non-commercial use only, which the derived products inherit.
* **Commercial licensing and permissions:** contact the author through GitHub (@AsasHusain).
* Versions released before this change (up to commit c5ab452, November 2026) were published
  under MIT and remain so for anyone who obtained them then; this licence applies from v0.4.1.
