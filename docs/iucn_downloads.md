# IUCN Red List range-data downloads

Checklist for pulling IUCN range maps under the website's 10,000-records-per-download cap.
Counts are from the IUCN assessment export of 2026-11-06; **check the count the site shows is < 10,000 before each download.**

**Every download:** format **GeoPackage** (one file; no 2 GB shapefile limit or 10-character field names; includes points). Save as `iucn_<ID>.zip` (e.g. `iucn_A01.zip`) into `data/raw/iucn/ranges/`, and fill in the date and the count the site showed. Raw files are licensed: never commit or redistribute them.

Birds are skipped everywhere (BirdLife BOTW 2025 is the bird source). Marine-only groups are skipped.

## Phase A: Pakistan now (2 downloads)

Land-regions filter. Tick **Jammu-Kashmir** explicitly: many Gilgit-Baltistan/AJK species are coded there, not under Pakistan.

| ID | Filters | Expected | Site count | Date |
|---|---|---|---|---|
| A01 | Land regions: Pakistan, Jammu-Kashmir, Afghanistan, Iran. No taxonomy filter (birds/fish included; dropped later) | 4,755 | | |
| A02 | Land region: India. Taxonomy: Mammalia, Reptilia, Amphibia, Magnoliopsida, Liliopsida, Pinopsida, Gnetopsida, Insecta | 5,880 | | |

A02 covers the eastern buffer (Thar, Punjab, LoC). Overlap with A01 is fine; duplicates are removed by species id.

## Phase B: global (for the world revision; not needed for Pakistan)

Taxonomy filter only, no region. Category = Red List category filter.

| ID | Filters | Expected | Site count | Date |
|---|---|---|---|---|
| B01 | Mammalia | 6,058 | | |
| B02 | Amphibia | 8,095 | | |
| B03 | Reptilia, category Least Concern only | 6,415 | | |
| B04 | Reptilia, every category except Least Concern | 3,965 | | |
| B05 | Pinopsida, Cycadopsida, Gnetopsida (conifers, cycads, Ephedra/Gnetum) | 1,063 | | |
| B06 | Liliopsida, category Least Concern only | 4,777 | | |
| B07 | Liliopsida, every category except Least Concern | 6,188 | | |
| B08 | Magnoliopsida orders: Malpighiales | 7,488 | | |
| B09 | Magnoliopsida orders: Fabales | 6,748 | | |
| B10 | Magnoliopsida orders: Myrtales | 6,163 | | |
| B11 | Magnoliopsida orders: Gentianales | 5,979 | | |
| B12 | Magnoliopsida orders: Ericales, Sapindales | 8,750 | | |
| B13 | Magnoliopsida orders: Lamiales, Caryophyllales | 6,644 | | |
| B14 | Magnoliopsida orders: Laurales, Asterales, Malvales | 8,484 | | |
| B15 | Magnoliopsida orders: Rosales, Magnoliales, Proteales, Apiales, Fagales | 9,189 | | |
| B16 | Magnoliopsida orders: Oxalidales, Solanales, Brassicales, Celastrales, Boraginales, Aquifoliales, Ranunculales, Santalales, Piperales, Saxifragales, Cucurbitales, Dipsacales, Cornales, Cardiopteridales, Dilleniales, Buxales, Vitales, Metteniusales, Geraniales, Chloranthales, Escalloniales, Canellales, Picramniales, Crossosomatales, Zygophyllales, Icacinales, Austrobaileyales, Huerteales, Paracryphiales, Nymphaeales, Garryales, Ceratophyllales, Gunnerales, Bruniales, Oncothecales, Amborellales, Berberidopsidales, Vahliales, Trochodendrales | 6,577 | | |
| B17 | Insecta orders: Odonata | 6,237 | | |
| B18 | Insecta orders: Lepidoptera, Coleoptera | 8,411 | | |
| B19 | Insecta orders: Hymenoptera, Orthoptera, Diptera, Phasmida, Hemiptera, Trichoptera, Mantodea, Plecoptera, Blattodea, Isoptera, Ephemeroptera, Dermaptera, Psocodea, Grylloblattodea, Neuroptera, Archaeognatha, Megaloptera, Siphonaptera, Thysanoptera | 5,861 | | |
| B20 | Polypodiopsida, Lycopodiopsida, Bryopsida, Jungermanniopsida, Marchantiopsida (ferns, lycophytes, bryophytes) | 2,774 | | |
| B21 | Gastropoda (land + freshwater snails; marine dropped later) | 8,539 | | |
| B22 | Arachnida, Clitellata, Diplopoda, Chilopoda | 1,696 | | |

## Phase C: later, only for freshwater ecoregions

| ID | Filters | Expected | Site count | Date |
|---|---|---|---|---|
| C01 | Actinopterygii orders: Perciformes, Cypriniformes | 9,082 | | |
| C02 | Actinopterygii orders: Siluriformes, Gobiiformes, Characiformes, Cichliformes | 9,260 | | |
| C03 | Actinopterygii orders: Cyprinodontiformes, Anguilliformes, Pleuronectiformes, Notassigned, Scorpaeniformes, Ophidiiformes, Stomiiformes, Clupeiformes, Tetraodontiformes, Lophiiformes, Atheriniformes, Syngnathiformes, Aulopiformes, Myctophiformes, Gymnotiformes, Salmoniformes, Osmeriformes, Osteoglossiformes, Gobiesociformes, Beloniformes, Gadiformes, Beryciformes, Synbranchiformes, Batrachoidiformes, Stephanoberyciformes, Mugiliformes | 9,499 | | |
| C04 | Actinopterygii orders: Gonorynchiformes, Gasterosteiformes, Cetomimiformes, Saccopharyngiformes, Acipenseriformes, Lampriformes, Notacanthiformes, Polypteriformes, Esociformes, Zeiformes, Ateleopodiformes, Albuliformes, Polymixiiformes, Percopsiformes, Elopiformes, Lepisosteiformes, Amiiformes | 287 | | |
| C05 | Malacostraca, Bivalvia | 4,116 | | |

Not every assessed species has a range map (many plants and Data Deficient species have none), so files hold fewer maps than records.
