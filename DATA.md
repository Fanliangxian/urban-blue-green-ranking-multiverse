# Data sources and redistribution boundary

This repository distributes code and documentation only. It does **not** redistribute raw or derived third-party geospatial data, uploaded Earth Engine assets, administrative boundaries or city-level analytical tables.

## City identity and reference geometry

- 2021 administrative-code identity frame: Ministry of Civil Affairs of the People's Republic of China, county-and-above administrative division codes.
- Reference administrative geometry: National Geomatics Center of China / Tianditu administrative-boundary vectors, CGCS2000, map approval number GS (2024) 0650. Obtain the data from the official provider and follow its terms.

## Measurement boundaries

- GCTB 2021 urban boundaries: https://doi.org/10.5281/zenodo.16418717 (CC BY 4.0).
- GUB 2018 physical urban boundaries derived from GAIA: https://data-starcloud.pcl.ac.cn/iearthdata/14; method article https://doi.org/10.1088/1748-9326/ab9be3. The study's permission basis was author-confirmed academic use and public release with article citation; users must independently comply with the provider's current terms.
- GHS-SMOD R2023A Urban Centres, epoch 2020: https://doi.org/10.2905/A0DF7A6F-49DE-46EA-9BDE-563437A6E2BA (EU reuse with acknowledgement).

## Land-cover measurement specifications

- ESA WorldCover 2021 v200: https://doi.org/10.5281/zenodo.7254221 (CC BY 4.0).
- Dynamic World V1: `GOOGLE/DYNAMICWORLD/V1`; catalog https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_DYNAMICWORLD_V1; method https://doi.org/10.1038/s41597-022-01307-4 (CC BY 4.0).

## City-characteristic covariates

- WorldPop 2020 population: `WorldPop/GP/100m/pop` (CC BY 4.0).
- Copernicus Global Land Service LC100 collection 3, epoch 2019: `COPERNICUS/Landcover/100m/Proba-V-C3/Global`.
- JRC Global Surface Water v1.4 Yearly History, 2021: `JRC/GSW1_4/YearlyHistory`.
- MERIT DEM v1.0.3: `MERIT/DEM/v1_0_3` (CC BY-NC 4.0 for non-commercial research).

## Expected local layout

The scripts resolve the repository root from their own location. Create `data/reference`, `data/derived` and `work` under the repository root. File-level prerequisites and producers are listed in `docs/SCRIPT_MAP.md`. Google Earth Engine CSV exports are downloaded into `data/derived` using the frozen filenames expected by the validators.

## Public-release boundary

The release excludes shapefiles, GeoPackages, rasters, Earth Engine exports, model tables and figure-source data. This is a deliberate licensing and size boundary, not evidence that those assets were absent from the analysis. SHA-256 provenance and numerical audits remain recorded in the research archive maintained by the authors.
