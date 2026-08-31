# Reproducibility guide

## 1. Environments

Use the Conda environment for registry, tabular analysis and most plotting. Use ArcGIS Pro Python only for `run_task6f_spatial_dependence_arcpy.py`, which requires `arcpy`. GDAL/OGR is needed for figure-source geometry and maps.

## 2. Configure external paths

Copy `config/paths.example.yaml` to a private file outside Git or set the documented environment variables. Never commit credentials or personal Earth Engine asset IDs.

## 3. Build the city frame

```powershell
python scripts/build_city_registry.py --output-dir data/reference
python scripts/build_analysis_registry.py
python scripts/validate_city_registry.py
```

The master registry contains 297 identities; the frozen analysis registry excludes one non-comparable discontinuous island-maritime entity and contains 296 cities.

## 4. Measure map combinations

Run the scripts in `gee/` in the Earth Engine Code Editor after replacing `YOUR_EE_PROJECT`. Download their CSV tasks into `data/derived`. Boundary geometries must be constructed from the cited providers using the study's assignment protocol; third-party vectors are not included here.

Build the 4 × 2 matrix after all boundary and land-cover summaries are present:

```powershell
python scripts/build_city_scenario_landcover_matrix.py `
  --project-dir . `
  --derived-dir data/derived `
  --output data/derived/R1_city_scenario_landcover_estimator_matrix.csv `
  --manifest data/derived/R1_city_scenario_landcover_estimator_matrix_manifest.json
```

## 5. Ranking and finite-multiverse decomposition

```powershell
python scripts/reproduce_analysis.py --stage rank
```

This runs the frozen rank-frame audit, ranking summaries, factorial decomposition, perturbation sensitivities and two lineage-balanced 3 × 2 designs.

## 6. City-characteristic models

After downloading the Task 6 Earth Engine batches:

```powershell
python scripts/integrate_task6_basic_covariates.py
python scripts/merge_validate_task6_national_covariates.py
python scripts/reproduce_analysis.py --stage models
& "C:\Program Files\ArcGIS\Pro\bin\Python\envs\arcgispro-py3\python.exe" scripts\run_task6f_spatial_dependence_arcpy.py
python scripts/build_task6g_submission_tables.py
```

The primary model is the frozen 284-city complete-case associational analysis. Scripts retain HC3 inference, exact partial R², the 15-test BH family, influence checks, leave-one-region-out refits, residual Moran tests, Conley bandwidth checks and regional cross-validation.

## 7. Figures

```powershell
python scripts/build_r2_figure_source_data.py
python scripts/reproduce_analysis.py --stage figures
```

The plotting scripts consume frozen source tables and do not re-estimate the main models. Figure 1 additionally requires the two Shangluo land-cover raster exports and the official administrative map source described in `DATA.md`.

## 8. Verification

```powershell
python scripts/check_release.py
python -m unittest discover -s tests -v
python scripts/build_manifest.py
```

The manifest provides a file-level SHA-256 inventory of the release package. Exact reproduction also depends on using the same upstream dataset versions and the frozen seeds retained in the scripts.
