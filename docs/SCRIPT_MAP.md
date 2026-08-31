# Script map

| Stage | Script | Main input | Main output or role |
|---|---|---|---|
| Registry | `build_city_registry.py` | MCA administrative-code source | 297-city identity frame |
| Registry | `build_analysis_registry.py` | Master registry | Frozen 296-city analysis registry |
| Measurement | `build_city_scenario_landcover_matrix.py` | Boundary and land-cover CSV summaries | Registered scenario × specification matrix |
| Task 5 | `audit_task5_ranking_frame.py` | Measurement matrix | Eight-setting common-set audit |
| Task 5 | `build_task5_ranking_instability.py` | Measurement matrix | Rank agreement and city instability tables |
| Task 5 | `decompose_task5_map_sensitivity.py` | Matrix and rank audit | Boundary/specification/interaction decomposition |
| Task 5 | `run_task5_stage3_sensitivity.py` | Measurement matrix | Six perturbation summaries |
| Task 5 | `run_task5_lineage_balance_sensitivity.py` | Matrix and frozen protocol | Two fixed-population 3 × 2 checks |
| Task 6 | `integrate_task6_basic_covariates.py` | GEE batch exports | Population, water and terrain table |
| Task 6 | `merge_validate_task6_national_covariates.py` | 50 GEE exports | Polycentricity and peri-urban farm table |
| Task 6 | `build_task6_model_frame_and_premodel_diagnostics.py` | Task 5 sensitivity and covariates | Frozen 284-city model frame |
| Task 6 | `run_task6_confirmatory_model.py` | Model frame | OLS, HC3, partial R², influence and CV outputs |
| Task 6 | `run_task6_component_models_fdr.py` | Model frame | Three component models and 15-test BH correction |
| Task 6 | `run_task6e_robustness.py` | Primary/component models | Alternative outcomes, influence and regional refits |
| Task 6 | `run_task6f_spatial_dependence_arcpy.py` | Model frame and reference geometry | Moran and Conley-HAC diagnostics |
| Figures | `build_r2_figure_source_data.py` | Frozen Task 5/6 outputs | Audited plot-ready tables |
| Figures | `plot_r2_figure1_study_area_measurement_scenarios.py` | Registry, boundaries and example rasters | Figure 1 |
| Figures | `plot_task5_publication_figures.py` | Task 5 source tables | Figure 2 only; the former Task 5 sensitivity graphic is not in the current supplement |
| Figures | `plot_task5_supplementary_figure_s1.py` | Task 5 factorial decomposition aggregate table | Current Supplementary Figure S1: outcome-scale and component decomposition diagnostics |
| Figures | `plot_r2_spatial_sensitivity_figures.py` | City sensitivity and map geometry | Figures 3 and 4 |
| Figures | `plot_r2_figure5_city_characteristics.py` | Association and CV source tables | Figure 5 |
| Figures | `plot_task6g_publication_figures.py` | Task 6 synthesis tables | Current Supplementary Figure S2: association-model robustness and residual spatial dependence |
| Figure style | `unified_figure_style.py` | Shared Matplotlib contract | 180-mm final width, typography, panel labels, axes, line weights, margins and palette used by all active figure scripts |

The repository does not include a one-click raw-data downloader because several sources require provider-specific access, Earth Engine execution, ArcGIS Pro or independent acceptance of licensing terms.
