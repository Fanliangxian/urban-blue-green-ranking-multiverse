"""Build the frozen Task 6 model frame and predictor-only premodel diagnostics.

This script does not fit an outcome regression or inspect coefficient p-values.
"""

import hashlib
import json
import math
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
DERIVED = ROOT / "data" / "derived"
REFERENCE = ROOT / "data" / "reference"
RUN_ID = "task6_model_frame_premodel_diagnostics_20260820_r1"
REGION_ORDER = [
    "east_china", "north_china", "northeast_china", "central_china",
    "south_china", "southwest_china", "northwest_china",
]
REGION_REFERENCE = "east_china"
RAW_PREDICTORS = [
    "population_2020_worldpop", "reference_area_km2",
    "polycentricity_index", "boundary_shape_complexity",
    "periurban_farm_fraction", "surface_water_density",
    "terrain_ruggedness_riley_mean_m",
]
MODEL_PREDICTORS = [
    "log_population_z", "log_area_z", "polycentricity_z",
    "shape_complexity_z", "periurban_farm_z", "water_density_z",
    "ruggedness_z",
]
OUTCOMES = [
    "total_rank_instability_rms", "boundary_rank_sensitivity_rms",
    "product_rank_sensitivity_rms", "interaction_rank_sensitivity_rms",
    "normalized_percentile_range", "normalized_percentile_iqr",
    "quartile_membership_entropy_bits",
]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(frame, path):
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def zscore(series):
    sd = series.std(ddof=1)
    if not np.isfinite(sd) or sd <= 0:
        raise ValueError(f"Cannot standardize {series.name}: SD={sd}")
    return (series - series.mean()) / sd


def vif_for_column(matrix, index):
    y = matrix[:, index]
    others = np.delete(matrix, index, axis=1)
    design = np.column_stack([np.ones(len(others)), others])
    beta, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    fitted = design @ beta
    ss_total = np.sum((y - y.mean()) ** 2)
    ss_resid = np.sum((y - fitted) ** 2)
    r2 = 1 - ss_resid / ss_total
    return math.inf if r2 >= 1 - 1e-12 else 1 / (1 - r2)


def main():
    failures = []
    cov_path = DERIVED / "R1_task6_covariates_296.csv"
    decomp_path = DERIVED / "R1_task5_city_factorial_sensitivity_decomposition.csv"
    instability_path = DERIVED / "R1_task5_city_rank_instability.csv"
    region_path = REFERENCE / "china_seven_macroregions.csv"
    registry_path = REFERENCE / "city_registry_analysis_2021.csv"

    cov = pd.read_csv(cov_path, dtype={"city_id": str, "prov_code": str})
    decomp = pd.read_csv(decomp_path, dtype={"city_id": str})
    instability = pd.read_csv(instability_path, dtype={"city_id": str})
    regions = pd.read_csv(region_path, dtype={"prov_code": str})
    registry = pd.read_csv(registry_path, dtype={"city_id": str, "prov_code": str})

    if len(regions) != 31 or regions["prov_code"].nunique() != 31:
        failures.append("macroregion map is not 31 unique province-level units")
    if set(regions["macroregion_code"]) != set(REGION_ORDER):
        failures.append("macroregion map does not contain the frozen seven regions")
    reference_regions = set(regions.loc[
        regions["model_reference_category"] == 1, "macroregion_code"
    ])
    if reference_regions != {REGION_REFERENCE}:
        failures.append("East China is not the unique frozen reference category")
    if set(registry["prov_code"]) != set(regions["prov_code"]):
        failures.append("macroregion map province codes do not match the registry")

    outcome = decomp.loc[
        (decomp["scale"] == "common285_normalized_rank_percentile")
        & (decomp["outcome_id"] == "blue_green_share")
    ].copy()
    if len(outcome) != 285 or outcome["city_id"].nunique() != 285:
        failures.append("primary Task 5 decomposition is not 285 unique cities")
    outcome["total_rank_instability_rms"] = np.sqrt(outcome["ss_total_within_city"] / 8)
    outcome["boundary_rank_sensitivity_rms"] = np.sqrt(outcome["ss_boundary"] / 8)
    outcome["product_rank_sensitivity_rms"] = np.sqrt(outcome["ss_product"] / 8)
    outcome["interaction_rank_sensitivity_rms"] = np.sqrt(
        outcome["ss_boundary_product_interaction"] / 8
    )
    rms_closure = (
        outcome["total_rank_instability_rms"] ** 2
        - outcome["boundary_rank_sensitivity_rms"] ** 2
        - outcome["product_rank_sensitivity_rms"] ** 2
        - outcome["interaction_rank_sensitivity_rms"] ** 2
    ).abs().max()
    if rms_closure > 1e-12:
        failures.append(f"RMS component closure error exceeds tolerance: {rms_closure}")

    sensitivity = instability.loc[
        (instability["analysis_set"] == "confirmatory_8")
        & (instability["outcome_id"] == "blue_green_share")
        & (instability["specifications_available"] == 8),
        [
            "city_id", "normalized_percentile_range",
            "normalized_percentile_iqr", "quartile_membership_entropy_bits",
        ],
    ].copy()
    if len(sensitivity) != 285 or sensitivity["city_id"].nunique() != 285:
        failures.append("confirmatory sensitivity outcomes are not 285 unique complete cities")

    frame = outcome[[
        "city_id", "total_rank_instability_rms",
        "boundary_rank_sensitivity_rms", "product_rank_sensitivity_rms",
        "interaction_rank_sensitivity_rms",
    ]].merge(sensitivity, on="city_id", validate="one_to_one")
    frame = frame.merge(cov, on="city_id", how="left", validate="one_to_one")
    frame = frame.merge(
        regions[[
            "prov_code", "macroregion_code", "macroregion_name_en",
            "macroregion_name_zh", "definition_version",
        ]],
        on="prov_code", how="left", validate="many_to_one",
    )
    if frame["macroregion_code"].isna().any():
        failures.append("one or more primary cities lack a macroregion")

    missing_pre = frame[RAW_PREDICTORS].isna().sum().to_dict()
    exclusion = frame.loc[frame[RAW_PREDICTORS].isna().any(axis=1), [
        "city_id", "city_name", "macroregion_code"
    ]].copy()
    exclusion["exclusion_reason"] = "structural_missing_periurban_farm_no_retained_center"
    model = frame.dropna(subset=RAW_PREDICTORS).copy()
    if len(model) != 284:
        failures.append(f"expected 284 complete cases, found {len(model)}")
    if exclusion["city_id"].tolist() != ["CN-630200"]:
        failures.append(f"unexpected complete-case exclusion: {exclusion['city_id'].tolist()}")

    positive_fields = ["population_2020_worldpop", "reference_area_km2"]
    if (model[positive_fields] <= 0).any().any():
        failures.append("population or area contains a non-positive value")
    model["log_population"] = np.log1p(model["population_2020_worldpop"])
    model["log_area"] = np.log(model["reference_area_km2"])
    transform_map = {
        "log_population_z": "log_population",
        "log_area_z": "log_area",
        "polycentricity_z": "polycentricity_index",
        "shape_complexity_z": "boundary_shape_complexity",
        "periurban_farm_z": "periurban_farm_fraction",
        "water_density_z": "surface_water_density",
        "ruggedness_z": "terrain_ruggedness_riley_mean_m",
    }
    for target, source in transform_map.items():
        model[target] = zscore(model[source])
    for outcome_name in OUTCOMES:
        model[outcome_name + "_z"] = zscore(model[outcome_name])

    for region in REGION_ORDER[1:]:
        model["region_" + region] = (model["macroregion_code"] == region).astype(int)
    model["model_frame_run_id"] = RUN_ID
    model["model_sample"] = "primary_common285_complete_case284"
    model["macroregion_reference"] = REGION_REFERENCE
    model = model.sort_values("city_id").reset_index(drop=True)

    model_output = DERIVED / "R1_task6_model_frame_284.csv"
    write_csv(model, model_output)
    exclusion_output = DERIVED / "R1_task6_model_frame_exclusions.csv"
    write_csv(exclusion, exclusion_output)

    region_counts = model.groupby([
        "macroregion_code", "macroregion_name_en", "macroregion_name_zh"
    ], as_index=False).agg(
        city_count=("city_id", "size"),
        province_count=("prov_code", "nunique"),
    )
    region_counts["is_model_reference"] = (
        region_counts["macroregion_code"] == REGION_REFERENCE
    ).astype(int)
    region_counts["leave_one_region_out_training_n"] = len(model) - region_counts["city_count"]
    region_counts_output = DERIVED / "R1_task6_macroregion_counts.csv"
    write_csv(region_counts, region_counts_output)

    diagnostic_variables = list(dict.fromkeys(
        OUTCOMES + RAW_PREDICTORS + ["log_population", "log_area"] + MODEL_PREDICTORS
    ))
    descriptive_rows = []
    for field in diagnostic_variables:
        values = model[field].astype(float)
        q1, q3 = values.quantile([0.25, 0.75])
        descriptive_rows.append({
            "variable": field,
            "n": int(values.notna().sum()),
            "missing_n": int(values.isna().sum()),
            "mean": values.mean(),
            "sd": values.std(ddof=1),
            "median": values.median(),
            "q1": q1,
            "q3": q3,
            "min": values.min(),
            "max": values.max(),
            "skewness": stats.skew(values, bias=False),
            "excess_kurtosis": stats.kurtosis(values, fisher=True, bias=False),
            "shapiro_w": stats.shapiro(values).statistic,
            "shapiro_p": stats.shapiro(values).pvalue,
            "iqr_outlier_n": int(((values < q1 - 1.5 * (q3 - q1)) | (values > q3 + 1.5 * (q3 - q1))).sum()),
            "absolute_z_gt_3_n": int((np.abs(zscore(values)) > 3).sum()),
        })
    descriptives = pd.DataFrame(descriptive_rows)
    descriptives_output = DERIVED / "R1_task6_premodel_descriptives.csv"
    write_csv(descriptives, descriptives_output)

    spearman_matrix = model[MODEL_PREDICTORS].corr(method="spearman")
    spearman_rows = []
    for i, left in enumerate(MODEL_PREDICTORS):
        for right in MODEL_PREDICTORS[i + 1:]:
            rho, p_value = stats.spearmanr(model[left], model[right])
            spearman_rows.append({
                "predictor_1": left, "predictor_2": right,
                "spearman_rho": rho, "two_sided_p_unadjusted_diagnostic": p_value,
                "absolute_rho_ge_0_8": int(abs(rho) >= 0.8),
            })
    spearman_long = pd.DataFrame(spearman_rows)
    spearman_output = DERIVED / "R1_task6_predictor_spearman.csv"
    write_csv(spearman_long, spearman_output)

    region_dummy_fields = ["region_" + region for region in REGION_ORDER[1:]]
    design_fields = MODEL_PREDICTORS + region_dummy_fields
    design = model[design_fields].to_numpy(float)
    vif_rows = [{
        "term": field,
        "vif": vif_for_column(design, index),
        "vif_ge_5": int(vif_for_column(design, index) >= 5),
        "vif_ge_10": int(vif_for_column(design, index) >= 10),
    } for index, field in enumerate(design_fields)]
    vif = pd.DataFrame(vif_rows)
    multicollinearity_output = DERIVED / "R1_task6_multicollinearity.csv"
    write_csv(vif, multicollinearity_output)

    standardized_design = np.column_stack([
        zscore(pd.Series(design[:, index])).to_numpy()
        for index in range(design.shape[1])
    ])
    singular_values = np.linalg.svd(standardized_design, compute_uv=False)
    condition_index = float(singular_values.max() / singular_values.min())
    x_with_intercept = np.column_stack([np.ones(len(model)), design])
    hat = np.einsum(
        "ij,jk,ik->i", x_with_intercept,
        np.linalg.pinv(x_with_intercept.T @ x_with_intercept), x_with_intercept
    )
    parameter_count = x_with_intercept.shape[1]
    leverage_threshold = 2 * parameter_count / len(model)
    severe_leverage_threshold = 3 * parameter_count / len(model)
    flag_rows = []
    for index, row in model.iterrows():
        flags = [
            field for field in MODEL_PREDICTORS
            if abs(float(row[field])) > 3
        ]
        outcome_flag = abs(float(row["total_rank_instability_rms_z"])) > 3
        high_leverage = hat[index] > leverage_threshold
        severe_leverage = hat[index] > severe_leverage_threshold
        if flags or outcome_flag or high_leverage:
            flag_rows.append({
                "city_id": row["city_id"], "city_name": row["city_name"],
                "macroregion_code": row["macroregion_code"],
                "predictor_absolute_z_gt_3": "|".join(flags),
                "primary_outcome_absolute_z_gt_3": int(outcome_flag),
                "predictor_design_leverage": hat[index],
                "leverage_gt_2p_over_n": int(high_leverage),
                "leverage_gt_3p_over_n": int(severe_leverage),
                "automatic_exclusion": 0,
            })
    flags = pd.DataFrame(flag_rows, columns=[
        "city_id", "city_name", "macroregion_code",
        "predictor_absolute_z_gt_3", "primary_outcome_absolute_z_gt_3",
        "predictor_design_leverage", "leverage_gt_2p_over_n",
        "leverage_gt_3p_over_n", "automatic_exclusion",
    ])
    flags_output = DERIVED / "R1_task6_premodel_flags.csv"
    write_csv(flags, flags_output)

    max_vif = float(vif["vif"].max())
    high_pairwise = int(spearman_long["absolute_rho_ge_0_8"].sum())
    severe_collinearity = max_vif >= 10 or condition_index >= 30
    if not np.isfinite(design).all() or np.linalg.matrix_rank(x_with_intercept) < parameter_count:
        failures.append("confirmatory predictor design matrix is non-finite or rank deficient")
    gate = "pass_with_prespecified_robust_diagnostics" if not failures else "do_not_fit"
    if severe_collinearity and not failures:
        gate = "review_collinearity_before_fit"

    output_paths = [
        model_output, exclusion_output, region_counts_output,
        descriptives_output, spearman_output, multicollinearity_output,
        flags_output,
    ]
    report = {
        "status": "pass" if not failures else "fail",
        "run_id": RUN_ID,
        "model_fit_performed": False,
        "coefficient_or_significance_screening_performed": False,
        "gate_6b_recommendation": gate,
        "failure_count": len(failures),
        "failures": failures,
        "registered_city_count": len(cov),
        "task5_common_city_count": len(frame),
        "complete_case_model_n": len(model),
        "excluded_complete_case_n": len(exclusion),
        "excluded_city_ids": exclusion["city_id"].tolist(),
        "missing_before_complete_case": {k: int(v) for k, v in missing_pre.items()},
        "macroregion_definition": "china_seven_geographic_regions_r1",
        "macroregion_reference": REGION_REFERENCE,
        "macroregion_counts": region_counts.to_dict(orient="records"),
        "rms_component_max_absolute_closure_error": float(rms_closure),
        "continuous_predictor_count": len(MODEL_PREDICTORS),
        "region_dummy_count": len(region_dummy_fields),
        "design_parameter_count_including_intercept": parameter_count,
        "design_matrix_rank": int(np.linalg.matrix_rank(x_with_intercept)),
        "maximum_vif": max_vif,
        "vif_ge_5_count": int(vif["vif_ge_5"].sum()),
        "vif_ge_10_count": int(vif["vif_ge_10"].sum()),
        "standardized_design_condition_index": condition_index,
        "predictor_pair_absolute_spearman_ge_0_8_count": high_pairwise,
        "flagged_city_count": len(flags),
        "high_leverage_city_count_2p_over_n": int((hat > leverage_threshold).sum()),
        "severe_leverage_city_count_3p_over_n": int((hat > severe_leverage_threshold).sum()),
        "leverage_threshold_2p_over_n": leverage_threshold,
        "leverage_threshold_3p_over_n": severe_leverage_threshold,
        "automatic_outlier_deletion": False,
        "cv_preprocessing_rule": "fit_transformations_and_standardization_within_each_training_fold",
        "software": {
            "python": platform.python_version(),
            "pandas": pd.__version__, "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "input_sha256": {
            path.name: sha256(path) for path in [
                cov_path, decomp_path, instability_path, region_path, registry_path
            ]
        },
        "output_sha256": {path.name: sha256(path) for path in output_paths},
    }
    audit_output = DERIVED / "R1_task6_premodel_diagnostics.json"
    audit_output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if not failures else 1)


if __name__ == "__main__":
    main()
