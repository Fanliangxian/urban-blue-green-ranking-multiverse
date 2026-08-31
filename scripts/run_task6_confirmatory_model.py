"""Run the frozen Task 6 primary confirmatory associational model.

Implements OLS, HC3, exact SSE-based partial R2, influence diagnostics,
leave-one-macroregion-out refits, and seven-fold spatial block CV without
outcome-driven variable selection.
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
INPUT = DERIVED / "R1_task6_model_frame_284.csv"
RUN_ID = "task6_primary_confirmatory_association_20260820_r1"
Y = "total_rank_instability_rms_z"
Y_RAW = "total_rank_instability_rms"
CONTINUOUS = [
    "log_population_z", "log_area_z", "polycentricity_z",
    "shape_complexity_z", "periurban_farm_z", "water_density_z",
    "ruggedness_z",
]
PRIMARY_FIVE = CONTINUOUS[2:]
REGIONS = [
    "central_china", "east_china", "north_china", "northeast_china",
    "northwest_china", "south_china", "southwest_china",
]
REGION_REFERENCE = "east_china"
REGION_TERMS = ["region_" + region for region in REGIONS if region != REGION_REFERENCE]
FULL_TERMS = CONTINUOUS + REGION_TERMS
BASELINE_TERMS = CONTINUOUS[:2] + REGION_TERMS


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(frame, path):
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def json_default(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def design(frame, terms):
    return np.column_stack([np.ones(len(frame)), frame[terms].to_numpy(float)])


def fit_ols(y, x):
    n, p = x.shape
    rank = np.linalg.matrix_rank(x)
    if rank != p:
        raise ValueError(f"Design is rank deficient: rank={rank}, p={p}")
    xtx_inv = np.linalg.inv(x.T @ x)
    beta = xtx_inv @ x.T @ y
    fitted = x @ beta
    residual = y - fitted
    sse = float(residual @ residual)
    centered = y - y.mean()
    sst = float(centered @ centered)
    df_resid = n - p
    mse = sse / df_resid
    hat = np.einsum("ij,jk,ik->i", x, xtx_inv, x)
    scaled_residual = residual / (1 - hat)
    meat = x.T @ ((scaled_residual ** 2)[:, None] * x)
    cov_hc3 = xtx_inv @ meat @ xtx_inv
    se_hc3 = np.sqrt(np.diag(cov_hc3))
    t_hc3 = beta / se_hc3
    p_hc3 = 2 * stats.t.sf(np.abs(t_hc3), df_resid)
    critical = stats.t.ppf(0.975, df_resid)
    ci_low = beta - critical * se_hc3
    ci_high = beta + critical * se_hc3
    r2 = 1 - sse / sst
    adjusted_r2 = 1 - (1 - r2) * (n - 1) / df_resid
    model_df = p - 1
    f_stat = ((sst - sse) / model_df) / mse
    f_p = stats.f.sf(f_stat, model_df, df_resid)
    return {
        "n": n, "p": p, "rank": rank, "beta": beta, "fitted": fitted,
        "residual": residual, "sse": sse, "sst": sst,
        "df_resid": df_resid, "mse": mse, "hat": hat,
        "xtx_inv": xtx_inv, "cov_hc3": cov_hc3, "se_hc3": se_hc3,
        "t_hc3": t_hc3, "p_hc3": p_hc3, "ci_low": ci_low,
        "ci_high": ci_high, "r2": r2, "adjusted_r2": adjusted_r2,
        "f_stat": f_stat, "f_p": f_p, "model_df": model_df,
    }


def partial_r2(full_sse, reduced_sse):
    return (reduced_sse - full_sse) / reduced_sse


def wald_block(fit, indices):
    beta = fit["beta"][indices]
    cov = fit["cov_hc3"][np.ix_(indices, indices)]
    statistic = float(beta.T @ np.linalg.pinv(cov) @ beta)
    df = len(indices)
    return statistic, df, float(stats.chi2.sf(statistic, df))


def coefficient_table(fit, terms, model_id):
    names = ["intercept"] + terms
    rows = []
    for index, term in enumerate(names):
        if term in PRIMARY_FIVE:
            role = "primary_explanatory_domain"
        elif term in CONTINUOUS[:2]:
            role = "control"
        elif term.startswith("region_"):
            role = "macroregion_fixed_effect"
        else:
            role = "intercept"
        rows.append({
            "run_id": RUN_ID, "model_id": model_id, "term": term,
            "role": role, "estimate": fit["beta"][index],
            "hc3_se": fit["se_hc3"][index], "hc3_t": fit["t_hc3"][index],
            "hc3_df": fit["df_resid"], "hc3_p_two_sided": fit["p_hc3"][index],
            "hc3_ci95_low": fit["ci_low"][index],
            "hc3_ci95_high": fit["ci_high"][index],
            "coefficient_scale": (
                "standardized_continuous_beta" if term in CONTINUOUS
                else "indicator_contrast_vs_east_china" if term.startswith("region_")
                else "standardized_outcome_intercept"
            ),
        })
    return pd.DataFrame(rows)


def training_transform(train, test):
    train_x = pd.DataFrame(index=train.index)
    test_x = pd.DataFrame(index=test.index)
    sources = [
        ("log_population", np.log1p(train["population_2020_worldpop"]),
         np.log1p(test["population_2020_worldpop"])),
        ("log_area", np.log(train["reference_area_km2"]),
         np.log(test["reference_area_km2"])),
        ("polycentricity", train["polycentricity_index"], test["polycentricity_index"]),
        ("shape_complexity", train["boundary_shape_complexity"], test["boundary_shape_complexity"]),
        ("periurban_farm", train["periurban_farm_fraction"], test["periurban_farm_fraction"]),
        ("water_density", train["surface_water_density"], test["surface_water_density"]),
        ("ruggedness", train["terrain_ruggedness_riley_mean_m"], test["terrain_ruggedness_riley_mean_m"]),
    ]
    parameters = {}
    for name, train_values, test_values in sources:
        mean = float(train_values.mean())
        sd = float(train_values.std(ddof=1))
        if not np.isfinite(sd) or sd <= 0:
            raise ValueError(f"Training fold has invalid SD for {name}: {sd}")
        train_x[name] = (train_values - mean) / sd
        test_x[name] = (test_values - mean) / sd
        parameters[name] = {"mean": mean, "sd": sd}
    return train_x, test_x, parameters


def main():
    failures = []
    frame = pd.read_csv(INPUT, dtype={"city_id": str, "prov_code": str})
    if len(frame) != 284 or frame["city_id"].nunique() != 284:
        failures.append("model frame is not 284 unique cities")
    required = [Y, Y_RAW, "macroregion_code"] + FULL_TERMS
    if frame[required].isna().any().any():
        failures.append("model frame contains missing confirmatory fields")
    if set(frame["macroregion_code"]) != set(REGIONS):
        failures.append("model frame does not contain the frozen seven regions")
    if failures:
        raise ValueError("; ".join(failures))

    y = frame[Y].to_numpy(float)
    x_full = design(frame, FULL_TERMS)
    x_baseline = design(frame, BASELINE_TERMS)
    full = fit_ols(y, x_full)
    baseline = fit_ols(y, x_baseline)
    coefficient = coefficient_table(full, FULL_TERMS, "primary_full_ols")

    partial_rows = []
    for term in CONTINUOUS:
        reduced_terms = [candidate for candidate in FULL_TERMS if candidate != term]
        reduced = fit_ols(y, design(frame, reduced_terms))
        partial_rows.append({
            "effect": term, "effect_type": "single_term",
            "df_removed": 1, "partial_r2": partial_r2(full["sse"], reduced["sse"]),
            "full_sse": full["sse"], "reduced_sse": reduced["sse"],
        })
    for effect, removed in [
        ("five_primary_explanatory_domains", PRIMARY_FIVE),
        ("macroregion_fixed_effects", REGION_TERMS),
    ]:
        reduced_terms = [term for term in FULL_TERMS if term not in removed]
        reduced = fit_ols(y, design(frame, reduced_terms))
        partial_rows.append({
            "effect": effect, "effect_type": "prespecified_block",
            "df_removed": len(removed),
            "partial_r2": partial_r2(full["sse"], reduced["sse"]),
            "full_sse": full["sse"], "reduced_sse": reduced["sse"],
        })
    partial = pd.DataFrame(partial_rows)
    partial_output = DERIVED / "R1_task6_primary_partial_r2.csv"
    write_csv(partial, partial_output)

    primary_indices = [1 + FULL_TERMS.index(term) for term in PRIMARY_FIVE]
    region_indices = [1 + FULL_TERMS.index(term) for term in REGION_TERMS]
    primary_wald = wald_block(full, primary_indices)
    region_wald = wald_block(full, region_indices)
    fit_rows = [{
        "run_id": RUN_ID, "model_id": "baseline_population_area_region",
        "n": baseline["n"], "parameter_count": baseline["p"],
        "r2": baseline["r2"], "adjusted_r2": baseline["adjusted_r2"],
        "classical_f": baseline["f_stat"], "model_df": baseline["model_df"],
        "residual_df": baseline["df_resid"], "classical_f_p": baseline["f_p"],
        "sse": baseline["sse"], "rmse": math.sqrt(baseline["mse"]),
        "five_domain_incremental_partial_r2": np.nan,
        "hc3_wald_chi2": np.nan, "hc3_wald_df": np.nan,
        "hc3_wald_p": np.nan,
    }, {
        "run_id": RUN_ID, "model_id": "primary_full_ols",
        "n": full["n"], "parameter_count": full["p"],
        "r2": full["r2"], "adjusted_r2": full["adjusted_r2"],
        "classical_f": full["f_stat"], "model_df": full["model_df"],
        "residual_df": full["df_resid"], "classical_f_p": full["f_p"],
        "sse": full["sse"], "rmse": math.sqrt(full["mse"]),
        "five_domain_incremental_partial_r2": partial.loc[
            partial["effect"] == "five_primary_explanatory_domains", "partial_r2"
        ].iloc[0],
        "hc3_wald_chi2": primary_wald[0], "hc3_wald_df": primary_wald[1],
        "hc3_wald_p": primary_wald[2],
    }]
    fit_table = pd.DataFrame(fit_rows)
    fit_output = DERIVED / "R1_task6_primary_model_fit.csv"
    write_csv(fit_table, fit_output)
    coefficient_output = DERIVED / "R1_task6_primary_model_coefficients.csv"
    write_csv(coefficient, coefficient_output)

    # Residual diagnostics and influence statistics for the unchanged primary fit.
    residual = full["residual"]
    hat = full["hat"]
    n, p = full["n"], full["p"]
    mse = full["mse"]
    internally_studentized = residual / np.sqrt(mse * (1 - hat))
    deleted_mse = (full["sse"] - residual ** 2 / (1 - hat)) / (full["df_resid"] - 1)
    deleted_mse = np.maximum(deleted_mse, np.finfo(float).eps)
    externally_studentized = residual / np.sqrt(deleted_mse * (1 - hat))
    cooks = (residual ** 2 / (p * mse)) * hat / ((1 - hat) ** 2)
    dffits = externally_studentized * np.sqrt(hat / (1 - hat))
    delta_beta = np.array([
        full["xtx_inv"] @ x_full[i] * residual[i] / (1 - hat[i])
        for i in range(n)
    ])
    dfbetas = delta_beta / np.sqrt(
        deleted_mse[:, None] * np.diag(full["xtx_inv"])[None, :]
    )
    max_abs_dfbeta = np.max(np.abs(dfbetas), axis=1)
    leverage_2 = 2 * p / n
    leverage_3 = 3 * p / n
    cook_threshold = 4 / n
    dffits_threshold = 2 * math.sqrt(p / n)
    dfbeta_threshold = 2 / math.sqrt(n)
    severe_union = (
        (hat > leverage_3) | (cooks > cook_threshold)
        | (np.abs(externally_studentized) > 3)
    )
    influence = frame[["city_id", "city_name", "macroregion_code"]].copy()
    influence["fitted_outcome_z"] = full["fitted"]
    influence["residual_z"] = residual
    influence["leverage"] = hat
    influence["internal_studentized_residual"] = internally_studentized
    influence["external_studentized_residual"] = externally_studentized
    influence["cooks_distance"] = cooks
    influence["dffits"] = dffits
    influence["max_absolute_dfbeta"] = max_abs_dfbeta
    influence["leverage_gt_2p_over_n"] = (hat > leverage_2).astype(int)
    influence["leverage_gt_3p_over_n"] = (hat > leverage_3).astype(int)
    influence["cooks_gt_4_over_n"] = (cooks > cook_threshold).astype(int)
    influence["absolute_external_studentized_gt_3"] = (
        np.abs(externally_studentized) > 3
    ).astype(int)
    influence["absolute_dffits_gt_threshold"] = (
        np.abs(dffits) > dffits_threshold
    ).astype(int)
    influence["max_absolute_dfbeta_gt_threshold"] = (
        max_abs_dfbeta > dfbeta_threshold
    ).astype(int)
    influence["prespecified_sensitivity_exclusion_union"] = severe_union.astype(int)
    influence["primary_fit_excluded"] = 0
    influence_output = DERIVED / "R1_task6_primary_influence_diagnostics.csv"
    write_csv(influence, influence_output)

    sensitivity_frame = frame.loc[~severe_union].copy()
    sensitivity = fit_ols(
        sensitivity_frame[Y].to_numpy(float), design(sensitivity_frame, FULL_TERMS)
    )
    sensitivity_coef = coefficient_table(
        sensitivity, FULL_TERMS, "sensitivity_excluding_prespecified_influence_union"
    )
    comparison = coefficient.loc[
        coefficient["term"].isin(CONTINUOUS),
        ["term", "estimate", "hc3_ci95_low", "hc3_ci95_high", "hc3_p_two_sided"],
    ].rename(columns={
        "estimate": "primary_estimate", "hc3_ci95_low": "primary_ci95_low",
        "hc3_ci95_high": "primary_ci95_high", "hc3_p_two_sided": "primary_p",
    }).merge(
        sensitivity_coef.loc[
            sensitivity_coef["term"].isin(CONTINUOUS),
            ["term", "estimate", "hc3_ci95_low", "hc3_ci95_high", "hc3_p_two_sided"],
        ].rename(columns={
            "estimate": "sensitivity_estimate",
            "hc3_ci95_low": "sensitivity_ci95_low",
            "hc3_ci95_high": "sensitivity_ci95_high",
            "hc3_p_two_sided": "sensitivity_p",
        }), on="term", validate="one_to_one"
    )
    comparison["absolute_change"] = (
        comparison["sensitivity_estimate"] - comparison["primary_estimate"]
    ).abs()
    comparison["sign_stable"] = (
        np.sign(comparison["sensitivity_estimate"])
        == np.sign(comparison["primary_estimate"])
    ).astype(int)
    comparison["primary_n"] = n
    comparison["sensitivity_n"] = len(sensitivity_frame)
    comparison_output = DERIVED / "R1_task6_influence_sensitivity_coefficients.csv"
    write_csv(comparison, comparison_output)

    # Classical residual checks; inference remains HC3 regardless of test result.
    e2_fit = fit_ols(residual ** 2, x_full)
    bp_lm = n * e2_fit["r2"]
    bp_df = p - 1
    bp_p = stats.chi2.sf(bp_lm, bp_df)
    reset_x = np.column_stack([x_full, full["fitted"] ** 2, full["fitted"] ** 3])
    reset = fit_ols(y, reset_x)
    reset_df = reset_x.shape[1] - x_full.shape[1]
    reset_f = ((full["sse"] - reset["sse"]) / reset_df) / reset["mse"]
    reset_p = stats.f.sf(reset_f, reset_df, reset["df_resid"])
    shapiro = stats.shapiro(residual)
    jarque = stats.jarque_bera(residual)
    durbin_watson = float(np.sum(np.diff(residual) ** 2) / np.sum(residual ** 2))
    residual_diag = pd.DataFrame([{
        "run_id": RUN_ID, "n": n,
        "shapiro_w": shapiro.statistic, "shapiro_p": shapiro.pvalue,
        "jarque_bera": jarque.statistic, "jarque_bera_p": jarque.pvalue,
        "breusch_pagan_lm": bp_lm, "breusch_pagan_df": bp_df,
        "breusch_pagan_p": bp_p, "ramsey_reset_f_fitted2_fitted3": reset_f,
        "ramsey_reset_df_num": reset_df, "ramsey_reset_df_den": reset["df_resid"],
        "ramsey_reset_p": reset_p, "durbin_watson_order_diagnostic": durbin_watson,
        "inference_action": "retain_frozen_linear_specification_and_report_HC3",
    }])
    residual_output = DERIVED / "R1_task6_primary_residual_diagnostics.csv"
    write_csv(residual_diag, residual_output)

    # Leave-one-macroregion-out coefficient stability with estimable remaining FEs.
    loor_rows = []
    for heldout in REGIONS:
        train = frame.loc[frame["macroregion_code"] != heldout].copy()
        available = sorted(train["macroregion_code"].unique())
        reference = REGION_REFERENCE if REGION_REFERENCE in available else available[0]
        dummy_terms = []
        for region in available:
            if region == reference:
                continue
            term = "loo_region_" + region
            train[term] = (train["macroregion_code"] == region).astype(int)
            dummy_terms.append(term)
        terms = CONTINUOUS + dummy_terms
        loor_fit = fit_ols(train[Y].to_numpy(float), design(train, terms))
        for term in CONTINUOUS:
            index = 1 + terms.index(term)
            loor_rows.append({
                "held_out_macroregion": heldout,
                "training_n": len(train), "training_region_reference": reference,
                "term": term, "estimate": loor_fit["beta"][index],
                "hc3_se": loor_fit["se_hc3"][index],
                "hc3_ci95_low": loor_fit["ci_low"][index],
                "hc3_ci95_high": loor_fit["ci_high"][index],
                "hc3_p_two_sided": loor_fit["p_hc3"][index],
            })
    loor = pd.DataFrame(loor_rows)
    primary_continuous = coefficient.set_index("term").loc[CONTINUOUS, "estimate"]
    loor_summary_rows = []
    for term in CONTINUOUS:
        values = loor.loc[loor["term"] == term, "estimate"]
        primary_estimate = float(primary_continuous[term])
        loor_summary_rows.append({
            "term": term, "primary_estimate": primary_estimate,
            "loor_min_estimate": values.min(), "loor_max_estimate": values.max(),
            "loor_mean_estimate": values.mean(), "loor_sd_estimate": values.std(ddof=1),
            "same_sign_fold_count": int((np.sign(values) == np.sign(primary_estimate)).sum()),
            "fold_count": len(values),
            "maximum_absolute_change_from_primary": float(
                np.max(np.abs(values - primary_estimate))
            ),
        })
    loor_summary = pd.DataFrame(loor_summary_rows)
    loor_output = DERIVED / "R1_task6_leave_one_macroregion_out_coefficients.csv"
    loor_summary_output = DERIVED / "R1_task6_leave_one_macroregion_out_summary.csv"
    write_csv(loor, loor_output)
    write_csv(loor_summary, loor_summary_output)

    # Seven-fold out-of-region CV. Region indicators are deliberately omitted.
    cv_rows = []
    prediction_rows = []
    for heldout in REGIONS:
        train = frame.loc[frame["macroregion_code"] != heldout].copy()
        test = frame.loc[frame["macroregion_code"] == heldout].copy()
        train_x, test_x, parameters = training_transform(train, test)
        train_y = train[Y_RAW].to_numpy(float)
        test_y = test[Y_RAW].to_numpy(float)
        fold_predictions = {}
        for model_id, fields in [
            ("baseline_population_area", ["log_population", "log_area"]),
            ("full_plus_five_domains", [
                "log_population", "log_area", "polycentricity",
                "shape_complexity", "periurban_farm", "water_density", "ruggedness",
            ]),
        ]:
            x_train = np.column_stack([np.ones(len(train)), train_x[fields].to_numpy(float)])
            x_test = np.column_stack([np.ones(len(test)), test_x[fields].to_numpy(float)])
            cv_fit = fit_ols(train_y, x_train)
            prediction = x_test @ cv_fit["beta"]
            fold_predictions[model_id] = prediction
            errors = test_y - prediction
            fold_sst = float(np.sum((test_y - test_y.mean()) ** 2))
            fold_r2 = np.nan if fold_sst == 0 else 1 - float(errors @ errors) / fold_sst
            cv_rows.append({
                "held_out_macroregion": heldout, "model_id": model_id,
                "training_n": len(train), "test_n": len(test),
                "fold_r2": fold_r2, "fold_mae": float(np.mean(np.abs(errors))),
                "fold_rmse": float(np.sqrt(np.mean(errors ** 2))),
                "region_fixed_effect_policy": "omitted_unseen_region_level_unestimable",
                "preprocessing_fit_on_training_only": 1,
                "training_preprocessing_parameters_json": json.dumps(parameters, sort_keys=True),
            })
        for row_index, (_, city) in enumerate(test.reset_index(drop=True).iterrows()):
            prediction_rows.append({
                "city_id": city["city_id"], "city_name": city["city_name"],
                "held_out_macroregion": heldout, "observed": test_y[row_index],
                "baseline_prediction": fold_predictions["baseline_population_area"][row_index],
                "full_prediction": fold_predictions["full_plus_five_domains"][row_index],
            })
    cv_folds = pd.DataFrame(cv_rows)
    cv_predictions = pd.DataFrame(prediction_rows).sort_values("city_id")
    observed = cv_predictions["observed"].to_numpy(float)
    total_sst = float(np.sum((observed - observed.mean()) ** 2))
    cv_summary_rows = []
    for model_id, prediction_field in [
        ("baseline_population_area", "baseline_prediction"),
        ("full_plus_five_domains", "full_prediction"),
    ]:
        error = observed - cv_predictions[prediction_field].to_numpy(float)
        cv_summary_rows.append({
            "model_id": model_id, "n": len(observed),
            "pooled_cv_r2": 1 - float(error @ error) / total_sst,
            "pooled_mae": float(np.mean(np.abs(error))),
            "pooled_rmse": float(np.sqrt(np.mean(error ** 2))),
        })
    cv_summary = pd.DataFrame(cv_summary_rows)
    base_cv = cv_summary.loc[
        cv_summary["model_id"] == "baseline_population_area"
    ].iloc[0]
    full_cv = cv_summary.loc[
        cv_summary["model_id"] == "full_plus_five_domains"
    ].iloc[0]
    incremental = pd.DataFrame([{
        "comparison": "full_minus_baseline",
        "delta_pooled_cv_r2": full_cv["pooled_cv_r2"] - base_cv["pooled_cv_r2"],
        "delta_mae_full_minus_baseline": full_cv["pooled_mae"] - base_cv["pooled_mae"],
        "delta_rmse_full_minus_baseline": full_cv["pooled_rmse"] - base_cv["pooled_rmse"],
        "interpretation_rule": "positive_delta_R2_and_negative_delta_error_support_incremental_prediction",
    }])
    cv_fold_output = DERIVED / "R1_task6_spatial_block_cv_folds.csv"
    cv_prediction_output = DERIVED / "R1_task6_spatial_block_cv_predictions.csv"
    cv_summary_output = DERIVED / "R1_task6_spatial_block_cv_summary.csv"
    cv_incremental_output = DERIVED / "R1_task6_spatial_block_cv_incremental.csv"
    write_csv(cv_folds, cv_fold_output)
    write_csv(cv_predictions, cv_prediction_output)
    write_csv(cv_summary, cv_summary_output)
    write_csv(incremental, cv_incremental_output)

    coefficient_checks = coefficient.set_index("term").loc[CONTINUOUS]
    if not np.all(np.isfinite(coefficient_checks[[
        "estimate", "hc3_se", "hc3_p_two_sided", "hc3_ci95_low", "hc3_ci95_high"
    ]])):
        failures.append("non-finite primary coefficient result")
    if full["rank"] != full["p"]:
        failures.append("primary design matrix is not full rank")
    if len(cv_predictions) != 284 or cv_predictions["city_id"].nunique() != 284:
        failures.append("spatial CV predictions are not one per model city")
    if len(loor) != 7 * len(CONTINUOUS):
        failures.append("leave-one-region output is incomplete")
    normal_equation_error = float(np.max(np.abs(x_full.T @ full["residual"])))
    if normal_equation_error > 1e-10:
        failures.append(f"OLS normal-equation residual too large: {normal_equation_error}")

    outputs = [
        fit_output, coefficient_output, partial_output, influence_output,
        comparison_output, residual_output, loor_output, loor_summary_output,
        cv_fold_output, cv_prediction_output, cv_summary_output,
        cv_incremental_output,
    ]
    manifest = {
        "status": "pass" if not failures else "fail",
        "run_id": RUN_ID, "claim_type": "associational",
        "causal_interpretation_prohibited": True,
        "failure_count": len(failures), "failures": failures,
        "primary_model": {
            "n": n, "parameter_count": p, "residual_df": full["df_resid"],
            "r2": full["r2"], "adjusted_r2": full["adjusted_r2"],
            "classical_f": full["f_stat"], "classical_f_p": full["f_p"],
            "hc3_primary_five_wald_chi2": primary_wald[0],
            "hc3_primary_five_wald_df": primary_wald[1],
            "hc3_primary_five_wald_p": primary_wald[2],
            "hc3_region_wald_chi2": region_wald[0],
            "hc3_region_wald_df": region_wald[1],
            "hc3_region_wald_p": region_wald[2],
            "five_domain_incremental_partial_r2": float(
                partial.loc[
                    partial["effect"] == "five_primary_explanatory_domains", "partial_r2"
                ].iloc[0]
            ),
        },
        "influence": {
            "leverage_2p_over_n": leverage_2, "leverage_3p_over_n": leverage_3,
            "cooks_4_over_n": cook_threshold,
            "dffits_2sqrt_p_over_n": dffits_threshold,
            "dfbeta_2_over_sqrt_n": dfbeta_threshold,
            "high_leverage_n": int((hat > leverage_2).sum()),
            "severe_leverage_n": int((hat > leverage_3).sum()),
            "cook_flag_n": int((cooks > cook_threshold).sum()),
            "external_studentized_gt_3_n": int((np.abs(externally_studentized) > 3).sum()),
            "prespecified_sensitivity_exclusion_union_n": int(severe_union.sum()),
            "primary_fit_automatic_deletion": False,
            "sensitivity_n": len(sensitivity_frame),
            "continuous_coefficient_sign_stable_n": int(comparison["sign_stable"].sum()),
        },
        "residual_diagnostics": residual_diag.iloc[0].to_dict(),
        "leave_one_macroregion_out": {
            "fold_count": 7,
            "all_continuous_terms_same_sign_all_folds_n": int(
                (loor_summary["same_sign_fold_count"] == 7).sum()
            ),
        },
        "spatial_block_cv": {
            "region_fixed_effects": "omitted_because_heldout_level_unestimable",
            "baseline_pooled_cv_r2": float(base_cv["pooled_cv_r2"]),
            "full_pooled_cv_r2": float(full_cv["pooled_cv_r2"]),
            "delta_pooled_cv_r2": float(incremental["delta_pooled_cv_r2"].iloc[0]),
            "baseline_mae": float(base_cv["pooled_mae"]),
            "full_mae": float(full_cv["pooled_mae"]),
            "delta_mae_full_minus_baseline": float(
                incremental["delta_mae_full_minus_baseline"].iloc[0]
            ),
        },
        "validation": {
            "normal_equation_max_absolute_error": normal_equation_error,
            "primary_design_rank": full["rank"],
            "one_cv_prediction_per_city": len(cv_predictions) == 284,
        },
        "software": {
            "python": platform.python_version(), "numpy": np.__version__,
            "pandas": pd.__version__, "scipy": scipy.__version__,
            "statsmodels": "not_installed_manual_auditable_linear_algebra",
        },
        "input_sha256": {INPUT.name: sha256(INPUT)},
        "output_sha256": {path.name: sha256(path) for path in outputs},
    }
    manifest_output = DERIVED / "R1_task6_confirmatory_model_manifest.json"
    manifest_output.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=json_default),
        encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, default=json_default))
    raise SystemExit(0 if not failures else 1)


if __name__ == "__main__":
    main()
