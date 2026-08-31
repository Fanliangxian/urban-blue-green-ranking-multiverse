"""Run frozen Task 6E robustness, component influence, and LOOR analyses."""

import hashlib
import json
import math
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy import stats

from run_task6_confirmatory_model import (
    CONTINUOUS, FULL_TERMS, PRIMARY_FIVE, REGION_REFERENCE, REGIONS,
    coefficient_table, design, fit_ols, partial_r2, wald_block,
)
from run_task6_component_models_fdr import bh_adjust


ROOT = Path(__file__).resolve().parents[1]
DERIVED = ROOT / "data" / "derived"
INPUT = DERIVED / "R1_task6_model_frame_284.csv"
PRIMARY_COMPONENT_FDR = DERIVED / "R1_task6_component_primary_family_fdr.csv"
RUN_ID = "task6e_robustness_20260820_r1"
ROBUST_OUTCOMES = [
    ("rank_primary", "total_rank_instability_rms_rank_z"),
    ("alternative_range", "normalized_percentile_range_z"),
    ("alternative_iqr", "normalized_percentile_iqr_z"),
    ("alternative_entropy", "quartile_membership_entropy_bits_z"),
]
COMPONENTS = [
    ("boundary", "boundary_rank_sensitivity_rms_z"),
    ("product", "product_rank_sensitivity_rms_z"),
    ("interaction", "interaction_rank_sensitivity_rms_z"),
]


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
    raise TypeError(type(value).__name__)


def model_outputs(frame, analysis_id, outcome):
    fit = fit_ols(frame[outcome].to_numpy(float), design(frame, FULL_TERMS))
    coefficients = coefficient_table(fit, FULL_TERMS, analysis_id)
    coefficients["run_id"] = RUN_ID
    coefficients.insert(1, "analysis_id", analysis_id)
    coefficients.insert(2, "outcome", outcome)
    reduced_terms = [term for term in FULL_TERMS if term not in PRIMARY_FIVE]
    reduced = fit_ols(frame[outcome].to_numpy(float), design(frame, reduced_terms))
    indices = [1 + FULL_TERMS.index(term) for term in PRIMARY_FIVE]
    wald = wald_block(fit, indices)
    fit_row = {
        "run_id": RUN_ID, "analysis_id": analysis_id, "outcome": outcome,
        "n": fit["n"], "parameter_count": fit["p"], "r2": fit["r2"],
        "adjusted_r2": fit["adjusted_r2"],
        "five_domain_partial_r2": partial_r2(fit["sse"], reduced["sse"]),
        "five_domain_hc3_wald_chi2": wald[0],
        "five_domain_hc3_wald_df": wald[1],
        "five_domain_hc3_wald_p": wald[2],
    }
    return fit, coefficients, fit_row


def fdr_family(coefficients, family_id):
    family = coefficients.loc[coefficients["term"].isin(PRIMARY_FIVE)].copy()
    family["fdr_family_id"] = family_id
    family["fdr_family_size"] = len(family)
    family["bh_q_value"] = bh_adjust(family["hc3_p_two_sided"].to_numpy(float))
    family["bh_reject_q_0_05"] = (family["bh_q_value"] <= 0.05).astype(int)
    return family.sort_values(["bh_q_value", "analysis_id", "term"]).reset_index(drop=True)


def influence_table(frame, component, outcome, fit):
    n, p = fit["n"], fit["p"]
    residual, hat = fit["residual"], fit["hat"]
    deleted_mse = (
        fit["sse"] - residual ** 2 / (1 - hat)
    ) / (fit["df_resid"] - 1)
    deleted_mse = np.maximum(deleted_mse, np.finfo(float).eps)
    external = residual / np.sqrt(deleted_mse * (1 - hat))
    cooks = (residual ** 2 / (p * fit["mse"])) * hat / (1 - hat) ** 2
    leverage_threshold = 3 * p / n
    cooks_threshold = 4 / n
    flag_leverage = hat > leverage_threshold
    flag_cooks = cooks > cooks_threshold
    flag_external = np.abs(external) > 3
    union = flag_leverage | flag_cooks | flag_external
    return pd.DataFrame({
        "run_id": RUN_ID, "component": component, "outcome": outcome,
        "city_id": frame["city_id"], "city_name": frame["city_name"],
        "macroregion_code": frame["macroregion_code"], "leverage": hat,
        "cooks_distance": cooks, "external_studentized_residual": external,
        "leverage_threshold_3p_over_n": leverage_threshold,
        "cooks_threshold_4_over_n": cooks_threshold,
        "flag_severe_leverage": flag_leverage.astype(int),
        "flag_cooks": flag_cooks.astype(int),
        "flag_abs_external_studentized_gt_3": flag_external.astype(int),
        "sensitivity_exclusion_union": union.astype(int),
    })


def main():
    failures = []
    frame = pd.read_csv(INPUT, dtype={"city_id": str})
    required = ["total_rank_instability_rms", "macroregion_code"] + FULL_TERMS
    required += [outcome for _, outcome in ROBUST_OUTCOMES[1:] + COMPONENTS]
    if len(frame) != 284 or frame["city_id"].nunique() != 284:
        failures.append("Task 6E frame is not 284 unique cities")
    if frame[required].isna().any().any():
        failures.append("Task 6E frame contains missing required fields")
    if failures:
        raise ValueError("; ".join(failures))

    raw = frame["total_rank_instability_rms"].to_numpy(float)
    percentile = (stats.rankdata(raw, method="average") - 1) / (len(frame) - 1)
    frame["total_rank_instability_rms_rank_z"] = (
        percentile - percentile.mean()
    ) / percentile.std(ddof=1)

    robust_coefficients, robust_fits = [], []
    normal_errors = {}
    for analysis_id, outcome in ROBUST_OUTCOMES:
        fit, coefficients, fit_row = model_outputs(frame, analysis_id, outcome)
        robust_coefficients.append(coefficients)
        robust_fits.append(fit_row)
        normal_errors[analysis_id] = float(np.max(np.abs(
            design(frame, FULL_TERMS).T @ fit["residual"]
        )))
    robust_coefficients = pd.concat(robust_coefficients, ignore_index=True)
    robust_fit_table = pd.DataFrame(robust_fits)
    rank_family = fdr_family(
        robust_coefficients.loc[robust_coefficients["analysis_id"] == "rank_primary"],
        "rank_transformed_primary_x_five_domains",
    )
    alternative_family = fdr_family(
        robust_coefficients.loc[robust_coefficients["analysis_id"] != "rank_primary"],
        "three_alternative_outcomes_x_five_domains",
    )

    influence_frames, sensitivity_coefficients = [], []
    sensitivity_fit_rows = []
    sensitivity_normal_errors = {}
    for component, outcome in COMPONENTS:
        full_fit = fit_ols(frame[outcome].to_numpy(float), design(frame, FULL_TERMS))
        influence = influence_table(frame, component, outcome, full_fit)
        influence_frames.append(influence)
        excluded_ids = set(influence.loc[
            influence["sensitivity_exclusion_union"] == 1, "city_id"
        ])
        sensitivity = frame.loc[~frame["city_id"].isin(excluded_ids)].copy()
        fit, coefficients, fit_row = model_outputs(
            sensitivity, component + "_influence_sensitivity", outcome
        )
        coefficients.insert(3, "component", component)
        coefficients["primary_n"] = len(frame)
        coefficients["sensitivity_n"] = len(sensitivity)
        coefficients["excluded_n"] = len(excluded_ids)
        sensitivity_coefficients.append(coefficients)
        fit_row["component"] = component
        fit_row["excluded_n"] = len(excluded_ids)
        sensitivity_fit_rows.append(fit_row)
        sensitivity_normal_errors[component] = float(np.max(np.abs(
            design(sensitivity, FULL_TERMS).T @ fit["residual"]
        )))
    influence = pd.concat(influence_frames, ignore_index=True)
    sensitivity_coefficients = pd.concat(sensitivity_coefficients, ignore_index=True)
    sensitivity_fits = pd.DataFrame(sensitivity_fit_rows)
    sensitivity_family = fdr_family(
        sensitivity_coefficients,
        "component_influence_sensitivity_x_five_domains",
    )
    primary_family = pd.read_csv(PRIMARY_COMPONENT_FDR, encoding="utf-8-sig")
    primary_compare = primary_family[[
        "component", "term", "estimate", "bh_q_value", "bh_reject_q_0_05"
    ]].rename(columns={
        "estimate": "primary_estimate", "bh_q_value": "primary_bh_q_value",
        "bh_reject_q_0_05": "primary_bh_reject_q_0_05",
    })
    sensitivity_family = sensitivity_family.merge(
        primary_compare, on=["component", "term"], validate="one_to_one"
    )
    sensitivity_family["sign_stable_vs_primary"] = (
        np.sign(sensitivity_family["estimate"])
        == np.sign(sensitivity_family["primary_estimate"])
    ).astype(int)
    sensitivity_family["fdr_rejection_stable_vs_primary"] = (
        sensitivity_family["bh_reject_q_0_05"]
        == sensitivity_family["primary_bh_reject_q_0_05"]
    ).astype(int)

    loor_rows = []
    full_component_estimates = {}
    for component, outcome in COMPONENTS:
        full = fit_ols(frame[outcome].to_numpy(float), design(frame, FULL_TERMS))
        full_component_estimates[component] = {
            term: float(full["beta"][1 + FULL_TERMS.index(term)]) for term in CONTINUOUS
        }
        for heldout in REGIONS:
            train = frame.loc[frame["macroregion_code"] != heldout].copy()
            available = sorted(train["macroregion_code"].unique())
            reference = REGION_REFERENCE if REGION_REFERENCE in available else available[0]
            dummy_terms = []
            for region in available:
                if region != reference:
                    term = "loo_region_" + region
                    train[term] = (train["macroregion_code"] == region).astype(int)
                    dummy_terms.append(term)
            terms = CONTINUOUS + dummy_terms
            fit = fit_ols(train[outcome].to_numpy(float), design(train, terms))
            for term in CONTINUOUS:
                index = 1 + terms.index(term)
                loor_rows.append({
                    "run_id": RUN_ID, "component": component, "outcome": outcome,
                    "held_out_macroregion": heldout, "training_n": len(train),
                    "training_region_reference": reference, "term": term,
                    "estimate": fit["beta"][index], "hc3_se": fit["se_hc3"][index],
                    "hc3_ci95_low": fit["ci_low"][index],
                    "hc3_ci95_high": fit["ci_high"][index],
                    "hc3_p_two_sided": fit["p_hc3"][index],
                })
    loor = pd.DataFrame(loor_rows)
    summary_rows = []
    for component, _ in COMPONENTS:
        for term in CONTINUOUS:
            values = loor.loc[
                (loor["component"] == component) & (loor["term"] == term), "estimate"
            ]
            primary_estimate = full_component_estimates[component][term]
            summary_rows.append({
                "run_id": RUN_ID, "component": component, "term": term,
                "primary_estimate": primary_estimate,
                "loor_min_estimate": values.min(), "loor_max_estimate": values.max(),
                "loor_mean_estimate": values.mean(),
                "loor_sd_estimate": values.std(ddof=1),
                "same_sign_fold_count": int((np.sign(values) == np.sign(primary_estimate)).sum()),
                "fold_count": len(values),
                "maximum_absolute_change_from_primary": float(
                    np.max(np.abs(values - primary_estimate))
                ),
            })
    loor_summary = pd.DataFrame(summary_rows)

    outputs = {
        "R1_task6e_robustness_model_fit.csv": robust_fit_table,
        "R1_task6e_robustness_coefficients.csv": robust_coefficients,
        "R1_task6e_rank_primary_family_fdr.csv": rank_family,
        "R1_task6e_alternative_primary_family_fdr.csv": alternative_family,
        "R1_task6e_component_influence_diagnostics.csv": influence,
        "R1_task6e_component_influence_sensitivity_model_fit.csv": sensitivity_fits,
        "R1_task6e_component_influence_sensitivity_coefficients.csv": sensitivity_coefficients,
        "R1_task6e_component_influence_fdr.csv": sensitivity_family,
        "R1_task6e_component_loor_coefficients.csv": loor,
        "R1_task6e_component_loor_summary.csv": loor_summary,
    }
    output_paths = []
    for name, table in outputs.items():
        path = DERIVED / name
        write_csv(table, path)
        output_paths.append(path)

    if len(rank_family) != 5 or len(alternative_family) != 15:
        failures.append("robustness FDR family size is incorrect")
    if len(influence) != 3 * 284:
        failures.append("component influence output is incomplete")
    if len(sensitivity_family) != 15:
        failures.append("component sensitivity FDR family is incomplete")
    if len(loor) != 3 * 7 * 7 or len(loor_summary) != 3 * 7:
        failures.append("component LOOR output is incomplete")
    if max(list(normal_errors.values()) + list(sensitivity_normal_errors.values())) > 1e-10:
        failures.append("normal-equation error exceeds tolerance")
    for family in [rank_family, alternative_family, sensitivity_family]:
        if not family["bh_q_value"].between(0, 1).all():
            failures.append("BH q-value outside [0,1]")

    excluded_counts = influence.groupby("component")[
        "sensitivity_exclusion_union"
    ].sum().astype(int).to_dict()
    manifest = {
        "status": "pass" if not failures else "fail", "run_id": RUN_ID,
        "claim_type": "associational_robustness",
        "robustness_does_not_replace_primary_model": True,
        "causal_interpretation_prohibited": True,
        "failure_count": len(failures), "failures": failures, "sample_n": 284,
        "rank_primary_fdr_family_size": len(rank_family),
        "rank_primary_fdr_rejection_count": int(rank_family["bh_reject_q_0_05"].sum()),
        "alternative_fdr_family_size": len(alternative_family),
        "alternative_fdr_rejection_count": int(alternative_family["bh_reject_q_0_05"].sum()),
        "component_influence_exclusion_counts": excluded_counts,
        "component_sensitivity_fdr_family_size": len(sensitivity_family),
        "component_sensitivity_fdr_rejection_count": int(
            sensitivity_family["bh_reject_q_0_05"].sum()
        ),
        "component_sensitivity_sign_stable_count": int(
            sensitivity_family["sign_stable_vs_primary"].sum()
        ),
        "component_sensitivity_fdr_decision_stable_count": int(
            sensitivity_family["fdr_rejection_stable_vs_primary"].sum()
        ),
        "component_loor_all_seven_same_sign_count": int(
            (loor_summary["same_sign_fold_count"] == 7).sum()
        ),
        "normal_equation_max_absolute_errors": normal_errors,
        "sensitivity_normal_equation_max_absolute_errors": sensitivity_normal_errors,
        "variable_selection_performed": False,
        "primary_or_component_main_fit_city_deletion": False,
        "software": {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__, "scipy": scipy.__version__},
        "input_sha256": {INPUT.name: sha256(INPUT),
                         PRIMARY_COMPONENT_FDR.name: sha256(PRIMARY_COMPONENT_FDR)},
        "output_sha256": {path.name: sha256(path) for path in output_paths},
    }
    manifest_path = DERIVED / "R1_task6e_robustness_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=json_default),
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, default=json_default))
    raise SystemExit(0 if not failures else 1)


if __name__ == "__main__":
    main()
