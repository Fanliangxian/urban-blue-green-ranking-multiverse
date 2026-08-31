"""Run frozen Task 6D component models and 15-test BH-FDR correction."""

import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import scipy

from run_task6_confirmatory_model import (
    CONTINUOUS, FULL_TERMS, PRIMARY_FIVE, REGION_TERMS,
    coefficient_table, design, fit_ols, partial_r2, wald_block,
)


ROOT = Path(__file__).resolve().parents[1]
DERIVED = ROOT / "data" / "derived"
INPUT = DERIVED / "R1_task6_model_frame_284.csv"
RUN_ID = "task6_component_models_fdr_20260820_r1"
OUTCOMES = [
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


def bh_adjust(p_values):
    values = np.asarray(p_values, dtype=float)
    m = len(values)
    order = np.argsort(values)
    ranked = values[order] * m / np.arange(1, m + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adjusted = np.empty(m, dtype=float)
    adjusted[order] = np.minimum(ranked, 1.0)
    return adjusted


def json_default(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    raise TypeError(type(value).__name__)


def main():
    failures = []
    frame = pd.read_csv(INPUT, dtype={"city_id": str})
    required = [outcome for _, outcome in OUTCOMES] + FULL_TERMS
    if len(frame) != 284 or frame["city_id"].nunique() != 284:
        failures.append("component model frame is not 284 unique cities")
    if frame[required].isna().any().any():
        failures.append("component model frame contains missing required fields")
    if failures:
        raise ValueError("; ".join(failures))

    x = design(frame, FULL_TERMS)
    coefficient_frames = []
    partial_rows = []
    fit_rows = []
    normal_equation_errors = {}
    for component, outcome in OUTCOMES:
        y = frame[outcome].to_numpy(float)
        fit = fit_ols(y, x)
        coefficient = coefficient_table(
            fit, FULL_TERMS, f"{component}_sensitivity_full_ols"
        )
        coefficient["run_id"] = RUN_ID
        coefficient.insert(1, "component", component)
        coefficient.insert(2, "outcome", outcome)
        coefficient_frames.append(coefficient)

        for term in CONTINUOUS:
            reduced_terms = [candidate for candidate in FULL_TERMS if candidate != term]
            reduced = fit_ols(y, design(frame, reduced_terms))
            partial_rows.append({
                "component": component, "outcome": outcome,
                "effect": term, "effect_type": "single_term", "df_removed": 1,
                "partial_r2": partial_r2(fit["sse"], reduced["sse"]),
                "full_sse": fit["sse"], "reduced_sse": reduced["sse"],
            })
        for effect, removed in [
            ("five_primary_explanatory_domains", PRIMARY_FIVE),
            ("macroregion_fixed_effects", REGION_TERMS),
        ]:
            reduced_terms = [term for term in FULL_TERMS if term not in removed]
            reduced = fit_ols(y, design(frame, reduced_terms))
            partial_rows.append({
                "component": component, "outcome": outcome,
                "effect": effect, "effect_type": "prespecified_block",
                "df_removed": len(removed),
                "partial_r2": partial_r2(fit["sse"], reduced["sse"]),
                "full_sse": fit["sse"], "reduced_sse": reduced["sse"],
            })
        primary_indices = [1 + FULL_TERMS.index(term) for term in PRIMARY_FIVE]
        region_indices = [1 + FULL_TERMS.index(term) for term in REGION_TERMS]
        primary_wald = wald_block(fit, primary_indices)
        region_wald = wald_block(fit, region_indices)
        block_partial = partial_rows[-2]["partial_r2"]
        region_partial = partial_rows[-1]["partial_r2"]
        fit_rows.append({
            "run_id": RUN_ID, "component": component, "outcome": outcome,
            "n": fit["n"], "parameter_count": fit["p"],
            "residual_df": fit["df_resid"], "r2": fit["r2"],
            "adjusted_r2": fit["adjusted_r2"], "classical_f": fit["f_stat"],
            "classical_f_p": fit["f_p"],
            "five_domain_partial_r2": block_partial,
            "five_domain_hc3_wald_chi2": primary_wald[0],
            "five_domain_hc3_wald_df": primary_wald[1],
            "five_domain_hc3_wald_p": primary_wald[2],
            "region_partial_r2": region_partial,
            "region_hc3_wald_chi2": region_wald[0],
            "region_hc3_wald_df": region_wald[1],
            "region_hc3_wald_p": region_wald[2],
        })
        normal_equation_errors[component] = float(
            np.max(np.abs(x.T @ fit["residual"]))
        )

    coefficients = pd.concat(coefficient_frames, ignore_index=True)
    partial = pd.DataFrame(partial_rows)
    fit_table = pd.DataFrame(fit_rows)

    family_mask = coefficients["term"].isin(PRIMARY_FIVE)
    family = coefficients.loc[family_mask].copy()
    if len(family) != 15:
        failures.append(f"FDR family expected 15 rows, found {len(family)}")
    family["fdr_family_id"] = "five_primary_domains_x_three_components"
    family["fdr_family_size"] = 15
    family["bh_q_value"] = bh_adjust(family["hc3_p_two_sided"].to_numpy(float))
    family["bh_reject_q_0_05"] = (family["bh_q_value"] <= 0.05).astype(int)
    family["expected_positive_direction"] = 1
    family["direction_matches_preregistered_positive"] = (
        family["estimate"] > 0
    ).astype(int)
    family = family.sort_values(["bh_q_value", "component", "term"]).reset_index(drop=True)
    family_partial = partial.loc[
        partial["effect"].isin(PRIMARY_FIVE),
        ["component", "effect", "partial_r2"],
    ].rename(columns={"effect": "term"})
    family = family.merge(
        family_partial, on=["component", "term"], validate="one_to_one"
    )

    coefficients["fdr_family_id"] = ""
    coefficients["fdr_family_size"] = np.nan
    coefficients["bh_q_value"] = np.nan
    coefficients["bh_reject_q_0_05"] = np.nan
    coefficients["direction_matches_preregistered_positive"] = np.nan
    family_lookup = family.set_index(["component", "term"])
    for index, row in coefficients.loc[family_mask].iterrows():
        values = family_lookup.loc[(row["component"], row["term"])]
        coefficients.loc[index, "fdr_family_id"] = values["fdr_family_id"]
        coefficients.loc[index, "fdr_family_size"] = values["fdr_family_size"]
        coefficients.loc[index, "bh_q_value"] = values["bh_q_value"]
        coefficients.loc[index, "bh_reject_q_0_05"] = values["bh_reject_q_0_05"]
        coefficients.loc[index, "direction_matches_preregistered_positive"] = values[
            "direction_matches_preregistered_positive"
        ]

    coefficient_output = DERIVED / "R1_task6_component_model_coefficients.csv"
    family_output = DERIVED / "R1_task6_component_primary_family_fdr.csv"
    fit_output = DERIVED / "R1_task6_component_model_fit.csv"
    partial_output = DERIVED / "R1_task6_component_partial_r2.csv"
    write_csv(coefficients, coefficient_output)
    write_csv(family, family_output)
    write_csv(fit_table, fit_output)
    write_csv(partial, partial_output)

    if not np.all(np.isfinite(family[[
        "estimate", "hc3_se", "hc3_p_two_sided", "bh_q_value",
        "partial_r2",
    ]])):
        failures.append("non-finite FDR-family result")
    if any(error > 1e-10 for error in normal_equation_errors.values()):
        failures.append(f"normal-equation error exceeds tolerance: {normal_equation_errors}")
    if family["bh_q_value"].min() < 0 or family["bh_q_value"].max() > 1:
        failures.append("BH q-values outside [0,1]")

    outputs = [coefficient_output, family_output, fit_output, partial_output]
    manifest = {
        "status": "pass" if not failures else "fail",
        "run_id": RUN_ID, "claim_type": "associational",
        "causal_interpretation_prohibited": True,
        "failure_count": len(failures), "failures": failures,
        "sample_n": len(frame), "component_model_count": len(OUTCOMES),
        "fdr_method": "Benjamini-Hochberg",
        "fdr_family": "five_primary_domains_x_three_components",
        "fdr_family_size": len(family), "fdr_alpha": 0.05,
        "fdr_rejection_count": int(family["bh_reject_q_0_05"].sum()),
        "fdr_positive_direction_rejection_count": int((
            (family["bh_reject_q_0_05"] == 1)
            & (family["direction_matches_preregistered_positive"] == 1)
        ).sum()),
        "fdr_negative_direction_rejection_count": int((
            (family["bh_reject_q_0_05"] == 1)
            & (family["direction_matches_preregistered_positive"] == 0)
        ).sum()),
        "model_fit": fit_table.to_dict(orient="records"),
        "normal_equation_max_absolute_errors": normal_equation_errors,
        "variable_selection_performed": False,
        "sample_or_outlier_deletion_performed": False,
        "software": {
            "python": platform.python_version(), "numpy": np.__version__,
            "pandas": pd.__version__, "scipy": scipy.__version__,
            "engine": "shared_auditable_numpy_scipy_ols_hc3",
        },
        "input_sha256": {INPUT.name: sha256(INPUT)},
        "output_sha256": {path.name: sha256(path) for path in outputs},
    }
    manifest_output = DERIVED / "R1_task6_component_models_fdr_manifest.json"
    manifest_output.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=json_default),
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, default=json_default))
    raise SystemExit(0 if not failures else 1)


if __name__ == "__main__":
    main()
