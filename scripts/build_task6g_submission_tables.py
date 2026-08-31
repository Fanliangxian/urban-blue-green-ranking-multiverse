"""Build traceable submission tables from frozen Task 6B--6F outputs."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DERIVED = ROOT / "data" / "derived"
RUN_ID = "task6g_submission_synthesis_20260820_r1"
TERMS = [
    "log_population_z", "log_area_z", "polycentricity_z", "shape_complexity_z",
    "periurban_farm_z", "water_density_z", "ruggedness_z",
]
PRIMARY_FIVE = TERMS[2:]
LABELS = {
    "log_population_z": "Population (log)", "log_area_z": "City area (log)",
    "polycentricity_z": "Polycentricity", "shape_complexity_z": "Boundary shape complexity",
    "periurban_farm_z": "Peri-urban farm fraction", "water_density_z": "Surface-water density",
    "ruggedness_z": "Terrain ruggedness",
}
COMPONENT_LABELS = {"boundary": "Boundary", "product": "Land-cover product",
                    "interaction": "Boundary × product"}


def path(name):
    return DERIVED / name


def read(name):
    return pd.read_csv(path(name), encoding="utf-8-sig")


def sha256(file_path):
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write(frame, name):
    output = path(name)
    frame.to_csv(output, index=False, encoding="utf-8-sig")
    return output


def main():
    failures = []
    source_names = [
        "R1_task6_primary_model_coefficients.csv", "R1_task6_primary_partial_r2.csv",
        "R1_task6_primary_model_fit.csv", "R1_task6_component_primary_family_fdr.csv",
        "R1_task6_component_model_fit.csv", "R1_task6e_component_influence_fdr.csv",
        "R1_task6e_component_loor_summary.csv", "R1_task6e_rank_primary_family_fdr.csv",
        "R1_task6e_alternative_primary_family_fdr.csv",
        "R1_task6f_conley_coefficients.csv", "R1_task6f_conley_component_fdr.csv",
        "R1_task6f_residual_moran.csv", "R1_task6_spatial_block_cv_summary.csv",
        "R1_task6_spatial_block_cv_incremental.csv",
    ]

    primary = read(source_names[0])
    partial = read(source_names[1]).rename(columns={"effect": "term"})
    conley = read("R1_task6f_conley_coefficients.csv")
    conley_primary = conley.loc[
        (conley["model_id"] == "primary_total") & (conley["cutoff_km"] == 500)
    ].copy()
    table1 = primary.loc[primary["term"].isin(TERMS)].copy()
    table1 = table1.merge(partial[["term", "partial_r2"]], on="term", validate="one_to_one")
    table1 = table1.merge(
        conley_primary[["term", "conley_se", "conley_p_two_sided", "conley_ci95_low",
                        "conley_ci95_high", "conley_ci_excludes_zero"]],
        on="term", validate="one_to_one",
    )
    table1["predictor"] = table1["term"].map(LABELS)
    table1["predictor_role"] = np.where(
        table1["term"].isin(PRIMARY_FIVE), "primary explanatory domain", "control"
    )
    table1["standardized_beta"] = table1["estimate"]
    table1["hc3_ci_excludes_zero"] = (
        (table1["hc3_ci95_low"] > 0) | (table1["hc3_ci95_high"] < 0)
    ).astype(int)
    table1["source_primary_coefficients"] = source_names[0]
    table1["source_partial_r2"] = source_names[1]
    table1["source_conley_coefficients"] = "R1_task6f_conley_coefficients.csv"
    table1["display_order"] = table1["term"].map({term: i + 1 for i, term in enumerate(TERMS)})
    table1 = table1.sort_values("display_order")[[
        "display_order", "term", "predictor", "predictor_role", "standardized_beta",
        "hc3_se", "hc3_ci95_low", "hc3_ci95_high", "hc3_p_two_sided",
        "hc3_ci_excludes_zero", "partial_r2", "conley_se", "conley_ci95_low",
        "conley_ci95_high", "conley_p_two_sided", "conley_ci_excludes_zero",
        "source_primary_coefficients", "source_partial_r2", "source_conley_coefficients",
    ]]

    component = read("R1_task6_component_primary_family_fdr.csv")
    conley_component = read("R1_task6f_conley_component_fdr.csv")
    conley_component = conley_component.loc[conley_component["cutoff_km"] == 500].copy()
    influence = read("R1_task6e_component_influence_fdr.csv")
    loor = read("R1_task6e_component_loor_summary.csv")
    table2 = component.merge(
        conley_component[["model_id", "term", "conley_se", "conley_p_two_sided",
                          "conley_ci95_low", "conley_ci95_high", "bh_q_value",
                          "bh_reject_q_0_05"]].rename(columns={
            "model_id": "component", "bh_q_value": "conley_bh_q_value",
            "bh_reject_q_0_05": "conley_bh_reject_q_0_05",
        }), on=["component", "term"], validate="one_to_one"
    )
    table2 = table2.merge(
        influence[["component", "term", "estimate", "bh_q_value", "bh_reject_q_0_05",
                   "sensitivity_n", "excluded_n", "sign_stable_vs_primary"]].rename(columns={
            "estimate": "influence_sensitivity_beta",
            "bh_q_value": "influence_sensitivity_bh_q_value",
            "bh_reject_q_0_05": "influence_sensitivity_bh_reject_q_0_05",
        }), on=["component", "term"], validate="one_to_one"
    )
    table2 = table2.merge(
        loor[["component", "term", "loor_min_estimate", "loor_max_estimate",
              "same_sign_fold_count", "fold_count"]],
        on=["component", "term"], validate="one_to_one"
    )
    table2["component_label"] = table2["component"].map(COMPONENT_LABELS)
    table2["predictor"] = table2["term"].map(LABELS)
    table2["standardized_beta"] = table2["estimate"]
    table2["hc3_bh_q_value"] = table2["bh_q_value"]
    table2["hc3_bh_reject_q_0_05"] = table2["bh_reject_q_0_05"]
    table2["source_component_model"] = "R1_task6_component_primary_family_fdr.csv"
    table2["source_conley"] = "R1_task6f_conley_component_fdr.csv"
    table2["source_influence"] = "R1_task6e_component_influence_fdr.csv"
    table2["source_loor"] = "R1_task6e_component_loor_summary.csv"
    component_order = {name: i for i, name in enumerate(["boundary", "product", "interaction"])}
    term_order = {term: i for i, term in enumerate(PRIMARY_FIVE)}
    table2["component_order"] = table2["component"].map(component_order)
    table2["predictor_order"] = table2["term"].map(term_order)
    table2 = table2.sort_values(["predictor_order", "component_order"])[[
        "component", "component_label", "term", "predictor", "standardized_beta",
        "hc3_se", "hc3_ci95_low", "hc3_ci95_high", "hc3_p_two_sided", "partial_r2",
        "hc3_bh_q_value", "hc3_bh_reject_q_0_05", "conley_se", "conley_ci95_low",
        "conley_ci95_high", "conley_p_two_sided", "conley_bh_q_value",
        "conley_bh_reject_q_0_05", "influence_sensitivity_beta",
        "influence_sensitivity_bh_q_value", "influence_sensitivity_bh_reject_q_0_05",
        "sensitivity_n", "excluded_n", "sign_stable_vs_primary", "loor_min_estimate",
        "loor_max_estimate", "same_sign_fold_count", "fold_count",
        "source_component_model", "source_conley", "source_influence", "source_loor",
    ]]

    primary_fit = read("R1_task6_primary_model_fit.csv")
    primary_fit = primary_fit.loc[primary_fit["model_id"] == "primary_full_ols"].iloc[0]
    component_fit = read("R1_task6_component_model_fit.csv")
    moran = read("R1_task6f_residual_moran.csv")
    moran4 = moran.loc[moran["weight_scheme"] == "knn4"].set_index("model_id")
    cv = read("R1_task6_spatial_block_cv_summary.csv").set_index("model_id")
    delta = read("R1_task6_spatial_block_cv_incremental.csv").iloc[0]
    performance_rows = [{
        "model_id": "primary_total", "model_label": "Total instability", "n": int(primary_fit["n"]),
        "r2": primary_fit["r2"], "adjusted_r2": primary_fit["adjusted_r2"],
        "five_domain_partial_r2": primary_fit["five_domain_incremental_partial_r2"],
        "five_domain_wald_p": primary_fit["hc3_wald_p"],
        "residual_moran_knn4_i": moran4.loc["primary_total", "moran_i"],
        "residual_moran_knn4_p": moran4.loc["primary_total", "permutation_p_two_sided"],
        "baseline_pooled_cv_r2": cv.loc["baseline_population_area", "pooled_cv_r2"],
        "full_pooled_cv_r2": cv.loc["full_plus_five_domains", "pooled_cv_r2"],
        "delta_pooled_cv_r2": delta["delta_pooled_cv_r2"],
    }]
    for _, row in component_fit.iterrows():
        component_id = row["component"]
        performance_rows.append({
            "model_id": component_id, "model_label": COMPONENT_LABELS[component_id],
            "n": int(row["n"]), "r2": row["r2"], "adjusted_r2": row["adjusted_r2"],
            "five_domain_partial_r2": row["five_domain_partial_r2"],
            "five_domain_wald_p": row["five_domain_hc3_wald_p"],
            "residual_moran_knn4_i": moran4.loc[component_id, "moran_i"],
            "residual_moran_knn4_p": moran4.loc[component_id, "permutation_p_two_sided"],
            "baseline_pooled_cv_r2": np.nan, "full_pooled_cv_r2": np.nan,
            "delta_pooled_cv_r2": np.nan,
        })
    table_s1 = pd.DataFrame(performance_rows)
    table_s1["source_model_fit"] = np.where(
        table_s1["model_id"] == "primary_total", "R1_task6_primary_model_fit.csv",
        "R1_task6_component_model_fit.csv"
    )
    table_s1["source_moran"] = "R1_task6f_residual_moran.csv"
    table_s1["source_cv"] = np.where(
        table_s1["model_id"] == "primary_total", "R1_task6_spatial_block_cv_summary.csv", ""
    )

    rank = read("R1_task6e_rank_primary_family_fdr.csv")
    alternatives = read("R1_task6e_alternative_primary_family_fdr.csv")
    outcome_robust = pd.concat([rank, alternatives], ignore_index=True)
    outcome_robust["analysis_family"] = "outcome_scale_robustness"
    outcome_robust["component"] = "primary_total"
    outcome_robust["sensitivity_n"] = 284
    outcome_robust["excluded_n"] = 0
    outcome_robust["sign_stable_vs_primary"] = np.nan
    outcome_robust["source_file"] = np.where(
        outcome_robust["analysis_id"] == "rank_primary",
        "R1_task6e_rank_primary_family_fdr.csv",
        "R1_task6e_alternative_primary_family_fdr.csv",
    )
    influence_robust = influence.copy()
    influence_robust["analysis_family"] = "component_influence_sensitivity"
    influence_robust["source_file"] = "R1_task6e_component_influence_fdr.csv"
    table_s2 = pd.concat([
        outcome_robust[["analysis_family", "analysis_id", "component", "outcome", "term",
                        "estimate", "hc3_se", "hc3_ci95_low", "hc3_ci95_high",
                        "hc3_p_two_sided", "bh_q_value", "bh_reject_q_0_05",
                        "sensitivity_n", "excluded_n", "sign_stable_vs_primary", "source_file"]],
        influence_robust[["analysis_family", "analysis_id", "component", "outcome", "term",
                          "estimate", "hc3_se", "hc3_ci95_low", "hc3_ci95_high",
                          "hc3_p_two_sided", "bh_q_value", "bh_reject_q_0_05",
                          "sensitivity_n", "excluded_n", "sign_stable_vs_primary", "source_file"]],
    ], ignore_index=True)
    table_s2["predictor"] = table_s2["term"].map(LABELS)

    claims = pd.DataFrame([
        ["C6G001", "analysis_population", "284-city complete-case analysis population",
         "R1_task6_model_frame_284.csv", "row count and unique city_id", "computationally_verified_human_approval_pending"],
        ["C6G002", "primary_model_fit", "adjusted R2 0.1819 and five-domain partial R2 0.1049",
         "R1_task6_primary_model_fit.csv", "model_id=primary_full_ols", "computationally_verified_human_approval_pending"],
        ["C6G003", "primary_coefficients", "polycentricity positive; peri-urban farm and ruggedness negative",
         "R1_task6_primary_model_coefficients.csv", "five primary explanatory-domain rows", "computationally_verified_human_approval_pending"],
        ["C6G004", "influence_sensitivity", "primary influence sensitivity retains six of seven continuous signs",
         "R1_task6_influence_sensitivity_coefficients.csv", "all seven continuous predictor rows", "computationally_verified_human_approval_pending"],
        ["C6G005", "spatial_cv", "full model improves pooled out-of-region R2 by 0.1478",
         "R1_task6_spatial_block_cv_incremental.csv", "comparison=full_minus_baseline", "computationally_verified_human_approval_pending"],
        ["C6G006", "component_fdr", "six of fifteen Task 6D component tests pass BH FDR",
         "R1_task6_component_primary_family_fdr.csv", "sum bh_reject_q_0_05", "computationally_verified_human_approval_pending"],
        ["C6G007", "outcome_robustness", "alternative outcomes preserve mixed polycentricity/farm/ruggedness directions",
         "R1_task6e_alternative_primary_family_fdr.csv", "all 15 rows", "computationally_verified_human_approval_pending"],
        ["C6G008", "spatial_dependence", "primary residual Moran I 0.1005 with permutation p 0.0065",
         "R1_task6f_residual_moran.csv", "primary_total and knn4", "computationally_verified_human_approval_pending"],
        ["C6G009", "spatial_hac", "primary five-domain inference stable under 500 km Conley HAC",
         "R1_task6f_conley_coefficients.csv", "primary_total cutoff 500 and five domains", "computationally_verified_human_approval_pending"],
        ["C6G010", "hypothesis_decision", "uniformly positive H3 is not supported",
         "R1_task6_primary_model_coefficients.csv", "mixed signs across five primary domains", "author_interpretation_human_approval_pending"],
    ], columns=["claim_id", "claim_type", "claim_summary", "evidence_file", "evidence_locator",
                "verification_status"])

    outputs = [
        write(table1, "R1_task6g_table1_primary_associations.csv"),
        write(table2, "R1_task6g_table2_component_associations.csv"),
        write(table_s1, "R1_task6g_tableS1_model_performance.csv"),
        write(table_s2, "R1_task6g_tableS2_robustness_summary.csv"),
        write(claims, "R1_task6g_claim_evidence_registry.csv"),
    ]
    if len(table1) != 7 or len(table2) != 15 or len(table_s1) != 4 or len(table_s2) != 35:
        failures.append("submission table cardinality mismatch")
    if table1[["standardized_beta", "hc3_ci95_low", "hc3_ci95_high", "partial_r2"]].isna().any().any():
        failures.append("primary submission table has missing numerical evidence")
    if not table2["sign_stable_vs_primary"].eq(1).all():
        failures.append("component influence directions are not all stable")
    manifest = {
        "status": "pass" if not failures else "fail", "run_id": RUN_ID,
        "failure_count": len(failures), "failures": failures,
        "analysis_population_n": 284, "new_model_fit_performed": False,
        "result_selection_performed": False, "causal_interpretation_prohibited": True,
        "table_rows": {"table1": len(table1), "table2": len(table2),
                       "tableS1": len(table_s1), "tableS2": len(table_s2),
                       "claim_registry": len(claims)},
        "input_sha256": {name: sha256(path(name)) for name in source_names},
        "output_sha256": {output.name: sha256(output) for output in outputs},
    }
    manifest_path = path("R1_task6g_submission_tables_manifest.json")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    raise SystemExit(0 if not failures else 1)


if __name__ == "__main__":
    main()
