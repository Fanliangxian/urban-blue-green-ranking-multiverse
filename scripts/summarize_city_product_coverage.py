"""Summarize the integrated matrix back into Task 3 coverage registries."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


PRIMARY_PRODUCTS = {
    "esa_worldcover_2021": {
        "estimator_id": "categorical_native_pixel_area",
        "acquisition_status": "official_gee_asset_v200_national_measurement_validated",
        "evidence_run_id": "worldcover2021_296_national_r1",
    },
    "dynamic_world_2021": {
        "estimator_id": "annual_mean_probability_area",
        "acquisition_status": "official_gee_collection_v1_annual_composite_validated",
        "evidence_run_id": "dynamicworld2021_probability_296_20260806_r3",
    },
}
FORMAL_SCENARIOS = {"gctb_core_2021", "gctb_system_2021", "ghs_uc_2020"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--product-registry", type=Path, required=True)
    parser.add_argument("--scenario-output", type=Path, required=True)
    args = parser.parse_args()

    matrix = pd.read_csv(args.matrix, dtype={"city_id": str, "prov_code": str})
    primary = matrix[
        matrix["estimator_role"].eq("primary")
        & matrix["landcover_dataset_id"].isin(PRIMARY_PRODUCTS)
    ].copy()
    primary["coverage_status"] = "boundary_unavailable"
    measured = primary["measurement_available"].eq(1)
    primary.loc[measured, "coverage_status"] = primary.loc[measured, "measurement_status"]
    scenario_columns = [
        "city_id", "city_name", "prov_name", "bnd_scn", "boundary_dataset_id",
        "boundary_lineage_group", "boundary_available", "boundary_missing_reason",
        "boundary_formal_ok", "landcover_dataset_id", "estimator_id",
        "measurement_available", "coverage_status", "valid_fraction",
        "unclassified_area_m2", "mass_balance_relative_error",
        "combination_formal_eligible", "relatively_independent_row",
        "dynamic_world_low_support_majority_flag", "dynamic_world_zero_observation_flag",
        "mean_observation_count_on_observed_area", "obs_0_fraction", "obs_1_4_fraction",
    ]
    scenario = primary[scenario_columns].sort_values(
        ["city_id", "bnd_scn", "landcover_dataset_id"]
    )
    args.scenario_output.parent.mkdir(parents=True, exist_ok=True)
    scenario.to_csv(args.scenario_output, index=False, encoding="utf-8-sig")

    registry = pd.read_csv(args.product_registry, dtype={"city_id": str})
    formal = primary[primary["bnd_scn"].isin(FORMAL_SCENARIOS)]
    summaries = {}
    for (city_id, dataset_id), group in formal.groupby(["city_id", "landcover_dataset_id"]):
        available = group[group["measurement_available"].eq(1)]
        measured_count = len(available)
        partial_count = int(available["measurement_status"].eq("partial_valid_coverage").sum())
        low_support = int(available["dynamic_world_low_support_majority_flag"].max()) if measured_count else 0
        zero_observation = int(available["dynamic_world_zero_observation_flag"].max()) if measured_count else 0
        if measured_count == 0:
            match_status = "not_measurable_no_formal_boundary_geometry"
            usable_status = "unavailable_no_formal_boundary_geometry"
            ambiguity_status = "not_assessed"
            valid_fraction = float("nan")
        elif partial_count:
            match_status = "available_measured"
            usable_status = "available_with_partial_valid_coverage"
            ambiguity_status = "low_observation_support" if low_support else "partial_valid_coverage"
            valid_fraction = float(available["valid_fraction"].min())
        else:
            match_status = "available_measured"
            usable_status = "available_matched"
            ambiguity_status = "low_observation_support" if low_support else "not_flagged"
            valid_fraction = float(available["valid_fraction"].min())
        summaries[(city_id, dataset_id)] = {
            "match_status": match_status,
            "usable_status": usable_status,
            "ambiguity_status": ambiguity_status,
            "observed_feature_count": measured_count,
            "valid_area_fraction": valid_fraction,
            "evidence_note": (
                f"Derived from the frozen integrated matrix across three formal boundary scenarios; "
                f"{measured_count}/3 scenarios measurable, {partial_count} partial-coverage rows, "
                f"low-support flag={low_support}, zero-observation flag={zero_observation}. "
                "Scenario-specific failures remain in city_scenario_landcover_coverage_status.csv."
            ),
        }

    for index, row in registry.iterrows():
        dataset_id = row["dataset_id"]
        if dataset_id not in PRIMARY_PRODUCTS:
            continue
        summary = summaries[(row["city_id"], dataset_id)]
        registry.at[index, "acquisition_status"] = PRIMARY_PRODUCTS[dataset_id]["acquisition_status"]
        registry.at[index, "match_status"] = summary["match_status"]
        registry.at[index, "usable_status"] = summary["usable_status"]
        registry.at[index, "ambiguity_status"] = summary["ambiguity_status"]
        registry.at[index, "observed_feature_count"] = summary["observed_feature_count"]
        registry.at[index, "valid_area_fraction"] = summary["valid_area_fraction"]
        registry.at[index, "evidence_run_id"] = PRIMARY_PRODUCTS[dataset_id]["evidence_run_id"]
        registry.at[index, "evidence_note"] = summary["evidence_note"]
    gub_mask = registry["dataset_id"].eq("gub_2018")
    registry.loc[gub_mask, "acquisition_status"] = (
        "local_file_frozen_author_confirmed_academic_use_citation_required"
    )
    registry.loc[gub_mask & registry["match_status"].eq("available_matched"), "usable_status"] = (
        "available_matched"
    )
    registry.loc[gub_mask & registry["match_status"].ne("available_matched"), "usable_status"] = (
        registry.loc[gub_mask & registry["match_status"].ne("available_matched"), "match_status"]
    )
    registry.loc[gub_mask, "evidence_note"] = (
        "Exact 2018 file and empirical assignment are frozen. The researcher reports prior direct "
        "author email permission for scientific use with citation of the GUB article; the email is "
        "not archived. Formal academic analysis is allowed, but source/derived vector redistribution "
        "is not assumed."
    )
    registry.to_csv(args.product_registry, index=False, encoding="utf-8-sig")

    assert len(scenario) == 296 * 4 * 2
    assert scenario[["city_id", "bnd_scn", "landcover_dataset_id"]].duplicated().sum() == 0
    assert len(summaries) == 296 * 2
    print(
        {
            "scenario_rows": len(scenario),
            "product_city_summaries_updated": len(summaries),
            "product_registry_rows": len(registry),
        }
    )


if __name__ == "__main__":
    main()
