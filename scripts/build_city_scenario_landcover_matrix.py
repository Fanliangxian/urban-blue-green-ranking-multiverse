"""Build the registered 296-city boundary-scenario x land-cover-estimator matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


SCENARIOS = {
    "gctb_core_2021": {
        "boundary_dataset_id": "gctb_2021",
        "boundary_lineage_group": "GLC_FCS30D_GISD30_GCTB",
        "boundary_license_ok": 1,
        "boundary_formal_ok": 1,
    },
    "gctb_system_2021": {
        "boundary_dataset_id": "gctb_2021",
        "boundary_lineage_group": "GLC_FCS30D_GISD30_GCTB",
        "boundary_license_ok": 1,
        "boundary_formal_ok": 1,
    },
    "gub_2018": {
        "boundary_dataset_id": "gub_2018",
        "boundary_lineage_group": "GAIA_GUB",
        "boundary_license_ok": 1,
        "boundary_formal_ok": 1,
    },
    "ghs_uc_2020": {
        "boundary_dataset_id": "ghs_uc_2020",
        "boundary_lineage_group": "GHS_SMOD",
        "boundary_license_ok": 1,
        "boundary_formal_ok": 1,
    },
}
ESTIMATORS = [
    {
        "landcover_dataset_id": "esa_worldcover_2021",
        "estimator_id": "categorical_native_pixel_area",
        "landcover_lineage_group": "WORLDCOVER",
        "estimator_role": "primary",
        "landcover_license_ok": 1,
        "landcover_formal_ok": 1,
    },
    {
        "landcover_dataset_id": "dynamic_world_2021",
        "estimator_id": "annual_mean_probability_area",
        "landcover_lineage_group": "DYNAMIC_WORLD",
        "estimator_role": "primary",
        "landcover_license_ok": 1,
        "landcover_formal_ok": 1,
    },
    {
        "landcover_dataset_id": "dynamic_world_2021",
        "estimator_id": "annual_modal_label_area",
        "landcover_lineage_group": "DYNAMIC_WORLD",
        "estimator_role": "sensitivity",
        "landcover_license_ok": 1,
        "landcover_formal_ok": 1,
    },
]
MEASUREMENT_COLUMNS = [
    "run_id", "uid", "geom_var", "measurement_status", "boundary_area_m2",
    "valid_area_m2", "raw_valid_fraction", "valid_fraction",
    "boundary_minus_valid_area_m2", "unclassified_area_m2",
    "raster_area_excess_m2", "raster_area_excess_fraction",
    "Blue_m2", "Green_m2", "Grey_m2", "Farm_m2", "Other_Uncertain_m2",
    "class_sum_m2", "mass_balance_error_m2", "mass_balance_relative_error",
    "Blue_share_full_boundary", "Green_share_full_boundary",
    "Grey_share_full_boundary", "Farm_share_full_boundary",
    "Other_Uncertain_share_full_boundary", "Blue_share_valid_area",
    "Green_share_valid_area", "Grey_share_valid_area", "Farm_share_valid_area",
    "Other_Uncertain_share_valid_area", "unmapped_valid_m2",
    "probability_closure_error_area_m2", "probability_closure_relative_error",
    "modal_tie_area_m2", "modal_tie_fraction_valid_area",
]
OBSERVATION_COLUMNS = [
    "obs_0_fraction", "obs_1_4_fraction", "obs_5_9_fraction",
    "obs_10_19_fraction", "obs_ge20_fraction",
    "mean_observation_count_on_observed_area",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_many(paths: list[Path]) -> pd.DataFrame:
    return pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)


def missing_reason_map(reference_dir: Path) -> dict[tuple[str, str], str]:
    reasons: dict[tuple[str, str], str] = {}
    gctb = pd.read_csv(reference_dir / "gctb_2021_city_coverage_296.csv")
    for row in gctb.itertuples():
        if int(row.core_entity_count) == 0:
            reasons[(row.city_id, "gctb_core_2021")] = str(row.match_status)
        if int(row.system_entity_count_min1km2_rel1pct) == 0:
            reasons[(row.city_id, "gctb_system_2021")] = str(row.match_status)
    gub = pd.read_csv(reference_dir / "gub_2018_city_coverage_296.csv")
    for row in gub.itertuples():
        if int(row.assigned_entity_count) == 0:
            reasons[(row.city_id, "gub_2018")] = str(row.match_status)
    ghs = pd.read_csv(reference_dir / "ghs_uc_2020_city_coverage_296.csv")
    for row in ghs.itertuples():
        if int(row.selected_primary_entity_count) == 0:
            reasons[(row.city_id, "ghs_uc_2020")] = str(row.match_status)
    return reasons


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-dir", type=Path, required=True)
    parser.add_argument("--derived-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    reference_dir = args.project_dir / "data" / "reference"
    registry_path = reference_dir / "city_registry_analysis_2021.csv"
    registry = pd.read_csv(registry_path, dtype={"city_id": str, "city_code": str, "prov_code": str})

    source_paths = (
        sorted(args.derived_dir.glob("R1_worldcover2021_296_*.csv"))
        + sorted(args.derived_dir.glob("R1_dw2021_prob_296_r3_*.csv"))
        + sorted(args.derived_dir.glob("R1_dw2021_modal_296_r1_*.csv"))
        + sorted(args.derived_dir.glob("R1_dw2021_obs_296_r1_*.csv"))
    )
    worldcover = read_many(sorted(args.derived_dir.glob("R1_worldcover2021_296_*.csv")))
    probability = read_many(sorted(args.derived_dir.glob("R1_dw2021_prob_296_r3_*.csv")))
    modal = read_many(sorted(args.derived_dir.glob("R1_dw2021_modal_296_r1_*.csv")))
    observations = read_many(sorted(args.derived_dir.glob("R1_dw2021_obs_296_r1_*.csv")))
    measurements = pd.concat([worldcover, probability, modal], ignore_index=True, sort=False)
    observations = observations[["uid", *OBSERVATION_COLUMNS]].drop_duplicates("uid")
    measurements = measurements.merge(observations, on="uid", how="left", validate="many_to_one")
    measurements.loc[
        measurements["landcover_dataset_id"].ne("dynamic_world_2021"), OBSERVATION_COLUMNS
    ] = np.nan

    grid_rows = []
    for city in registry.itertuples(index=False):
        for scenario_id, scenario in SCENARIOS.items():
            for estimator in ESTIMATORS:
                grid_rows.append(
                    {
                        "city_id": city.city_id,
                        "city_name": city.city_name,
                        "prov_code": city.prov_code,
                        "prov_name": city.prov_name,
                        "legacy_291_member": city.legacy_291_member,
                        "bnd_scn": scenario_id,
                        **scenario,
                        **estimator,
                    }
                )
    grid = pd.DataFrame(grid_rows)
    keep = [
        "city_id", "bnd_scn", "landcover_dataset_id", "estimator_id",
        *[column for column in MEASUREMENT_COLUMNS if column in measurements.columns],
        *OBSERVATION_COLUMNS,
    ]
    measurements = measurements[keep]
    key = ["city_id", "bnd_scn", "landcover_dataset_id", "estimator_id"]
    matrix = grid.merge(measurements, on=key, how="left", validate="one_to_one")
    reasons = missing_reason_map(reference_dir)
    matrix["boundary_available"] = matrix["uid"].notna().astype(int)
    matrix["boundary_missing_reason"] = [
        "" if available else reasons.get((city, scenario), "boundary_unavailable_unknown_reason")
        for city, scenario, available in zip(
            matrix["city_id"], matrix["bnd_scn"], matrix["boundary_available"]
        )
    ]
    matrix["measurement_available"] = matrix["measurement_status"].notna().astype(int)
    matrix["combination_formal_eligible"] = (
        matrix["boundary_available"].eq(1)
        & matrix["boundary_formal_ok"].eq(1)
        & matrix["landcover_formal_ok"].eq(1)
    ).astype(int)
    matrix["confirmatory_main_row"] = (
        matrix["combination_formal_eligible"].eq(1)
        & matrix["estimator_role"].eq("primary")
    ).astype(int)
    matrix["direct_lineage_coupling"] = (
        matrix["boundary_lineage_group"].eq(matrix["landcover_lineage_group"])
    ).astype(int)
    matrix["relatively_independent_row"] = (
        matrix["confirmatory_main_row"].eq(1)
        & matrix["direct_lineage_coupling"].eq(0)
    ).astype(int)
    matrix["dynamic_world_low_support_majority_flag"] = (
        matrix["landcover_dataset_id"].eq("dynamic_world_2021")
        & matrix["obs_1_4_fraction"].gt(0.5)
    ).astype(int)
    matrix["dynamic_world_zero_observation_flag"] = (
        matrix["landcover_dataset_id"].eq("dynamic_world_2021")
        & matrix["obs_0_fraction"].gt(0)
    ).astype(int)
    matrix["combination_id"] = (
        matrix["city_id"] + "|" + matrix["bnd_scn"] + "|"
        + matrix["landcover_dataset_id"] + "|" + matrix["estimator_id"]
    )
    first_columns = [
        "combination_id", "city_id", "city_name", "prov_code", "prov_name",
        "legacy_291_member", "bnd_scn", "boundary_dataset_id",
        "boundary_lineage_group", "boundary_available", "boundary_missing_reason",
        "boundary_license_ok", "boundary_formal_ok", "landcover_dataset_id",
        "estimator_id", "estimator_role", "landcover_lineage_group",
        "landcover_license_ok", "landcover_formal_ok", "measurement_available",
        "combination_formal_eligible", "confirmatory_main_row",
        "direct_lineage_coupling", "relatively_independent_row",
        "dynamic_world_low_support_majority_flag",
        "dynamic_world_zero_observation_flag",
    ]
    matrix = matrix[first_columns + [column for column in matrix.columns if column not in first_columns]]
    matrix = matrix.sort_values(key).reset_index(drop=True)

    checks = {
        "registered_cities_296": matrix["city_id"].nunique() == 296,
        "rows_3552": len(matrix) == 3552,
        "combination_id_unique": matrix["combination_id"].is_unique,
        "measured_rows_3474": int(matrix["measurement_available"].sum()) == 3474,
        "explicit_missing_rows_78": int(matrix["measurement_available"].eq(0).sum()) == 78,
        "missing_rows_have_reason": matrix.loc[
            matrix["measurement_available"].eq(0), "boundary_missing_reason"
        ].ne("").all(),
        "missing_rows_not_formal_eligible": matrix.loc[
            matrix["measurement_available"].eq(0), "combination_formal_eligible"
        ].eq(0).all(),
        "gub_author_confirmed_academic_use_included": matrix.loc[
            matrix["bnd_scn"].eq("gub_2018") & matrix["boundary_available"].eq(1),
            "combination_formal_eligible",
        ].eq(1).all(),
        "modal_not_confirmatory_main": matrix.loc[
            matrix["estimator_id"].eq("annual_modal_label_area"), "confirmatory_main_row"
        ].eq(0).all(),
    }
    if not all(checks.values()):
        raise ValueError(f"Matrix checks failed: {checks}")
    checks = {key: bool(value) for key, value in checks.items()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    matrix.to_csv(args.output, index=False, encoding="utf-8-sig")
    manifest = {
        "status": "pass",
        "checks": checks,
        "metrics": {
            "rows": int(len(matrix)),
            "registered_cities": int(matrix["city_id"].nunique()),
            "measured_rows": int(matrix["measurement_available"].sum()),
            "explicit_missing_rows": int(matrix["measurement_available"].eq(0).sum()),
            "formal_eligible_rows": int(matrix["combination_formal_eligible"].sum()),
            "confirmatory_main_rows": int(matrix["confirmatory_main_row"].sum()),
            "relatively_independent_rows": int(matrix["relatively_independent_row"].sum()),
            "dynamic_world_low_support_rows": int(
                matrix["dynamic_world_low_support_majority_flag"].sum()
            ),
            "dynamic_world_zero_observation_rows": int(
                matrix["dynamic_world_zero_observation_flag"].sum()
            ),
        },
        "output": {"path": str(args.output), "sha256": sha256(args.output)},
        "sources": {str(path): sha256(path) for path in [registry_path, *source_paths]},
        "policy_sources": {
            str(args.project_dir / "config" / "boundary_products.yaml"): sha256(
                args.project_dir / "config" / "boundary_products.yaml"
            ),
            str(args.project_dir / "data" / "reference" / "lineage_registry.csv"): sha256(
                args.project_dir / "data" / "reference" / "lineage_registry.csv"
            ),
        },
        "gub_academic_use": {
            "status": "author_confirmed_academic_use_citation_required",
            "evidence": "researcher reports direct author email; original email not archived",
            "redistribution": "source_and_derived_vector_redistribution_not_authorized",
            "decision_id": "D-032",
        },
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
