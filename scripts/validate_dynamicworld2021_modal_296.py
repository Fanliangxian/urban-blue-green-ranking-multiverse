"""Validate Dynamic World annual modal-label sensitivity exports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


EXPECTED_SCENARIO_ROWS = {
    "gctb_core_2021": 287,
    "gctb_system_2021": 287,
    "gub_2018": 294,
    "ghs_uc_2020": 290,
}
EXPECTED_BATCHES = {"b1_11_29", "b2_31_39", "b3_41_49", "b4_50_65"}
CLASS_STEMS = ["Blue", "Green", "Grey", "Farm", "Other_Uncertain"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    modal_paths = sorted(args.input_dir.glob("R1_dw2021_modal_296_r1_*.csv"))
    probability_paths = sorted(args.input_dir.glob("R1_dw2021_prob_296_r3_*.csv"))
    if not modal_paths or not probability_paths:
        raise FileNotFoundError("Modal or probability Dynamic World CSV files are missing")
    modal = pd.concat([pd.read_csv(path) for path in modal_paths], ignore_index=True)
    probability = pd.concat([pd.read_csv(path) for path in probability_paths], ignore_index=True)
    scenario_rows = modal.groupby("bnd_scn").size().to_dict()
    combinations = set(zip(modal["bnd_scn"], modal["batch_id"]))
    expected_combinations = {
        (scenario, batch) for scenario in EXPECTED_SCENARIO_ROWS for batch in EXPECTED_BATCHES
    }
    numeric = modal.iloc[:, 11:].apply(pd.to_numeric, errors="coerce")
    full_share_sum = modal[[f"{stem}_share_full_boundary" for stem in CLASS_STEMS]].sum(axis=1)
    valid_share_sum = modal[[f"{stem}_share_valid_area" for stem in CLASS_STEMS]].sum(axis=1)
    full_closure = (
        full_share_sum + modal["unclassified_area_m2"] / modal["boundary_area_m2"] - 1
    ).abs()
    valid_closure = (valid_share_sum - 1).abs()
    matched = modal[["uid", "valid_area_m2"]].merge(
        probability[["uid", "valid_area_m2"]],
        on="uid",
        how="left",
        suffixes=("_modal", "_probability"),
        validate="one_to_one",
    )
    valid_area_difference = (
        matched["valid_area_m2_modal"] - matched["valid_area_m2_probability"]
    ).abs()
    checks = {
        "sixteen_files": len(modal_paths) == 16,
        "rows_1158": len(modal) == 1158,
        "represented_cities_294": modal["city_id"].nunique() == 294,
        "uids_unique": modal["uid"].is_unique,
        "expected_scenario_rows": scenario_rows == EXPECTED_SCENARIO_ROWS,
        "all_scenario_batch_combinations": combinations == expected_combinations,
        "no_null_cells": not modal.isna().any().any(),
        "all_numeric_finite": bool(np.isfinite(numeric.to_numpy(dtype=float)).all()),
        "run_id_frozen": modal["run_id"].nunique() == 1,
        "dataset_id_frozen": modal["landcover_dataset_id"].eq("dynamic_world_2021").all(),
        "estimator_id_frozen": modal["estimator_id"].eq("annual_modal_label_area").all(),
        "no_low_or_zero_valid_coverage": modal["valid_fraction"].ge(0.95).all(),
        "raster_excess_fraction_at_most_0_001": modal["raster_area_excess_fraction"].le(0.001).all(),
        "mass_balance_relative_error_at_most_0_001": modal[
            "mass_balance_relative_error"
        ].abs().le(0.001).all(),
        "modal_tie_area_nonnegative": modal["modal_tie_area_m2"].ge(0).all(),
        "modal_tie_fraction_bounded": modal["modal_tie_fraction_valid_area"].between(0, 1).all(),
        "valid_share_closure_at_most_0_001": valid_closure.le(0.001).all(),
        "full_share_closure_at_most_0_001": full_closure.le(0.001).all(),
        "all_probability_rows_matched": matched["valid_area_m2_probability"].notna().all(),
        "valid_area_matches_probability_within_1m2": valid_area_difference.le(1).all(),
    }
    result = {
        "status": "pass" if all(checks.values()) else "fail",
        "checks": {key: bool(value) for key, value in checks.items()},
        "metrics": {
            "files": len(modal_paths),
            "rows": int(len(modal)),
            "unique_cities": int(modal["city_id"].nunique()),
            "unique_uids": int(modal["uid"].nunique()),
            "scenario_rows": {key: int(value) for key, value in scenario_rows.items()},
            "measurement_status_counts": {
                str(key): int(value) for key, value in modal["measurement_status"].value_counts().items()
            },
            "min_valid_fraction": float(modal["valid_fraction"].min()),
            "max_raster_excess_fraction": float(modal["raster_area_excess_fraction"].max()),
            "max_abs_mass_balance_relative_error": float(
                modal["mass_balance_relative_error"].abs().max()
            ),
            "max_abs_valid_share_closure_error": float(valid_closure.max()),
            "max_abs_full_share_closure_error": float(full_closure.max()),
            "median_modal_tie_fraction_valid_area": float(
                modal["modal_tie_fraction_valid_area"].median()
            ),
            "max_modal_tie_fraction_valid_area": float(
                modal["modal_tie_fraction_valid_area"].max()
            ),
            "max_valid_area_difference_vs_probability_m2": float(valid_area_difference.max()),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
