"""Validate the 16 Dynamic World 2021 national probability exports."""

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

    paths = sorted(args.input_dir.glob("R1_dw2021_prob_296_r3_*.csv"))
    if not paths:
        raise FileNotFoundError("No Dynamic World national r3 CSV files found")
    frames = []
    files = []
    for path in paths:
        frame = pd.read_csv(path)
        frame["source_file"] = path.name
        frames.append(frame)
        files.append(
            {
                "file": path.name,
                "rows": int(len(frame)),
                "cities": int(frame["city_id"].nunique()),
                "uids": int(frame["uid"].nunique()),
                "scenario": frame["bnd_scn"].dropna().unique().tolist(),
                "batch": frame["batch_id"].dropna().unique().tolist(),
                "null_cells": int(frame.isna().sum().sum()),
            }
        )
    data = pd.concat(frames, ignore_index=True)
    scenario_rows = data.groupby("bnd_scn").size().to_dict()
    scenario_cities = data.groupby("bnd_scn")["city_id"].nunique().to_dict()
    combinations = set(zip(data["bnd_scn"], data["batch_id"]))
    expected_combinations = {
        (scenario, batch) for scenario in EXPECTED_SCENARIO_ROWS for batch in EXPECTED_BATCHES
    }
    numeric = data[data.columns[11:-1]].apply(pd.to_numeric, errors="coerce")
    full_share_sum = data[[f"{stem}_share_full_boundary" for stem in CLASS_STEMS]].sum(axis=1)
    valid_share_sum = data[[f"{stem}_share_valid_area" for stem in CLASS_STEMS]].sum(axis=1)
    full_closure = (
        full_share_sum + data["unclassified_area_m2"] / data["boundary_area_m2"] - 1
    ).abs()
    valid_closure = (valid_share_sum - 1).abs()

    hard_checks = {
        "sixteen_files": len(paths) == 16,
        "total_rows_1158": len(data) == 1158,
        "represented_cities_294": data["city_id"].nunique() == 294,
        "structural_missing_cities_absent": not data["city_id"].isin(
            ["CN-540300", "CN-540400"]
        ).any(),
        "expected_scenario_rows": scenario_rows == EXPECTED_SCENARIO_ROWS,
        "all_scenario_batch_combinations": combinations == expected_combinations,
        "uid_unique": data["uid"].is_unique,
        "no_null_cells": not data.isna().any().any(),
        "all_numeric_finite": bool(np.isfinite(numeric.to_numpy(dtype=float)).all()),
        "run_id_frozen": data["run_id"].nunique() == 1,
        "dataset_id_frozen": data["landcover_dataset_id"].eq("dynamic_world_2021").all(),
        "estimator_frozen": data["estimator_id"].eq("annual_mean_probability_area").all(),
        "no_low_or_zero_valid_coverage": data["valid_fraction"].ge(0.95).all(),
        "raster_excess_fraction_at_most_0_001": data["raster_area_excess_fraction"].le(0.001).all(),
        "mass_balance_relative_error_at_most_0_001": data["mass_balance_relative_error"].abs().le(0.001).all(),
        "probability_closure_relative_error_at_most_0_001": data[
            "probability_closure_relative_error"
        ].le(0.001).all(),
        "valid_share_closure_at_most_0_001": valid_closure.le(0.001).all(),
        "full_share_closure_at_most_0_001": full_closure.le(0.001).all(),
    }
    diagnostic_checks = {
        "all_rows_complete_valid_coverage": data["valid_fraction"].ge(0.999).all(),
        "partial_rows_only_nagqu": set(
            data.loc[data["valid_fraction"].lt(0.999), "city_id"]
        ).issubset({"CN-540600"}),
    }
    hard_pass = all(hard_checks.values())
    has_partial = not diagnostic_checks["all_rows_complete_valid_coverage"]
    result = {
        "status": "pass_with_partial_coverage_flags" if hard_pass and has_partial else (
            "pass" if hard_pass else "fail"
        ),
        "hard_checks": {key: bool(value) for key, value in hard_checks.items()},
        "diagnostic_checks": {key: bool(value) for key, value in diagnostic_checks.items()},
        "files": files,
        "metrics": {
            "files": len(paths),
            "rows": int(len(data)),
            "unique_cities": int(data["city_id"].nunique()),
            "unique_uids": int(data["uid"].nunique()),
            "scenario_rows": {key: int(value) for key, value in scenario_rows.items()},
            "scenario_cities": {key: int(value) for key, value in scenario_cities.items()},
            "measurement_status_counts": {
                str(key): int(value) for key, value in data["measurement_status"].value_counts().items()
            },
            "min_valid_fraction": float(data["valid_fraction"].min()),
            "max_raster_excess_fraction": float(data["raster_area_excess_fraction"].max()),
            "max_abs_mass_balance_relative_error": float(data["mass_balance_relative_error"].abs().max()),
            "max_probability_closure_relative_error": float(
                data["probability_closure_relative_error"].max()
            ),
            "max_abs_valid_share_closure_error": float(valid_closure.max()),
            "max_abs_full_share_closure_error": float(full_closure.max()),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["status"] == "fail":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
