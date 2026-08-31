"""Validate national Dynamic World observation-support exports against main measurements."""

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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    obs_paths = sorted(args.input_dir.glob("R1_dw2021_obs_296_r1_*.csv"))
    main_paths = sorted(args.input_dir.glob("R1_dw2021_prob_296_r3_*.csv"))
    if not obs_paths or not main_paths:
        raise FileNotFoundError("Observation or main Dynamic World CSV files are missing")
    obs = pd.concat([pd.read_csv(path) for path in obs_paths], ignore_index=True)
    main = pd.concat([pd.read_csv(path) for path in main_paths], ignore_index=True)
    merged = obs.merge(
        main[["uid", "valid_area_m2", "unclassified_area_m2", "valid_fraction"]],
        on="uid",
        how="left",
        validate="one_to_one",
    )
    bin_fraction_columns = [
        "obs_0_fraction", "obs_1_4_fraction", "obs_5_9_fraction",
        "obs_10_19_fraction", "obs_ge20_fraction",
    ]
    observed_difference = (merged["observed_area_m2"] - merged["valid_area_m2"]).abs()
    zero_difference = (merged["obs_0_area_m2"] - merged["unclassified_area_m2"]).abs()
    positive_zero_mask = merged["obs_0_area_m2"].gt(0)
    checks = {
        "sixteen_files": len(obs_paths) == 16,
        "rows_1158": len(obs) == 1158,
        "uids_unique": obs["uid"].is_unique,
        "cities_294": obs["city_id"].nunique() == 294,
        "scenario_rows": obs.groupby("bnd_scn").size().to_dict() == EXPECTED_SCENARIO_ROWS,
        "no_null_cells": not obs.isna().any().any(),
        "all_numeric_finite": bool(np.isfinite(obs.iloc[:, 5:].to_numpy(dtype=float)).all()),
        "all_main_rows_matched": merged["valid_area_m2"].notna().all(),
        "observed_area_matches_main_within_1m2": observed_difference.le(1).all(),
        "positive_zero_area_matches_unclassified_within_pixel_edge_tolerance": zero_difference[
            positive_zero_mask
        ].le(1000).all(),
        "observation_bin_closure_at_most_0_001": obs[
            "observation_bin_closure_relative_error"
        ].le(0.001).all(),
        "fraction_bins_close_at_most_0_001": (
            obs[bin_fraction_columns].sum(axis=1) - 1
        ).abs().le(0.001).all(),
    }
    result = {
        "status": "pass" if all(checks.values()) else "fail",
        "checks": {key: bool(value) for key, value in checks.items()},
        "metrics": {
            "files": len(obs_paths),
            "rows": int(len(obs)),
            "unique_cities": int(obs["city_id"].nunique()),
            "min_mean_observation_count": float(
                obs["mean_observation_count_on_observed_area"].min()
            ),
            "median_mean_observation_count": float(
                obs["mean_observation_count_on_observed_area"].median()
            ),
            "max_obs_0_fraction": float(obs["obs_0_fraction"].max()),
            "max_obs_1_4_fraction": float(obs["obs_1_4_fraction"].max()),
            "rows_with_any_zero_observation_area": int(obs["obs_0_fraction"].gt(0).sum()),
            "rows_with_more_than_half_area_at_1_4_observations": int(
                obs["obs_1_4_fraction"].gt(0.5).sum()
            ),
            "max_observed_area_difference_m2": float(observed_difference.max()),
            "max_positive_zero_area_difference_m2": float(
                zero_difference[positive_zero_mask].max()
            ),
            "max_observation_bin_closure_relative_error": float(
                obs["observation_bin_closure_relative_error"].max()
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
