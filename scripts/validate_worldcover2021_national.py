"""Validate four immutable national WorldCover 2021 GEE exports."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
DERIVED = PROJECT / "data" / "derived"
INVENTORY = DERIVED / "measurement_boundaries_r1_inventory.csv"
PROTOCOL = PROJECT / "config" / "landcover_measurement_protocol.yaml"
OUTPUT = DERIVED / "R1_worldcover2021_296_validation.json"
SCENARIOS = {
    "gctb_core_2021": 287,
    "gctb_system_2021": 287,
    "gub_2018": 294,
    "ghs_uc_2020": 290,
}
INPUTS = {
    scenario: DERIVED / f"R1_worldcover2021_296_{scenario}.csv"
    for scenario in SCENARIOS
}
CLASS_FIELDS = ["Blue_m2", "Green_m2", "Grey_m2", "Farm_m2", "Other_Uncertain_m2"]
VALID_SHARE_FIELDS = [field.replace("_m2", "_share_valid_area") for field in CLASS_FIELDS]
NONNEGATIVE_FIELDS = [
    "boundary_area_m2", "valid_area_m2", "unclassified_area_m2",
    "raster_area_excess_m2", "raster_area_excess_fraction",
    "unmapped_valid_m2", "class_sum_m2", *CLASS_FIELDS,
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def number(row: dict[str, str], field: str) -> float:
    try:
        value = float(row[field])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid {field} for {row.get('uid')}") from exc
    if not math.isfinite(value):
        raise ValueError(f"Non-finite {field} for {row.get('uid')}")
    return value


def main() -> None:
    for path in (*INPUTS.values(), INVENTORY, PROTOCOL):
        if not path.exists():
            raise FileNotFoundError(path)
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    mass_tolerance = protocol["area_balance_relative_tolerance"]
    excess_tolerance = protocol["valid_area_boundary_excess_relative_tolerance"]
    inventory_rows = read_rows(INVENTORY)
    expected_keys = {
        scenario: {
            row["uid"]
            for row in inventory_rows
            if row["boundary_scenario_id"] == scenario
            and row["geometry_variant"] == "raw_product_geometry"
            and row["coverage_status"] == "available_matched"
        }
        for scenario in SCENARIOS
    }
    all_rows: list[dict[str, str]] = []
    file_audit = {}
    scenario_metrics = {}
    for scenario, expected_count in SCENARIOS.items():
        path = INPUTS[scenario]
        data = read_rows(path)
        keys = [row.get("uid", "") for row in data]
        if len(data) != expected_count or len(keys) != len(set(keys)):
            raise ValueError(f"Count or UID uniqueness failure for {scenario}")
        if set(keys) != expected_keys[scenario]:
            raise ValueError(
                f"Key mismatch for {scenario}; missing={sorted(expected_keys[scenario] - set(keys))[:10]}, "
                f"extra={sorted(set(keys) - expected_keys[scenario])[:10]}"
            )
        max_mass = 0.0
        max_share_error = 0.0
        min_raw_valid = math.inf
        max_raw_valid = 0.0
        max_excess = 0.0
        negative_signed_difference_count = 0
        for row in data:
            uid = row["uid"]
            if row.get("run_id") != "worldcover2021_296_20260806_r1":
                raise ValueError(f"Unexpected run ID for {uid}")
            if row.get("bnd_scn") != scenario or row.get("geom_var") != "raw_product_geometry":
                raise ValueError(f"Scenario or geometry mismatch for {uid}")
            if row.get("landcover_dataset_id") != "esa_worldcover_2021":
                raise ValueError(f"Product mismatch for {uid}")
            if row.get("measurement_status") != "complete_valid_coverage":
                raise ValueError(f"Unexpected coverage status for {uid}")
            expected_formal = "0" if scenario == "gub_2018" else "1"
            if row.get("formal_ok") != expected_formal:
                raise ValueError(f"Formal eligibility changed for {uid}")
            for field in NONNEGATIVE_FIELDS:
                if number(row, field) < 0:
                    raise ValueError(f"Negative {field} for {uid}")
            boundary = number(row, "boundary_area_m2")
            valid = number(row, "valid_area_m2")
            raw_fraction = number(row, "raw_valid_fraction")
            capped_fraction = number(row, "valid_fraction")
            if boundary <= 0 or valid <= 0:
                raise ValueError(f"Non-positive denominator for {uid}")
            if abs(raw_fraction - valid / boundary) > 1e-12:
                raise ValueError(f"Raw valid fraction mismatch for {uid}")
            if abs(capped_fraction - min(1.0, max(0.0, raw_fraction))) > 1e-12:
                raise ValueError(f"Capped valid fraction mismatch for {uid}")
            class_sum = sum(number(row, field) for field in CLASS_FIELDS)
            unmapped = number(row, "unmapped_valid_m2")
            mass_error = abs(class_sum + unmapped - valid) / valid
            max_mass = max(max_mass, mass_error)
            if mass_error > mass_tolerance or unmapped != 0:
                raise ValueError(f"Class mass or native legend failure for {uid}")
            share_error = abs(sum(number(row, field) for field in VALID_SHARE_FIELDS) - 1)
            max_share_error = max(max_share_error, share_error)
            if share_error > mass_tolerance:
                raise ValueError(f"Valid-area shares do not close for {uid}")
            signed = number(row, "boundary_minus_valid_area_m2")
            if abs(signed - (boundary - valid)) > max(1e-6, boundary * 1e-12):
                raise ValueError(f"Signed area diagnostic mismatch for {uid}")
            expected_unclassified = max(0.0, signed)
            expected_excess = max(0.0, -signed)
            if abs(number(row, "unclassified_area_m2") - expected_unclassified) > max(1e-6, boundary * 1e-12):
                raise ValueError(f"Unclassified area mismatch for {uid}")
            if abs(number(row, "raster_area_excess_m2") - expected_excess) > max(1e-6, boundary * 1e-12):
                raise ValueError(f"Raster excess mismatch for {uid}")
            excess_fraction = number(row, "raster_area_excess_fraction")
            if excess_fraction > excess_tolerance:
                raise ValueError(f"Raster excess exceeds tolerance for {uid}")
            min_raw_valid = min(min_raw_valid, raw_fraction)
            max_raw_valid = max(max_raw_valid, raw_fraction)
            max_excess = max(max_excess, excess_fraction)
            negative_signed_difference_count += int(signed < 0)
        file_audit[scenario] = {
            "path": str(path), "sha256": sha256(path), "row_count": len(data),
            "unique_uid_count": len(set(keys)),
        }
        scenario_metrics[scenario] = {
            "row_count": len(data),
            "minimum_raw_valid_fraction": min_raw_valid,
            "maximum_raw_valid_fraction": max_raw_valid,
            "maximum_raster_area_excess_fraction": max_excess,
            "negative_boundary_minus_valid_row_count": negative_signed_difference_count,
            "maximum_mass_balance_relative_error_recomputed": max_mass,
            "maximum_valid_share_closure_error": max_share_error,
            "maximum_unmapped_valid_area_m2": 0.0,
            "formal_analysis_eligible": scenario != "gub_2018",
        }
        all_rows.extend(data)
    all_keys = [row["uid"] for row in all_rows]
    if len(all_rows) != 1158 or len(all_keys) != len(set(all_keys)):
        raise ValueError("Combined national UID inventory is not 1,158 unique rows")
    result = {
        "validation_status": "passed",
        "validated_on": "2026-08-06",
        "run_id": "worldcover2021_296_20260806_r1",
        "landcover_dataset_id": "esa_worldcover_2021",
        "total_row_count": len(all_rows),
        "total_unique_uid_count": len(set(all_keys)),
        "registered_city_count": 296,
        "files": file_audit,
        "scenario_metrics": scenario_metrics,
        "mass_balance_relative_tolerance": mass_tolerance,
        "boundary_excess_relative_tolerance": excess_tolerance,
        "formal_boundary_scenario_row_count": sum(
            metric["row_count"] for metric in scenario_metrics.values()
            if metric["formal_analysis_eligible"]
        ),
        "gub_technical_only_row_count": scenario_metrics["gub_2018"]["row_count"],
        "scientific_interpretation": "national_empirical_coverage_and_measurement_validation_not_accuracy_validation",
    }
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
