"""Merge and audit the 50 frozen Task 6A national GEE exports."""

import csv
import hashlib
import json
import math
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DERIVED = ROOT / "data" / "derived"
REFERENCE = ROOT / "data" / "reference"
EXPECTED_RUN = "task6_polycentricity_periurban_farm_national296_20260818_r1"
EXPECTED_COLUMNS = [
    "run_id", "city_id", "city_name", "urban_grid_m",
    "urban_density_threshold_people_km2", "minimum_center_population",
    "center_count", "retained_center_population", "center_population_hhi",
    "polycentricity_index", "periurban_buffer_m", "periurban_ring_area_m2",
    "periurban_crop_area_equivalent_m2", "periurban_farm_fraction",
]
PILOT_IDS = [
    "CN-110000", "CN-230100", "CN-310000", "CN-420100",
    "CN-440100", "CN-440300", "CN-500000", "CN-510100",
    "CN-540100", "CN-610100", "CN-620100", "CN-650100",
]
COMPARE_FIELDS = EXPECTED_COLUMNS[3:]


def read_rows(path):
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return reader.fieldnames, list(reader)


def write_rows(path, fieldnames, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def close(a, b, rel=1e-12, abs_=1e-9):
    return math.isclose(float(a), float(b), rel_tol=rel, abs_tol=abs_)


def summary(values):
    values = sorted(values)
    return {
        "min": min(values),
        "median": statistics.median(values),
        "max": max(values),
    }


def main():
    failures = []
    registry_fields, registry = read_rows(REFERENCE / "city_registry_analysis_2021.csv")
    basic_fields, basic = read_rows(DERIVED / "R1_task6_basic_covariates_296.csv")
    registry_ids = [row["city_id"] for row in registry]
    sorted_ids = sorted(registry_ids)
    basic_by_id = {row["city_id"]: row for row in basic}
    if len(registry) != 296 or len(set(registry_ids)) != 296:
        failures.append("analysis registry is not 296 unique cities")
    if set(basic_by_id) != set(registry_ids):
        failures.append("basic covariates do not match the registered 296 cities")

    expected_names = [
        f"R1_task6_polycentricity_periurban_farm_national_r1_batch_{i:03d}.csv"
        for i in range(1, 51)
    ]
    actual_paths = sorted(DERIVED.glob(
        "R1_task6_polycentricity_periurban_farm_national_r1_batch_*.csv"
    ))
    actual_names = [path.name for path in actual_paths]
    if actual_names != expected_names:
        failures.append("national batch filenames are missing, extra, or non-contiguous")

    merged = []
    hashes = {}
    batch_audit = []
    for batch_number, name in enumerate(expected_names, start=1):
        path = DERIVED / name
        if not path.exists():
            continue
        hashes[name] = sha256(path)
        fields, rows = read_rows(path)
        if fields != EXPECTED_COLUMNS:
            failures.append(f"{name}: schema differs from frozen schema")
        expected_count = 2 if batch_number == 50 else 6
        expected_batch_ids = sorted_ids[(batch_number - 1) * 6:batch_number * 6]
        row_ids = [row.get("city_id", "") for row in rows]
        if len(rows) != expected_count:
            failures.append(f"{name}: expected {expected_count} rows, found {len(rows)}")
        # Earth Engine preserves batch membership but does not guarantee CSV
        # feature order. Require exact unique membership; sort the merged table
        # deterministically below instead of treating row order as substantive.
        if len(set(row_ids)) != len(row_ids) or set(row_ids) != set(expected_batch_ids):
            failures.append(
                f"{name}: IDs differ from frozen registry batch; "
                f"expected {expected_batch_ids}, found {row_ids}"
            )
        for row in rows:
            row["source_batch"] = f"{batch_number:03d}"
            merged.append(row)
        batch_audit.append({
            "batch": f"{batch_number:03d}",
            "row_count": len(rows),
            "first_city_id_sorted": min(row_ids) if row_ids else None,
            "last_city_id_sorted": max(row_ids) if row_ids else None,
        })

    merged_by_id = {row.get("city_id"): row for row in merged}
    merged_ids = [row.get("city_id") for row in merged]
    if len(merged) != 296:
        failures.append(f"expected 296 merged rows, found {len(merged)}")
    if len(set(merged_ids)) != len(merged_ids):
        failures.append("duplicate city IDs found across national batches")
    if set(merged_ids) != set(registry_ids):
        failures.append("national batch city IDs do not equal the registered 296")

    zero_center_ids = []
    retained_fractions = []
    numeric_values = {field: [] for field in [
        "center_count", "polycentricity_index", "periurban_farm_fraction"
    ]}
    for city_id, row in merged_by_id.items():
        if row.get("run_id") != EXPECTED_RUN:
            failures.append(f"{city_id}: wrong run_id")
        for field, expected in {
            "urban_grid_m": 1000,
            "urban_density_threshold_people_km2": 1500,
            "minimum_center_population": 50000,
            "periurban_buffer_m": 5000,
        }.items():
            if not close(row.get(field, "nan"), expected):
                failures.append(f"{city_id}: {field} is not frozen value {expected}")
        try:
            numbers = {field: float(row[field]) for field in COMPARE_FIELDS}
        except (KeyError, TypeError, ValueError):
            failures.append(f"{city_id}: missing or nonnumeric measurement")
            continue
        if not all(math.isfinite(value) for value in numbers.values()):
            failures.append(f"{city_id}: non-finite measurement")
            continue
        k = numbers["center_count"]
        retained = numbers["retained_center_population"]
        hhi = numbers["center_population_hhi"]
        poly = numbers["polycentricity_index"]
        ring = numbers["periurban_ring_area_m2"]
        crop = numbers["periurban_crop_area_equivalent_m2"]
        farm = numbers["periurban_farm_fraction"]
        if k < 0 or not close(k, round(k)):
            failures.append(f"{city_id}: invalid center_count {k}")
        if retained < -1e-9 or ring < -1e-9 or crop < -1e-9:
            failures.append(f"{city_id}: negative population or area")
        city_population = float(basic_by_id[city_id]["population_2020_worldpop"])
        if retained > city_population * (1 + 1e-6):
            failures.append(f"{city_id}: retained centers exceed full city population")
        retained_fractions.append(retained / city_population if city_population > 0 else 0)
        if k == 0:
            zero_center_ids.append(city_id)
            if not (close(retained, 0) and close(hhi, 0) and close(poly, 0)):
                failures.append(f"{city_id}: zero-center identity failed")
        elif k == 1:
            if retained < 50000 or not (close(hhi, 1) and close(poly, 0)):
                failures.append(f"{city_id}: one-center identity failed")
        else:
            if retained < k * 50000 - 1e-6:
                failures.append(f"{city_id}: retained population below center minimum total")
            if hhi < 1 / k - 1e-12 or hhi > 1 + 1e-12:
                failures.append(f"{city_id}: HHI outside [1/k, 1]")
            if not close(poly, 1 - hhi):
                failures.append(f"{city_id}: polycentricity != 1 - HHI")
        if crop > ring * (1 + 1e-12):
            failures.append(f"{city_id}: crop-equivalent area exceeds ring area")
        if ring == 0:
            if not (close(crop, 0) and close(farm, 0)):
                failures.append(f"{city_id}: zero-ring identity failed")
        elif not close(farm, crop / ring):
            failures.append(f"{city_id}: periurban farm fraction identity failed")
        if farm < -1e-12 or farm > 1 + 1e-12:
            failures.append(f"{city_id}: periurban farm fraction outside [0,1]")
        numeric_values["center_count"].append(k)
        numeric_values["polycentricity_index"].append(poly)
        numeric_values["periurban_farm_fraction"].append(farm)

    pilot_comparisons = 0
    for city_id in PILOT_IDS:
        pilot_path = DERIVED / (
            "R1_task6_polycentricity_periurban_farm_pilot_r4_" +
            city_id.removeprefix("CN-") + ".csv"
        )
        _, pilot_rows = read_rows(pilot_path)
        if len(pilot_rows) != 1 or city_id not in merged_by_id:
            failures.append(f"{city_id}: pilot comparison unavailable")
            continue
        for field in COMPARE_FIELDS:
            if not close(pilot_rows[0][field], merged_by_id[city_id][field]):
                failures.append(f"{city_id}: national {field} differs from r4 pilot")
        pilot_comparisons += 1

    merged.sort(key=lambda row: row["city_id"])
    raw_output = DERIVED / "R1_task6_polycentricity_periurban_farm_296.csv"
    write_rows(raw_output, EXPECTED_COLUMNS + ["source_batch"], merged)

    combined_fields = [
        "city_id", "prov_code", "prov_name", "city_name",
        "reference_area_km2", "reference_perimeter_km",
        "population_2020_worldpop", "boundary_shape_complexity",
        "polycentricity_index", "periurban_farm_fraction",
        "periurban_farm_defined",
        "surface_water_density", "terrain_ruggedness_riley_mean_m",
        "center_count", "retained_center_population", "center_population_hhi",
        "periurban_ring_area_m2", "periurban_crop_area_equivalent_m2",
        "task6_measurement_run_id", "task6_source_batch",
    ]
    registry_by_id = {row["city_id"]: row for row in registry}
    combined = []
    for city_id in sorted_ids:
        base = basic_by_id[city_id]
        urban = merged_by_id.get(city_id, {})
        reg = registry_by_id[city_id]
        combined.append({
            "city_id": city_id,
            "prov_code": reg["prov_code"],
            "prov_name": reg["prov_name"],
            "city_name": reg["city_name"],
            "reference_area_km2": base["reference_area_km2"],
            "reference_perimeter_km": base["reference_perimeter_km"],
            "population_2020_worldpop": base["population_2020_worldpop"],
            "boundary_shape_complexity": base["boundary_shape_complexity"],
            "polycentricity_index": urban.get("polycentricity_index", ""),
            # A zero-centre city has no 5 km outer ring. Preserve the raw GEE
            # zero in the measurement table, but encode the derived fraction
            # as structurally missing rather than misclassifying 0/0 as no farm.
            "periurban_farm_fraction": (
                urban.get("periurban_farm_fraction", "")
                if float(urban.get("periurban_ring_area_m2", 0) or 0) > 0
                else ""
            ),
            "periurban_farm_defined": (
                "1" if float(urban.get("periurban_ring_area_m2", 0) or 0) > 0 else "0"
            ),
            "surface_water_density": base["surface_water_density"],
            "terrain_ruggedness_riley_mean_m": base["terrain_ruggedness_riley_mean_m"],
            "center_count": urban.get("center_count", ""),
            "retained_center_population": urban.get("retained_center_population", ""),
            "center_population_hhi": urban.get("center_population_hhi", ""),
            "periurban_ring_area_m2": urban.get("periurban_ring_area_m2", ""),
            "periurban_crop_area_equivalent_m2": urban.get(
                "periurban_crop_area_equivalent_m2", ""
            ),
            "task6_measurement_run_id": urban.get("run_id", ""),
            "task6_source_batch": urban.get("source_batch", ""),
        })
    combined_output = DERIVED / "R1_task6_covariates_296.csv"
    write_rows(combined_output, combined_fields, combined)

    missing_by_field = {
        field: sum(row[field] == "" for row in combined)
        for field in combined_fields
    }
    expected_missing = {
        field: (len(zero_center_ids) if field == "periurban_farm_fraction" else 0)
        for field in combined_fields
    }
    if missing_by_field != expected_missing:
        failures.append(
            "combined covariate missingness differs from the expected "
            f"zero-centre structural pattern: {missing_by_field}"
        )

    _, task5_rows = read_rows(
        DERIVED / "R1_task5_city_factorial_sensitivity_decomposition.csv"
    )
    primary_ids = {
        row["city_id"] for row in task5_rows
        if row["scale"] == "common285_normalized_rank_percentile"
        and row["outcome_id"] == "blue_green_share"
    }
    primary_structural_missing = sorted(primary_ids.intersection(zero_center_ids))
    primary_complete_case_n = len(primary_ids) - len(primary_structural_missing)

    report = {
        "status": "pass" if not failures else "fail",
        "gate_6a_recommendation": "pass" if not failures else "do_not_pass",
        "file_count": len(actual_paths),
        "row_count": len(merged),
        "unique_city_id_count": len(set(merged_ids)),
        "registered_city_match": set(merged_ids) == set(registry_ids),
        "pilot_city_exact_comparisons": pilot_comparisons,
        "failure_count": len(failures),
        "failures": failures,
        "zero_center_city_count": len(zero_center_ids),
        "zero_center_city_ids": sorted(zero_center_ids),
        "periurban_farm_structural_missing_count": len(zero_center_ids),
        "primary_common_city_count": len(primary_ids),
        "primary_complete_case_count": primary_complete_case_n,
        "primary_complete_case_excluded_city_ids": primary_structural_missing,
        "summaries": {
            field: summary(values) for field, values in numeric_values.items()
        } | {"retained_population_fraction": summary(retained_fractions)},
        "missing_by_combined_field": missing_by_field,
        "batch_audit": batch_audit,
        "input_sha256": hashes,
        "output_sha256": {
            raw_output.name: sha256(raw_output),
            combined_output.name: sha256(combined_output),
        },
    }
    audit_output = DERIVED / "R1_task6_national_covariates_296_audit.json"
    audit_output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if not failures else 1)


if __name__ == "__main__":
    main()
