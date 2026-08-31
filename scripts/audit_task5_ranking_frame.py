import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MATRIX = ROOT / "data" / "derived" / "R1_city_scenario_landcover_estimator_matrix.csv"
OUTPUT = ROOT / "data" / "derived" / "R1_task5_ranking_frame_audit.json"

BOUNDARIES = (
    "gctb_core_2021",
    "gctb_system_2021",
    "gub_2018",
    "ghs_uc_2020",
)
ESTIMATORS = (
    ("esa_worldcover_2021", "categorical_native_pixel_area"),
    ("dynamic_world_2021", "annual_mean_probability_area"),
)


def main():
    with MATRIX.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    registered = {row["city_id"] for row in rows}
    eligible_by_spec = defaultdict(set)
    missing_reason_counts = Counter()

    for row in rows:
        estimator = (row["landcover_dataset_id"], row["estimator_id"])
        if row["bnd_scn"] not in BOUNDARIES or estimator not in ESTIMATORS:
            continue
        spec = f"{row['bnd_scn']}|{row['landcover_dataset_id']}|{row['estimator_id']}"
        if row["confirmatory_main_row"] == "1" and row["measurement_available"] == "1":
            eligible_by_spec[spec].add(row["city_id"])
        else:
            reason = row["boundary_missing_reason"] or "not_confirmatory_eligible"
            missing_reason_counts[reason] += 1

    expected_specs = {
        f"{boundary}|{dataset}|{estimator}"
        for boundary in BOUNDARIES
        for dataset, estimator in ESTIMATORS
    }
    if set(eligible_by_spec) != expected_specs:
        raise ValueError("Confirmatory specification set does not match the frozen 4 x 2 design")

    common = set.intersection(*(eligible_by_spec[spec] for spec in sorted(expected_specs)))
    union = set.union(*(eligible_by_spec[spec] for spec in sorted(expected_specs)))
    audit = {
        "status": "pass",
        "audit_scope": "eligibility_and_missingness_only_no_ecological_values_or_ranks_computed",
        "registered_city_count": len(registered),
        "confirmatory_specification_count": len(expected_specs),
        "eligible_city_count_by_specification": {
            spec: len(eligible_by_spec[spec]) for spec in sorted(expected_specs)
        },
        "cities_measurable_in_any_confirmatory_specification": len(union),
        "cities_measurable_in_all_confirmatory_specifications": len(common),
        "cities_not_in_common_intersection_count": len(registered - common),
        "cities_not_in_common_intersection": sorted(registered - common),
        "missing_reason_row_counts": dict(sorted(missing_reason_counts.items())),
    }
    OUTPUT.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
