import csv
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data" / "derived" / "R1_task6_reference_admin2021_asset_audit.csv"
OUTPUT = ROOT / "data" / "derived" / "R1_task6_reference_admin2021_asset_audit_validation.json"


def truthy(value):
    return str(value).strip().lower() in {"true", "1"}


def main():
    with INPUT.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    ids = [row["city_id"] for row in rows]
    codes = [row["src_code"] for row in rows]
    failures = []
    if len(rows) != 296:
        failures.append(f"row_count={len(rows)}")
    if len(set(ids)) != 296 or any(not value for value in ids):
        failures.append(f"unique_city_id_count={len(set(ids))}")
    if len(set(codes)) != 296 or any(not value for value in codes):
        failures.append(f"unique_src_code_count={len(set(codes))}")
    for row in rows:
        numeric = [float(row[key]) for key in (
            "raw_area_m2", "area_m2", "polygon_extraction_relative_area_difference",
            "perimeter_m", "shape_complexity_log_ratio"
        )]
        # Raw LineString children are retained as a diagnostic. Eligibility is
        # determined by the polygon-only normalized geometry and area closure.
        if (row["normalized_geometry_type"] not in {"Polygon", "MultiPolygon"}
                or row["src_year"] not in {"2021", "2021.0"}
                or not truthy(row["positive_area"]) or not truthy(row["positive_perimeter"])
                or not all(math.isfinite(value) for value in numeric)
                or numeric[0] <= 0 or numeric[1] <= 0 or numeric[2] > 1e-10
                or numeric[3] <= 0 or numeric[4] < -1e-10):
            failures.append(row["city_id"] or "missing_city_id")
    result = {
        "status": "pass" if not failures else "fail",
        "row_count": len(rows),
        "unique_city_id_count": len(set(ids)),
        "unique_src_code_count": len(set(codes)),
        "failure_count": len(failures),
        "failures": failures,
        "raw_non_polygon_child_city_count": sum(
            int(float(row["non_polygon_child_count"])) > 0 for row in rows
        ),
        "raw_non_polygon_child_city_ids": [
            row["city_id"] for row in rows
            if int(float(row["non_polygon_child_count"])) > 0
        ],
        "maximum_polygon_extraction_relative_area_difference": max(
            float(row["polygon_extraction_relative_area_difference"]) for row in rows
        ),
        "frozen_relative_area_tolerance": 1e-10,
    }
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
