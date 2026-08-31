"""Validate the frozen 2021 city registry without network access."""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path


REQUIRED_COLUMNS = {
    "city_id", "prov_code", "city_code", "prov_name", "city_name",
    "city_type", "reference_year", "reference_date", "in_scope",
    "admin_geometry_key", "legacy_291_member", "source_authority",
    "source_title", "source_release_date", "canonical_source_url",
    "retrieval_url", "retrieval_note",
}
LEGACY_MISSING = {"三沙市", "日喀则市", "昌都市", "林芝市", "山南市", "那曲市"}


def validate(registry_path: Path, sha_path: Path) -> list[str]:
    errors: list[str] = []
    with registry_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    columns = set(rows[0]) if rows else set()
    if columns != REQUIRED_COLUMNS:
        errors.append(f"column mismatch: {sorted(columns ^ REQUIRED_COLUMNS)}")
    if len(rows) != 297:
        errors.append(f"row count is {len(rows)}, expected 297")
    if len({row["city_id"] for row in rows}) != len(rows):
        errors.append("city_id is not unique")
    if len({row["city_code"] for row in rows}) != len(rows):
        errors.append("city_code is not unique")
    if sum(row["city_type"] == "municipality" for row in rows) != 4:
        errors.append("municipality count is not 4")
    if sum(row["city_type"] == "prefecture_level_city" for row in rows) != 293:
        errors.append("prefecture-level city count is not 293")
    if {row["city_name"] for row in rows}.isdisjoint(LEGACY_MISSING):
        errors.append("legacy-missing diagnostic cities are absent")
    if not LEGACY_MISSING <= {row["city_name"] for row in rows}:
        errors.append("not all six legacy-missing cities are retained")
    if any(row["city_id"] != "CN-" + row["city_code"] for row in rows):
        errors.append("city_id format mismatch")
    if any(row["admin_geometry_key"] != row["city_code"] for row in rows):
        errors.append("administrative geometry key mismatch")
    if any(row["reference_year"] != "2021" for row in rows):
        errors.append("reference year mismatch")
    expected_hash = sha_path.read_text(encoding="ascii").split()[0]
    actual_hash = hashlib.sha256(registry_path.read_bytes()).hexdigest()
    if actual_hash != expected_hash:
        errors.append("SHA-256 mismatch")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("registry", type=Path)
    parser.add_argument("sha256", type=Path)
    args = parser.parse_args()
    errors = validate(args.registry, args.sha256)
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        raise SystemExit(1)
    print("VALID: 297 unique city identities; 293 prefecture-level cities + 4 municipalities")


if __name__ == "__main__":
    main()
