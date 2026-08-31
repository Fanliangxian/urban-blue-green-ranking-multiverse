"""Derive the active 296-city analysis registry from the frozen 297-city frame."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MASTER = ROOT / "data" / "reference" / "city_registry_2021.csv"
ANALYSIS = ROOT / "data" / "reference" / "city_registry_analysis_2021.csv"
EXCLUDED = ROOT / "data" / "reference" / "city_registry_exclusions_2021.csv"
HASH_FILE = ROOT / "data" / "reference" / "city_registry_analysis_2021.sha256"

EXCLUSIONS = {
    "CN-460300": {
        "decision_id": "D-013",
        "decision_date": "2026-08-05",
        "reason_code": "noncomparable_discontinuous_island_maritime_entity",
        "reason": (
            "Dispersed island-maritime administrative entity without a comparable "
            "continuous land-based municipal assignment frame in the frozen source."
        ),
    }
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    with MASTER.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
        base_fields = list(rows[0])
    if len(rows) != 297:
        raise ValueError(f"Frozen master frame must contain 297 identities, found {len(rows)}")

    included = [row for row in rows if row["city_id"] not in EXCLUSIONS]
    excluded = [row for row in rows if row["city_id"] in EXCLUSIONS]
    if len(included) != 296 or len(excluded) != 1:
        raise ValueError("Expected 296 included identities and exactly one exclusion")

    analysis_fields = base_fields + ["analysis_registry_version", "analysis_decision_id"]
    with ANALYSIS.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=analysis_fields)
        writer.writeheader()
        for row in included:
            writer.writerow({**row, "analysis_registry_version": "2.0", "analysis_decision_id": "D-013"})

    exclusion_fields = base_fields + ["decision_id", "decision_date", "reason_code", "reason"]
    with EXCLUDED.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=exclusion_fields)
        writer.writeheader()
        for row in excluded:
            writer.writerow({**row, **EXCLUSIONS[row["city_id"]]})

    digest = sha256(ANALYSIS)
    HASH_FILE.write_text(f"{digest}  {ANALYSIS.name}\n", encoding="ascii")
    print(f"Wrote {len(included)} analysis identities; SHA256={digest}")


if __name__ == "__main__":
    main()

