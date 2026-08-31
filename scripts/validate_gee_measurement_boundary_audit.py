"""Validate the exported GEE audit against the frozen local boundary manifest."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
AUDIT = PROJECT / "data" / "derived" / "R1_measurement_boundary_asset_audit.csv"
MANIFEST = PROJECT / "data" / "derived" / "measurement_boundaries_r1_manifest.json"
PROTOCOL = PROJECT / "config" / "analysis_protocol.yaml"
OUTPUT = PROJECT / "data" / "derived" / "R1_measurement_boundary_asset_audit_validation.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def integer(row: dict[str, str], field: str) -> int:
    value = row.get(field, "")
    if value == "":
        raise ValueError(f"Missing audit field: {field}")
    return int(float(value))


def main() -> None:
    for path in (AUDIT, MANIFEST, PROTOCOL):
        if not path.exists():
            raise FileNotFoundError(path)
    with AUDIT.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise ValueError(f"Expected one GEE audit row, found {len(rows)}")
    row = rows[0]
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    protocol_text = PROTOCOL.read_text(encoding="utf-8")
    match = re.search(r"^\s*gee_asset_id:\s*(\S+)\s*$", protocol_text, re.MULTILINE)
    if not match or match.group(1) == "PENDING_USER_UPLOAD":
        raise ValueError("A concrete measurement-boundary GEE asset ID is not frozen")
    expected_asset = match.group(1)

    expected_counts = {
        "expected_feature_count": manifest["output_geometry_feature_count"],
        "observed_feature_count": manifest["output_geometry_feature_count"],
        "observed_uid_unique_count": manifest["output_uid_unique_count"],
        "null_uid_count": 0,
        "nonpositive_boundary_area_count": 0,
        "gctb_core_raw_city_count": 287,
        "gctb_system_raw_city_count": 287,
        "gub_raw_city_count": 294,
        "ghs_uc_raw_city_count": 290,
    }
    disagreements = {}
    for field, expected in expected_counts.items():
        observed = integer(row, field)
        if observed != expected:
            disagreements[field] = {"expected": expected, "observed": observed}
    if row.get("asset_id") != expected_asset:
        disagreements["asset_id"] = {"expected": expected_asset, "observed": row.get("asset_id")}
    local_manifest_sha = sha256(MANIFEST)
    if row.get("local_manifest_sha256") != local_manifest_sha:
        disagreements["local_manifest_sha256"] = {
            "expected": local_manifest_sha,
            "observed": row.get("local_manifest_sha256"),
        }
    if disagreements:
        raise ValueError(f"GEE audit disagrees with frozen local evidence: {disagreements}")

    validation = {
        "validation_status": "passed",
        "validated_on": "2026-08-06",
        "asset_id": expected_asset,
        "audit_run_id": row.get("audit_run_id"),
        "audit_csv": str(AUDIT),
        "audit_csv_sha256": sha256(AUDIT),
        "local_manifest": str(MANIFEST),
        "local_manifest_sha256": local_manifest_sha,
        "validated_counts": expected_counts,
        "disagreement_count": 0,
    }
    OUTPUT.write_text(json.dumps(validation, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(validation, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
