"""Build the independent 2021 registry of 297 Chinese city identities.

The authoritative source is the Ministry of Civil Affairs (MCA) 2021
county-and-above administrative code table. The original MCA page is retired,
so the script retrieves two preserved HTML copies while retaining both the
canonical official URL and actual retrieval URLs in the outputs.

The registry is selected from administrative codes and names only. No urban
boundary or land-cover product participates in inclusion.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import re
import urllib.request
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Iterable, Sequence


CANONICAL_OFFICIAL_URL = (
    "https://www.mca.gov.cn/article/sj/xzqh/1980/202203/"
    "20220300040708.shtml"
)
MIRROR_URLS = (
    "https://www.mzwu.com/article.asp?id=4997",
    "https://www.mzwu.com/article.asp?id=4998",
)
REFERENCE_YEAR = 2021
REFERENCE_DATE = "2021-12-31"
OFFICIAL_RELEASE_DATE = "2022-03-21"
MUNICIPALITY_CODES = {"110000", "120000", "310000", "500000"}
MAINLAND_PROVINCE_PREFIXES = {
    "11", "12", "13", "14", "15", "21", "22", "23", "31", "32",
    "33", "34", "35", "36", "37", "41", "42", "43", "44", "45",
    "46", "50", "51", "52", "53", "54", "61", "62", "63", "64",
    "65",
}
LEGACY_291_MISSING_NAMES = {
    "三沙市", "日喀则市", "昌都市", "林芝市", "山南市", "那曲市"
}


@dataclass(frozen=True)
class AdminRecord:
    code: str
    name: str
    retrieval_url: str


class _VisibleTextParser(HTMLParser):
    """Extract visible text and preserve row-like HTML breaks."""

    BREAK_TAGS = {"br", "div", "p", "tr", "li", "h1", "h2", "h3"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() in self.BREAK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in self.BREAK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        self.parts.append(data)

    def text(self) -> str:
        return "".join(self.parts)


def fetch_html(url: str, timeout_seconds: int = 60) -> str:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 city-registry-research/1.0"},
    )
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        raw = response.read()
    return raw.decode("utf-8")


def parse_admin_records(page_html: str, retrieval_url: str) -> list[AdminRecord]:
    parser = _VisibleTextParser()
    parser.feed(page_html)
    records: list[AdminRecord] = []
    row_pattern = re.compile(r"^\s*(\d{6})\s+([\u3400-\u9fff][^\s]*)\s*$")
    for raw_line in parser.text().splitlines():
        normalized = html.unescape(raw_line).replace("\xa0", " ").strip()
        match = row_pattern.match(normalized)
        if match:
            records.append(
                AdminRecord(
                    code=match.group(1),
                    name=match.group(2).strip(),
                    retrieval_url=retrieval_url,
                )
            )
    return records


def merge_source_records(groups: Iterable[Iterable[AdminRecord]]) -> list[AdminRecord]:
    """Merge exact repeated rows while preserving conflicting source rows.

    The preserved HTML contains a known transcription defect around Sansha:
    subordinate district names repeat the city code 460300. We retain those
    raw source rows for audit instead of silently choosing a name. Selection
    and uniqueness are validated later at the target city level.
    """
    by_code_and_name: dict[tuple[str, str], AdminRecord] = {}
    for group in groups:
        for record in group:
            by_code_and_name.setdefault((record.code, record.name), record)
    return sorted(
        by_code_and_name.values(), key=lambda row: (row.code, row.name)
    )


def build_registry(records: Sequence[AdminRecord]) -> list[dict[str, str]]:
    province_names = {
        row.code: row.name
        for row in records
        if row.code.endswith("0000") and row.code[:2] in MAINLAND_PROVINCE_PREFIXES
    }
    expected_province_codes = {prefix + "0000" for prefix in MAINLAND_PROVINCE_PREFIXES}
    if set(province_names) != expected_province_codes:
        missing = sorted(expected_province_codes - set(province_names))
        extra = sorted(set(province_names) - expected_province_codes)
        raise ValueError(f"Province table mismatch; missing={missing}, extra={extra}")

    selected: list[tuple[AdminRecord, str]] = []
    for row in records:
        if row.code in MUNICIPALITY_CODES:
            selected.append((row, "municipality"))
        elif (
            row.code[:2] in MAINLAND_PROVINCE_PREFIXES
            and row.code[2:4] != "00"
            and row.code[4:6] == "00"
            and row.name.endswith("市")
        ):
            selected.append((row, "prefecture_level_city"))

    registry: list[dict[str, str]] = []
    for row, city_type in sorted(selected, key=lambda item: item[0].code):
        prov_code = row.code[:2] + "0000"
        registry.append(
            {
                "city_id": f"CN-{row.code}",
                "prov_code": prov_code,
                "city_code": row.code,
                "prov_name": province_names[prov_code],
                "city_name": row.name,
                "city_type": city_type,
                "reference_year": str(REFERENCE_YEAR),
                "reference_date": REFERENCE_DATE,
                "in_scope": "1",
                "admin_geometry_key": row.code,
                "legacy_291_member": (
                    "0" if row.name in LEGACY_291_MISSING_NAMES else "1"
                ),
                "source_authority": "Ministry of Civil Affairs of the PRC",
                "source_title": "2021年中华人民共和国县以上行政区划代码",
                "source_release_date": OFFICIAL_RELEASE_DATE,
                "canonical_source_url": CANONICAL_OFFICIAL_URL,
                "retrieval_url": row.retrieval_url,
                "retrieval_note": "preserved copy of retired official table",
            }
        )
    validate_registry(registry)
    return registry


def validate_registry(registry: Sequence[dict[str, str]]) -> None:
    if len(registry) != 297:
        raise ValueError(f"Expected 297 cities, found {len(registry)}")
    city_ids = [row["city_id"] for row in registry]
    city_codes = [row["city_code"] for row in registry]
    if len(city_ids) != len(set(city_ids)):
        raise ValueError("city_id is not unique")
    if len(city_codes) != len(set(city_codes)):
        raise ValueError("city_code is not unique")
    type_counts = {
        city_type: sum(row["city_type"] == city_type for row in registry)
        for city_type in {row["city_type"] for row in registry}
    }
    if type_counts != {"prefecture_level_city": 293, "municipality": 4}:
        raise ValueError(f"Unexpected city type counts: {type_counts}")
    if any(not row["admin_geometry_key"] for row in registry):
        raise ValueError("Missing administrative geometry key")
    names = {row["city_name"] for row in registry}
    if not LEGACY_291_MISSING_NAMES <= names:
        raise ValueError("One or more legacy-missing cities are absent")


def write_csv(path: Path, rows: Sequence[dict[str, str]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_sha256(path: Path, hash_path: Path) -> str:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    hash_path.write_text(f"{digest}  {path.name}\n", encoding="ascii")
    return digest


def compare_legacy(registry: Sequence[dict[str, str]], legacy_path: Path) -> list[dict[str, str]]:
    with legacy_path.open("r", encoding="utf-8-sig", newline="") as handle:
        legacy_rows = list(csv.DictReader(handle))
    new_keys = {(row["prov_name"], row["city_name"]) for row in registry}
    old_keys = {(row["prov_name"], row["pref_name"]) for row in legacy_rows}
    report: list[dict[str, str]] = []
    for province, city in sorted(new_keys | old_keys):
        in_new = (province, city) in new_keys
        in_old = (province, city) in old_keys
        report.append(
            {
                "prov_name": province,
                "city_name": city,
                "in_new_official_registry": str(int(in_new)),
                "in_legacy_297_list": str(int(in_old)),
                "comparison_status": "match" if in_new and in_old else (
                    "new_only" if in_new else "legacy_only"
                ),
            }
        )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--legacy", type=Path)
    parser.add_argument("--source-dir", type=Path)
    args = parser.parse_args()

    parsed_groups = [
        parse_admin_records(fetch_html(url), url) for url in MIRROR_URLS
    ]
    source_records = merge_source_records(parsed_groups)
    registry = build_registry(source_records)

    output_dir = args.output_dir
    registry_path = output_dir / "city_registry_2021.csv"
    registry_fields = list(registry[0].keys())
    write_csv(registry_path, registry, registry_fields)
    digest = write_sha256(registry_path, output_dir / "city_registry_2021.sha256")

    if args.source_dir:
        source_rows = [
            {
                "admin_code": row.code,
                "admin_name": row.name,
                "retrieval_url": row.retrieval_url,
                "canonical_source_url": CANONICAL_OFFICIAL_URL,
                "reference_date": REFERENCE_DATE,
            }
            for row in source_records
        ]
        source_path = args.source_dir / "mca_2021_county_and_above_codes.csv"
        write_csv(source_path, source_rows, list(source_rows[0].keys()))
        write_sha256(source_path, args.source_dir / "mca_2021_county_and_above_codes.sha256")

    if args.legacy:
        comparison = compare_legacy(registry, args.legacy)
        write_csv(
            output_dir / "legacy_297_comparison.csv",
            comparison,
            list(comparison[0].keys()),
        )

    print(f"registry_rows={len(registry)}")
    print("prefecture_level_cities=293")
    print("municipalities=4")
    print(f"sha256={digest}")


if __name__ == "__main__":
    main()
