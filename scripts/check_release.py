from __future__ import annotations

import ast
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {".py", ".js", ".md", ".json", ".yaml", ".yml", ".cff", ".txt", ".csv"}
REQUIRED = [
    "README.md", "DATA.md", "LICENSE", "CITATION.cff", "environment.yml",
    "requirements.txt", "docs/REPRODUCIBILITY.md", "docs/SCRIPT_MAP.md",
    "docs/GITHUB_UPLOAD_TUTORIAL_CN.md", "scripts/reproduce_analysis.py",
    "scripts/build_task5_ranking_instability.py", "scripts/run_task6_confirmatory_model.py",
    "scripts/plot_r2_figure5_city_characteristics.py", "gee/R1_worldcover2021_national_measurement.js",
]
FORBIDDEN = [
    re.compile(r"[A-Za-z]:[\\/](?:Users|Research|Datasets)[\\/]", re.I),
    re.compile(r"projects/ee-[^/\\s]+/assets", re.I),
    re.compile(r"service[_-]?account", re.I),
    re.compile(r"private[_-]?key", re.I),
    re.compile(r"(?:api[_-]?key|access[_-]?token)\\s*[:=]", re.I),
]


def audit() -> list[str]:
    errors = []
    for rel in REQUIRED:
        if not (ROOT / rel).exists():
            errors.append(f"missing required file: {rel}")
    for path in ROOT.rglob("*"):
        if not path.is_file() or any(part in {".git", "__pycache__"} for part in path.parts):
            continue
        if path.stat().st_size > 95 * 1024 * 1024:
            errors.append(f"oversized file: {path.relative_to(ROOT)}")
        if path.suffix.lower() in TEXT_SUFFIXES or path.name == "LICENSE":
            text = path.read_text(encoding="utf-8", errors="replace")
            for pattern in FORBIDDEN:
                if pattern.search(text):
                    errors.append(f"forbidden local/private pattern in {path.relative_to(ROOT)}")
                    break
        if path.suffix == ".py":
            try:
                ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
            except SyntaxError as exc:
                errors.append(f"Python syntax error in {path.relative_to(ROOT)}: {exc}")
    return errors


def main() -> int:
    errors = audit()
    if errors:
        print("RELEASE CHECK FAILED")
        for error in errors:
            print(f"- {error}")
        return 1
    print("RELEASE CHECK PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
