import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DERIVED = ROOT / "data" / "derived"
REGISTRY = ROOT / "data" / "reference" / "city_registry_analysis_2021.csv"
ASSET_AUDIT = DERIVED / "R1_task6_reference_admin2021_asset_audit.csv"
OUTPUT = DERIVED / "R1_task6_basic_covariates_296.csv"
AUDIT = DERIVED / "R1_task6_basic_covariates_296_audit.json"
PATTERNS = {
    "population": "R1_task6_worldpop2020_population_b*.csv",
    "water": "R1_task6_jrcgsw2021_permanent_water_b*.csv",
    "terrain": "R1_task6_meritdem_terrain_ruggedness_b*.csv",
}


def read_rows(path):
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def finite(value, field, city_id):
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{field} is missing/non-numeric for {city_id}: {value!r}")
    if not math.isfinite(number):
        raise ValueError(f"{field} is non-finite for {city_id}")
    return number


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_product(label, expected_ids):
    files = sorted(DERIVED.glob(PATTERNS[label]))
    if len(files) != 4:
        raise ValueError(f"{label}: expected 4 batch files, found {len(files)}")
    rows = [row for path in files for row in read_rows(path)]
    ids = [row["city_id"] for row in rows]
    counts = Counter(ids)
    if len(rows) != 296 or set(ids) != expected_ids or any(n != 1 for n in counts.values()):
        raise ValueError({
            "product": label, "rows": len(rows), "unique": len(counts),
            "missing": sorted(expected_ids - set(ids)),
            "duplicates": sorted(k for k, n in counts.items() if n != 1),
        })
    return {row["city_id"]: row for row in rows}, files


def main():
    registry_rows = read_rows(REGISTRY)
    registry = {row["city_id"]: row for row in registry_rows}
    expected_ids = set(registry)
    if len(expected_ids) != 296:
        raise ValueError(f"Registry unique city count is {len(expected_ids)}")

    geometry_rows = read_rows(ASSET_AUDIT)
    geometry = {row["city_id"]: row for row in geometry_rows}
    if len(geometry_rows) != 296 or set(geometry) != expected_ids:
        raise ValueError("Reference geometry audit does not cover registered 296 exactly")

    population, population_files = load_product("population", expected_ids)
    water, water_files = load_product("water", expected_ids)
    terrain, terrain_files = load_product("terrain", expected_ids)

    output_rows = []
    blank_permanent_water_as_zero = []
    for city_id in sorted(expected_ids):
        g = geometry[city_id]
        p = population[city_id]
        w = water[city_id]
        t = terrain[city_id]
        area_m2 = finite(g["area_m2"], "area_m2", city_id)
        perimeter_m = finite(g["perimeter_m"], "perimeter_m", city_id)
        shape = finite(g["shape_complexity_log_ratio"], "shape_complexity", city_id)
        pop = finite(p["population_2020"], "population_2020", city_id)
        valid_water = finite(w["water_valid_area_m2"], "water_valid_area_m2", city_id)
        permanent_raw = w.get("permanent_water_area_m2", "")
        if permanent_raw in (None, ""):
            permanent = 0.0
            blank_permanent_water_as_zero.append(city_id)
        else:
            permanent = finite(permanent_raw, "permanent_water_area_m2", city_id)
        ruggedness = finite(
            t["terrain_ruggedness_riley_m"], "terrain_ruggedness_riley_m", city_id
        )
        if not (area_m2 > 0 and perimeter_m > 0 and shape >= -1e-10 and pop >= 0
                and valid_water > 0 and 0 <= permanent <= valid_water
                and permanent <= area_m2 * 1.01 and ruggedness >= 0):
            raise ValueError(f"Physical range failure for {city_id}")
        output_rows.append({
            "city_id": city_id,
            "prov_code": registry[city_id]["prov_code"],
            "prov_name": registry[city_id]["prov_name"],
            "city_name": registry[city_id]["city_name"],
            "reference_area_km2": area_m2 / 1e6,
            "reference_perimeter_km": perimeter_m / 1e3,
            "boundary_shape_complexity": shape,
            "population_2020_worldpop": pop,
            "surface_water_permanent_area_km2": permanent / 1e6,
            "surface_water_classified_support_area_km2": valid_water / 1e6,
            "surface_water_density": permanent / area_m2,
            "surface_water_classified_support_fraction": min(valid_water / area_m2, 1.0),
            "terrain_ruggedness_riley_mean_m": ruggedness,
        })

    fields = list(output_rows[0])
    with OUTPUT.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(output_rows)

    def summary(field):
        values = sorted(float(row[field]) for row in output_rows)
        return {
            "min": values[0], "median": values[len(values) // 2], "max": values[-1]
        }

    all_files = [ASSET_AUDIT] + population_files + water_files + terrain_files
    audit = {
        "status": "pass",
        "row_count": len(output_rows),
        "unique_city_id_count": len({row["city_id"] for row in output_rows}),
        "product_batch_counts": {"population": 4, "water": 4, "terrain": 4},
        "blank_permanent_water_interpreted_as_zero_count": len(blank_permanent_water_as_zero),
        "blank_permanent_water_interpreted_as_zero_city_ids": blank_permanent_water_as_zero,
        "summaries": {
            field: summary(field) for field in (
                "reference_area_km2", "boundary_shape_complexity",
                "population_2020_worldpop", "surface_water_density",
                "surface_water_classified_support_fraction",
                "terrain_ruggedness_riley_mean_m",
            )
        },
        "input_sha256": {path.name: sha256(path) for path in all_files},
        "output_sha256": sha256(OUTPUT),
    }
    AUDIT.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
