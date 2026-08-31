import csv
import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MATRIX = ROOT / "data" / "derived" / "R1_city_scenario_landcover_estimator_matrix.csv"
AUDIT = ROOT / "data" / "derived" / "R1_task5_ranking_frame_audit.json"
OUT = ROOT / "data" / "derived"
BOUNDARIES = ("gctb_core_2021", "gctb_system_2021", "gub_2018", "ghs_uc_2020")
PRODUCTS = (
    ("esa_worldcover_2021", "categorical_native_pixel_area"),
    ("dynamic_world_2021", "annual_mean_probability_area"),
)
OUTCOMES = {
    "blue_green_share": ("descending", None),
    "green_share": ("descending", "Green_share_full_boundary"),
    "blue_share": ("descending", "Blue_share_full_boundary"),
    "grey_share": ("ascending", "Grey_share_full_boundary"),
}
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 20260811


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def average_ranks(values, descending):
    ordered = sorted(values.items(), key=lambda item: (-item[1], item[0]) if descending else (item[1], item[0]))
    ranks = {}
    i = 0
    while i < len(ordered):
        j = i + 1
        while j < len(ordered) and ordered[j][1] == ordered[i][1]:
            j += 1
        rank = ((i + 1) + j) / 2
        for city_id, _ in ordered[i:j]:
            ranks[city_id] = rank
        i = j
    return ranks


def percentile(values, p):
    ordered = sorted(values)
    position = (len(ordered) - 1) * p
    low, high = math.floor(position), math.ceil(position)
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def decompose(grid):
    grand = sum(grid.values()) / 8
    boundary_means = {b: sum(grid[(b, p)] for p in PRODUCTS) / 2 for b in BOUNDARIES}
    product_means = {p: sum(grid[(b, p)] for b in BOUNDARIES) / 4 for p in PRODUCTS}
    ss_boundary = 2 * sum((boundary_means[b] - grand) ** 2 for b in BOUNDARIES)
    ss_product = 4 * sum((product_means[p] - grand) ** 2 for p in PRODUCTS)
    ss_interaction = sum(
        (grid[(b, p)] - boundary_means[b] - product_means[p] + grand) ** 2
        for b in BOUNDARIES for p in PRODUCTS
    )
    ss_total = sum((value - grand) ** 2 for value in grid.values())
    return {
        "ss_boundary": ss_boundary,
        "ss_product": ss_product,
        "ss_boundary_product_interaction": ss_interaction,
        "ss_total_within_city": ss_total,
        "closure_error": ss_total - ss_boundary - ss_product - ss_interaction,
    }


def write_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    common = set(audit["cities_not_in_common_intersection"])
    with MATRIX.open("r", encoding="utf-8-sig", newline="") as handle:
        source = list(csv.DictReader(handle))
    all_cities = {row["city_id"] for row in source}
    common = all_cities - common
    if len(common) != 285:
        raise ValueError(f"Expected 285 common cities, found {len(common)}")

    values = defaultdict(dict)
    metadata = {}
    for row in source:
        product = (row["landcover_dataset_id"], row["estimator_id"])
        if row["city_id"] not in common or row["bnd_scn"] not in BOUNDARIES or product not in PRODUCTS:
            continue
        if row["confirmatory_main_row"] != "1":
            raise ValueError("Common-city confirmatory cell is not eligible")
        blue, green = float(row["Blue_share_full_boundary"]), float(row["Green_share_full_boundary"])
        outcome_values = {
            "blue_green_share": blue + green,
            "green_share": green,
            "blue_share": blue,
            "grey_share": float(row["Grey_share_full_boundary"]),
        }
        for outcome, value in outcome_values.items():
            values[outcome][(row["city_id"], row["bnd_scn"], product)] = value
        metadata[row["city_id"]] = (row["city_name"], row["prov_name"], row["legacy_291_member"])

    expected_cells = 285 * 4 * 2
    for outcome in OUTCOMES:
        if len(values[outcome]) != expected_cells:
            raise ValueError(f"{outcome} has {len(values[outcome])} cells, expected {expected_cells}")

    rank_values = defaultdict(dict)
    for outcome, (direction, _) in OUTCOMES.items():
        for boundary in BOUNDARIES:
            for product in PRODUCTS:
                cell = {city: values[outcome][(city, boundary, product)] for city in common}
                ranks = average_ranks(cell, direction == "descending")
                for city, rank in ranks.items():
                    rank_values[outcome][(city, boundary, product)] = (285 - rank) / 284

    city_rows = []
    components_by_group = defaultdict(list)
    for scale, data in (("raw_full_boundary_share", values), ("common285_normalized_rank_percentile", rank_values)):
        for outcome in OUTCOMES:
            for city in sorted(common):
                grid = {(b, p): data[outcome][(city, b, p)] for b in BOUNDARIES for p in PRODUCTS}
                parts = decompose(grid)
                total = parts["ss_total_within_city"]
                proportions = {
                    component.replace("ss_", "proportion_"): (value / total if total > 0 else None)
                    for component, value in parts.items() if component.startswith("ss_") and component != "ss_total_within_city"
                }
                city_name, prov_name, legacy = metadata[city]
                row = {
                    "scale": scale, "outcome_id": outcome, "city_id": city, "city_name": city_name,
                    "prov_name": prov_name, "legacy_291_member": legacy, **parts, **proportions,
                }
                city_rows.append(row)
                components_by_group[(scale, outcome)].append(row)

    city_path = OUT / "R1_task5_city_factorial_sensitivity_decomposition.csv"
    write_csv(city_path, city_rows)

    rng = random.Random(BOOTSTRAP_SEED)
    aggregate_rows = []
    component_fields = ("ss_boundary", "ss_product", "ss_boundary_product_interaction")
    for (scale, outcome), rows in sorted(components_by_group.items()):
        total_ss = sum(row["ss_total_within_city"] for row in rows)
        point = {component: sum(row[component] for row in rows) / total_ss for component in component_fields}
        replicates = {component: [] for component in component_fields}
        for _ in range(BOOTSTRAP_REPLICATES):
            sampled = [rows[rng.randrange(len(rows))] for _ in rows]
            denominator = sum(row["ss_total_within_city"] for row in sampled)
            for component in component_fields:
                replicates[component].append(sum(row[component] for row in sampled) / denominator)
        for component in component_fields:
            aggregate_rows.append({
                "scale": scale, "outcome_id": outcome,
                "component": component.removeprefix("ss_"), "variance_share": point[component],
                "cluster_bootstrap_ci95_low": percentile(replicates[component], 0.025),
                "cluster_bootstrap_ci95_high": percentile(replicates[component], 0.975),
                "city_clusters": len(rows), "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                "bootstrap_seed": BOOTSTRAP_SEED,
            })
    aggregate_path = OUT / "R1_task5_factorial_sensitivity_aggregate.csv"
    write_csv(aggregate_path, aggregate_rows)

    max_closure = max(abs(row["closure_error"]) for row in city_rows)
    manifest = {
        "status": "pass", "protocol_id": "ranking_instability_r1_stage2",
        "generated_on": "2026-08-11", "common_city_count": 285,
        "balanced_cells_per_outcome": expected_cells, "outcomes": list(OUTCOMES),
        "scales": ["raw_full_boundary_share", "common285_normalized_rank_percentile"],
        "bootstrap": {"unit": "city", "replicates": BOOTSTRAP_REPLICATES, "seed": BOOTSTRAP_SEED, "ci": "percentile_95"},
        "maximum_absolute_orthogonal_decomposition_closure_error": max_closure,
        "source_matrix_sha256": sha256(MATRIX),
        "outputs": {
            city_path.name: {"sha256": sha256(city_path), "rows": len(city_rows)},
            aggregate_path.name: {"sha256": sha256(aggregate_path), "rows": len(aggregate_rows)},
        },
        "interpretation": "Shares partition within-city variation across the frozen 4x2 map design; they are not causal effects or accuracy estimates.",
    }
    manifest_path = OUT / "R1_task5_factorial_sensitivity_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

