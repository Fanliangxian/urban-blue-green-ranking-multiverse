import csv
import hashlib
import itertools
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path

from build_task5_ranking_instability import average_ranks, kendall_tau_b, pearson


ROOT = Path(__file__).resolve().parents[1]
MATRIX = ROOT / "data" / "derived" / "R1_city_scenario_landcover_estimator_matrix.csv"
OUT = ROOT / "data" / "derived"
ALL_BOUNDARIES = ("gctb_core_2021", "gctb_system_2021", "gub_2018", "ghs_uc_2020")
MAIN_ESTIMATORS = (
    ("esa_worldcover_2021", "categorical_native_pixel_area"),
    ("dynamic_world_2021", "annual_mean_probability_area"),
)
ALL_ESTIMATORS = MAIN_ESTIMATORS + (("dynamic_world_2021", "annual_modal_label_area"),)
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 20260811

SCENARIOS = (
    {"id": "baseline_full_boundary", "boundaries": ALL_BOUNDARIES, "estimators": MAIN_ESTIMATORS, "denominator": "full", "filter": "all"},
    {"id": "valid_classified_area_denominator", "boundaries": ALL_BOUNDARIES, "estimators": MAIN_ESTIMATORS, "denominator": "valid", "filter": "all"},
    {"id": "all_estimators_12_specifications", "boundaries": ALL_BOUNDARIES, "estimators": ALL_ESTIMATORS, "denominator": "full", "filter": "all"},
    {"id": "legacy_291_members", "boundaries": ALL_BOUNDARIES, "estimators": MAIN_ESTIMATORS, "denominator": "full", "filter": "legacy"},
    {"id": "exclude_gub_2018", "boundaries": tuple(b for b in ALL_BOUNDARIES if b != "gub_2018"), "estimators": MAIN_ESTIMATORS, "denominator": "full", "filter": "all"},
    {"id": "exclude_dynamic_world_low_support_cities", "boundaries": ALL_BOUNDARIES, "estimators": MAIN_ESTIMATORS, "denominator": "full", "filter": "exclude_low_support"},
)


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def percentile(values, p):
    x = sorted(values)
    pos = (len(x) - 1) * p
    lo, hi = math.floor(pos), math.ceil(pos)
    return x[lo] if lo == hi else x[lo] + (x[hi] - x[lo]) * (pos - lo)


def generic_decompose(grid, boundaries, estimators):
    n_b, n_e = len(boundaries), len(estimators)
    grand = sum(grid.values()) / (n_b * n_e)
    means_b = {b: sum(grid[(b, e)] for e in estimators) / n_e for b in boundaries}
    means_e = {e: sum(grid[(b, e)] for b in boundaries) / n_b for e in estimators}
    ss_b = n_e * sum((means_b[b] - grand) ** 2 for b in boundaries)
    ss_e = n_b * sum((means_e[e] - grand) ** 2 for e in estimators)
    ss_i = sum((grid[(b, e)] - means_b[b] - means_e[e] + grand) ** 2 for b in boundaries for e in estimators)
    total = sum((v - grand) ** 2 for v in grid.values())
    return ss_b, ss_e, ss_i, total


def write_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    with MATRIX.open("r", encoding="utf-8-sig", newline="") as handle:
        source = list(csv.DictReader(handle))
    row_index = {(r["city_id"], r["bnd_scn"], (r["landcover_dataset_id"], r["estimator_id"])): r for r in source}
    registered = sorted({r["city_id"] for r in source})
    low_support = {r["city_id"] for r in source if r["dynamic_world_low_support_majority_flag"] == "1"}
    legacy = {r["city_id"] for r in source if r["legacy_291_member"] == "1"}
    rng = random.Random(BOOTSTRAP_SEED)
    aggregate_rows, summary_rows = [], []

    for scenario in SCENARIOS:
        specs = list(itertools.product(scenario["boundaries"], scenario["estimators"]))
        candidates = set(registered)
        if scenario["filter"] == "legacy":
            candidates &= legacy
        elif scenario["filter"] == "exclude_low_support":
            candidates -= low_support
        common = []
        for city in sorted(candidates):
            rows = [row_index[(city, b, e)] for b, e in specs]
            if all(r["measurement_available"] == "1" and r["combination_formal_eligible"] == "1" for r in rows):
                common.append(city)

        raw = {}
        for city in common:
            for b, e in specs:
                row = row_index[(city, b, e)]
                suffix = "full_boundary" if scenario["denominator"] == "full" else "valid_area"
                raw[(city, b, e)] = float(row[f"Blue_share_{suffix}"]) + float(row[f"Green_share_{suffix}"])
        ranked = {}
        per_spec_ranks = {}
        for b, e in specs:
            values = {city: raw[(city, b, e)] for city in common}
            ranks = average_ranks(values, True)
            per_spec_ranks[(b, e)] = ranks
            for city, rank in ranks.items():
                ranked[(city, b, e)] = (len(common) - rank) / (len(common) - 1)

        pairs = []
        for spec_a, spec_b in itertools.combinations(specs, 2):
            xa = [per_spec_ranks[spec_a][c] for c in common]
            xb = [per_spec_ranks[spec_b][c] for c in common]
            k10 = math.ceil(0.1 * len(common))
            top_a = {c for c in common if per_spec_ranks[spec_a][c] <= k10}
            top_b = {c for c in common if per_spec_ranks[spec_b][c] <= k10}
            pairs.append((pearson(xa, xb), kendall_tau_b(xa, xb), len(top_a & top_b) / len(top_a | top_b)))
        city_ranges = [max(ranked[(c, b, e)] for b, e in specs) - min(ranked[(c, b, e)] for b, e in specs) for c in common]
        summary_rows.append({
            "scenario_id": scenario["id"], "city_count": len(common), "specification_count": len(specs),
            "median_pairwise_spearman_rho": statistics.median(x[0] for x in pairs),
            "minimum_pairwise_spearman_rho": min(x[0] for x in pairs),
            "median_pairwise_kendall_tau_b": statistics.median(x[1] for x in pairs),
            "median_top_decile_jaccard": statistics.median(x[2] for x in pairs),
            "median_city_percentile_range": statistics.median(city_ranges),
            "maximum_city_percentile_range": max(city_ranges),
        })

        for scale, data in (("raw_share", raw), ("normalized_rank_percentile", ranked)):
            city_parts = []
            for city in common:
                grid = {(b, e): data[(city, b, e)] for b, e in specs}
                city_parts.append(generic_decompose(grid, scenario["boundaries"], scenario["estimators"]))
            point_den = sum(x[3] for x in city_parts)
            point = [sum(x[i] for x in city_parts) / point_den for i in range(3)]
            boot = [[], [], []]
            for _ in range(BOOTSTRAP_REPLICATES):
                sample = [city_parts[rng.randrange(len(city_parts))] for _ in city_parts]
                den = sum(x[3] for x in sample)
                for i in range(3):
                    boot[i].append(sum(x[i] for x in sample) / den)
            for i, component in enumerate(("boundary", "landcover_estimator", "boundary_by_estimator_interaction")):
                aggregate_rows.append({
                    "scenario_id": scenario["id"], "scale": scale, "component": component,
                    "variance_share": point[i], "cluster_bootstrap_ci95_low": percentile(boot[i], 0.025),
                    "cluster_bootstrap_ci95_high": percentile(boot[i], 0.975), "city_count": len(common),
                    "boundary_levels": len(scenario["boundaries"]), "estimator_levels": len(scenario["estimators"]),
                    "bootstrap_replicates": BOOTSTRAP_REPLICATES, "bootstrap_seed": BOOTSTRAP_SEED,
                })

    summary_path = OUT / "R1_task5_stage3_sensitivity_summary.csv"
    aggregate_path = OUT / "R1_task5_stage3_sensitivity_decomposition.csv"
    write_csv(summary_path, summary_rows)
    write_csv(aggregate_path, aggregate_rows)
    manifest = {
        "status": "pass", "protocol_id": "ranking_instability_r1_stage3", "generated_on": "2026-08-11",
        "scenario_count": len(SCENARIOS), "scenarios": [r["scenario_id"] for r in summary_rows],
        "bootstrap": {"unit": "city", "replicates": BOOTSTRAP_REPLICATES, "seed": BOOTSTRAP_SEED},
        "source_matrix_sha256": sha256(MATRIX),
        "outputs": {
            summary_path.name: {"sha256": sha256(summary_path), "rows": len(summary_rows)},
            aggregate_path.name: {"sha256": sha256(aggregate_path), "rows": len(aggregate_rows)},
        },
        "caveat": "Factor shares are finite-design descriptions and are not directly comparable when factor level counts differ.",
    }
    manifest_path = OUT / "R1_task5_stage3_sensitivity_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

