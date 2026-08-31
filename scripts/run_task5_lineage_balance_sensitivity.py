"""Run the prespecified Task 5 lineage-balance sensitivity on the fixed 285 cities."""

import csv
import hashlib
import itertools
import json
import math
import platform
import random
import statistics
from pathlib import Path

from build_task5_ranking_instability import average_ranks, kendall_tau_b, pearson


ROOT = Path(__file__).resolve().parents[1]
MATRIX = ROOT / "data" / "derived" / "R1_city_scenario_landcover_estimator_matrix.csv"
COMMON_AUDIT = ROOT / "data" / "derived" / "R1_task5_ranking_frame_audit.json"
REVISION_PROTOCOL = ROOT / "config" / "manuscript_revision_r2.yaml"
OUT = ROOT / "data" / "derived"

LANDCOVER_LEVELS = (
    ("esa_worldcover_2021", "categorical_native_pixel_area"),
    ("dynamic_world_2021", "annual_mean_probability_area"),
)
DESIGNS = {
    "core_gub_ghs_3x2": ("gctb_core_2021", "gub_2018", "ghs_uc_2020"),
    "system_gub_ghs_3x2": ("gctb_system_2021", "gub_2018", "ghs_uc_2020"),
}
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 20260811
COMPONENTS = (
    "boundary",
    "landcover_source_estimator",
    "boundary_by_source_estimator_interaction",
)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def percentile(values, probability):
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    low, high = math.floor(position), math.ceil(position)
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def specification_id(boundary, landcover):
    dataset, estimator = landcover
    return f"{boundary}|{dataset}|{estimator}"


def decompose_balanced_grid(grid, boundaries):
    boundary_count = len(boundaries)
    landcover_count = len(LANDCOVER_LEVELS)
    expected = boundary_count * landcover_count
    if len(grid) != expected:
        raise ValueError(f"Balanced grid has {len(grid)} cells, expected {expected}")
    grand = sum(grid.values()) / expected
    boundary_means = {
        boundary: sum(grid[(boundary, landcover)] for landcover in LANDCOVER_LEVELS)
        / landcover_count
        for boundary in boundaries
    }
    landcover_means = {
        landcover: sum(grid[(boundary, landcover)] for boundary in boundaries)
        / boundary_count
        for landcover in LANDCOVER_LEVELS
    }
    ss_boundary = landcover_count * sum(
        (boundary_means[boundary] - grand) ** 2 for boundary in boundaries
    )
    ss_landcover = boundary_count * sum(
        (landcover_means[landcover] - grand) ** 2
        for landcover in LANDCOVER_LEVELS
    )
    ss_interaction = sum(
        (
            grid[(boundary, landcover)]
            - boundary_means[boundary]
            - landcover_means[landcover]
            + grand
        )
        ** 2
        for boundary in boundaries
        for landcover in LANDCOVER_LEVELS
    )
    ss_total = sum((value - grand) ** 2 for value in grid.values())
    return {
        "boundary": ss_boundary,
        "landcover_source_estimator": ss_landcover,
        "boundary_by_source_estimator_interaction": ss_interaction,
        "total": ss_total,
        "closure_error": ss_total - ss_boundary - ss_landcover - ss_interaction,
    }


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"Refusing to write empty output: {path}")
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_fixed_common_cities(source_rows):
    audit = json.loads(COMMON_AUDIT.read_text(encoding="utf-8"))
    registered = {row["city_id"] for row in source_rows}
    excluded = set(audit["cities_not_in_common_intersection"])
    common = sorted(registered - excluded)
    if len(registered) != 296:
        raise ValueError(f"Expected 296 registered cities, found {len(registered)}")
    if len(common) != 285:
        raise ValueError(f"Expected fixed Task 5 common set of 285, found {len(common)}")
    return common


def main():
    with MATRIX.open("r", encoding="utf-8-sig", newline="") as handle:
        source = list(csv.DictReader(handle))
    common = load_fixed_common_cities(source)
    common_set = set(common)
    relevant_boundaries = {boundary for boundaries in DESIGNS.values() for boundary in boundaries}
    row_index = {}
    for row in source:
        landcover = (row["landcover_dataset_id"], row["estimator_id"])
        if (
            row["city_id"] in common_set
            and row["bnd_scn"] in relevant_boundaries
            and landcover in LANDCOVER_LEVELS
        ):
            key = (row["city_id"], row["bnd_scn"], landcover)
            if key in row_index:
                raise ValueError(f"Duplicate confirmatory cell: {key}")
            if row["measurement_available"] != "1" or row["combination_formal_eligible"] != "1":
                raise ValueError(f"Fixed common city has unavailable cell: {key}")
            if row["confirmatory_main_row"] != "1":
                raise ValueError(f"Fixed common city has non-confirmatory cell: {key}")
            row_index[key] = row

    expected_index_rows = 285 * len(relevant_boundaries) * len(LANDCOVER_LEVELS)
    if len(row_index) != expected_index_rows:
        raise ValueError(f"Indexed {len(row_index)} cells, expected {expected_index_rows}")

    summary_rows = []
    decomposition_rows = []
    maximum_closure_error = 0.0
    rng = random.Random(BOOTSTRAP_SEED)

    for design_id, boundaries in DESIGNS.items():
        specifications = list(itertools.product(boundaries, LANDCOVER_LEVELS))
        raw_values = {}
        for city in common:
            for boundary, landcover in specifications:
                row = row_index[(city, boundary, landcover)]
                raw_values[(city, boundary, landcover)] = (
                    float(row["Blue_share_full_boundary"])
                    + float(row["Green_share_full_boundary"])
                )

        ranks_by_specification = {}
        normalized_percentiles = {}
        for boundary, landcover in specifications:
            values = {
                city: raw_values[(city, boundary, landcover)] for city in common
            }
            ranks = average_ranks(values, descending=True)
            ranks_by_specification[(boundary, landcover)] = ranks
            for city, rank in ranks.items():
                normalized_percentiles[(city, boundary, landcover)] = (285 - rank) / 284

        pairwise = []
        top_decile_cutoff = math.ceil(0.10 * len(common))
        top_quintile_cutoff = math.ceil(0.20 * len(common))
        for specification_a, specification_b in itertools.combinations(specifications, 2):
            ranks_a = ranks_by_specification[specification_a]
            ranks_b = ranks_by_specification[specification_b]
            values_a = [ranks_a[city] for city in common]
            values_b = [ranks_b[city] for city in common]
            top10_a = {city for city in common if ranks_a[city] <= top_decile_cutoff}
            top10_b = {city for city in common if ranks_b[city] <= top_decile_cutoff}
            top20_a = {city for city in common if ranks_a[city] <= top_quintile_cutoff}
            top20_b = {city for city in common if ranks_b[city] <= top_quintile_cutoff}
            pairwise.append(
                {
                    "spearman": pearson(values_a, values_b),
                    "kendall": kendall_tau_b(values_a, values_b),
                    "top10_jaccard": len(top10_a & top10_b) / len(top10_a | top10_b),
                    "top20_jaccard": len(top20_a & top20_b) / len(top20_a | top20_b),
                }
            )

        city_ranges = [
            max(
                normalized_percentiles[(city, boundary, landcover)]
                for boundary, landcover in specifications
            )
            - min(
                normalized_percentiles[(city, boundary, landcover)]
                for boundary, landcover in specifications
            )
            for city in common
        ]
        summary_rows.append(
            {
                "design_id": design_id,
                "city_count": len(common),
                "specification_count": len(specifications),
                "pairwise_comparison_count": len(pairwise),
                "median_pairwise_spearman_rho": statistics.median(
                    row["spearman"] for row in pairwise
                ),
                "minimum_pairwise_spearman_rho": min(
                    row["spearman"] for row in pairwise
                ),
                "median_pairwise_kendall_tau_b": statistics.median(
                    row["kendall"] for row in pairwise
                ),
                "median_top_decile_jaccard": statistics.median(
                    row["top10_jaccard"] for row in pairwise
                ),
                "median_top_quintile_jaccard": statistics.median(
                    row["top20_jaccard"] for row in pairwise
                ),
                "median_city_percentile_range": statistics.median(city_ranges),
                "maximum_city_percentile_range": max(city_ranges),
            }
        )

        for scale, values in (
            ("raw_full_boundary_share", raw_values),
            ("common285_normalized_rank_percentile", normalized_percentiles),
        ):
            city_components = []
            for city in common:
                grid = {
                    (boundary, landcover): values[(city, boundary, landcover)]
                    for boundary, landcover in specifications
                }
                parts = decompose_balanced_grid(grid, boundaries)
                maximum_closure_error = max(
                    maximum_closure_error, abs(parts["closure_error"])
                )
                city_components.append(parts)
            total_ss = sum(parts["total"] for parts in city_components)
            if total_ss <= 0:
                raise ValueError(f"Non-positive total SS for {design_id}, {scale}")
            point_estimates = {
                component: sum(parts[component] for parts in city_components) / total_ss
                for component in COMPONENTS
            }
            bootstrap = {component: [] for component in COMPONENTS}
            for _ in range(BOOTSTRAP_REPLICATES):
                sampled = [
                    city_components[rng.randrange(len(city_components))]
                    for _ in city_components
                ]
                denominator = sum(parts["total"] for parts in sampled)
                if denominator <= 0:
                    raise ValueError("Bootstrap sample has non-positive total SS")
                for component in COMPONENTS:
                    bootstrap[component].append(
                        sum(parts[component] for parts in sampled) / denominator
                    )
            for component in COMPONENTS:
                decomposition_rows.append(
                    {
                        "design_id": design_id,
                        "scale": scale,
                        "component": component,
                        "variance_share": point_estimates[component],
                        "cluster_bootstrap_ci95_low": percentile(
                            bootstrap[component], 0.025
                        ),
                        "cluster_bootstrap_ci95_high": percentile(
                            bootstrap[component], 0.975
                        ),
                        "city_count": len(common),
                        "boundary_levels": len(boundaries),
                        "landcover_levels": len(LANDCOVER_LEVELS),
                        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                        "bootstrap_seed": BOOTSTRAP_SEED,
                    }
                )

    summary_path = OUT / "R2_task5_lineage_balance_summary.csv"
    decomposition_path = OUT / "R2_task5_lineage_balance_decomposition.csv"
    write_csv(summary_path, summary_rows)
    write_csv(decomposition_path, decomposition_rows)

    manifest_designs = {}
    for design_id, boundaries in DESIGNS.items():
        manifest_designs[design_id] = {
            "boundaries": list(boundaries),
            "landcover_levels": [
                {"dataset_id": dataset, "estimator_id": estimator}
                for dataset, estimator in LANDCOVER_LEVELS
            ],
            "specifications": [
                specification_id(boundary, landcover)
                for boundary, landcover in itertools.product(boundaries, LANDCOVER_LEVELS)
            ],
        }
    manifest = {
        "status": "pass",
        "protocol_id": "task5_lineage_balance_sensitivity_r2",
        "generated_on": "2026-08-20",
        "inferential_role": "prespecified_sensitivity_to_gctb_lineage_representation",
        "task6_refit": False,
        "common_city_definition": "frozen_task5_common_intersection_across_all_eight_confirmatory_settings",
        "common_city_count": len(common),
        "common_city_ids": common,
        "designs": manifest_designs,
        "ranking": {
            "direction": "descending_blue_plus_green_share",
            "ties": "average_rank_exact_stored_numeric_equality",
            "normalized_percentile": "(285-average_rank)/284",
            "top_decile_cutoff": top_decile_cutoff,
            "top_quintile_cutoff": top_quintile_cutoff,
        },
        "decomposition": {
            "method": "balanced_within_city_two_way_orthogonal_sums_of_squares",
            "scales": [
                "raw_full_boundary_share",
                "common285_normalized_rank_percentile",
            ],
            "components": list(COMPONENTS),
            "maximum_absolute_closure_error": maximum_closure_error,
        },
        "bootstrap": {
            "unit": "city",
            "replicates": BOOTSTRAP_REPLICATES,
            "seed": BOOTSTRAP_SEED,
            "ci": "percentile_95",
        },
        "software": {"python": platform.python_version()},
        "inputs": {
            MATRIX.name: {"sha256": sha256(MATRIX)},
            COMMON_AUDIT.name: {"sha256": sha256(COMMON_AUDIT)},
            "config/manuscript_revision_r2.yaml": {"sha256": sha256(REVISION_PROTOCOL)},
        },
        "outputs": {
            summary_path.name: {"sha256": sha256(summary_path), "rows": len(summary_rows)},
            decomposition_path.name: {
                "sha256": sha256(decomposition_path),
                "rows": len(decomposition_rows),
            },
        },
        "interpretation_caveat": (
            "Finite-design sensitivity to GCTB lineage representation; this is not an accuracy, "
            "truth, causal-effect, or replacement-primary-analysis test."
        ),
    }
    manifest_path = OUT / "R2_task5_lineage_balance_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
