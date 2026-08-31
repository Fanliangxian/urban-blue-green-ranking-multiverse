import csv
import hashlib
import itertools
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MATRIX = ROOT / "data" / "derived" / "R1_city_scenario_landcover_estimator_matrix.csv"
OUT = ROOT / "data" / "derived"
BOUNDARIES = ("gctb_core_2021", "gctb_system_2021", "gub_2018", "ghs_uc_2020")
CONFIRMATORY = {
    ("esa_worldcover_2021", "categorical_native_pixel_area"),
    ("dynamic_world_2021", "annual_mean_probability_area"),
}
SENSITIVITY = CONFIRMATORY | {("dynamic_world_2021", "annual_modal_label_area")}
OUTCOMES = {
    "blue_green_share": ("descending", None),
    "green_share": ("descending", "Green_share_full_boundary"),
    "blue_share": ("descending", "Blue_share_full_boundary"),
    "grey_share": ("ascending", "Grey_share_full_boundary"),
}


def f(value):
    return float(value) if value not in (None, "") else None


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def average_ranks(values, descending):
    ordered = sorted(values.items(), key=lambda item: (-item[1], item[0]) if descending else (item[1], item[0]))
    ranks = {}
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][1] == ordered[index][1]:
            end += 1
        rank = ((index + 1) + end) / 2
        for city_id, _ in ordered[index:end]:
            ranks[city_id] = rank
        index = end
    return ranks


def pearson(xs, ys):
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    numerator = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    denominator = math.sqrt(sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys))
    return numerator / denominator if denominator else None


def kendall_tau_b(xs, ys):
    concordant = discordant = tie_x = tie_y = 0
    for i in range(len(xs) - 1):
        for j in range(i + 1, len(xs)):
            dx, dy = xs[i] - xs[j], ys[i] - ys[j]
            if dx == 0 and dy == 0:
                continue
            if dx == 0:
                tie_x += 1
            elif dy == 0:
                tie_y += 1
            elif dx * dy > 0:
                concordant += 1
            else:
                discordant += 1
    denominator = math.sqrt((concordant + discordant + tie_x) * (concordant + discordant + tie_y))
    return (concordant - discordant) / denominator if denominator else None


def quantile(values, p):
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * p
    low, high = math.floor(position), math.ceil(position)
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def quartile(percentile):
    if percentile >= 0.75:
        return 1
    if percentile >= 0.50:
        return 2
    if percentile >= 0.25:
        return 3
    return 4


def entropy(categories):
    counts = Counter(categories)
    n = len(categories)
    return -sum((count / n) * math.log(count / n, 2) for count in counts.values())


def write_csv(path, rows, fields):
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main():
    with MATRIX.open("r", encoding="utf-8-sig", newline="") as handle:
        source = list(csv.DictReader(handle))

    selected = []
    for row in source:
        estimator = (row["landcover_dataset_id"], row["estimator_id"])
        if row["bnd_scn"] not in BOUNDARIES or estimator not in SENSITIVITY:
            continue
        if row["measurement_available"] != "1" or row["combination_formal_eligible"] != "1":
            continue
        row = dict(row)
        row["specification_id"] = f"{row['bnd_scn']}|{row['landcover_dataset_id']}|{row['estimator_id']}"
        row["analysis_layer"] = "confirmatory" if estimator in CONFIRMATORY else "estimator_sensitivity"
        row["blue_green_share"] = f(row["Blue_share_full_boundary"]) + f(row["Green_share_full_boundary"])
        row["green_share"] = f(row["Green_share_full_boundary"])
        row["blue_share"] = f(row["Blue_share_full_boundary"])
        row["grey_share"] = f(row["Grey_share_full_boundary"])
        selected.append(row)

    by_spec = defaultdict(list)
    for row in selected:
        by_spec[row["specification_id"]].append(row)

    ranking_rows = []
    for spec, rows in sorted(by_spec.items()):
        for outcome, (direction, _) in OUTCOMES.items():
            values = {row["city_id"]: row[outcome] for row in rows}
            ranks = average_ranks(values, direction == "descending")
            n = len(values)
            top10_cut = math.ceil(0.10 * n)
            top20_cut = math.ceil(0.20 * n)
            bottom20_start = n - math.ceil(0.20 * n) + 1
            metadata = {row["city_id"]: row for row in rows}
            for city_id, value in values.items():
                rank = ranks[city_id]
                percentile = (n - rank) / (n - 1)
                row = metadata[city_id]
                ranking_rows.append({
                    "analysis_layer": row["analysis_layer"], "outcome_id": outcome,
                    "specification_id": spec, "bnd_scn": row["bnd_scn"],
                    "landcover_dataset_id": row["landcover_dataset_id"], "estimator_id": row["estimator_id"],
                    "city_id": city_id, "city_name": row["city_name"], "prov_name": row["prov_name"],
                    "legacy_291_member": row["legacy_291_member"], "n_ranked": n, "outcome_value": value,
                    "average_rank": rank, "normalized_percentile": percentile,
                    "top_decile": int(rank <= top10_cut), "top_quintile": int(rank <= top20_cut),
                    "bottom_quintile": int(rank >= bottom20_start), "quartile": quartile(percentile),
                    "dynamic_world_low_support_majority_flag": row["dynamic_world_low_support_majority_flag"],
                    "dynamic_world_zero_observation_flag": row["dynamic_world_zero_observation_flag"],
                })

    ranking_path = OUT / "R1_task5_rankings_long.csv"
    ranking_fields = list(ranking_rows[0])
    write_csv(ranking_path, ranking_rows, ranking_fields)

    indexed = defaultdict(dict)
    for row in ranking_rows:
        indexed[(row["outcome_id"], row["specification_id"])][row["city_id"]] = row

    agreement_rows = []
    spec_sets = {
        "confirmatory_8": sorted({r["specification_id"] for r in ranking_rows if r["analysis_layer"] == "confirmatory"}),
        "all_estimators_sensitivity_12": sorted({r["specification_id"] for r in ranking_rows}),
    }
    for analysis_set, specs in spec_sets.items():
        for outcome in OUTCOMES:
            for spec_a, spec_b in itertools.combinations(specs, 2):
                a, b = indexed[(outcome, spec_a)], indexed[(outcome, spec_b)]
                common = sorted(set(a) & set(b))
                xa = [a[c]["average_rank"] for c in common]
                xb = [b[c]["average_rank"] for c in common]
                top10_a, top10_b = {c for c in common if a[c]["top_decile"]}, {c for c in common if b[c]["top_decile"]}
                top20_a, top20_b = {c for c in common if a[c]["top_quintile"]}, {c for c in common if b[c]["top_quintile"]}
                agreement_rows.append({
                    "analysis_set": analysis_set, "outcome_id": outcome, "specification_a": spec_a,
                    "specification_b": spec_b, "common_city_n": len(common),
                    "spearman_rho": pearson(xa, xb), "kendall_tau_b": kendall_tau_b(xa, xb),
                    "top_decile_jaccard": len(top10_a & top10_b) / len(top10_a | top10_b) if top10_a | top10_b else None,
                    "top_quintile_jaccard": len(top20_a & top20_b) / len(top20_a | top20_b) if top20_a | top20_b else None,
                })
    agreement_path = OUT / "R1_task5_pairwise_rank_agreement.csv"
    write_csv(agreement_path, agreement_rows, list(agreement_rows[0]))

    grouped = defaultdict(list)
    for analysis_set, specs in spec_sets.items():
        allowed = set(specs)
        for row in ranking_rows:
            if row["specification_id"] in allowed:
                grouped[(analysis_set, row["outcome_id"], row["city_id"])].append(row)
    city_rows = []
    for (analysis_set, outcome, city_id), rows in sorted(grouped.items()):
        ps = [r["normalized_percentile"] for r in rows]
        qs = [r["quartile"] for r in rows]
        first = rows[0]
        city_rows.append({
            "analysis_set": analysis_set, "outcome_id": outcome, "city_id": city_id,
            "city_name": first["city_name"], "prov_name": first["prov_name"],
            "legacy_291_member": first["legacy_291_member"], "specifications_available": len(rows),
            "median_normalized_percentile": statistics.median(ps), "normalized_percentile_range": max(ps) - min(ps),
            "normalized_percentile_iqr": quantile(ps, 0.75) - quantile(ps, 0.25),
            "normalized_percentile_sd": statistics.stdev(ps) if len(ps) > 1 else None,
            "top_decile_specification_share": statistics.fmean(r["top_decile"] for r in rows),
            "top_quintile_specification_share": statistics.fmean(r["top_quintile"] for r in rows),
            "bottom_quintile_specification_share": statistics.fmean(r["bottom_quintile"] for r in rows),
            "quartile_membership_count": len(set(qs)), "quartile_membership_entropy_bits": entropy(qs),
        })
    city_path = OUT / "R1_task5_city_rank_instability.csv"
    write_csv(city_path, city_rows, list(city_rows[0]))

    output_paths = (ranking_path, agreement_path, city_path)
    manifest = {
        "status": "pass", "protocol_id": "ranking_instability_r1", "generated_on": "2026-08-11",
        "source_matrix_sha256": sha256(MATRIX),
        "metrics": {
            "formal_measured_rows_selected": len(selected), "ranking_rows": len(ranking_rows),
            "confirmatory_specifications": len({r["specification_id"] for r in selected if r["analysis_layer"] == "confirmatory"}),
            "sensitivity_additional_specifications": len({r["specification_id"] for r in selected if r["analysis_layer"] == "estimator_sensitivity"}),
            "all_estimators_sensitivity_specifications": len(spec_sets["all_estimators_sensitivity_12"]),
            "pairwise_agreement_rows": len(agreement_rows), "city_instability_rows": len(city_rows),
        },
        "outputs": {path.name: {"sha256": sha256(path), "bytes": path.stat().st_size} for path in output_paths},
        "notes": [
            "Spearman correlation is Pearson correlation of average ranks on pairwise common cities.",
            "Kendall tau-b explicitly adjusts for ties.",
            "Top-set thresholds preserve all exact outcome ties and may therefore contain more than the nominal ceiling count.",
        ],
    }
    manifest_path = OUT / "R1_task5_ranking_instability_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
