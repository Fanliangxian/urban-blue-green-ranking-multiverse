"""Build audited, plot-ready source tables for revision Figures 1--5.

This script deliberately performs no plotting.  All ranking denominators, geometry-only
representative-city selection, and model sample distinctions are materialized here so
that later figure scripts remain presentation-only.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import os
import platform
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau


ROOT = Path(__file__).resolve().parents[1]
DERIVED = ROOT / "data" / "derived"
REFERENCE = ROOT / "data" / "reference"
OUT = DERIVED / "figure_source_data"
PROXY = ROOT / "work" / "r3_gdal_proxy"
PROXY.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("GDAL_PAM_PROXY_DIR", str(PROXY))

from osgeo import gdal, ogr  # noqa: E402


MATRIX = DERIVED / "R1_city_scenario_landcover_estimator_matrix.csv"
REGISTRY = REFERENCE / "city_registry_analysis_2021.csv"
DECOMP = DERIVED / "R1_task5_city_factorial_sensitivity_decomposition.csv"
MODEL = DERIVED / "R1_task6_model_frame_284.csv"
PRIMARY = DERIVED / "R1_task6g_table1_primary_associations.csv"
COMPONENT = DERIVED / "R1_task6g_table2_component_associations.csv"
CV_FOLDS = DERIVED / "R1_task6_spatial_block_cv_folds.csv"
CV_SUMMARY = DERIVED / "R1_task6_spatial_block_cv_summary.csv"
ADMIN = DERIVED / "task4_admin_assignment_296.gpkg"
BOUNDARIES = DERIVED / "measurement_boundaries_r1.gpkg"

BOUNDARY_LEVELS = ("gctb_core_2021", "gctb_system_2021", "gub_2018", "ghs_uc_2020")
PRODUCT_LEVELS = (
    ("esa_worldcover_2021", "categorical_native_pixel_area"),
    ("dynamic_world_2021", "annual_mean_probability_area"),
)
HAIDONG = "CN-630200"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temp, index=False, encoding="utf-8-sig", line_terminator="\n")
    temp.replace(path)
    check = pd.read_csv(path)
    if len(check) != len(frame) or list(check.columns) != list(frame.columns):
        raise RuntimeError(f"Round-trip validation failed: {path}")


def write_json(path: Path, value: dict) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)
    json.loads(path.read_text(encoding="utf-8"))


def point_count(geometry) -> int:
    children = geometry.GetGeometryCount()
    if not children:
        return geometry.GetPointCount()
    return sum(point_count(geometry.GetGeometryRef(i)) for i in range(children))


def read_admin_geometry():
    dataset = ogr.Open(str(ADMIN), 0)
    if dataset is None:
        raise RuntimeError(f"OGR could not open {ADMIN}")
    layer = dataset.GetLayerByName("admin_assignment_296_aea")
    spatial_ref = layer.GetSpatialRef()
    records = {}
    validity = {"feature_count": 0, "null_geometry_count": 0, "empty_geometry_count": 0,
                "invalid_geometry_count": 0}
    for feature in layer:
        validity["feature_count"] += 1
        city_id = feature.GetField("city_id")
        geometry = feature.GetGeometryRef()
        if geometry is None:
            validity["null_geometry_count"] += 1
            continue
        if geometry.IsEmpty():
            validity["empty_geometry_count"] += 1
            continue
        if not geometry.IsValid():
            validity["invalid_geometry_count"] += 1
        minx, maxx, miny, maxy = geometry.GetEnvelope()
        point = geometry.PointOnSurface()
        records[city_id] = {
            "map_x_aea_m": point.GetX(),
            "map_y_aea_m": point.GetY(),
            "admin_bbox_width_m": maxx - minx,
            "admin_bbox_height_m": maxy - miny,
            "admin_area_m2": geometry.GetArea(),
            "admin_vertex_count": point_count(geometry),
        }
    audit = {
        **validity,
        "layer": layer.GetName(),
        "geometry_type": ogr.GeometryTypeToName(layer.GetGeomType()),
        "crs_name": spatial_ref.GetName() if spatial_ref else None,
        "crs_wkt": spatial_ref.ExportToWkt() if spatial_ref else None,
    }
    dataset = None
    return records, audit


def read_boundary_geometry():
    dataset = ogr.Open(str(BOUNDARIES), 0)
    if dataset is None:
        raise RuntimeError(f"OGR could not open {BOUNDARIES}")
    layer = dataset.GetLayerByName("measurement_boundaries_aea")
    spatial_ref = layer.GetSpatialRef()
    records = {}
    validity = {"raw_feature_count": 0, "null_geometry_count": 0, "empty_geometry_count": 0,
                "invalid_geometry_count": 0}
    for feature in layer:
        if feature.GetField("geom_var") != "raw_product_geometry":
            continue
        validity["raw_feature_count"] += 1
        city_id, scenario = feature.GetField("city_id"), feature.GetField("bnd_scn")
        geometry = feature.GetGeometryRef()
        if geometry is None:
            validity["null_geometry_count"] += 1
            continue
        if geometry.IsEmpty():
            validity["empty_geometry_count"] += 1
            continue
        if not geometry.IsValid():
            validity["invalid_geometry_count"] += 1
        key = (city_id, scenario)
        if key in records:
            raise ValueError(f"Duplicate raw boundary geometry: {key}")
        records[key] = {"area_m2": geometry.GetArea(), "vertex_count": point_count(geometry)}
    audit = {
        **validity,
        "layer": layer.GetName(),
        "filter": "geom_var == raw_product_geometry",
        "geometry_type": ogr.GeometryTypeToName(layer.GetGeomType()),
        "crs_name": spatial_ref.GetName() if spatial_ref else None,
        "crs_wkt": spatial_ref.ExportToWkt() if spatial_ref else None,
    }
    dataset = None
    return records, audit


def fixed_common_values(matrix: pd.DataFrame):
    confirmatory = matrix.loc[matrix.confirmatory_main_row.eq(1)].copy()
    counts = confirmatory.groupby("city_id").size()
    common = set(counts[counts.eq(8)].index)
    if len(common) != 285:
        raise ValueError(f"Expected fixed common set of 285, got {len(common)}")
    selected = confirmatory.loc[confirmatory.city_id.isin(common)].copy()
    selected["specification_id"] = (
        selected.bnd_scn + "|" + selected.landcover_dataset_id + "|" + selected.estimator_id
    )
    selected["blue_green_share"] = (
        selected.Blue_share_full_boundary + selected.Green_share_full_boundary
    )
    if len(selected) != 285 * 8 or selected.specification_id.nunique() != 8:
        raise ValueError("The fixed 285 x 8 confirmatory matrix is not rectangular")
    return confirmatory, selected, common


def ranks_for_common(selected: pd.DataFrame):
    specifications = sorted(selected.specification_id.unique())
    ranks = {}
    percentiles = {}
    top10 = {}
    top20 = {}
    for specification in specifications:
        part = selected.loc[selected.specification_id.eq(specification), ["city_id", "blue_green_share"]]
        series = part.set_index("city_id").blue_green_share
        rank = series.rank(method="average", ascending=False)
        ranks[specification] = rank
        percentiles[specification] = (285 - rank) / 284
        top10[specification] = set(rank.index[rank <= 29])
        top20[specification] = set(rank.index[rank <= 57])
    return specifications, ranks, percentiles, top10, top20


def build_figure1(registry, confirmatory, common, admin_records, boundary_records):
    availability = confirmatory.groupby("city_id").size().reindex(registry.city_id, fill_value=0).astype(int)
    model_ids = set(pd.read_csv(MODEL).city_id)
    rows = registry[["city_id", "city_name", "prov_name", "legacy_291_member"]].copy()
    rows["confirmatory_setting_count"] = rows.city_id.map(availability).astype(int)
    rows["map_status"] = np.select(
        [rows.confirmatory_setting_count.eq(8), rows.confirmatory_setting_count.eq(0)],
        ["complete_8", "unrepresented_0"], default="partial_1_to_7")
    rows["task6_status"] = np.select(
        [rows.city_id.isin(model_ids), rows.city_id.eq(HAIDONG)],
        ["model_complete_case", "structural_covariate_missing"],
        default="not_in_task5_common_set")
    for field in ("map_x_aea_m", "map_y_aea_m", "admin_area_m2"):
        rows[field] = rows.city_id.map(lambda x: admin_records[x][field])

    candidates = []
    median_log_ratio_values = []
    intermediate = {}
    for city_id in sorted(common):
        boundary_items = [boundary_records.get((city_id, b)) for b in BOUNDARY_LEVELS]
        if any(item is None or item["area_m2"] <= 0 for item in boundary_items):
            continue
        areas = [item["area_m2"] for item in boundary_items]
        log_area_ratio = math.log(max(areas) / min(areas))
        median_log_ratio_values.append(log_area_ratio)
        admin = admin_records[city_id]
        intermediate[city_id] = {
            "raw_boundary_count": 4,
            "aspect_penalty": abs(math.log(admin["admin_bbox_width_m"] / admin["admin_bbox_height_m"])),
            "log1p_total_boundary_vertices": math.log1p(sum(x["vertex_count"] for x in boundary_items)),
            "boundary_log_area_ratio": log_area_ratio,
            "boundary_areas_m2": dict(zip(BOUNDARY_LEVELS, areas)),
            "boundary_vertex_counts": dict(zip(BOUNDARY_LEVELS, [x["vertex_count"] for x in boundary_items])),
        }
    if len(intermediate) != 285:
        raise ValueError(f"Expected 285 geometry-eligible representative-city candidates, got {len(intermediate)}")
    national_median = float(np.median(median_log_ratio_values))
    score = pd.DataFrame.from_dict(intermediate, orient="index")
    score.index.name = "city_id"
    score["area_ratio_typicality_penalty"] = abs(score.boundary_log_area_ratio - national_median)
    for source, target in [
        ("aspect_penalty", "aspect_penalty_percentile_rank"),
        ("log1p_total_boundary_vertices", "complexity_penalty_percentile_rank"),
        ("area_ratio_typicality_penalty", "area_ratio_penalty_percentile_rank"),
    ]:
        score[target] = score[source].rank(method="average", pct=True, ascending=True)
    score["score_sum"] = score[[
        "aspect_penalty_percentile_rank", "complexity_penalty_percentile_rank",
        "area_ratio_penalty_percentile_rank"]].sum(axis=1)
    score = score.sort_values(["score_sum"], kind="mergesort")
    score = score.reset_index().sort_values(["score_sum", "city_id"], kind="mergesort")
    names = registry.set_index("city_id")[["city_name", "prov_name"]].to_dict("index")
    candidates = []
    for record in score.to_dict("records"):
        record.update(names[record["city_id"]])
        candidates.append(record)
    selected = candidates[0]
    audit = {
        "protocol_decision": "D-054",
        "selection_uses_outcomes": False,
        "eligible_city_count": 285,
        "eligibility": "fixed Task 5 common set and four non-empty raw measurement boundaries",
        "score_rule": "minimize equal-weight sum of three within-candidate percentile ranks; tie-break city_id ascending",
        "national_median_boundary_log_area_ratio": national_median,
        "selected_city_id": selected["city_id"],
        "selected_city_name": selected["city_name"],
        "selected_prov_name": selected["prov_name"],
        "candidate_scores": candidates,
    }
    return rows, audit


def build_figure2(specifications, ranks, top10, top20):
    rank_rows, top_rows = [], []
    for a, b in itertools.combinations(specifications, 2):
        xa, xb = ranks[a].sort_index(), ranks[b].sort_index()
        rank_rows.append({
            "specification_a": a, "specification_b": b, "common_city_n": 285,
            "spearman_rho": xa.corr(xb, method="pearson"),
            "kendall_tau_b": kendalltau(xa.to_numpy(), xb.to_numpy(), variant="b")[0],
        })
        for group, cutoff, groups in [
            ("top_10_percent", 29, top10), ("top_20_percent", 57, top20)
        ]:
            ga, gb = groups[a], groups[b]
            top_rows.append({
                "specification_a": a, "specification_b": b, "group": group,
                "common_city_n": 285, "nominal_rank_cutoff": cutoff,
                "members_a_including_ties": len(ga), "members_b_including_ties": len(gb),
                "intersection_n": len(ga & gb), "union_n": len(ga | gb),
                "jaccard": len(ga & gb) / len(ga | gb),
            })
    return pd.DataFrame(rank_rows), pd.DataFrame(top_rows)


def build_figure3_4(registry, common, percentiles, top10, admin_records):
    metadata = registry.set_index("city_id")[["city_name", "prov_name", "legacy_291_member"]]
    model_ids = set(pd.read_csv(MODEL).city_id)
    decomp = pd.read_csv(DECOMP)
    decomp = decomp.loc[(decomp.scale == "common285_normalized_rank_percentile") &
                        (decomp.outcome_id == "blue_green_share")].set_index("city_id")
    if set(decomp.index) != set(common):
        raise ValueError("Task 5 decomposition does not match the fixed common 285")
    f3, f4 = [], []
    for city_id in sorted(common):
        values = [percentiles[s][city_id] for s in sorted(percentiles)]
        top_count = sum(city_id in top10[s] for s in top10)
        row = decomp.loc[city_id]
        total = math.sqrt(max(float(row.ss_total_within_city), 0) / 8)
        base = {
            "city_id": city_id, **metadata.loc[city_id].to_dict(),
            "map_x_aea_m": admin_records[city_id]["map_x_aea_m"],
            "map_y_aea_m": admin_records[city_id]["map_y_aea_m"],
            "task6_model_included": city_id in model_ids,
        }
        f3.append({
            **base, "normalized_percentile_range": max(values) - min(values),
            "top_decile_count": int(top_count), "top_decile_frequency": top_count / 8,
            "total_rank_instability_rms": total,
        })
        components = {
            "boundary": math.sqrt(max(float(row.ss_boundary), 0) / 8),
            "product": math.sqrt(max(float(row.ss_product), 0) / 8),
            "interaction": math.sqrt(max(float(row.ss_boundary_product_interaction), 0) / 8),
        }
        maximum = max(components.values())
        winners = [name for name, value in components.items() if abs(value - maximum) <= 1e-12]
        f4.append({
            **base, "boundary_rms": components["boundary"], "product_rms": components["product"],
            "interaction_rms": components["interaction"], "total_rms": total,
            "dominant_source": winners[0] if len(winners) == 1 else "tie",
        })
    return pd.DataFrame(f3), pd.DataFrame(f4)


def build_top_decile_leader_summary(figure3, common, top10):
    """Summarize existing top-decile memberships without adding inference."""
    counts_from_sets = pd.Series({
        city_id: sum(city_id in top10[specification] for specification in sorted(top10))
        for city_id in sorted(common)
    }, dtype="int64")
    counts_from_figure3 = (
        figure3.set_index("city_id")["top_decile_count"]
        .reindex(counts_from_sets.index)
        .astype("int64")
    )
    pd.testing.assert_series_equal(
        counts_from_figure3,
        counts_from_sets,
        check_names=False,
        check_dtype=True,
    )
    if len(counts_from_sets) != 285 or not counts_from_sets.between(0, 8).all():
        raise ValueError("Top-decile membership counts must cover 285 cities and lie in 0..8")

    ever = int(counts_from_sets.gt(0).sum())
    consistent = int(counts_from_sets.eq(8).sum())
    conditional = int(counts_from_sets.between(1, 7).sum())
    if consistent + conditional != ever:
        raise ValueError("Consistent and conditional leaders do not partition ever-leaders")

    return pd.DataFrame([
        {
            "category": "ever_top_decile",
            "city_count": ever,
            "denominator_n": 285,
            "denominator_definition": "fixed eight-setting common set",
            "specification_count": 8,
            "membership_rule": "top_decile_count >= 1",
        },
        {
            "category": "consistent_top_decile",
            "city_count": consistent,
            "denominator_n": ever,
            "denominator_definition": "cities entering the top decile in at least one specification",
            "specification_count": 8,
            "membership_rule": "top_decile_count == 8",
        },
        {
            "category": "conditional_top_decile",
            "city_count": conditional,
            "denominator_n": ever,
            "denominator_definition": "cities entering the top decile in at least one specification",
            "specification_count": 8,
            "membership_rule": "1 <= top_decile_count <= 7",
        },
    ])


def build_figure5():
    primary, component = pd.read_csv(PRIMARY), pd.read_csv(COMPONENT)
    rows = []
    for record in primary.to_dict("records"):
        rows.append({
            "analysis_family": "primary_total", "component": "total", "component_label": "Total instability",
            "display_order": record["display_order"], "term": record["term"], "predictor": record["predictor"],
            "predictor_role": record["predictor_role"], "standardized_beta": record["standardized_beta"],
            "hc3_se": record["hc3_se"], "hc3_ci95_low": record["hc3_ci95_low"],
            "hc3_ci95_high": record["hc3_ci95_high"], "hc3_p_two_sided": record["hc3_p_two_sided"],
            "partial_r2": record["partial_r2"], "hc3_bh_q_value": np.nan, "hc3_bh_reject_q_0_05": np.nan,
            "conley_se": record["conley_se"], "conley_ci95_low": record["conley_ci95_low"],
            "conley_ci95_high": record["conley_ci95_high"], "conley_p_two_sided": record["conley_p_two_sided"],
            "conley_bh_q_value": np.nan, "conley_bh_reject_q_0_05": np.nan, "analysis_n": 284,
        })
    component_order = {"boundary": 1, "product": 2, "interaction": 3}
    predictor_order = {name: i + 1 for i, name in enumerate(
        ["polycentricity_z", "shape_complexity_z", "periurban_farm_z", "water_density_z", "ruggedness_z"])}
    for record in component.to_dict("records"):
        rows.append({
            "analysis_family": "component", "component": record["component"],
            "component_label": record["component_label"],
            "display_order": component_order[record["component"]] * 10 + predictor_order[record["term"]],
            "term": record["term"], "predictor": record["predictor"], "predictor_role": "five_city_domain",
            "standardized_beta": record["standardized_beta"], "hc3_se": record["hc3_se"],
            "hc3_ci95_low": record["hc3_ci95_low"], "hc3_ci95_high": record["hc3_ci95_high"],
            "hc3_p_two_sided": record["hc3_p_two_sided"], "partial_r2": record["partial_r2"],
            "hc3_bh_q_value": record["hc3_bh_q_value"],
            "hc3_bh_reject_q_0_05": record["hc3_bh_reject_q_0_05"], "conley_se": record["conley_se"],
            "conley_ci95_low": record["conley_ci95_low"], "conley_ci95_high": record["conley_ci95_high"],
            "conley_p_two_sided": record["conley_p_two_sided"], "conley_bh_q_value": record["conley_bh_q_value"],
            "conley_bh_reject_q_0_05": record["conley_bh_reject_q_0_05"], "analysis_n": 284,
        })
    associations = pd.DataFrame(rows).sort_values(["analysis_family", "display_order", "component"])

    folds, summary = pd.read_csv(CV_FOLDS), pd.read_csv(CV_SUMMARY)
    cv_rows = []
    for record in folds.to_dict("records"):
        cv_rows.append({
            "estimate_scope": "held_out_region", "model_id": record["model_id"],
            "held_out_macroregion": record["held_out_macroregion"], "training_n": record["training_n"],
            "test_n": record["test_n"], "r2": record["fold_r2"], "mae": record["fold_mae"],
            "rmse": record["fold_rmse"],
        })
    for record in summary.to_dict("records"):
        cv_rows.append({
            "estimate_scope": "pooled", "model_id": record["model_id"],
            "held_out_macroregion": "all_seven_regions_pooled", "training_n": np.nan,
            "test_n": record["n"], "r2": record["pooled_cv_r2"], "mae": record["pooled_mae"],
            "rmse": record["pooled_rmse"],
        })
    return associations, pd.DataFrame(cv_rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    registry, matrix = pd.read_csv(REGISTRY), pd.read_csv(MATRIX)
    if len(registry) != 296 or registry.city_id.nunique() != 296:
        raise ValueError("Registered population is not 296 unique cities")
    confirmatory, selected, common = fixed_common_values(matrix)
    admin_records, admin_audit = read_admin_geometry()
    boundary_records, boundary_audit = read_boundary_geometry()
    if set(admin_records) != set(registry.city_id):
        raise ValueError("Administrative geometry IDs do not match the registry")

    figure1, representative = build_figure1(
        registry, confirmatory, common, admin_records, boundary_records)
    specs, ranks, percentiles, top10, top20 = ranks_for_common(selected)
    figure2_rank, figure2_top = build_figure2(specs, ranks, top10, top20)
    figure3, figure4 = build_figure3_4(registry, common, percentiles, top10, admin_records)
    leader_summary = build_top_decile_leader_summary(figure3, common, top10)
    figure5_assoc, figure5_cv = build_figure5()

    products = {
        "R2_figure1_city_status.csv": figure1,
        "R2_figure2_rank_agreement.csv": figure2_rank,
        "R2_figure2_top_group_agreement.csv": figure2_top,
        "R2_figure3_city_instability.csv": figure3,
        "R2_figure4_component_sensitivity.csv": figure4,
        "R2_figure5_associations.csv": figure5_assoc,
        "R2_figure5_regional_cv.csv": figure5_cv,
        "R3_top_decile_leader_summary.csv": leader_summary,
    }
    for name, frame in products.items():
        write_csv(OUT / name, frame)
    representative_path = OUT / "R2_figure1_representative_city.json"
    write_json(representative_path, representative)

    source_paths = [MATRIX, REGISTRY, DECOMP, MODEL, PRIMARY, COMPONENT, CV_FOLDS,
                    CV_SUMMARY, ADMIN, BOUNDARIES]
    output_entries = []
    for name, frame in products.items():
        path = OUT / name
        output_entries.append({
            "path": relative(path), "sha256": sha256(path), "row_count": len(frame),
            "columns": list(frame.columns), "missing_by_column": {
                key: int(value) for key, value in frame.isna().sum().items() if value
            },
        })
    output_entries.append({
        "path": relative(representative_path), "sha256": sha256(representative_path),
        "record_type": "geometry_only_representative_city_audit",
    })
    manifest = {
        "status": "pass", "protocol_decision": "D-054", "generated_on": "2026-08-22",
        "builder_script": relative(Path(__file__).resolve()),
        "builder_script_sha256": sha256(Path(__file__).resolve()),
        "task5_common_city_count": 285, "task6_model_city_count": 284,
        "rank_denominator": "fixed 285 cities for all eight specifications",
        "top_group_rule": "average ranks; ceil(0.10*n)=29 and ceil(0.20*n)=57; exact ties retained",
        "top_decile_leader_summary_status": "descriptive_not_inferential",
        "rms_rule": "sqrt(within-city component sum of squares / 8 specifications)",
        "geometry_audit": {"administrative": admin_audit, "measurement_boundaries": boundary_audit},
        "sources": [{"path": relative(path), "sha256": sha256(path)} for path in source_paths],
        "outputs": output_entries,
        "software": {
            "python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__,
            "gdal": gdal.VersionInfo("RELEASE_NAME"),
        },
        "figure_consumers": {
            "figure1": ["R2_figure1_city_status.csv", "R2_figure1_representative_city.json"],
            "figure2": ["R2_figure2_rank_agreement.csv", "R2_figure2_top_group_agreement.csv"],
            "figure3": ["R2_figure3_city_instability.csv"],
            "figure4": ["R2_figure4_component_sensitivity.csv"],
            "figure5": ["R2_figure5_associations.csv", "R2_figure5_regional_cv.csv"],
            "results": ["R3_top_decile_leader_summary.csv"],
        },
    }
    manifest_path = OUT / "R2_figure_source_manifest.json"
    write_json(manifest_path, manifest)
    print(json.dumps({
        "status": "pass", "selected_representative_city": representative["selected_city_id"],
        "figure1_status_counts": figure1.map_status.value_counts().to_dict(),
        "figure2_median_spearman": figure2_rank.spearman_rho.median(),
        "figure3_city_count": len(figure3), "figure4_city_count": len(figure4),
        "top_decile_leader_counts": leader_summary.set_index("category").city_count.to_dict(),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
