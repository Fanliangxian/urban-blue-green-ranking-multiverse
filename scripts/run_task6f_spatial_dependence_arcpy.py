"""Run frozen Task 6F residual spatial-dependence and Conley-HAC audit."""

import hashlib
import json
import math
import platform
from pathlib import Path

import arcpy
import numpy as np
import pandas as pd
import scipy
from scipy import stats

from run_task6_confirmatory_model import (
    CONTINUOUS, FULL_TERMS, PRIMARY_FIVE, design, fit_ols,
)
from run_task6_component_models_fdr import bh_adjust


ROOT = Path(__file__).resolve().parents[1]
DERIVED = ROOT / "data" / "derived"
MODEL_FRAME = DERIVED / "R1_task6_model_frame_284.csv"
GEOMETRY = DERIVED / "task6_reference_admin2021" / "task6_reference_admin2021.shp"
PRIMARY_COEFFICIENTS = DERIVED / "R1_task6_primary_model_coefficients.csv"
COMPONENT_COEFFICIENTS = DERIVED / "R1_task6_component_model_coefficients.csv"
RUN_ID = "task6f_spatial_dependence_20260820_r1"
SEED = 20260820
PERMUTATIONS = 9999
EARTH_RADIUS_KM = 6371.0088
MORAN_SCHEMES = ["knn4", "knn8", "band500km"]
CONLEY_CUTOFFS = [250, 500, 1000]
MODELS = [
    ("primary_total", "total_rank_instability_rms_z", "primary"),
    ("boundary", "boundary_rank_sensitivity_rms_z", "component"),
    ("product", "product_rank_sensitivity_rms_z", "component"),
    ("interaction", "interaction_rank_sensitivity_rms_z", "component"),
]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(frame, path):
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def json_default(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    raise TypeError(type(value).__name__)


def geometry_component_hashes(path):
    hashes = {}
    for candidate in sorted(path.parent.glob(path.stem + ".*")):
        if candidate.is_file() and candidate.suffix.lower() not in {".lock"}:
            hashes[candidate.name] = sha256(candidate)
    return hashes


def extract_centroids_and_audit(path):
    description = arcpy.Describe(str(path))
    source_sr = description.spatialReference
    projected_sr = arcpy.SpatialReference(102025)
    wgs84 = arcpy.SpatialReference(4326)
    if source_sr is None or source_sr.name in {"Unknown", ""}:
        raise ValueError("Reference geometry has unknown CRS")
    fields = {field.name for field in arcpy.ListFields(str(path))}
    if "city_id" not in fields:
        raise ValueError("Reference geometry lacks city_id")

    records = []
    null_count = empty_count = nonpositive_count = 0
    multipart_count = 0
    with arcpy.da.SearchCursor(str(path), ["city_id", "city_name", "SHAPE@"]) as cursor:
        for city_id, city_name, geometry in cursor:
            if geometry is None:
                null_count += 1
                continue
            if geometry.pointCount == 0:
                empty_count += 1
                continue
            projected = geometry.projectAs(projected_sr)
            if projected.area <= 0:
                nonpositive_count += 1
            multipart_count += int(projected.isMultipart)
            centroid_projected = projected.trueCentroid
            centroid_wgs84 = arcpy.PointGeometry(
                centroid_projected, projected_sr
            ).projectAs(wgs84).firstPoint
            records.append({
                "city_id": city_id, "city_name_geometry": city_name,
                "centroid_lon": centroid_wgs84.X,
                "centroid_lat": centroid_wgs84.Y,
                "projected_area_m2": projected.area,
                "multipart": int(projected.isMultipart),
            })

    check_table = "memory\\task6f_check_geometry"
    if arcpy.Exists(check_table):
        arcpy.management.Delete(check_table)
    arcpy.management.CheckGeometry(str(path), check_table, "OGC")
    check_fields = {field.name for field in arcpy.ListFields(check_table)}
    check_rows = []
    if {"FEATURE_ID", "PROBLEM"}.issubset(check_fields):
        with arcpy.da.SearchCursor(check_table, ["FEATURE_ID", "PROBLEM"]) as cursor:
            check_rows = [(feature_id, problem) for feature_id, problem in cursor]
    feature_issue_count = sum(
        1 for feature_id, _ in check_rows if feature_id is not None and feature_id >= 0
    )
    dataset_warning_count = sum(
        1 for feature_id, _ in check_rows if feature_id is None or feature_id < 0
    )
    arcpy.management.Delete(check_table)
    table = pd.DataFrame(records)
    audit = {
        "source_path": str(path), "source_row_count": int(arcpy.management.GetCount(str(path))[0]),
        "source_shape_type": description.shapeType,
        "source_crs_name": source_sr.name,
        "source_crs_wkid": source_sr.factoryCode,
        "centroid_projection_name": projected_sr.name,
        "centroid_projection_wkid": projected_sr.factoryCode,
        "centroid_projection_linear_unit": projected_sr.linearUnitName,
        "null_geometry_count": null_count, "empty_geometry_count": empty_count,
        "nonpositive_projected_area_count": nonpositive_count,
        "multipart_count": multipart_count,
        "ogc_feature_geometry_issue_count": feature_issue_count,
        "check_geometry_dataset_warning_count": dataset_warning_count,
        "check_geometry_dataset_warning_class": (
            "feature_id_minus_one_dataset_level_warning" if dataset_warning_count else "none"
        ),
        "city_id_unique_count": int(table["city_id"].nunique()),
        "city_id_duplicate_count": int(table["city_id"].duplicated().sum()),
    }
    return table, audit


def haversine_matrix(lon_deg, lat_deg):
    lon = np.radians(np.asarray(lon_deg, dtype=float))
    lat = np.radians(np.asarray(lat_deg, dtype=float))
    dlon = lon[:, None] - lon[None, :]
    dlat = lat[:, None] - lat[None, :]
    a = np.sin(dlat / 2) ** 2 + (
        np.cos(lat[:, None]) * np.cos(lat[None, :]) * np.sin(dlon / 2) ** 2
    )
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def knn_adjacency(distances, k):
    directed = np.zeros(distances.shape, dtype=bool)
    nearest = np.argsort(distances, axis=1)[:, 1:k + 1]
    directed[np.arange(len(distances))[:, None], nearest] = True
    return directed | directed.T


def row_standardize(adjacency):
    degree = adjacency.sum(axis=1)
    weights = np.zeros(adjacency.shape, dtype=float)
    nonisolated = degree > 0
    weights[nonisolated] = adjacency[nonisolated].astype(float) / degree[nonisolated, None]
    return weights


def connected_component_count(adjacency):
    unseen = set(range(len(adjacency)))
    count = 0
    while unseen:
        count += 1
        stack = [unseen.pop()]
        while stack:
            node = stack.pop()
            neighbors = set(np.flatnonzero(adjacency[node])) & unseen
            unseen.difference_update(neighbors)
            stack.extend(neighbors)
    return count


def moran_i(values, weights):
    centered = values - values.mean()
    n = len(centered)
    return (n / weights.sum()) * float(centered @ weights @ centered) / float(centered @ centered)


def moran_permutation(values, weights, permutation_indices):
    observed = moran_i(values, weights)
    expected = -1 / (len(values) - 1)
    centered = values - values.mean()
    permuted = centered[permutation_indices]
    numerators = np.einsum("bi,ij,bj->b", permuted, weights, permuted, optimize=True)
    simulated = (len(values) / weights.sum()) * numerators / float(centered @ centered)
    p_two = (1 + np.sum(np.abs(simulated - expected) >= abs(observed - expected))) / (
        len(simulated) + 1
    )
    return observed, expected, float(simulated.mean()), float(simulated.std(ddof=1)), float(p_two)


def conley_covariance(x, residual, xtx_inv, distances, cutoff_km):
    kernel = np.clip(1 - distances / cutoff_km, 0, 1)
    score_cross = residual[:, None] * kernel * residual[None, :]
    meat = x.T @ score_cross @ x
    n, p = x.shape
    covariance = (n / (n - p)) * xtx_inv @ meat @ xtx_inv
    covariance = (covariance + covariance.T) / 2
    return covariance, kernel


def coefficient_source_table():
    primary = pd.read_csv(PRIMARY_COEFFICIENTS, encoding="utf-8-sig")
    primary["spatial_model_id"] = "primary_total"
    component = pd.read_csv(COMPONENT_COEFFICIENTS, encoding="utf-8-sig")
    component["spatial_model_id"] = component["component"]
    return pd.concat([
        primary[["spatial_model_id", "term", "estimate", "hc3_se", "hc3_p_two_sided",
                 "hc3_ci95_low", "hc3_ci95_high"]],
        component[["spatial_model_id", "term", "estimate", "hc3_se", "hc3_p_two_sided",
                   "hc3_ci95_low", "hc3_ci95_high"]],
    ], ignore_index=True)


def main():
    failures = []
    frame = pd.read_csv(MODEL_FRAME, dtype={"city_id": str})
    centroids, geometry_audit = extract_centroids_and_audit(GEOMETRY)
    if geometry_audit["source_row_count"] != 296:
        failures.append("reference geometry does not contain 296 rows")
    for key in ["null_geometry_count", "empty_geometry_count", "nonpositive_projected_area_count",
                "ogc_feature_geometry_issue_count", "city_id_duplicate_count"]:
        if geometry_audit[key] != 0:
            failures.append(f"geometry audit {key} is {geometry_audit[key]}")
    merged = frame.merge(centroids, on="city_id", how="left", validate="one_to_one", indicator=True)
    if len(merged) != 284 or (merged["_merge"] != "both").any():
        failures.append("284-city model-to-geometry join is not complete one-to-one")
    merged = merged.drop(columns="_merge")
    if failures:
        raise ValueError("; ".join(failures))

    distances = haversine_matrix(merged["centroid_lon"], merged["centroid_lat"])
    np.fill_diagonal(distances, 0)
    adjacencies = {
        "knn4": knn_adjacency(distances, 4),
        "knn8": knn_adjacency(distances, 8),
        "band500km": (distances <= 500) & (distances > 0),
    }
    weights = {name: row_standardize(adjacency) for name, adjacency in adjacencies.items()}

    weight_audit_rows, edge_rows = [], []
    for scheme in MORAN_SCHEMES:
        adjacency = adjacencies[scheme]
        degree = adjacency.sum(axis=1)
        for i, j in zip(*np.where(np.triu(adjacency, 1))):
            edge_rows.append({
                "run_id": RUN_ID, "weight_scheme": scheme,
                "city_id_i": merged.iloc[i]["city_id"],
                "city_id_j": merged.iloc[j]["city_id"],
                "great_circle_distance_km": distances[i, j],
            })
        weight_audit_rows.append({
            "run_id": RUN_ID, "weight_scheme": scheme, "n": len(merged),
            "undirected_edge_count": int(np.triu(adjacency, 1).sum()),
            "isolate_count": int((degree == 0).sum()),
            "isolated_city_ids": "|".join(merged.loc[degree == 0, "city_id"].tolist()),
            "connected_component_count_including_isolates": connected_component_count(adjacency),
            "min_degree": int(degree.min()), "median_degree": float(np.median(degree)),
            "max_degree": int(degree.max()),
            "minimum_nonzero_distance_km": float(distances[distances > 0].min()),
            "maximum_edge_distance_km": float(distances[adjacency].max()),
            "nonisolated_row_sum_max_abs_error": float(np.max(
                np.abs(weights[scheme].sum(axis=1)[degree > 0] - 1)
            )),
        })
    weight_audit = pd.DataFrame(weight_audit_rows)
    edges = pd.DataFrame(edge_rows)

    rng = np.random.RandomState(SEED)
    permutation_indices = np.vstack([rng.permutation(len(merged)) for _ in range(PERMUTATIONS)])
    x = design(merged, FULL_TERMS)
    fits = {}
    moran_rows = []
    for model_id, outcome, family in MODELS:
        fit = fit_ols(merged[outcome].to_numpy(float), x)
        fits[model_id] = fit
        for scheme in MORAN_SCHEMES:
            result = moran_permutation(fit["residual"], weights[scheme], permutation_indices)
            moran_rows.append({
                "run_id": RUN_ID, "model_id": model_id, "outcome": outcome,
                "outcome_family": family, "weight_scheme": scheme,
                "n": len(merged), "moran_i": result[0],
                "randomization_expected_i": result[1],
                "permutation_mean_i": result[2], "permutation_sd_i": result[3],
                "permutation_replicates": PERMUTATIONS, "random_seed": SEED,
                "permutation_p_two_sided": result[4],
            })
    moran = pd.DataFrame(moran_rows)
    moran["component_bh_q_value"] = np.nan
    moran["component_bh_reject_q_0_05"] = np.nan
    for scheme in MORAN_SCHEMES:
        mask = (moran["weight_scheme"] == scheme) & (moran["outcome_family"] == "component")
        q = bh_adjust(moran.loc[mask, "permutation_p_two_sided"].to_numpy(float))
        moran.loc[mask, "component_bh_q_value"] = q
        moran.loc[mask, "component_bh_reject_q_0_05"] = (q <= 0.05).astype(int)

    source_coefficients = coefficient_source_table()
    coefficient_rows, wald_rows = [], []
    names = ["intercept"] + FULL_TERMS
    primary_indices = [1 + FULL_TERMS.index(term) for term in PRIMARY_FIVE]
    for cutoff in CONLEY_CUTOFFS:
        for model_id, outcome, family in MODELS:
            fit = fits[model_id]
            covariance, kernel = conley_covariance(
                x, fit["residual"], fit["xtx_inv"], distances, cutoff
            )
            covariance_min_eigenvalue = float(np.linalg.eigvalsh(covariance).min())
            if covariance_min_eigenvalue < -1e-10:
                failures.append(
                    f"Conley covariance is not positive semidefinite for {model_id} at {cutoff} km"
                )
            diagonal = np.diag(covariance)
            if np.any(diagonal <= 0):
                failures.append(f"non-positive Conley variance for {model_id} at {cutoff} km")
            se = np.sqrt(np.maximum(diagonal, np.finfo(float).eps))
            z = fit["beta"] / se
            p = 2 * stats.norm.sf(np.abs(z))
            ci_low = fit["beta"] - stats.norm.ppf(0.975) * se
            ci_high = fit["beta"] + stats.norm.ppf(0.975) * se
            for index, term in enumerate(names):
                coefficient_rows.append({
                    "run_id": RUN_ID, "model_id": model_id, "outcome": outcome,
                    "outcome_family": family, "cutoff_km": cutoff,
                    "kernel": "bartlett", "term": term,
                    "estimate": fit["beta"][index], "conley_se": se[index],
                    "conley_z": z[index], "conley_p_two_sided": p[index],
                    "conley_ci95_low": ci_low[index], "conley_ci95_high": ci_high[index],
                    "conley_ci_excludes_zero": int(ci_low[index] > 0 or ci_high[index] < 0),
                    "positive_weight_pair_count_including_diagonal": int((kernel > 0).sum()),
                })
            beta_block = fit["beta"][primary_indices]
            cov_block = covariance[np.ix_(primary_indices, primary_indices)]
            statistic = float(beta_block.T @ np.linalg.pinv(cov_block) @ beta_block)
            wald_rows.append({
                "run_id": RUN_ID, "model_id": model_id, "outcome": outcome,
                "cutoff_km": cutoff, "block": "five_primary_domains",
                "wald_chi2": statistic, "wald_df": len(primary_indices),
                "wald_p": float(stats.chi2.sf(statistic, len(primary_indices))),
                "covariance_min_eigenvalue": covariance_min_eigenvalue,
                "covariance_psd_tolerance_minus_1e_10": int(
                    covariance_min_eigenvalue >= -1e-10
                ),
            })
    conley = pd.DataFrame(coefficient_rows)
    # Merge by a temporary model-key without exposing duplicate coefficient estimates.
    source = source_coefficients.rename(columns={"spatial_model_id": "model_id"})
    conley = conley.merge(
        source.drop(columns="estimate").rename(columns={
            "hc3_ci95_low": "main_hc3_ci95_low", "hc3_ci95_high": "main_hc3_ci95_high"
        }), on=["model_id", "term"], how="left", validate="many_to_one"
    )
    conley["hc3_ci_excludes_zero"] = (
        (conley["main_hc3_ci95_low"] > 0) | (conley["main_hc3_ci95_high"] < 0)
    ).astype(int)
    conley["ci_exclusion_stable_vs_hc3"] = (
        conley["conley_ci_excludes_zero"] == conley["hc3_ci_excludes_zero"]
    ).astype(int)
    wald = pd.DataFrame(wald_rows)

    component_family = conley.loc[
        (conley["outcome_family"] == "component") & conley["term"].isin(PRIMARY_FIVE)
    ].copy()
    component_family["fdr_family_id"] = "five_domains_x_three_components_within_cutoff"
    component_family["fdr_family_size"] = 15
    component_family["bh_q_value"] = np.nan
    component_family["bh_reject_q_0_05"] = np.nan
    for cutoff in CONLEY_CUTOFFS:
        mask = component_family["cutoff_km"] == cutoff
        q = bh_adjust(component_family.loc[mask, "conley_p_two_sided"].to_numpy(float))
        component_family.loc[mask, "bh_q_value"] = q
        component_family.loc[mask, "bh_reject_q_0_05"] = (q <= 0.05).astype(int)
    component_family = component_family.sort_values(["cutoff_km", "bh_q_value", "model_id", "term"])

    # Validate exact reproduction of previously frozen OLS estimates.
    previous = source_coefficients.set_index(["spatial_model_id", "term"])["estimate"]
    coefficient_differences = []
    for model_id, fit in fits.items():
        for index, term in enumerate(names):
            coefficient_differences.append(abs(fit["beta"][index] - previous.loc[(model_id, term)]))
    max_coefficient_difference = float(max(coefficient_differences))
    if max_coefficient_difference > 1e-10:
        failures.append("Task 6F did not reproduce frozen OLS coefficients")
    if len(moran) != 12:
        failures.append("Moran output does not contain 12 model-scheme rows")
    if len(conley) != 4 * 3 * (1 + len(FULL_TERMS)):
        failures.append("Conley coefficient output has unexpected size")
    if len(component_family) != 45:
        failures.append("Conley component FDR output does not contain three 15-test families")

    output_tables = {
        "R1_task6f_spatial_weights_audit.csv": weight_audit,
        "R1_task6f_spatial_edges.csv": edges,
        "R1_task6f_residual_moran.csv": moran,
        "R1_task6f_conley_coefficients.csv": conley,
        "R1_task6f_conley_primary_block_wald.csv": wald,
        "R1_task6f_conley_component_fdr.csv": component_family,
    }
    output_paths = []
    for name, table in output_tables.items():
        path = DERIVED / name
        write_csv(table, path)
        output_paths.append(path)

    primary_moran = moran.loc[
        (moran["model_id"] == "primary_total") & (moran["weight_scheme"] == "knn4")
    ].iloc[0]
    component_moran_primary = moran.loc[
        (moran["outcome_family"] == "component") & (moran["weight_scheme"] == "knn4")
    ]
    main_conley = conley.loc[
        (conley["cutoff_km"] == 500) & conley["term"].isin(PRIMARY_FIVE)
    ]
    main_component_fdr = component_family.loc[component_family["cutoff_km"] == 500]
    manifest = {
        "status": "pass" if not failures else "fail", "run_id": RUN_ID,
        "claim_type": "associational_spatial_robustness",
        "causal_interpretation_prohibited": True,
        "failure_count": len(failures), "failures": failures,
        "sample_n": len(merged), "geometry_audit": geometry_audit,
        "distance_method": "WGS84 spherical haversine between equal-area projected polygon true centroids",
        "moran_permutations": PERMUTATIONS, "random_seed": SEED,
        "primary_knn4_moran_i": primary_moran["moran_i"],
        "primary_knn4_moran_p_two_sided": primary_moran["permutation_p_two_sided"],
        "component_knn4_moran_bh_rejection_count": int(
            component_moran_primary["component_bh_reject_q_0_05"].sum()
        ),
        "conley_primary_cutoff_km": 500,
        "conley_500_all_models_primary_domain_ci_exclusion_stable_n_of_20": int(
            main_conley["ci_exclusion_stable_vs_hc3"].sum()
        ),
        "conley_500_primary_model_domain_ci_exclusion_stable_n_of_5": int(
            main_conley.loc[main_conley["model_id"] == "primary_total",
                            "ci_exclusion_stable_vs_hc3"].sum()
        ),
        "conley_500_component_fdr_rejection_count": int(
            main_component_fdr["bh_reject_q_0_05"].sum()
        ),
        "ols_max_absolute_coefficient_reproduction_error": max_coefficient_difference,
        "predictor_selection_performed": False, "city_deletion_performed": False,
        "software": {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__, "scipy": scipy.__version__,
                     "arcpy_product": arcpy.GetInstallInfo().get("ProductName"),
                     "arcpy_version": arcpy.GetInstallInfo().get("Version")},
        "input_sha256": {
            MODEL_FRAME.name: sha256(MODEL_FRAME),
            PRIMARY_COEFFICIENTS.name: sha256(PRIMARY_COEFFICIENTS),
            COMPONENT_COEFFICIENTS.name: sha256(COMPONENT_COEFFICIENTS),
            "geometry_components": geometry_component_hashes(GEOMETRY),
        },
        "output_sha256": {path.name: sha256(path) for path in output_paths},
    }
    manifest_path = DERIVED / "R1_task6f_spatial_dependence_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=json_default),
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, default=json_default))
    raise SystemExit(0 if not failures else 1)


if __name__ == "__main__":
    main()
