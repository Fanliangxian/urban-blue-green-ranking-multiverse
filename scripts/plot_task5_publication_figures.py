"""Create the revised main Figure 2 for Revision R15.

The plotting layer consumes the frozen R2 source tables. It does not recompute
city rankings, top groups, or variance components. The current supplementary
Figure S1 has its own independent script; the former Task 5 sensitivity graphic
is not part of the current supplementary information.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MPL_CACHE = ROOT / "work" / "mplconfig_r5_figure2"
MPL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE))

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image
from unified_figure_style import (
    ANNOTATION_SIZE,
    AXIS_LABEL_SIZE,
    DATA_LINEWIDTH,
    DPI,
    FIGURE_WIDTH_MM,
    GRID_LINEWIDTH,
    LEGEND_SIZE,
    PANEL_TITLE_SIZE,
    TICK_LABEL_SIZE,
    TITLE_PAD,
    apply_style,
)


DERIVED = ROOT / "data" / "derived"
SOURCE = DERIVED / "figure_source_data"
OUT = DERIVED / "figures" / "r2"
OUT.mkdir(parents=True, exist_ok=True)

PAIRWISE = SOURCE / "R2_figure2_rank_agreement.csv"
TOP_GROUP = SOURCE / "R2_figure2_top_group_agreement.csv"
CITY = SOURCE / "R2_figure3_city_instability.csv"
DECOMP = DERIVED / "R1_task5_factorial_sensitivity_aggregate.csv"

MAIN_STEM = OUT / "Figure2_ranking_instability"
MANIFEST = OUT / "R2_figure2_manifest.json"

BOUNDARY_ORDER = ("gctb_core_2021", "gctb_system_2021", "gub_2018", "ghs_uc_2020")
PRODUCT_ORDER = (
    ("esa_worldcover_2021", "categorical_native_pixel_area"),
    ("dynamic_world_2021", "annual_mean_probability_area"),
)
SPEC_ORDER = [f"{b}|{d}|{e}" for b in BOUNDARY_ORDER for d, e in PRODUCT_ORDER]
SPEC_LABELS = {
    f"{b}|{d}|{e}": f"{bl}\u2013{pl}"
    for b, bl in zip(BOUNDARY_ORDER, ("Core", "System", "GUB", "GHS"))
    for (d, e), pl in zip(PRODUCT_ORDER, ("WC", "DW"))
}
COMPONENTS = ("boundary", "product", "interaction")
COMPONENT_LABELS = {
    "boundary": "Boundary",
    "product": "Product (land-cover specification)",
    "interaction": "Interaction",
}
COLORS = {"boundary": "#3F67C6", "product": "#B82E6B", "interaction": "#7A7A7A"}
MAIN_PANELS = {
    "a": "pairwise Spearman rank correlation",
    "b": "pairwise top-decile Jaccard overlap",
    "c": "within-city rank-percentile-range empirical cumulative distribution",
    "d": "boundary, land-cover specification and interaction decomposition",
}
ABBREVIATION_DEFINITIONS = {
    "Core": "GCTB core 2021 boundary scenario",
    "System": "GCTB system 2021 boundary scenario",
    "GUB": "Global Urban Boundary 2018 boundary scenario",
    "GHS": "GHS-UC 2020 Urban Centre boundary scenario",
    "WC": "ESA WorldCover 2021 categorical native-label area",
    "DW": "Dynamic World 2021 annual-mean probability area",
}
BLUE = "#3F67C6"
MAGENTA = "#B82E6B"
MUTED = "#59666D"
WIDTH_MM = FIGURE_WIDTH_MM
HEIGHT_MM = 145

BOUNDARY_CMAP = LinearSegmentedColormap.from_list(
    "boundary_blue", ["#EEF2FB", "#3F67C6"]
)
PRODUCT_CMAP = LinearSegmentedColormap.from_list(
    "product_magenta", ["#F7E8EE", "#B82E6B"]
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def style() -> None:
    apply_style(mpl)


def component_name(value: str) -> str:
    return value.replace("boundary_product_interaction", "interaction") \
                .replace("landcover_source_estimator", "product") \
                .replace("boundary_by_source_estimator_interaction", "interaction")


def matrix_from_rows(rows: list[dict[str, str]], value_field: str,
                     filter_key: str | None = None, filter_value: str | None = None) -> np.ndarray:
    matrix = np.eye(len(SPEC_ORDER), dtype=float)
    positions = {spec: i for i, spec in enumerate(SPEC_ORDER)}
    for row in rows:
        if filter_key is not None and row[filter_key] != filter_value:
            continue
        i = positions[row["specification_a"]]
        j = positions[row["specification_b"]]
        value = float(row[value_field])
        matrix[i, j] = matrix[j, i] = value
    if not np.isfinite(matrix).all():
        raise ValueError("Pairwise matrix contains non-finite values")
    return matrix


def heatmap(ax, colorbar_ax, matrix: np.ndarray, title: str, colorbar_label: str,
            vmin: float, vmax: float, cmap) -> None:
    diagonal_mask = np.eye(matrix.shape[0], dtype=bool)
    display_matrix = np.ma.masked_where(diagonal_mask, matrix)
    # ArcGIS Pro's bundled matplotlib predates ``mpl.colormaps``; the
    # backwards-compatible ``cm.get_cmap`` accepts both names and objects.
    display_cmap = mpl.cm.get_cmap(cmap).copy()
    display_cmap.set_bad("#F2F2F2")
    image = ax.imshow(display_matrix, vmin=vmin, vmax=vmax,
                      cmap=display_cmap, aspect="equal")
    labels = [SPEC_LABELS[s] for s in SPEC_ORDER]
    ax.set_xticks(range(8)); ax.set_yticks(range(8))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.tick_params(length=0, pad=1.5)
    threshold = vmin + 0.62 * (vmax - vmin)
    for i in range(8):
        for j in range(8):
            if diagonal_mask[i, j]:
                ax.text(j, i, "—", ha="center", va="center",
                        color="#666666", fontsize=ANNOTATION_SIZE - 0.4)
                continue
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center",
                        color="white" if matrix[i, j] > threshold else "#222222",
                        fontsize=ANNOTATION_SIZE - 0.4)
    colorbar = ax.figure.colorbar(image, cax=colorbar_ax)
    colorbar.set_label(colorbar_label, fontsize=AXIS_LABEL_SIZE)
    colorbar.ax.tick_params(labelsize=TICK_LABEL_SIZE - 0.4, length=2)
    ax.set_title(title, loc="left", fontweight="bold", fontsize=PANEL_TITLE_SIZE, pad=TITLE_PAD)


def export(fig, stem: Path) -> list[Path]:
    outputs = [stem.with_suffix(suffix) for suffix in (".pdf", ".svg", ".png", ".tiff")]
    fig.savefig(outputs[0], facecolor="white", bbox_inches=None)
    fig.savefig(outputs[1], facecolor="white", bbox_inches=None)
    fig.savefig(outputs[2], dpi=DPI, facecolor="white", bbox_inches=None)
    with Image.open(outputs[2]) as preview:
        preview.convert("RGB").save(outputs[2], format="PNG", dpi=(DPI, DPI), optimize=True)
        preview.convert("RGB").save(outputs[3], format="TIFF", compression="tiff_lzw", dpi=(DPI, DPI))
    return outputs


def build_main_figure() -> tuple[plt.Figure, dict[str, object]]:
    pairwise = read_csv(PAIRWISE)
    top = read_csv(TOP_GROUP)
    city = read_csv(CITY)
    decomp = read_csv(DECOMP)

    rank_matrix = matrix_from_rows(pairwise, "spearman_rho")
    top_matrix = matrix_from_rows(top, "jaccard", "group", "top_10_percent")
    ranges = np.sort(np.array([float(row["normalized_percentile_range"])
                               for row in city if row["normalized_percentile_range"] != ""], dtype=float))
    if ranges.size != 285:
        raise ValueError(f"Expected 285 city ranges, found {ranges.size}")

    # Use explicit physical placements for the four panels.  The original
    # panel-a heatmap was deliberately large; preserving that map footprint
    # while making panel b identical requires avoiding a constrained-layout
    # grid that shrinks square heatmaps to accommodate asymmetric labels.
    # The two heatmaps and their colorbars therefore share exactly the same
    # width and height, while panels c and d occupy the corresponding lower
    # row spans.
    fig = plt.figure(figsize=(WIDTH_MM / 25.4, HEIGHT_MM / 25.4), constrained_layout=False)
    top_y, top_h = 0.526, 0.413
    map_w, cbar_w = 0.333, 0.018
    left_x, right_x = 0.090, 0.605
    cbar_gap = 0.015
    ax_a = fig.add_axes([left_x, top_y, map_w, top_h])
    cbar_a = fig.add_axes([left_x + map_w + cbar_gap, top_y, cbar_w, top_h])
    ax_b = fig.add_axes([right_x, top_y, map_w, top_h])
    cbar_b = fig.add_axes([right_x + map_w + cbar_gap, top_y, cbar_w, top_h])
    bottom_y, bottom_h = 0.075, 0.355
    group_w = map_w + cbar_gap + cbar_w
    ax_c = fig.add_axes([left_x, bottom_y, group_w, bottom_h])
    ax_d = fig.add_axes([right_x, bottom_y, group_w, bottom_h])

    heatmap(ax_a, cbar_a, rank_matrix, "a   Whole-list rank agreement", "Spearman ρ", 0.35, 1.0, BOUNDARY_CMAP)
    heatmap(ax_b, cbar_b, top_matrix, "b   Leading-group overlap", "Top-10% Jaccard", 0.0, 0.70, PRODUCT_CMAP)

    ecdf_y = np.arange(1, ranges.size + 1) / ranges.size
    median_range = float(np.median(ranges))
    ax_c.plot(ranges, ecdf_y, color=MAGENTA, lw=DATA_LINEWIDTH + 0.4)
    ax_c.axvline(median_range, color=BLUE, lw=DATA_LINEWIDTH + 0.1, ls="--")
    ax_c.text(median_range + 0.025, 0.12, f"Median {median_range:.2f}",
              color=BLUE, fontsize=ANNOTATION_SIZE)
    ax_c.set(xlim=(0, 1), ylim=(0, 1.01),
             xlabel="Within-city percentile range across 8 maps",
             ylabel="Cumulative share of cities")
    ax_c.grid(axis="both", color="#DDDDDD", lw=GRID_LINEWIDTH)
    ax_c.set_title("c   City rank intervals are often broad", loc="left",
                   fontweight="bold", fontsize=PANEL_TITLE_SIZE, pad=TITLE_PAD)
    ax_c.text(0.99, 0.04, "n = 285", transform=ax_c.transAxes, ha="right",
              color=MUTED, fontsize=ANNOTATION_SIZE)

    y_pos = {"boundary": 2, "product": 1, "interaction": 0}
    scale_specs = (("raw_full_boundary_share", "o", -0.12),
                   ("common285_normalized_rank_percentile", "s", 0.12))
    for scale, marker, offset in scale_specs:
        subset = [row for row in decomp if row["scale"] == scale and row["outcome_id"] == "blue_green_share"]
        for row in subset:
            comp = component_name(row["component"])
            if comp not in COMPONENTS:
                continue
            value = float(row["variance_share"])
            low = float(row["cluster_bootstrap_ci95_low"])
            high = float(row["cluster_bootstrap_ci95_high"])
            ax_d.errorbar(value, y_pos[comp] + offset, xerr=[[value - low], [high - value]],
                          fmt=marker, color=COLORS[comp], mec="white", mew=0.5,
                          ms=5, capsize=2, lw=DATA_LINEWIDTH)
    ax_d.set_yticks([0, 1, 2]); ax_d.set_yticklabels(["Interaction", "Product", "Boundary"])
    ax_d.set(xlim=(0, 0.70), xlabel="Share of within-city map-set variation")
    ax_d.grid(axis="x", color="#DDDDDD", lw=GRID_LINEWIDTH)
    ax_d.legend(handles=[
        Line2D([], [], marker="o", linestyle="", color="#555555", label="Mapped share"),
        Line2D([], [], marker="s", linestyle="", color="#555555", label="Rank percentile"),
    ], loc="lower right", fontsize=LEGEND_SIZE)
    ax_d.set_title("d   Boundary vs product contribution", loc="left",
                   fontweight="bold", fontsize=PANEL_TITLE_SIZE, pad=TITLE_PAD)

    # Panel c and panel a share the same left placement by construction.

    metadata = {
        "city_count": int(ranges.size), "specification_count": 8,
        "pairwise_count": 28, "median_spearman": float(np.median(rank_matrix[np.triu_indices(8, 1)])),
        "minimum_spearman": float(np.min(rank_matrix[np.triu_indices(8, 1)])),
        "median_top10_jaccard": float(np.median(top_matrix[np.triu_indices(8, 1)])),
        "median_city_percentile_range": median_range,
    }
    return fig, metadata


def main() -> None:
    style()
    required = [PAIRWISE, TOP_GROUP, CITY, DECOMP]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing Figure 2 source data: " + "; ".join(missing))

    figure, main_metadata = build_main_figure()
    main_outputs = export(figure, MAIN_STEM)
    plt.close(figure)
    all_outputs = main_outputs
    manifest = {
        "status": "pass_human_gate_r9_3_pending", "backend": "Python matplotlib",
        "generated_on": "2026-08-23", "figure_width_mm": WIDTH_MM,
        "shared_style": {"module": "scripts/unified_figure_style.py", "version": "2026-08-28"},
        "main_figure": "Figure2_ranking_instability",
        "main_metadata": main_metadata,
        "main_panels": MAIN_PANELS,
        "abbreviation_definitions": ABBREVIATION_DEFINITIONS,
        "product_label_definition": "Product denotes the land-cover specification component.",
        "specification_order": SPEC_ORDER,
        "specification_labels": SPEC_LABELS,
        "heatmap_color_semantics": {
            "scope": "off-diagonal map-pair comparisons only",
            "spearman": "whole-list rank correlation; independent 0.35..1.00 scale",
            "top10_jaccard": "leading-group overlap; independent 0.00..0.70 scale",
        },
        "diagonal_semantics": "self-comparison masked as non-comparison",
        "component_colors": COLORS,
        "sensitivity_scope": "Prespecified perturbations and lineage-balanced 3x2 checks are reported separately; this script generates Figure 2 only.",
        "source_hashes": {str(path.relative_to(ROOT)).replace("\\", "/"): sha256(path) for path in required},
        "outputs": [{"path": str(path.relative_to(ROOT)).replace("\\", "/"),
                     "sha256": sha256(path), "bytes": path.stat().st_size} for path in all_outputs],
    }
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
