"""Create Task 6G publication figures from frozen synthesis tables."""

import hashlib
import json
import os
from pathlib import Path

_MPL_CONFIG = Path(__file__).resolve().parents[1] / "data" / "derived" / "figures" / ".mplconfig"
_MPL_CONFIG.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CONFIG))

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from unified_figure_style import (
    ANNOTATION_SIZE,
    AXES_LINEWIDTH,
    AXIS_LABEL_SIZE,
    DATA_LINEWIDTH,
    DPI as SHARED_DPI,
    FIGURE_WIDTH_MM,
    GRID_LINEWIDTH,
    LEGEND_SIZE,
    PANEL_TITLE_SIZE,
    TICK_LABEL_SIZE,
    TITLE_PAD,
    apply_style,
)


ROOT = Path(__file__).resolve().parents[1]
DERIVED = ROOT / "data" / "derived"
FIGURES = DERIVED / "figures" / "task6"
FIGURES.mkdir(parents=True, exist_ok=True)
RUN_ID = "task6g_publication_figures_20260820_r1"
TABLE1 = DERIVED / "R1_task6g_table1_primary_associations.csv"
TABLE2 = DERIVED / "R1_task6g_table2_component_associations.csv"
TABLE_S1 = DERIVED / "R1_task6g_tableS1_model_performance.csv"
TABLE_S2 = DERIVED / "R1_task6g_tableS2_robustness_summary.csv"
MORAN = DERIVED / "R1_task6f_residual_moran.csv"

PRIMARY_TERMS = [
    "polycentricity_z", "shape_complexity_z", "periurban_farm_z",
    "water_density_z", "ruggedness_z",
]
SHORT_LABELS = {
    "polycentricity_z": "Polycentricity",
    "shape_complexity_z": "Shape complexity",
    "periurban_farm_z": "Peri-urban farm",
    "water_density_z": "Water density",
    "ruggedness_z": "Ruggedness",
}
COMPONENTS = ["boundary", "product", "interaction"]
COMPONENT_LABELS = {
    "boundary": "Boundary", "product": "Product", "interaction": "Interaction",
}
COLORS = {
    "primary": "#3F67C6", "conley": "#B82E6B", "boundary": "#3F67C6",
    "product": "#B82E6B", "interaction": "#7A7A7A", "neutral": "#777777",
}
WIDTH_MM = FIGURE_WIDTH_MM
DPI = SHARED_DPI
S2_COMPONENT_COLORS = {
    "boundary": "#3F67C6",
    "product": "#B82E6B",
    "interaction": "#7A7A7A",
    "total": "#BBA2D7",
}
S2_INFLUENCE_BLUE = "#B4C2E5"  # panel-a blue at standardized coefficient -0.18
S2_LOOR_LIGHT = "#F4D7E2"
S2_LOOR_DEEP = "#C45C86"  # intentionally lighter than the primary #B82E6B
S2_CONLEY_NONE = "#E6E6E6"
S2_CONLEY_PRESENT = "#B0B0B0"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def style():
    apply_style(mpl)


def export(fig, stem):
    fig.savefig(FIGURES / f"{stem}.pdf", facecolor="white", bbox_inches=None)
    fig.savefig(FIGURES / f"{stem}.svg", facecolor="white", bbox_inches=None)
    fig.savefig(FIGURES / f"{stem}.png", dpi=DPI, facecolor="white", bbox_inches=None)
    with Image.open(FIGURES / f"{stem}.png") as image:
        rgb = image.convert("RGB")
        rgb.save(FIGURES / f"{stem}.png", format="PNG", dpi=(DPI, DPI))
        rgb.save(FIGURES / f"{stem}.tiff", format="TIFF", compression="tiff_lzw",
                 dpi=(DPI, DPI))


def main_figure(table1, table2, performance):
    fig = plt.figure(figsize=(WIDTH_MM / 25.4, 135 / 25.4), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, width_ratios=(0.88, 1.25), height_ratios=(1.22, 0.78))
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[1, :])

    primary = table1.set_index("term").loc[PRIMARY_TERMS].reset_index()
    y = np.arange(len(primary))[::-1]
    for index, row in primary.iterrows():
        yy = y[index]
        beta = row["standardized_beta"]
        ax_a.plot([row["hc3_ci95_low"], row["hc3_ci95_high"]], [yy + 0.09] * 2,
                  color=COLORS["primary"], lw=DATA_LINEWIDTH + 0.4, solid_capstyle="round")
        ax_a.plot(beta, yy + 0.09, "o", color=COLORS["primary"], ms=4.6,
                  label="HC3" if index == 0 else None)
        ax_a.plot([row["conley_ci95_low"], row["conley_ci95_high"]], [yy - 0.09] * 2,
                  color=COLORS["conley"], lw=DATA_LINEWIDTH)
        ax_a.plot(beta, yy - 0.09, "D", mfc="white", mec=COLORS["conley"], mew=0.8,
                  ms=3.5, label="Conley 500 km" if index == 0 else None)
    ax_a.axvline(0, color="#999999", lw=AXES_LINEWIDTH, ls="--", zorder=0)
    ax_a.set_yticks(y)
    ax_a.set_yticklabels([SHORT_LABELS[term] for term in PRIMARY_TERMS])
    ax_a.set_xlim(-0.67, 0.42)
    ax_a.set_xlabel("Standardized coefficient (95% CI)")
    ax_a.grid(axis="x", color="#E6E6E6", lw=GRID_LINEWIDTH)
    ax_a.legend(loc="lower right", fontsize=LEGEND_SIZE)
    ax_a.set_title("a   Primary associations are mixed in direction", loc="left", fontsize=PANEL_TITLE_SIZE, fontweight="bold", pad=TITLE_PAD)

    offsets = {"boundary": 0.22, "product": 0.0, "interaction": -0.22}
    marker = {"boundary": "o", "product": "s", "interaction": "D"}
    y_base = {term: (len(PRIMARY_TERMS) - 1 - i) for i, term in enumerate(PRIMARY_TERMS)}
    for component in COMPONENTS:
        subset = table2.loc[table2["component"] == component].set_index("term")
        for term in PRIMARY_TERMS:
            row = subset.loc[term]
            yy = y_base[term] + offsets[component]
            beta = row["standardized_beta"]
            filled = int(row["hc3_bh_reject_q_0_05"]) == 1
            ax_b.plot([row["hc3_ci95_low"], row["hc3_ci95_high"]], [yy, yy],
                      color=COLORS[component], lw=DATA_LINEWIDTH)
            ax_b.plot(beta, yy, marker[component], ms=4.5, mec=COLORS[component], mew=0.9,
                      mfc=COLORS[component] if filled else "white",
                      label=COMPONENT_LABELS[component] if term == PRIMARY_TERMS[0] else None)
    ax_b.axvline(0, color="#999999", lw=AXES_LINEWIDTH, ls="--", zorder=0)
    ax_b.set_yticks([y_base[term] for term in PRIMARY_TERMS])
    ax_b.set_yticklabels([SHORT_LABELS[term] for term in PRIMARY_TERMS])
    ax_b.set_xlim(-0.68, 0.43)
    ax_b.set_xlabel("Standardized coefficient (HC3 95% CI)")
    ax_b.grid(axis="x", color="#E6E6E6", lw=GRID_LINEWIDTH)
    ax_b.legend(loc="upper left", ncol=3, fontsize=LEGEND_SIZE, columnspacing=0.8, handletextpad=0.3)
    ax_b.text(0.01, -0.18, "Filled marker: BH q ≤ 0.05 (15-test component family)",
              transform=ax_b.transAxes, fontsize=ANNOTATION_SIZE, color="#555555")
    ax_b.set_title("b   Associations differ among map-sensitivity components", loc="left", fontsize=PANEL_TITLE_SIZE, fontweight="bold", pad=TITLE_PAD)

    order = ["primary_total", "boundary", "product", "interaction"]
    labels = ["Total", "Boundary", "Product", "Interaction"]
    perf = performance.set_index("model_id").loc[order]
    xx = np.arange(len(order))
    width = 0.32
    ax_c.bar(xx - width / 2, perf["adjusted_r2"], width, color="#6B8FB3",
             label="Adjusted $R^2$")
    ax_c.bar(xx + width / 2, perf["five_domain_partial_r2"], width, color="#D9A441",
             label="Five-domain partial $R^2$")
    for xpos, value in zip(xx - width / 2, perf["adjusted_r2"]):
        ax_c.text(xpos, value + 0.008, f"{value:.2f}", ha="center", va="bottom", fontsize=ANNOTATION_SIZE)
    for xpos, value in zip(xx + width / 2, perf["five_domain_partial_r2"]):
        ax_c.text(xpos, value + 0.008, f"{value:.2f}", ha="center", va="bottom", fontsize=ANNOTATION_SIZE)
    delta = float(perf.loc["primary_total", "delta_pooled_cv_r2"])
    base = float(perf.loc["primary_total", "baseline_pooled_cv_r2"])
    full = float(perf.loc["primary_total", "full_pooled_cv_r2"])
    ax_c.text(4.68, 0.245, f"Out-of-region pooled CV $R^2$\nPopulation + area: {base:.3f}\n+ five domains: {full:.3f}\nΔ$R^2$ = {delta:+.3f}",
              ha="right", va="top", fontsize=ANNOTATION_SIZE,
              bbox=dict(boxstyle="round,pad=0.35", fc="#F5F5F5", ec="#CCCCCC", lw=0.6))
    ax_c.set_xticks(xx)
    ax_c.set_xticklabels(labels)
    ax_c.set_xlim(-0.5, 4.85)
    ax_c.set_ylim(0, 0.265)
    ax_c.set_ylabel("Model information")
    ax_c.grid(axis="y", color="#E6E6E6", lw=GRID_LINEWIDTH)
    ax_c.legend(loc="upper left", ncol=2, fontsize=LEGEND_SIZE)
    ax_c.set_title("c   Five city domains add explanatory and limited out-of-region predictive information",
                   loc="left", fontsize=PANEL_TITLE_SIZE, fontweight="bold", pad=TITLE_PAD)

    export(fig, "Figure_Task6_city_characteristics")
    plt.close(fig)


def supplementary_figure(table2, robustness, moran):
    fig = plt.figure(figsize=(WIDTH_MM / 25.4, 140 / 25.4), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, width_ratios=(0.98, 1.02),
                            height_ratios=(1.20, 0.80), wspace=0.04, hspace=0.14)
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[1, :])

    outcome = robustness.loc[robustness["analysis_family"] == "outcome_scale_robustness"].copy()
    analysis_order = ["rank_primary", "alternative_range", "alternative_iqr", "alternative_entropy"]
    analysis_labels = ["Rank-transformed RMS", "Percentile range", "Percentile IQR", "Quartile entropy"]
    matrix = np.full((len(analysis_order), len(PRIMARY_TERMS)), np.nan)
    rejected = np.zeros_like(matrix, dtype=bool)
    for i, analysis_id in enumerate(analysis_order):
        subset = outcome.loc[outcome["analysis_id"] == analysis_id].set_index("term")
        for j, term in enumerate(PRIMARY_TERMS):
            row = subset.loc[term]
            matrix[i, j] = row["estimate"]
            rejected[i, j] = int(row["bh_reject_q_0_05"]) == 1
    blue_red = mpl.colors.LinearSegmentedColormap.from_list(
        "s2_blue_red", ["#3F67C6", "#F7F7F7", "#B82E6B"]
    )
    image = ax_a.imshow(matrix, cmap=blue_red, vmin=-0.5, vmax=0.5, aspect="auto")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax_a.text(j, i, f"{matrix[i, j]:+.2f}", ha="center", va="center", fontsize=ANNOTATION_SIZE,
                      color="white" if abs(matrix[i, j]) > 0.28 else "#222222")
            if rejected[i, j]:
                ax_a.plot(j, i - 0.31, "o", ms=2.6, mfc="#111111", mec="#111111")
    ax_a.set_xticks(range(len(PRIMARY_TERMS)))
    ax_a.set_xticklabels([SHORT_LABELS[t] for t in PRIMARY_TERMS], rotation=35, ha="right")
    ax_a.set_yticks(range(len(analysis_order)))
    ax_a.set_yticklabels(analysis_labels)
    # Keep the colorbar immediately adjacent to panel a without allocating a
    # separate layout column; this also lets panel b move left with the bar.
    cax = ax_a.inset_axes([1.015, 0.0, 0.045, 1.0], transform=ax_a.transAxes)
    colorbar = fig.colorbar(image, cax=cax)
    colorbar.set_label("Standardized coefficient", fontsize=AXIS_LABEL_SIZE)
    colorbar.ax.tick_params(labelsize=TICK_LABEL_SIZE)
    ax_a.text(0, -0.27, "Black dot: BH q ≤ 0.05 within frozen outcome family",
              transform=ax_a.transAxes, fontsize=ANNOTATION_SIZE, color="#555555")
    ax_a.set_title("a   Mixed directions persist across outcome scales", loc="left", fontsize=PANEL_TITLE_SIZE, fontweight="bold", pad=TITLE_PAD)

    stability_rows = []
    for component in COMPONENTS:
        subset = table2.loc[table2["component"] == component].set_index("term")
        for term in PRIMARY_TERMS:
            row = subset.loc[term]
            stability_rows.append({
                "label": f"{component[0].upper()} · {SHORT_LABELS[term]}",
                "influence": float(row["sign_stable_vs_primary"]),
                "loor": float(row["same_sign_fold_count"]) / float(row["fold_count"]),
                "conley": float(row["conley_bh_reject_q_0_05"]),
                "loor_text": f"{int(row['same_sign_fold_count'])}/7",
            })
    stability = pd.DataFrame(stability_rows)
    stability_matrix = stability[["influence", "loor", "conley"]].to_numpy()
    # Column-specific palette: light cyan for influence sign, a pink series
    # for same-sign fold stability, and pale blue for Conley FDR support.
    light_pink = mpl.colors.to_rgba(S2_LOOR_LIGHT)
    deep_pink = mpl.colors.to_rgba(S2_LOOR_DEEP)
    cell_colors = np.zeros((*stability_matrix.shape, 4))
    for i in range(stability_matrix.shape[0]):
        cell_colors[i, 0] = mpl.colors.to_rgba(S2_INFLUENCE_BLUE)
        strength = np.clip(stability_matrix[i, 1], 0.0, 1.0)
        cell_colors[i, 1] = [
            light_pink[k] + strength * (deep_pink[k] - light_pink[k])
            for k in range(4)
        ]
        cell_colors[i, 2] = mpl.colors.to_rgba(
            S2_CONLEY_PRESENT if stability_matrix[i, 2] > 0 else S2_CONLEY_NONE
        )
    ax_b.imshow(cell_colors, vmin=0, vmax=1, aspect="auto")
    for i, row in stability.iterrows():
        texts = ["yes" if row["influence"] == 1 else "no", row["loor_text"],
                 "q≤.05" if row["conley"] == 1 else "—"]
        for j, text in enumerate(texts):
            value = stability_matrix[i, j]
            ax_b.text(j, i, text, ha="center", va="center", fontsize=ANNOTATION_SIZE,
                      color="white" if value > 0.72 else "#333333")
    ax_b.set_xticks([0, 1, 2])
    ax_b.set_xticklabels(["Influence\nsign", "LOOR\nsame sign", "Conley\nFDR"])
    ax_b.set_yticks(range(len(stability)))
    ax_b.set_yticklabels(stability["label"], fontsize=TICK_LABEL_SIZE)
    ax_b.set_title("b   Component robustness", loc="left", fontsize=PANEL_TITLE_SIZE, fontweight="bold", pad=TITLE_PAD)

    scheme_order = ["knn4", "knn8", "band500km"]
    scheme_labels = ["4-nearest", "8-nearest", "500 km band"]
    model_order = ["primary_total", "boundary", "product", "interaction"]
    model_labels = {"primary_total": "Total", **COMPONENT_LABELS}
    model_colors = {"primary_total": S2_COMPONENT_COLORS["total"],
                    "boundary": S2_COMPONENT_COLORS["boundary"],
                    "product": S2_COMPONENT_COLORS["product"],
                    "interaction": S2_COMPONENT_COLORS["interaction"]}
    markers = {"primary_total": "o", "boundary": "s", "product": "^", "interaction": "D"}
    for model_id in model_order:
        subset = moran.loc[moran["model_id"] == model_id].set_index("weight_scheme").loc[scheme_order]
        values = subset["moran_i"].to_numpy(float)
        ax_c.plot(range(3), values, color=model_colors[model_id], lw=DATA_LINEWIDTH, alpha=0.8)
        for x, (_, row) in enumerate(subset.iterrows()):
            p_value = row["permutation_p_two_sided"] if model_id == "primary_total" else row["component_bh_q_value"]
            filled = float(p_value) <= 0.05
            ax_c.plot(x, row["moran_i"], markers[model_id], ms=5.0,
                      mec=model_colors[model_id], mew=0.9,
                      mfc=model_colors[model_id] if filled else "white",
                      label=model_labels[model_id] if x == 0 else None)
    ax_c.axhline(-1 / 283, color="#999999", lw=AXES_LINEWIDTH, ls="--", label="Randomization expectation")
    ax_c.set_xticks(range(3))
    ax_c.set_xticklabels(scheme_labels)
    ax_c.set_ylabel("Residual Moran's $I$")
    ax_c.set_ylim(-0.035, 0.19)
    ax_c.grid(axis="y", color="#E6E6E6", lw=GRID_LINEWIDTH)
    ax_c.legend(loc="upper right", ncol=5, fontsize=LEGEND_SIZE, columnspacing=0.9, handletextpad=0.3)
    ax_c.text(0.01, 0.92, "Filled marker: p ≤ 0.05 for total; BH q ≤ 0.05 for component family",
              transform=ax_c.transAxes, fontsize=ANNOTATION_SIZE, color="#555555")
    ax_c.set_title("c   Residual spatial dependence is concentrated in product and interaction sensitivity",
                   loc="left", fontsize=PANEL_TITLE_SIZE, fontweight="bold", pad=TITLE_PAD)

    # Match the lower panel's physical width to the effective first-row span
    # (panel a through panel b), including any leftward movement of panel b.
    fig.canvas.draw()
    pos_a, pos_b, pos_c = ax_a.get_position(), ax_b.get_position(), ax_c.get_position()
    ax_c.set_in_layout(False)
    ax_c.set_position([pos_a.x0, pos_c.y0, pos_b.x1 - pos_a.x0, pos_c.height])

    export(fig, "Figure_S_task6_robustness")
    plt.close(fig)


def main():
    style()
    table1 = pd.read_csv(TABLE1, encoding="utf-8-sig")
    table2 = pd.read_csv(TABLE2, encoding="utf-8-sig")
    performance = pd.read_csv(TABLE_S1, encoding="utf-8-sig")
    robustness = pd.read_csv(TABLE_S2, encoding="utf-8-sig")
    moran = pd.read_csv(MORAN, encoding="utf-8-sig")
    main_figure(table1, table2, performance)
    supplementary_figure(table2, robustness, moran)

    inputs = [TABLE1, TABLE2, TABLE_S1, TABLE_S2, MORAN]
    files = sorted(path.name for path in FIGURES.iterdir() if path.is_file())
    manifest = {
        "status": "pass", "run_id": RUN_ID, "backend": "Python matplotlib",
        "shared_style": {"module": "scripts/unified_figure_style.py", "version": "2026-08-28"},
        "generated_on": "2026-08-20", "analysis_population_n": 284,
        "new_statistical_model_fit": False,
        "figure_contract": {
            "core_conclusion": (
                "City characteristics add associational information about map sensitivity, "
                "but directions are mixed and component-specific; the primary pattern is "
                "robust to outcome scale and spatial covariance despite residual clustering."
            ),
            "archetype": "quantitative_grid_with_hero_forest_plot",
            "main_width_mm": WIDTH_MM, "main_height_mm": 135,
            "supplement_width_mm": WIDTH_MM, "supplement_height_mm": 140,
            "formats": ["pdf", "svg", "png_600dpi", "tiff_600dpi_lzw_rgb"],
        },
        "supplementary_figure_s2_label_key": {
            "B": "Boundary", "P": "Product (land-cover specification)",
            "I": "Interaction",
        },
        "supplementary_figure_s2_palette": {
            "panel_a": "#3F67C6 to #B82E6B through a neutral midpoint",
            "panel_b_influence": S2_INFLUENCE_BLUE,
            "panel_b_loor": [S2_LOOR_LIGHT, S2_LOOR_DEEP],
            "panel_b_conley": [S2_CONLEY_NONE, S2_CONLEY_PRESENT],
            "panel_c": S2_COMPONENT_COLORS,
        },
        "source_data": [str(source.relative_to(ROOT)).replace("\\", "/") for source in inputs],
        "source_sha256": {source.name: sha256(source) for source in inputs},
        "files": files,
    }
    manifest_path = FIGURES / "task6_figure_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    raster_exports = []
    for stem in ["Figure_Task6_city_characteristics", "Figure_S_task6_robustness"]:
        for suffix in ["png", "tiff"]:
            raster_path = FIGURES / f"{stem}.{suffix}"
            with Image.open(raster_path) as image:
                dpi = image.info.get("dpi", (600, 600))
                raster_exports.append({
                    "file": raster_path.name, "width_px": image.width, "height_px": image.height,
                    "mode": image.mode, "dpi_x": float(dpi[0]), "dpi_y": float(dpi[1]),
                    "has_alpha": "A" in image.mode,
                })
    qa = {
        "status": "pending_visual_review", "backend_exclusive": True,
        "source_preflight_status": "pending", "visual_review_status": "pending",
        "raster_exports": raster_exports,
        "numeric_source_rows": {"table1": len(table1), "table2": len(table2),
                                "tableS1": len(performance), "tableS2": len(robustness),
                                "moran": len(moran)},
    }
    (FIGURES / "task6_figure_qa.json").write_text(
        json.dumps(qa, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
