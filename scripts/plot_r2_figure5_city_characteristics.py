"""Create the revised Figure 5 from frozen association and regional-CV tables."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MPL_CONFIG = ROOT / "work" / "mplconfig_r2_figure5"
MPL_CONFIG.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG))

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from PIL import Image
from unified_figure_style import (
    ANNOTATION_SIZE,
    AXIS_LABEL_SIZE,
    AXES_LINEWIDTH,
    DATA_LINEWIDTH,
    FIGURE_WIDTH_MM as SHARED_FIGURE_WIDTH_MM,
    GRID_LINEWIDTH,
    LEGEND_SIZE,
    PANEL_TITLE_SIZE,
    TICK_LABEL_SIZE,
    TITLE_PAD,
    apply_style,
)


SOURCE = ROOT / "data" / "derived" / "figure_source_data"
ASSOC = SOURCE / "R2_figure5_associations.csv"
REGIONAL_CV = SOURCE / "R2_figure5_regional_cv.csv"
OUT = ROOT / "data" / "derived" / "figures" / "r2"
OUT.mkdir(parents=True, exist_ok=True)
STEM = OUT / "Figure5_city_characteristics_sensitivity"
MANIFEST = OUT / "R2_figure5_manifest.json"
DPI = 600
FIGURE_WIDTH_MM = SHARED_FIGURE_WIDTH_MM
FIGURE_HEIGHT_MM = 126

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
    "boundary": "Boundary",
    "product": "Product",
    "interaction": "Interaction",
}
COLORS = {
    "primary": "#3F67C6",
    "primary_conley": "#B82E6B",
    "boundary": "#3F67C6",
    "product": "#B82E6B",
    "interaction": "#7A7A7A",
    "baseline": "#3F67C6",
    "full": "#B82E6B",
}
REGION_ORDER = [
    "north_china", "northeast_china", "east_china", "central_china",
    "south_china", "southwest_china", "northwest_china",
]
REGION_LABELS = {
    "north_china": "North China",
    "northeast_china": "Northeast China",
    "east_china": "East China",
    "central_china": "Central China",
    "south_china": "South China",
    "southwest_china": "Southwest China",
    "northwest_china": "Northwest China",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def style():
    apply_style(mpl)


def validate(assoc, cv):
    required_assoc = {
        "analysis_family", "component", "component_label", "term", "predictor",
        "standardized_beta", "hc3_ci95_low", "hc3_ci95_high",
        "hc3_bh_reject_q_0_05", "conley_ci95_low", "conley_ci95_high", "analysis_n",
    }
    required_cv = {"estimate_scope", "model_id", "held_out_macroregion", "r2", "test_n"}
    if not required_assoc <= set(assoc.columns):
        raise ValueError(f"Association table missing columns: {required_assoc - set(assoc.columns)}")
    if not required_cv <= set(cv.columns):
        raise ValueError(f"Regional-CV table missing columns: {required_cv - set(cv.columns)}")
    primary = assoc.loc[assoc.analysis_family.eq("primary_total")]
    component = assoc.loc[assoc.analysis_family.eq("component")]
    if len(primary) != 7 or len(component) != 15:
        raise ValueError("Frozen association table must contain seven primary and fifteen component rows")
    if set(primary.term) != set([
        "log_population_z", "log_area_z", *PRIMARY_TERMS,
    ]):
        raise ValueError("Primary association terms changed")
    if set(component.component) != set(COMPONENTS) or set(component.term) != set(PRIMARY_TERMS):
        raise ValueError("Component association grid changed")
    if set(primary.analysis_n) != {284} or set(component.analysis_n) != {284}:
        raise ValueError("Figure 5 association denominator changed")
    held = cv.loc[cv.estimate_scope.eq("held_out_region")]
    pooled = cv.loc[cv.estimate_scope.eq("pooled")]
    if len(held) != 14 or len(pooled) != 2:
        raise ValueError("Regional CV must contain seven paired folds and two pooled estimates")
    if set(held.held_out_macroregion) != set(REGION_ORDER):
        raise ValueError("Regional CV macroregion set changed")
    if set(held.model_id) != {"baseline_population_area", "full_plus_five_domains"}:
        raise ValueError("Regional CV model pair changed")
    if not np.isfinite(assoc[["standardized_beta", "hc3_ci95_low", "hc3_ci95_high",
                              "conley_ci95_low", "conley_ci95_high"]].to_numpy()).all():
        raise ValueError("Association table contains non-finite estimates")
    if not np.isfinite(cv[["r2"]].to_numpy()).all():
        raise ValueError("Regional CV table contains non-finite R2 values")


def draw_primary(ax, assoc):
    primary = assoc.loc[assoc.analysis_family.eq("primary_total") &
                        assoc.term.isin(PRIMARY_TERMS)].set_index("term").loc[PRIMARY_TERMS]
    y = np.arange(len(PRIMARY_TERMS))[::-1]
    for index, term in enumerate(PRIMARY_TERMS):
        row = primary.loc[term]
        yy = y[index]
        beta = row.standardized_beta
        ax.plot([row.hc3_ci95_low, row.hc3_ci95_high], [yy + 0.085, yy + 0.085],
                color=COLORS["primary"], lw=DATA_LINEWIDTH + 0.4, solid_capstyle="round")
        ax.plot(beta, yy + 0.085, "o", color=COLORS["primary"], ms=4.6,
                label="HC3 95% CI" if index == 0 else None)
        ax.plot([row.conley_ci95_low, row.conley_ci95_high], [yy - 0.085, yy - 0.085],
                color=COLORS["primary_conley"], lw=DATA_LINEWIDTH)
        ax.plot(beta, yy - 0.085, "D", mfc="white", mec=COLORS["primary_conley"], mew=0.8,
                ms=3.5, label="Conley 500 km" if index == 0 else None)
    ax.axvline(0, color="#999999", lw=AXES_LINEWIDTH, ls="--", zorder=0)
    ax.set_yticks(y)
    ax.set_yticklabels([SHORT_LABELS[t] for t in PRIMARY_TERMS])
    ax.set_xlim(-0.67, 0.42)
    ax.set_xlabel("Standardized coefficient (95% CI)")
    ax.grid(axis="x", color="#E6E6E6", lw=GRID_LINEWIDTH)
    ax.legend(loc="lower right", fontsize=LEGEND_SIZE, handlelength=1.4)
    ax.set_title("a   Primary associations are mixed in direction", loc="left",
                 fontsize=PANEL_TITLE_SIZE, fontweight="bold", pad=TITLE_PAD)


def draw_components(ax, assoc):
    offsets = {"boundary": 0.22, "product": 0.0, "interaction": -0.22}
    markers = {"boundary": "o", "product": "s", "interaction": "D"}
    y_base = {term: len(PRIMARY_TERMS) - 1 - i for i, term in enumerate(PRIMARY_TERMS)}
    for component in COMPONENTS:
        subset = assoc.loc[(assoc.analysis_family.eq("component")) &
                           (assoc.component.eq(component))].set_index("term")
        for term in PRIMARY_TERMS:
            row = subset.loc[term]
            yy = y_base[term] + offsets[component]
            filled = int(row.hc3_bh_reject_q_0_05) == 1
            ax.plot([row.hc3_ci95_low, row.hc3_ci95_high], [yy, yy],
                    color=COLORS[component], lw=DATA_LINEWIDTH)
            ax.plot(row.standardized_beta, yy, markers[component], ms=4.6,
                    mec=COLORS[component], mew=0.9,
                    mfc=COLORS[component] if filled else "white",
                    label=COMPONENT_LABELS[component] if term == PRIMARY_TERMS[0] else None)
    ax.axvline(0, color="#999999", lw=AXES_LINEWIDTH, ls="--", zorder=0)
    ax.set_yticks([y_base[t] for t in PRIMARY_TERMS])
    ax.set_yticklabels([SHORT_LABELS[t] for t in PRIMARY_TERMS])
    ax.set_xlim(-0.68, 0.43)
    ax.set_xlabel("Standardized coefficient (HC3 95% CI)")
    ax.grid(axis="x", color="#E6E6E6", lw=GRID_LINEWIDTH)
    ax.legend(loc="upper left", ncol=3, fontsize=LEGEND_SIZE, columnspacing=0.7,
              handletextpad=0.25)
    ax.set_title("b   Component-specific associations", loc="left",
                 fontsize=PANEL_TITLE_SIZE, fontweight="bold", pad=TITLE_PAD)


def draw_cv(ax, cv):
    held = cv.loc[cv.estimate_scope.eq("held_out_region")]
    pooled = cv.loc[cv.estimate_scope.eq("pooled")]
    rows = REGION_ORDER + ["all_seven_regions_pooled"]
    labels = [REGION_LABELS[r] for r in REGION_ORDER] + ["Pooled"]
    y = np.arange(len(rows))[::-1]
    baseline = held.loc[held.model_id.eq("baseline_population_area")].set_index("held_out_macroregion")
    full = held.loc[held.model_id.eq("full_plus_five_domains")].set_index("held_out_macroregion")
    for index, region in enumerate(REGION_ORDER):
        yy = y[index]
        b = float(baseline.loc[region, "r2"])
        f = float(full.loc[region, "r2"])
        ax.plot([b, f], [yy, yy], color="#A3A9AD", lw=DATA_LINEWIDTH, zorder=1)
        ax.plot(b, yy, "o", ms=4.6, mfc="white", mec=COLORS["baseline"], mew=0.9,
                label="Population + area" if index == 0 else None, zorder=3)
        ax.plot(f, yy, "o", ms=4.6, mfc=COLORS["full"], mec=COLORS["full"],
                mew=0.8, label="+ five city domains" if index == 0 else None, zorder=3)
        ax.text(b, yy + 0.17, f"{b:.3f}", ha="center", va="bottom", fontsize=ANNOTATION_SIZE - 0.3,
                color=COLORS["baseline"])
        ax.text(f, yy - 0.17, f"{f:.3f}", ha="center", va="top", fontsize=ANNOTATION_SIZE - 0.3,
                color=COLORS["full"])
    yy = y[-1]
    b = float(pooled.loc[pooled.model_id.eq("baseline_population_area"), "r2"].iloc[0])
    f = float(pooled.loc[pooled.model_id.eq("full_plus_five_domains"), "r2"].iloc[0])
    ax.axhline(yy + 0.5, color="#D7D7D7", lw=GRID_LINEWIDTH)
    ax.plot([b, f], [yy, yy], color="#555555", lw=DATA_LINEWIDTH + 0.2, zorder=1)
    ax.plot(b, yy, "o", ms=5.0, mfc="white", mec=COLORS["baseline"], mew=1.0, zorder=3)
    ax.plot(f, yy, "o", ms=5.0, mfc=COLORS["full"], mec=COLORS["full"], mew=0.9, zorder=3)
    ax.text(b, yy + 0.17, f"{b:.3f}", ha="center", va="bottom", fontsize=ANNOTATION_SIZE - 0.3,
            color=COLORS["baseline"])
    ax.text(f, yy - 0.17, f"{f:.3f}", ha="center", va="top", fontsize=ANNOTATION_SIZE - 0.3,
            color=COLORS["full"])
    ax.axvline(0, color="#777777", lw=0.8, ls="--", zorder=0)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_ylim(-0.42, len(rows) - 0.45)
    all_values = cv["r2"].to_numpy(float)
    xmin = float(all_values.min()) - 0.15
    xmax = float(all_values.max()) + 0.15
    ax.set_xlim(xmin, xmax)
    ax.set_xlabel("Held-out-region R² (full range shown)")
    ax.grid(axis="x", color="#E6E6E6", lw=GRID_LINEWIDTH)
    ax.legend(loc="upper left", fontsize=LEGEND_SIZE, ncol=2, columnspacing=0.8,
              handletextpad=0.3)
    ax.text(0.01, 0.02,
            "Negative R² indicates worse prediction than the held-out-region mean; folds are not significance tests.",
            transform=ax.transAxes, fontsize=ANNOTATION_SIZE, color="#555555", va="bottom")
    ax.set_title("c   Transportability is heterogeneous across held-out macroregions",
                 loc="left", fontsize=PANEL_TITLE_SIZE, fontweight="bold", pad=TITLE_PAD)


def export(fig):
    outputs = {
        "pdf": STEM.with_suffix(".pdf"),
        "svg": STEM.with_suffix(".svg"),
        "png": STEM.with_suffix(".png"),
        "tiff": STEM.with_suffix(".tiff"),
    }
    fig.savefig(outputs["pdf"], bbox_inches=None, facecolor="white")
    fig.savefig(outputs["svg"], bbox_inches=None, facecolor="white")
    fig.savefig(outputs["png"], dpi=DPI, bbox_inches=None, facecolor="white")
    fig.savefig(outputs["tiff"], dpi=DPI, bbox_inches=None, facecolor="white",
                pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)
    with Image.open(outputs["png"]) as image:
        image.convert("RGB").save(outputs["png"], format="PNG", dpi=(DPI, DPI))
    with Image.open(outputs["tiff"]) as image:
        image.convert("RGB").save(outputs["tiff"], format="TIFF",
                                   compression="tiff_lzw", dpi=(DPI, DPI))
    return outputs


def main():
    style()
    assoc = pd.read_csv(ASSOC, encoding="utf-8-sig")
    cv = pd.read_csv(REGIONAL_CV, encoding="utf-8-sig")
    validate(assoc, cv)
    fig = plt.figure(figsize=(FIGURE_WIDTH_MM / 25.4, FIGURE_HEIGHT_MM / 25.4),
                     constrained_layout=True)
    grid = fig.add_gridspec(2, 2, width_ratios=(0.94, 1.20), height_ratios=(1.06, 0.94))
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[1, :])
    draw_primary(ax_a, assoc)
    draw_components(ax_b, assoc)
    draw_cv(ax_c, cv)
    fig.canvas.draw()
    outputs = export(fig)

    manifest = {
        "status": "pass_human_gate_r9_5_pending",
        "generated_on": "2026-08-28",
        "backend": "Python matplotlib",
        "shared_style": {"module": "scripts/unified_figure_style.py", "version": "2026-08-28"},
        "new_statistical_model_fit": False,
        "analysis_population_n": 284,
        "component_test_family_n": int((assoc.analysis_family == "component").sum()),
        "component_bh_reject_n": int(assoc.loc[assoc.analysis_family == "component",
                                                   "hc3_bh_reject_q_0_05"].sum()),
        "product_label_definition": "Product denotes the land-cover specification component.",
        "figure_width_mm": FIGURE_WIDTH_MM,
        "figure_height_mm": FIGURE_HEIGHT_MM,
        "panels": {
            "a": "five identical primary city-domain point estimates with HC3 and Conley 500-km covariance intervals",
            "b": "fifteen component coefficients with HC3 intervals and BH q <= 0.05 fill",
            "c": "seven held-out-region paired R2 estimates plus two pooled estimates",
        },
        "panel_a_point_estimate_shared_between_covariance_estimators": True,
        "panel_c_geometry": "paired point-line estimates; no bars",
        "component_colors": COLORS,
        "figure_visual_encoding": {
            "panel_a": {
                "HC3": COLORS["primary"],
                "Conley_500_km": COLORS["primary_conley"],
            },
            "panel_b": {
                "Boundary": COLORS["boundary"],
                "Product": COLORS["product"],
                "Interaction": COLORS["interaction"],
            },
            "panel_c": {
                "Population_plus_area": COLORS["baseline"],
                "plus_five_city_domains": COLORS["full"],
            },
        },
        "regional_cv": {
            "held_out_region_rows": int((cv.estimate_scope == "held_out_region").sum()),
            "pooled_rows": int((cv.estimate_scope == "pooled").sum()),
            "negative_r2_min": float(cv.r2.min()),
            "baseline_pooled_r2": float(cv.loc[(cv.estimate_scope == "pooled") &
                                                (cv.model_id == "baseline_population_area"), "r2"].iloc[0]),
            "full_pooled_r2": float(cv.loc[(cv.estimate_scope == "pooled") &
                                            (cv.model_id == "full_plus_five_domains"), "r2"].iloc[0]),
        },
        "source_sha256": {rel(path): sha256(path) for path in [ASSOC, REGIONAL_CV, Path(__file__).resolve()]},
        "outputs": [{"path": rel(path), "sha256": sha256(path), "bytes": path.stat().st_size}
                    for path in outputs.values()],
        "qa_status": "pass_automated_checks_human_gate_r9_5_pending",
    }
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")
    print(json.dumps({"status": manifest["status"], "outputs": len(outputs),
                      "negative_r2_min": manifest["regional_cv"]["negative_r2_min"]},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
