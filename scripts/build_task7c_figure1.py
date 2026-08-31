"""Build the public-facing supplementary measurement-workflow figure.

The script uses only frozen counts and labels. It performs no statistical analysis.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

# Keep Matplotlib's font cache inside the writable project workspace.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_MPL_CACHE = _PROJECT_ROOT / "work" / "mplconfig_task7c"
_MPL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CACHE))

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from PIL import Image
from unified_figure_style import (
    ANNOTATION_SIZE,
    AXES_LINEWIDTH,
    DATA_LINEWIDTH,
    DPI as SHARED_DPI,
    FIGURE_WIDTH_MM as SHARED_FIGURE_WIDTH_MM,
    PANEL_LABEL_SIZE,
    PANEL_TITLE_SIZE,
    apply_style,
)


ROOT = _PROJECT_ROOT
OUT = ROOT / "data" / "derived" / "figures" / "r2"
STEM = OUT / "Figure_S1_measurement_workflow"
WIDTH_MM = SHARED_FIGURE_WIDTH_MM
HEIGHT_MM = 105
DPI = SHARED_DPI

NAVY = "#24445C"
BLUE = "#5B91B4"
GREEN = "#6E9F78"
SAND = "#D7B46A"
PALE_BLUE = "#E8F1F6"
PALE_GREEN = "#EBF3EC"
PALE_SAND = "#F7F0DF"
PALE_GREY = "#F3F4F5"
GREY = "#5D6870"
LIGHT_LINE = "#C9D1D6"
WHITE = "#FFFFFF"

CAPTION = (
    "Supplementary Figure S1 | Lineage-aware measurement multiverse, analysis populations and "
    "two-stage inference. An independently registered population of 296 cities was "
    "evaluated against four urban-boundary definitions and two land-cover estimators. "
    "Of these, 294 cities were represented in at least one confirmatory setting and "
    "285 in all eight settings. Rank-stability analysis quantified rank agreement and decomposed "
    "within-city variation into boundary, product and interaction components. Total "
    "and component RMS sensitivity connected rank stability to the 284-city "
    "associational analysis. Arrows indicate data flow, not causal effects."
)

ALT_TEXT = (
    "Three-panel flow diagram. Panel a shows 296 registered cities, 294 represented "
    "in at least one map setting, 285 represented in all eight map settings and 284 "
    "complete cases. Panel b crosses four urban boundaries with WorldCover categorical "
    "area and Dynamic World annual mean probability area. Panel c shows Task 5 rank "
    "agreement and factorial decomposition feeding total and component RMS measures, "
    "which are analysed associationally with independent city characteristics."
)


def mm_to_in(value: float) -> float:
    return value / 25.4


def add_box(ax, xy, width, height, facecolor, title, body="", edgecolor=LIGHT_LINE,
            title_color=NAVY, title_size=7.2, body_size=6.2, linewidth=0.8):
    x, y = xy
    box = FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.008,rounding_size=0.012",
        facecolor=facecolor, edgecolor=edgecolor, linewidth=linewidth,
        transform=ax.transAxes, clip_on=False,
    )
    ax.add_patch(box)
    ax.text(x + width / 2, y + height * 0.68, title, transform=ax.transAxes,
            ha="center", va="center", fontsize=title_size, color=title_color,
            fontweight="bold", linespacing=1.1)
    if body:
        ax.text(x + width / 2, y + height * 0.28, body, transform=ax.transAxes,
                ha="center", va="center", fontsize=body_size, color=GREY,
                linespacing=1.18)
    return box


def add_arrow(ax, start, end, color=NAVY, linewidth=DATA_LINEWIDTH, mutation_scale=8):
    arrow = FancyArrowPatch(
        start, end, transform=ax.transAxes, arrowstyle="-|>",
        mutation_scale=mutation_scale, linewidth=linewidth, color=color,
        shrinkA=2, shrinkB=2, connectionstyle="arc3,rad=0",
    )
    ax.add_patch(arrow)
    return arrow


def panel_label(ax, x, y, label):
    ax.text(x, y, label, transform=ax.transAxes, ha="left", va="top",
            fontsize=PANEL_LABEL_SIZE, fontweight="bold", color="#111111")


def build_figure():
    apply_style(mpl)
    fig = plt.figure(figsize=(mm_to_in(WIDTH_MM), mm_to_in(HEIGHT_MM)))
    ax = fig.add_axes([0.015, 0.035, 0.97, 0.93])
    ax.set_axis_off()

    panel_label(ax, 0.005, 0.985, "a")
    ax.text(0.035, 0.985, "Study population", transform=ax.transAxes,
            ha="left", va="top", fontsize=PANEL_TITLE_SIZE, fontweight="bold", color=NAVY)

    pop_boxes = [
        (0.035, 0.81, PALE_BLUE, "296 registered", "Independent 2021\ncity identity frame"),
        (0.035, 0.61, PALE_BLUE, "294 represented", "At least one of\neight settings"),
        (0.035, 0.41, PALE_GREEN, "285 common cities", "All eight settings\nRank-stability set"),
        (0.035, 0.21, PALE_SAND, "284 complete cases", "Association models\nno imputation"),
    ]
    for x, y, fc, title, body in pop_boxes:
        add_box(ax, (x, y), 0.205, 0.135, fc, title, body)
    add_arrow(ax, (0.1375, 0.81), (0.1375, 0.745))
    add_arrow(ax, (0.1375, 0.61), (0.1375, 0.545))
    add_arrow(ax, (0.1375, 0.41), (0.1375, 0.345))
    ax.text(0.155, 0.775, "2 absent from every setting", transform=ax.transAxes,
            ha="left", va="center", fontsize=ANNOTATION_SIZE, color=GREY)
    ax.text(0.155, 0.575, "9 additional incomplete across settings", transform=ax.transAxes,
            ha="left", va="center", fontsize=ANNOTATION_SIZE, color=GREY)
    ax.text(0.155, 0.375, "1 structural covariate missingness (Haidong)", transform=ax.transAxes,
            ha="left", va="center", fontsize=ANNOTATION_SIZE, color=GREY)

    ax.plot([0.275, 0.275], [0.08, 0.955], transform=ax.transAxes,
            color=LIGHT_LINE, linewidth=0.8)

    panel_label(ax, 0.295, 0.985, "b")
    ax.text(0.325, 0.985, "4 × 2 measurement multiverse", transform=ax.transAxes,
            ha="left", va="top", fontsize=PANEL_TITLE_SIZE, fontweight="bold", color=NAVY)

    ax.text(0.39, 0.905, "Urban boundary definitions", transform=ax.transAxes,
            ha="center", va="center", fontsize=ANNOTATION_SIZE, fontweight="bold", color=GREY)
    boundaries = ["GCTB core\n2021", "GCTB system\n2021", "GUB\n2018", "GHS-UC\n2020"]
    bx = [0.295, 0.385, 0.475, 0.565]
    for x, name in zip(bx, boundaries):
        add_box(ax, (x, 0.75), 0.078, 0.105, PALE_BLUE, name, "", title_size=6.0)

    ax.text(0.47, 0.675, "×", transform=ax.transAxes, ha="center", va="center",
            fontsize=PANEL_TITLE_SIZE + 3, color=NAVY, fontweight="bold")
    add_box(ax, (0.315, 0.56), 0.135, 0.09, PALE_GREEN, "WorldCover 2021",
            "categorical area", title_size=6.4, body_size=5.7)
    add_box(ax, (0.49, 0.56), 0.135, 0.09, PALE_GREEN, "Dynamic World 2021",
            "annual mean probability area", title_size=6.4, body_size=5.4)

    # Eight-cell matrix, deliberately schematic and count-based.
    matrix_x, matrix_y = 0.325, 0.32
    cell_w, cell_h = 0.068, 0.047
    for row in range(4):
        for col in range(2):
            color = BLUE if col == 0 else GREEN
            rect = Rectangle(
                (matrix_x + col * (cell_w + 0.02), matrix_y + (3 - row) * (cell_h + 0.012)),
                cell_w, cell_h, transform=ax.transAxes,
                facecolor=color, edgecolor=WHITE, linewidth=1.2, alpha=0.9,
            )
            ax.add_patch(rect)
    ax.text(0.455, 0.43, "8 confirmatory settings", transform=ax.transAxes,
            ha="center", va="center", fontsize=PANEL_TITLE_SIZE, fontweight="bold", color=NAVY)
    ax.text(0.455, 0.275, "One blue–green rank distribution per city\nwithin the finite, preregistered set",
            transform=ax.transAxes, ha="center", va="center", fontsize=ANNOTATION_SIZE,
            color=GREY, linespacing=1.25)

    ax.plot([0.66, 0.66], [0.08, 0.955], transform=ax.transAxes,
            color=LIGHT_LINE, linewidth=0.8)

    panel_label(ax, 0.68, 0.985, "c")
    ax.text(0.71, 0.985, "Integrated two-stage analysis", transform=ax.transAxes,
            ha="left", va="top", fontsize=PANEL_TITLE_SIZE, fontweight="bold", color=NAVY)

    add_box(ax, (0.70, 0.73), 0.255, 0.15, PALE_GREEN,
            "Rank-stability analysis (n = 285)",
            "Rank agreement · Top-group overlap\nWithin-city movement · 4 × 2 decomposition",
            title_size=7.0, body_size=5.9)
    add_arrow(ax, (0.8275, 0.73), (0.8275, 0.655), color=GREEN, linewidth=1.3)
    add_box(ax, (0.725, 0.50), 0.205, 0.125, PALE_SAND,
            "City-specific RMS bridge",
            "Total · boundary · product · interaction",
            title_size=7.0, body_size=5.8, edgecolor=SAND)
    add_arrow(ax, (0.8275, 0.50), (0.8275, 0.425), color=SAND, linewidth=1.3)
    add_box(ax, (0.70, 0.20), 0.255, 0.19, PALE_BLUE,
            "City-sensitivity analysis (n = 284)",
            "Independent city characteristics\nOLS + HC3 · FDR · influence · spatial robustness\nAssociational; no product-accuracy claim",
            title_size=6.9, body_size=5.7)

    # Cross-panel data-flow arrows.
    add_arrow(ax, (0.24, 0.4775), (0.285, 0.4775), color=NAVY, linewidth=1.2)
    add_arrow(ax, (0.635, 0.47), (0.69, 0.79), color=NAVY, linewidth=1.2)

    ax.text(0.5, 0.055,
            "Arrows denote registered data flow and derived outcomes—not causal pathways. Independent reference validation is outside this manuscript.",
            transform=ax.transAxes, ha="center", va="center", fontsize=ANNOTATION_SIZE,
            color=GREY)
    return fig


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fig = build_figure()
    outputs = {
        "pdf": STEM.with_suffix(".pdf"),
        "svg": STEM.with_suffix(".svg"),
        "png": STEM.with_suffix(".png"),
        "tiff": STEM.with_suffix(".tiff"),
    }
    fig.savefig(outputs["pdf"], bbox_inches=None)
    fig.savefig(outputs["svg"], bbox_inches=None)
    fig.savefig(outputs["png"], dpi=DPI, bbox_inches=None, pil_kwargs={"optimize": True})
    fig.savefig(outputs["tiff"], dpi=DPI, bbox_inches=None,
                pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)

    # Matplotlib may emit an unused alpha channel. Freeze both raster deliverables
    # as explicit white-background RGB while retaining 600-dpi metadata.
    with Image.open(outputs["png"]) as image:
        image.convert("RGB").save(outputs["png"], dpi=(DPI, DPI), optimize=True)
    with Image.open(outputs["tiff"]) as image:
        image.convert("RGB").save(outputs["tiff"], dpi=(DPI, DPI), compression="tiff_lzw")

    manifest = {
        "status": "generated_human_visual_approval_pending",
        "figure_id": "R2FS001",
        "backend": "Python_matplotlib_only",
        "shared_style": {"module": "scripts/unified_figure_style.py", "version": "2026-08-28"},
        "width_mm": WIDTH_MM,
        "height_mm": HEIGHT_MM,
        "raster_dpi": DPI,
        "population_cascade": [296, 294, 285, 284],
        "source_script": "scripts/build_task7c_figure1.py",
        "source_evidence": [
            "data/derived/R1_task5_ranking_frame_audit.json",
            "config/task6_city_sensitivity_protocol.yaml",
            "data/derived/R1_task7c_consistency_manifest.json",
        ],
        "caption": CAPTION,
        "alt_text": ALT_TEXT,
        "image_integrity": "vector_drawn_no_source_images_no_synthetic_data",
        "outputs": {
            key: {"path": path.relative_to(ROOT).as_posix(), "sha256": sha256(path), "bytes": path.stat().st_size}
            for key, path in outputs.items()
        },
    }
    (OUT / "R2_figure_s1_workflow_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
