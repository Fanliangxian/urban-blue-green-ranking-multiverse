"""Create the current Supplementary Figure S1 from frozen Task 5 tables.

Supplementary Figure S1 is the outcome-scale and component-decomposition
diagnostic retained in the R15.3 supplementary information.  It is deliberately
independent of the main Figure 2/sensitivity plotting script so that the
supplementary-figure number and source code remain one-to-one.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MPL_CACHE = ROOT / "work" / "mplconfig_task5_s1"
MPL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE))

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from PIL import Image
from unified_figure_style import (
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
SOURCE = DERIVED / "R1_task5_factorial_sensitivity_aggregate.csv"
OUT = DERIVED / "figures" / "task5"
OUT.mkdir(parents=True, exist_ok=True)
STEM = OUT / "Figure_S1_task5_component_decomposition"
MANIFEST = OUT / "task5_s1_figure_manifest.json"

SCALE_ORDER = ("raw_full_boundary_share", "common285_normalized_rank_percentile")
SCALE_LABELS = {
    "raw_full_boundary_share": "Mapped composition",
    "common285_normalized_rank_percentile": "Common-set rank percentile",
}
OUTCOME_ORDER = ("grey_share", "blue_share", "green_share", "blue_green_share")
OUTCOME_LABELS = {
    "grey_share": "Grey",
    "blue_share": "Blue",
    "green_share": "Green",
    "blue_green_share": "Blue + green",
}
COMPONENT_ORDER = ("boundary", "product", "boundary_product_interaction")
COMPONENT_LABELS = {
    "boundary": "Boundary",
    "product": "Product",
    "boundary_product_interaction": "Interaction",
}
# Frozen requested palette: blue, magenta and the Figure S1 interaction grey.
COLORS = {
    "boundary": "#3F67C6",
    "product": "#B82E6B",
    "boundary_product_interaction": "#7A7A7A",
}
WIDTH_MM = FIGURE_WIDTH_MM
HEIGHT_MM = 82


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def style() -> None:
    apply_style(mpl)


def build_figure(rows: list[dict[str, str]]) -> plt.Figure:
    expected = {(scale, outcome, component)
                for scale in SCALE_ORDER for outcome in OUTCOME_ORDER
                for component in COMPONENT_ORDER}
    observed = {(row["scale"], row["outcome_id"], row["component"])
                for row in rows}
    missing = expected - observed
    if missing:
        raise ValueError(f"Missing Task 5 aggregate rows: {sorted(missing)}")

    lookup = {(row["scale"], row["outcome_id"], row["component"]): row
              for row in rows}
    fig = plt.figure(figsize=(WIDTH_MM / 25.4, HEIGHT_MM / 25.4), constrained_layout=True)
    grid = fig.add_gridspec(1, 2, width_ratios=(1, 1), wspace=0.035)
    axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])]
    y_base = np.arange(len(OUTCOME_ORDER))[::-1]
    # Offsets keep the three component estimates legible without implying a
    # further grouping variable.
    offsets = {"boundary": 0.22, "product": 0.0,
               "boundary_product_interaction": -0.22}

    for ax, scale, label in zip(axes, SCALE_ORDER, ("a", "b")):
        for outcome_index, outcome in enumerate(OUTCOME_ORDER):
            for component in COMPONENT_ORDER:
                row = lookup[(scale, outcome, component)]
                value = float(row["variance_share"])
                low = float(row["cluster_bootstrap_ci95_low"])
                high = float(row["cluster_bootstrap_ci95_high"])
                y = y_base[outcome_index] + offsets[component]
                ax.errorbar(
                    value, y,
                    xerr=[[value - low], [high - value]],
                    fmt="o", ms=5.2, capsize=2.4, capthick=0.8,
                    lw=DATA_LINEWIDTH, color=COLORS[component],
                    markeredgewidth=0.5,
                )
        ax.set_title(f"{label}   {SCALE_LABELS[scale]}", pad=TITLE_PAD,
                     fontsize=PANEL_TITLE_SIZE, loc="left", fontweight="bold")
        ax.set_xlim(0, 0.80)
        ax.set_ylim(-0.55, len(OUTCOME_ORDER) - 0.45)
        ax.set_yticks(y_base)
        ax.set_yticklabels([OUTCOME_LABELS[o] for o in OUTCOME_ORDER])
        ax.set_xlabel("Share of within-city map-set variation")
        ax.grid(axis="x", color="#DDDDDD", lw=GRID_LINEWIDTH)
        ax.set_axisbelow(True)

    axes[0].legend(
        handles=[Line2D([0], [0], marker="o", color=COLORS[c],
                        markerfacecolor=COLORS[c], markeredgecolor=COLORS[c],
                        lw=DATA_LINEWIDTH, markersize=5.2, label=COMPONENT_LABELS[c])
                 for c in COMPONENT_ORDER],
        loc="lower right", fontsize=LEGEND_SIZE, handlelength=1.4,
        borderaxespad=0.2,
    )
    return fig


def export(fig: plt.Figure) -> list[Path]:
    outputs = [STEM.with_suffix(suffix)
               for suffix in (".pdf", ".svg", ".png", ".tiff")]
    fig.savefig(outputs[0], facecolor="white", bbox_inches=None)
    fig.savefig(outputs[1], facecolor="white", bbox_inches=None)
    fig.savefig(outputs[2], dpi=DPI, facecolor="white", bbox_inches=None)
    with Image.open(outputs[2]) as image:
        rgb = image.convert("RGB")
        rgb.save(outputs[2], format="PNG", dpi=(DPI, DPI), optimize=True)
        rgb.save(outputs[3], format="TIFF", compression="tiff_lzw",
                 dpi=(DPI, DPI))
    return outputs


def main() -> None:
    style()
    if not SOURCE.exists():
        raise FileNotFoundError(f"Missing source table: {SOURCE}")
    rows = read_csv(SOURCE)
    figure = build_figure(rows)
    outputs = export(figure)
    plt.close(figure)
    manifest = {
        "status": "pass",
        "backend": "Python matplotlib",
        "shared_style": {"module": "scripts/unified_figure_style.py", "version": "2026-08-28"},
        "figure_width_mm": WIDTH_MM,
        "figure_height_mm": HEIGHT_MM,
        "figure_id": "Supplementary Figure S1",
        "figure_contract": "Outcome-scale and component decomposition diagnostics",
        "source_data": str(SOURCE.relative_to(ROOT)),
        "scales": list(SCALE_ORDER),
        "outcomes": list(OUTCOME_ORDER),
        "components": list(COMPONENT_ORDER),
        "colors": COLORS,
        "bootstrap_replicates": 2000,
        "bootstrap_unit": "city cluster",
        "outputs": {path.suffix.lstrip("."): {
            "path": str(path.relative_to(ROOT)), "sha256": sha256(path)
        } for path in outputs},
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
