"""Create the spatial sensitivity bridge (Figures 3 and 4) from frozen R2 tables.

The maps use the same official administrative base, projected city coordinates and
South China Sea inset as the revised study-area figure.  Figure 3 shows overall
ranking instability; Figure 4 shows the three orthogonal sensitivity components
and the component that is largest for each city.  No statistical values are
re-estimated in this script.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MPL_CONFIG = ROOT / "work" / "mplconfig_r2_spatial"
GDAL_PROXY = ROOT / "work" / "r2_spatial_gdal_proxy"
MPL_CONFIG.mkdir(parents=True, exist_ok=True)
GDAL_PROXY.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG))
os.environ.setdefault("GDAL_PAM_PROXY_DIR", str(GDAL_PROXY))
os.environ.setdefault("PROJ_LIB", r"C:\Program Files\ArcGIS\Pro\Resources\pedata\gdaldata")

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, ListedColormap, Normalize
from matplotlib.lines import Line2D
from PIL import Image
from osgeo import ogr

from unified_figure_style import (
    ANNOTATION_SIZE,
    AXIS_LABEL_SIZE,
    BASE_FILL,
    DATA_LINEWIDTH,
    DPI as SHARED_DPI,
    FIGURE_WIDTH_MM as SHARED_FIGURE_WIDTH_MM,
    GRID_LINEWIDTH,
    LEGEND_SIZE,
    MAP_DASH_LINEWIDTH,
    MAP_NATIONAL_LINEWIDTH,
    MAP_PROVINCE_LINEWIDTH,
    PANEL_TITLE_SIZE,
    TICK_LABEL_SIZE,
    TITLE_PAD,
    apply_style,
)

from plot_r2_figure1_study_area_measurement_scenarios import (
    BOUNDARIES,
    MAP_APPROVAL_NUMBER,
    draw_line_geometry,
    draw_polygon_geometry,
    load_official_features,
    projected_rectangle_bounds,
    spatial_ref_epsg,
)


SOURCE = ROOT / "data" / "derived" / "figure_source_data"
FIGURES = ROOT / "data" / "derived" / "figures" / "r2"
FIGURES.mkdir(parents=True, exist_ok=True)
FIGURE3_SOURCE = SOURCE / "R2_figure3_city_instability.csv"
FIGURE4_SOURCE = SOURCE / "R2_figure4_component_sensitivity.csv"
FIGURE3_STEM = FIGURES / "Figure3_geography_rank_instability"
FIGURE4_STEM = FIGURES / "Figure4_geographic_decomposition"
MANIFEST_PATH = FIGURES / "R2_spatial_figures_manifest.json"

WIDTH_MM = SHARED_FIGURE_WIDTH_MM
FIGURE3_HEIGHT_MM = 76
FIGURE4_HEIGHT_MM = 116
DPI = SHARED_DPI
TEXT = "#17252E"
MUTED = "#59666D"
PALE = BASE_FILL
FIG3_PALE = "#F5F6F7"
PROVINCE_EDGE = "#B7C0C5"
NATIONAL_EDGE = "#7A858B"
HAIDONG_EDGE = "#B82E6B"
BOUNDARY_COLOR = "#3F67C6"
PRODUCT_COLOR = "#B82E6B"
INTERACTION_COLOR = "#7A7A7A"
COMPONENT_COLORS = {
    "boundary": BOUNDARY_COLOR,
    "product": PRODUCT_COLOR,
    "interaction": INTERACTION_COLOR,
    "tie": "#666666",
}
FIG4_DOMINANT_COLORS = {
    "boundary": "#6F8DD6",
    "product": "#C45F8B",
    "interaction": "#9A9A9A",
    "tie": "#A5A5A5",
}
COMPONENT_LABELS = {
    "boundary": "Boundary",
    "product": "Product (land-cover specification)",
    "interaction": "Interaction",
    "tie": "Tie",
}
DOMINANT_LEGEND_LABELS = {
    "boundary": "Boundary",
    "product": "Product",
    "interaction": "Interaction",
    "tie": "Tie",
}
TOP_DECILE_ZERO_COLOR = "#B0B0B0"
TOP_DECILE_ORDERED_COLORS = [
    "#DCE5F7", "#C8D5F0", "#B4C5E9", "#9FB4E1",
    "#8AA3D9", "#718FD0", "#587AC7", "#3F67C6",
]
TOP_DECILE_TICKS = np.arange(9)
FIGURE4_COMPONENT_LIMITS = (0.0, 0.4)
FIGURE4_COMPONENT_TICKS = np.arange(0.0, 0.41, 0.1)
MAP_EXTENT_LONLAT = (72, 136, 17, 55)
INSET_EXTENT_LONLAT = (105, 125, 3, 25)
FIG3_RANGE_CMAP = LinearSegmentedColormap.from_list(
    "figure3_blue_to_magenta", ["#6F8DD6", "#C45F8B"]
)
FIG3_RMS_CMAP = LinearSegmentedColormap.from_list(
    "figure3_magenta", ["#F3DCE5", "#B82E6B"]
)
FIG4_BLUE_CMAP = LinearSegmentedColormap.from_list(
    "figure4_blue", ["#EDF2FB", "#3F67C6"]
)
FIG4_MAGENTA_CMAP = LinearSegmentedColormap.from_list(
    "figure4_magenta", ["#F3DCE5", "#B82E6B"]
)
FIG4_PURPLE_CMAP = LinearSegmentedColormap.from_list(
    "figure4_purple", ["#F0EAF6", "#BBA2D7"]
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def style() -> None:
    apply_style(mpl)


def load_base():
    dataset = ogr.Open(str(BOUNDARIES), 0)
    if dataset is None:
        raise FileNotFoundError(f"Could not open {BOUNDARIES}")
    layer = dataset.GetLayerByName("measurement_boundaries_aea")
    target_ref = layer.GetSpatialRef()
    if target_ref is None:
        raise ValueError("Measurement boundary layer has no spatial reference")
    target_ref = target_ref.Clone()
    target_ref.SetDataAxisToSRSAxisMapping([1, 2])
    dataset = None
    provinces, dashes = load_official_features(target_ref)
    national_outline = provinces[0].Clone()
    for geometry in provinces[1:]:
        national_outline = national_outline.Union(geometry)
    extent = projected_rectangle_bounds(*MAP_EXTENT_LONLAT, target_ref)
    inset_extent = projected_rectangle_bounds(*INSET_EXTENT_LONLAT, target_ref)
    return provinces, national_outline, dashes, extent, inset_extent


def _draw_province_base(ax, provinces, national_outline=None, fill_color=None):
    if fill_color is None:
        fill_color = PALE
    for geometry in provinces:
        draw_polygon_geometry(ax, geometry, fill_color, PROVINCE_EDGE,
                              linewidth=MAP_PROVINCE_LINEWIDTH, zorder=0)
    if national_outline is not None:
        draw_polygon_geometry(ax, national_outline, "none", NATIONAL_EDGE,
                              linewidth=MAP_NATIONAL_LINEWIDTH, zorder=1)


def _configure_map(ax, extent, frame=None):
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect("equal")
    # Keep the geographic frame optically aligned to the top of its subplot;
    # otherwise Matplotlib centers an equal-aspect map and leaves a large
    # unused band above the map when the colorbar occupies the lower margin.
    ax.set_anchor("N")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        if frame is not None:
            spine.set_visible(frame)
        spine.set_color("#9AA5AA")
        spine.set_linewidth(0.35)


def _draw_inset(ax, frame, provinces, national_outline, dashes, inset_extent,
                value_column, cmap, norm, edgecolor="white", size=4.0,
                complete_frame=False, fill_color=None):
    inset = ax.inset_axes([0.775, 0.018, 0.205, 0.245])
    _draw_province_base(inset, provinces, national_outline, fill_color)
    for geometry in dashes:
        draw_line_geometry(inset, geometry, "#59666D", linewidth=MAP_DASH_LINEWIDTH, zorder=2)
    inset.scatter(frame["map_x_aea_m"], frame["map_y_aea_m"],
                  c=frame[value_column], cmap=cmap, norm=norm,
                  s=size, edgecolors=edgecolor, linewidths=0.15, zorder=4)
    _configure_map(inset, inset_extent, frame=True if complete_frame else None)
    return inset


def draw_continuous_map(ax, frame, provinces, national_outline, dashes, extent, inset_extent,
                        value_column, cmap, norm, title, colorbar_label,
                        outline_haidong=False, colorbar_ticks=None,
                        show_frame=None, colorbar_pad=0.035, title_x=0.0,
                        complete_inset_frame=False, base_fill=None,
                        title_pad=TITLE_PAD):
    _draw_province_base(ax, provinces, national_outline, base_fill)
    image = ax.scatter(frame["map_x_aea_m"], frame["map_y_aea_m"],
                       c=frame[value_column], cmap=cmap, norm=norm,
                       s=8.5, edgecolors="white", linewidths=0.26, zorder=4)
    if outline_haidong:
        haidong = frame.loc[frame.city_id.eq("CN-630200")]
        ax.scatter(haidong["map_x_aea_m"], haidong["map_y_aea_m"],
                   s=19, facecolors="none", edgecolors=HAIDONG_EDGE,
                   linewidths=0.65, zorder=5)
    _configure_map(ax, extent, frame=show_frame)
    _draw_inset(ax, frame, provinces, national_outline, dashes, inset_extent,
                value_column, cmap, norm, size=3.2,
                complete_frame=complete_inset_frame, fill_color=base_fill)
    ax.set_title(title, loc="left", x=title_x, fontweight="bold", fontsize=PANEL_TITLE_SIZE, pad=title_pad)
    colorbar = ax.figure.colorbar(image, ax=ax, orientation="horizontal",
                                  fraction=0.046, pad=colorbar_pad, aspect=27,
                                  ticks=colorbar_ticks)
    colorbar.set_label(colorbar_label, fontsize=AXIS_LABEL_SIZE, labelpad=1.2)
    colorbar.ax.tick_params(labelsize=TICK_LABEL_SIZE - 0.4, length=2, width=0.45)
    return image


def draw_discrete_map(ax, frame, provinces, national_outline, dashes,
                      extent, inset_extent, show_frame=True,
                      colorbar_pad=0.035, title_x=0.0,
                      complete_inset_frame=False, base_fill=None,
                      title_pad=TITLE_PAD):
    cmap = ListedColormap([TOP_DECILE_ZERO_COLOR, *TOP_DECILE_ORDERED_COLORS])
    norm = BoundaryNorm(np.arange(-0.5, 9.5, 1), cmap.N)
    _draw_province_base(ax, provinces, national_outline, base_fill)
    image = ax.scatter(frame["map_x_aea_m"], frame["map_y_aea_m"],
                       c=frame["top_decile_count"], cmap=cmap, norm=norm,
                       s=8.5, edgecolors="white", linewidths=0.26, zorder=4)
    _configure_map(ax, extent, frame=show_frame)
    _draw_inset(ax, frame, provinces, national_outline, dashes, inset_extent,
                "top_decile_count", cmap, norm, size=3.2,
                complete_frame=complete_inset_frame, fill_color=base_fill)
    ax.set_title("b   Top-10% membership count across 8 settings",
                 loc="left", x=title_x, fontweight="bold", fontsize=PANEL_TITLE_SIZE, pad=title_pad)
    colorbar = ax.figure.colorbar(image, ax=ax, orientation="horizontal",
                                  fraction=0.046, pad=colorbar_pad, aspect=27,
                                  ticks=TOP_DECILE_TICKS)
    colorbar.set_label("Top-10% count (0–8)",
                       fontsize=AXIS_LABEL_SIZE, labelpad=1.2)
    colorbar.ax.tick_params(labelsize=TICK_LABEL_SIZE - 0.4, length=2, width=0.45)
    return image


def draw_dominant_map(ax, frame, provinces, national_outline, dashes,
                      extent, inset_extent, title="Dominant sensitivity component by city",
                      show_frame=None, complete_inset_frame=False,
                      base_fill=None, title_x=0.0, title_pad=TITLE_PAD):
    categories = ["boundary", "product", "interaction", "tie"]
    cmap = ListedColormap([FIG4_DOMINANT_COLORS[c] for c in categories])
    norm = BoundaryNorm(np.arange(-0.5, len(categories) + 0.5, 1), len(categories))
    codes = frame["dominant_source"].map({c: i for i, c in enumerate(categories)})
    _draw_province_base(ax, provinces, national_outline, base_fill)
    ax.scatter(frame["map_x_aea_m"], frame["map_y_aea_m"],
               c=codes, cmap=cmap, norm=norm, s=8.5,
               edgecolors="white", linewidths=0.26, zorder=4)
    _configure_map(ax, extent, frame=show_frame)
    inset = ax.inset_axes([0.775, 0.018, 0.205, 0.245])
    _draw_province_base(inset, provinces, national_outline, base_fill)
    for geometry in dashes:
        draw_line_geometry(inset, geometry, "#59666D", linewidth=MAP_DASH_LINEWIDTH, zorder=2)
    inset.scatter(frame["map_x_aea_m"], frame["map_y_aea_m"],
                  c=codes, cmap=cmap, norm=norm, s=3.2,
                  edgecolors="white", linewidths=0.15, zorder=4)
    _configure_map(inset, inset_extent, frame=True if complete_inset_frame else None)
    ax.set_title(title, loc="left", x=title_x, fontweight="bold", fontsize=PANEL_TITLE_SIZE, pad=title_pad)
    counts = frame["dominant_source"].value_counts().to_dict()
    handles = []
    for category in categories:
        if category == "tie" and counts.get(category, 0) == 0:
            continue
        handles.append(Line2D([], [], marker="o", linestyle="",
                              markerfacecolor=FIG4_DOMINANT_COLORS[category],
                              markeredgecolor="white", markersize=4.2,
                              label=f"{DOMINANT_LEGEND_LABELS[category]} (n={counts.get(category, 0)})"))
    ax.legend(handles=handles, loc="lower left", frameon=False, fontsize=LEGEND_SIZE,
              handletextpad=0.35, borderaxespad=0.2)
    return counts


def export_figure(fig, stem: Path):
    outputs = {
        "pdf": stem.with_suffix(".pdf"),
        "svg": stem.with_suffix(".svg"),
        "png": stem.with_suffix(".png"),
        "tiff": stem.with_suffix(".tiff"),
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


def validate_sources(f3: pd.DataFrame, f4: pd.DataFrame):
    required_f3 = {
        "city_id", "city_name", "prov_name", "map_x_aea_m", "map_y_aea_m",
        "task6_model_included", "normalized_percentile_range", "top_decile_count",
        "total_rank_instability_rms",
    }
    required_f4 = {
        "city_id", "city_name", "prov_name", "map_x_aea_m", "map_y_aea_m",
        "task6_model_included", "boundary_rms", "product_rms", "interaction_rms",
        "total_rms", "dominant_source",
    }
    if not required_f3 <= set(f3.columns):
        raise ValueError(f"Figure 3 source missing columns: {required_f3 - set(f3.columns)}")
    if not required_f4 <= set(f4.columns):
        raise ValueError(f"Figure 4 source missing columns: {required_f4 - set(f4.columns)}")
    if len(f3) != 285 or f3.city_id.nunique() != 285:
        raise ValueError("Figure 3 must contain 285 unique common-set cities")
    if len(f4) != 285 or f4.city_id.nunique() != 285:
        raise ValueError("Figure 4 must contain 285 unique common-set cities")
    for frame, columns in ((f3, ["normalized_percentile_range", "total_rank_instability_rms"]),
                           (f4, ["boundary_rms", "product_rms", "interaction_rms", "total_rms"])):
        if frame[columns].isna().any().any():
            raise ValueError("Spatial source contains missing sensitivity values")
        if (frame[columns] < 0).any().any():
            raise ValueError("Spatial source contains negative sensitivity values")
        if not np.isfinite(frame[columns].to_numpy()).all():
            raise ValueError("Spatial source contains non-finite sensitivity values")
    if not f3["top_decile_count"].between(0, 8).all():
        raise ValueError("Top-decile counts must lie in 0..8")
    np.testing.assert_allclose(
        f4.total_rms.to_numpy() ** 2,
        (f4[["boundary_rms", "product_rms", "interaction_rms"]].to_numpy() ** 2).sum(axis=1),
        atol=1e-14,
        rtol=1e-12,
    )
    if set(f4.dominant_source) - {"boundary", "product", "interaction", "tie"}:
        raise ValueError("Unexpected dominant-source category")
    if not np.isfinite(f3[["map_x_aea_m", "map_y_aea_m"]].to_numpy()).all():
        raise ValueError("Figure 3 map coordinates are not finite")
    if not np.isfinite(f4[["map_x_aea_m", "map_y_aea_m"]].to_numpy()).all():
        raise ValueError("Figure 4 map coordinates are not finite")


def build_figure3(f3, provinces, national_outline, dashes, extent, inset_extent):
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH_MM / 25.4, FIGURE3_HEIGHT_MM / 25.4),
                             constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.01, h_pad=0.01,
                                    wspace=0.015, hspace=0.01)
    draw_continuous_map(
        axes[0], f3, provinces, national_outline, dashes, extent, inset_extent,
        "normalized_percentile_range", FIG3_RANGE_CMAP, Normalize(0, 1),
        "a   Within-city percentile range", "Percentile range (0–1)",
        show_frame=False, colorbar_pad=0.012, title_x=0.035,
        complete_inset_frame=True, base_fill=FIG3_PALE, title_pad=TITLE_PAD,
    )
    draw_discrete_map(
        axes[1], f3, provinces, national_outline, dashes, extent, inset_extent,
        show_frame=False, colorbar_pad=0.012, title_x=0.035,
        complete_inset_frame=True, base_fill=FIG3_PALE, title_pad=TITLE_PAD,
    )
    draw_continuous_map(
        axes[2], f3, provinces, national_outline, dashes, extent, inset_extent,
        "total_rank_instability_rms", FIG3_RMS_CMAP, Normalize(0, 0.4),
        "c   Total rank-instability RMS", "Total RMS (percentile units)",
        outline_haidong=True, show_frame=False, colorbar_pad=0.012,
        title_x=0.070, complete_inset_frame=True, base_fill=FIG3_PALE,
        title_pad=TITLE_PAD,
    )
    # Equal-aspect maps occupy only the height required by the national extent.
    # With a short three-panel canvas, constrained layout otherwise centers the
    # row too low and leaves a conspicuous white band above it.  Freeze the
    # layout and lift the row (including colorbars and inset axes) as one unit.
    fig.canvas.draw()
    fig.set_constrained_layout(False)
    lift = 0.21
    moved = set()
    for ax in axes:
        children = [ax, *getattr(ax, "child_axes", [])]
        for child in children:
            position = child.get_position()
            child.set_position([position.x0, position.y0 + lift,
                                position.width, position.height])
            moved.add(id(child))
    for ax in fig.axes:
        if id(ax) in moved:
            continue
        position = ax.get_position()
        ax.set_position([position.x0, position.y0 + lift,
                         position.width, position.height])
    return fig


def build_figure4(f4, provinces, national_outline, dashes, extent, inset_extent):
    fig, axes = plt.subplots(2, 2, figsize=(WIDTH_MM / 25.4, FIGURE4_HEIGHT_MM / 25.4),
                             constrained_layout=True)
    flat = axes.ravel()
    draw_continuous_map(
        flat[0], f4, provinces, national_outline, dashes, extent, inset_extent,
        "boundary_rms", FIG4_BLUE_CMAP, Normalize(*FIGURE4_COMPONENT_LIMITS),
        "a   Boundary RMS sensitivity", "Boundary RMS (0–0.40)",
        colorbar_ticks=FIGURE4_COMPONENT_TICKS,
        show_frame=False, complete_inset_frame=True, base_fill=FIG3_PALE,
        title_x=0.035, title_pad=TITLE_PAD,
    )
    draw_continuous_map(
        flat[1], f4, provinces, national_outline, dashes, extent, inset_extent,
        "product_rms", FIG4_MAGENTA_CMAP, Normalize(*FIGURE4_COMPONENT_LIMITS),
        "b   Land-cover specification RMS sensitivity", "Product component RMS (0–0.40)",
        colorbar_ticks=FIGURE4_COMPONENT_TICKS,
        show_frame=False, complete_inset_frame=True, base_fill=FIG3_PALE,
        title_x=0.035, title_pad=TITLE_PAD,
    )
    draw_continuous_map(
        flat[2], f4, provinces, national_outline, dashes, extent, inset_extent,
        "interaction_rms", FIG4_PURPLE_CMAP, Normalize(*FIGURE4_COMPONENT_LIMITS),
        "c   Interaction RMS sensitivity", "Interaction RMS (0–0.40)",
        colorbar_ticks=FIGURE4_COMPONENT_TICKS,
        show_frame=False, complete_inset_frame=True, base_fill=FIG3_PALE,
        title_x=0.035, title_pad=TITLE_PAD,
    )
    counts = draw_dominant_map(
        flat[3], f4, provinces, national_outline, dashes, extent, inset_extent,
        title="d   Dominant sensitivity component by city",
        show_frame=False, complete_inset_frame=True, base_fill=FIG3_PALE,
        title_x=0.035, title_pad=TITLE_PAD,
    )
    return fig, counts


def main():
    style()
    f3 = pd.read_csv(FIGURE3_SOURCE, encoding="utf-8-sig")
    f4 = pd.read_csv(FIGURE4_SOURCE, encoding="utf-8-sig")
    validate_sources(f3, f4)
    provinces, national_outline, dashes, extent, inset_extent = load_base()
    outputs3 = export_figure(
        build_figure3(f3, provinces, national_outline, dashes, extent, inset_extent),
        FIGURE3_STEM,
    )
    figure4, dominant_counts = build_figure4(
        f4, provinces, national_outline, dashes, extent, inset_extent
    )
    outputs4 = export_figure(figure4, FIGURE4_STEM)

    source_paths = [FIGURE3_SOURCE, FIGURE4_SOURCE, Path(__file__).resolve()]
    output_paths = list(outputs3.values()) + list(outputs4.values())
    manifest = {
        "status": "pass_human_gate_r9_4_pending",
        "generated_on": "2026-08-28",
        "backend": "Python matplotlib + GDAL/OGR",
        "shared_style": {"module": "scripts/unified_figure_style.py", "version": "2026-08-28"},
        "new_statistical_model_fit": False,
        "map_source": "National Geomatics Center of China 2024 administrative data",
        "map_approval_number": MAP_APPROVAL_NUMBER,
        "projection": "official AEA projection stored in measurement_boundaries_r1.gpkg",
        "population_denominators": {
            "figure3_cities": 285,
            "figure4_cities": 285,
            "figure3_task6_model_included": int(f3.task6_model_included.sum()),
            "figure4_task6_model_included": int(f4.task6_model_included.sum()),
        },
        "figure3_ranges": {
            "normalized_percentile_range": [float(f3.normalized_percentile_range.min()),
                                             float(f3.normalized_percentile_range.max())],
            "top_decile_count": [int(f3.top_decile_count.min()), int(f3.top_decile_count.max())],
            "total_rank_instability_rms": [float(f3.total_rank_instability_rms.min()),
                                            float(f3.total_rank_instability_rms.max())],
        },
        "figure3_top_decile_encoding": {
            "raw_count_range": [0, 8],
            "raw_count_ticks": [int(value) for value in TOP_DECILE_TICKS],
            "zero_color": TOP_DECILE_ZERO_COLOR,
            "ordered_colors_1_to_8": TOP_DECILE_ORDERED_COLORS,
            "category_counts": {
                str(int(key)): int(value)
                for key, value in f3.top_decile_count.value_counts().sort_index().items()
            },
            "interpretation": "0 = never in the top decile; 1-7 = conditional membership; 8 = consistent membership across the eight frozen settings",
        },
        "figure3_visual_encoding": {
            "panel_a_cmap_endpoints": ["#6F8DD6", "#C45F8B"],
            "panel_b_scale_type": "discrete ordinal bins (0..8), not a continuous gradient",
            "panel_b_zero_color": TOP_DECILE_ZERO_COLOR,
            "panel_b_ordered_colors_1_to_8": TOP_DECILE_ORDERED_COLORS,
            "panel_c_cmap_endpoints": ["#F3DCE5", "#B82E6B"],
            "base_fill": FIG3_PALE,
            "province_edge": PROVINCE_EDGE,
            "national_edge": NATIONAL_EDGE,
            "main_map_frame": "hidden",
            "inset_frame": "complete rectangle",
        },
        "figure4_ranges": {
            column: [float(f4[column].min()), float(f4[column].max())]
            for column in ["boundary_rms", "product_rms", "interaction_rms", "total_rms"]
        },
        "figure4_shared_component_scale": {
            "limits": list(FIGURE4_COMPONENT_LIMITS),
            "ticks": [round(float(value), 10) for value in FIGURE4_COMPONENT_TICKS],
            "applies_to": ["boundary_rms", "product_rms", "interaction_rms"],
            "scale_type": "common absolute RMS scale",
        },
        "figure4_visual_encoding": {
            "panel_a_cmap_endpoints": ["#EDF2FB", "#3F67C6"],
            "panel_b_cmap_endpoints": ["#F3DCE5", "#B82E6B"],
            "panel_c_cmap_endpoints": ["#F0EAF6", "#BBA2D7"],
            "panel_d_category_colors": {
                "Boundary": FIG4_DOMINANT_COLORS["boundary"],
                "Product": FIG4_DOMINANT_COLORS["product"],
                "Interaction": FIG4_DOMINANT_COLORS["interaction"],
            },
            "base_fill": FIG3_PALE,
            "province_edge": PROVINCE_EDGE,
            "national_edge": NATIONAL_EDGE,
            "main_map_frame": "hidden",
            "inset_frame": "complete rectangle",
        },
        "dominant_source_counts": {str(key): int(value) for key, value in dominant_counts.items()},
        "dominant_source_tie_count": int(dominant_counts.get("tie", 0)),
        "component_colors": COMPONENT_COLORS,
        "product_label_definition": "Product denotes the land-cover specification component.",
        "source_sha256": {rel(path): sha256(path) for path in source_paths},
        "outputs": [{"path": rel(path), "sha256": sha256(path), "bytes": path.stat().st_size}
                    for path in output_paths],
        "qa_status": "pass_automated_checks_human_gate_r9_4_pending",
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                             encoding="utf-8")
    print(json.dumps({"status": manifest["status"], "figure3_outputs": len(outputs3),
                      "figure4_outputs": len(outputs4),
                      "dominant_source_counts": manifest["dominant_source_counts"]},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
