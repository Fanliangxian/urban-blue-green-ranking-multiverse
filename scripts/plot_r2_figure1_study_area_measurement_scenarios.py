"""Plot the revised main Figure 1 from audited R3 and GEE source data."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MPL_CACHE = ROOT / "work" / "mplconfig_r2_figure1"
GDAL_PROXY = ROOT / "work" / "r4_gdal_proxy"
MPL_CACHE.mkdir(parents=True, exist_ok=True)
GDAL_PROXY.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE))
os.environ.setdefault("GDAL_PAM_PROXY_DIR", str(GDAL_PROXY))
os.environ.setdefault("PROJ_LIB", os.environ.get(
    "UBGR_PROJ_LIB", r"C:\Program Files\ArcGIS\Pro\Resources\pedata\gdaldata"
))

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Polygon as MplPolygon
from PIL import Image
from osgeo import gdal, ogr, osr
from unified_figure_style import (
    ANNOTATION_SIZE,
    AXIS_LABEL_SIZE,
    BASE_FILL,
    DPI as SHARED_DPI,
    FIGURE_WIDTH_MM as SHARED_FIGURE_WIDTH_MM,
    LEGEND_SIZE,
    MAP_DASH_LINEWIDTH,
    MAP_PROVINCE_LINEWIDTH,
    PANEL_TITLE_SIZE,
    SUBPANEL_TITLE_SIZE,
    TICK_LABEL_SIZE,
    TITLE_PAD,
    apply_style,
)


SOURCE = ROOT / "data" / "derived" / "figure_source_data"
LANDCOVER = SOURCE / "figure1_landcover_example"
STATUS = SOURCE / "R2_figure1_city_status.csv"
REPRESENTATIVE = SOURCE / "R2_figure1_representative_city.json"
BOUNDARIES = ROOT / "data" / "derived" / "measurement_boundaries_r1.gpkg"
WC = LANDCOVER / "R2_figure1_worldcover2021_shangluo.tif"
DW = LANDCOVER / "R2_figure1_dynamicworld2021_bluegreen_probability_shangluo.tif"
GEE_MANIFEST = LANDCOVER / "R2_figure1_landcover_example_manifest.csv"
S1_MANIFEST = ROOT / "data" / "derived" / "figures" / "r2" / "R2_figure_s1_workflow_manifest.json"
OUT = ROOT / "data" / "derived" / "figures" / "r2"
STEM = OUT / "Figure1_study_area_measurement_scenarios"
COMBINED_MANIFEST = OUT / "R2_figure1_manifest.json"

# The official administrative source is intentionally supplied by the user at
# runtime and is not redistributed with the release package.  The public
# README documents UBGR_ADMIN_2024_ROOT; the repository-relative fallback is
# useful for a portable local layout.
OFFICIAL_ROOT = Path(os.environ.get(
    "UBGR_ADMIN_2024_ROOT", str(ROOT / "data" / "reference" / "admin2024")
))
PROVINCES = OFFICIAL_ROOT / "省.shp"
TEN_DASH = OFFICIAL_ROOT / "十段线.shp"
MAP_APPROVAL_NUMBER = "GS (2024) 0650"

WIDTH_MM = SHARED_FIGURE_WIDTH_MM
HEIGHT_MM = 132
DPI = SHARED_DPI
WHITE, TEXT, MUTED = "#FFFFFF", "#17252E", "#59666D"
OUTLINE, PALE = "#4C5A61", BASE_FILL
STATUS_COLORS = {"complete_8": "#3F67C6", "partial_1_to_7": "#B82E6B", "unrepresented_0": "#111111"}
BOUNDARY_COLORS = {
    "gctb_core_2021": "#0072B2", "gctb_system_2021": "#56B4E9",
    "gub_2018": "#009E73", "ghs_uc_2020": "#CC79A7",
}
BOUNDARY_LABELS = {
    "gctb_core_2021": "GCTB core (2021)",
    "gctb_system_2021": "GCTB system (2021)",
    "gub_2018": "GUB (2018)", "ghs_uc_2020": "GHS-UC (2020)",
}
BOUNDARY_LEVELS = ("gctb_core_2021", "gctb_system_2021", "gub_2018", "ghs_uc_2020")
WC_COLORS = ["#0072B2", "#009E73", "#6B6B6B", "#E69F00", "#D9D9D9"]
WC_LABELS = ["Blue", "Green", "Grey", "Farm", "Other"]

CAPTION = (
    "Figure 1 | Study population and the frozen measurement multiverse. "
    "a, Locations and confirmatory-map coverage of 296 registered Chinese cities; "
    "Haidong is outlined because it has complete map coverage but a structurally missing "
    "covariate in the association models. b, Four equal-extent urban-boundary definitions "
    "for Shangluo. Shangluo was selected using a prespecified outcome-blind display rule "
    "based on geometry and boundary typicality; rank outcomes were not used. GCTB core and system "
    "share a data lineage. c, Co-registered land-cover measurement specifications. WorldCover is a categorical "
    "native-label product summarized by pixel area, whereas Dynamic World shows the "
    "continuous 2021 annual-mean blue-green probability contribution used for probability "
    "area. The two specifications differ in both source product and estimator architecture; "
    "figures shorten this axis to the product component. The panels therefore do not show "
    "equivalent class maps. Administrative base map: National Geomatics Center of China, map approval "
    "number GS (2024) 0650."
)

ALT_TEXT = (
    "Three-part map figure. Panel a maps 296 Chinese cities, distinguishing 285 with all "
    "eight map settings, nine with partial coverage and two unrepresented cities; Haidong "
    "has a separate outline. Panel b shows four separate urban-boundary shapes for Shangluo "
    "at one extent. Panel c compares categorical WorldCover classes with a continuous "
    "Dynamic World annual-mean blue-green probability surface."
)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rel(path):
    return path.relative_to(ROOT).as_posix()


def spatial_ref_epsg(epsg):
    ref = osr.SpatialReference()
    ref.ImportFromEPSG(epsg)
    ref.SetDataAxisToSRSAxisMapping([1, 2])
    return ref


def transformed_geometry(geometry, source_ref, target_ref):
    source = source_ref.Clone()
    source.SetDataAxisToSRSAxisMapping([2, 1])
    target = target_ref.Clone()
    target.SetDataAxisToSRSAxisMapping([1, 2])
    transformation = osr.CoordinateTransformation(source, target)
    clone = geometry.Clone()

    def transform_points(item):
        if item.GetGeometryCount():
            for index in range(item.GetGeometryCount()):
                transform_points(item.GetGeometryRef(index))
            return
        for index in range(item.GetPointCount()):
            x, y, z = item.GetPoint(index)
            easting, northing, height = transformation.TransformPoint(x, y, z)
            item.SetPoint(index, easting, northing, height)

    # Transform point-by-point because ArcGIS Pro's bundled GDAL applies feature-
    # level axis mappings inconsistently in Geometry.Transform(). Explicit source
    # and target data-axis mappings make TransformPoint deterministic.
    transform_points(clone)
    clone.AssignSpatialReference(target)
    return clone


def polygons(geometry):
    flat = ogr.GT_Flatten(geometry.GetGeometryType())
    if flat == ogr.wkbPolygon:
        yield geometry
    elif flat in (ogr.wkbMultiPolygon, ogr.wkbGeometryCollection):
        for index in range(geometry.GetGeometryCount()):
            yield from polygons(geometry.GetGeometryRef(index))


def lines(geometry):
    flat = ogr.GT_Flatten(geometry.GetGeometryType())
    if flat in (ogr.wkbLineString, ogr.wkbLinearRing):
        yield geometry
    elif geometry.GetGeometryCount():
        for index in range(geometry.GetGeometryCount()):
            yield from lines(geometry.GetGeometryRef(index))


def draw_polygon_geometry(ax, geometry, facecolor, edgecolor, linewidth=0.4, alpha=1.0, zorder=1):
    for polygon in polygons(geometry):
        ring = polygon.GetGeometryRef(0)
        coordinates = [(ring.GetX(i), ring.GetY(i)) for i in range(ring.GetPointCount())]
        if len(coordinates) >= 3:
            ax.add_patch(MplPolygon(coordinates, closed=True, facecolor=facecolor,
                                    edgecolor=edgecolor, linewidth=linewidth,
                                    alpha=alpha, zorder=zorder))


def draw_line_geometry(ax, geometry, color, linewidth=0.6, zorder=2):
    for line in lines(geometry):
        coordinates = [(line.GetX(i), line.GetY(i)) for i in range(line.GetPointCount())]
        if coordinates:
            xs, ys = zip(*coordinates)
            ax.plot(xs, ys, color=color, linewidth=linewidth, zorder=zorder)


def projected_rectangle_bounds(lon_min, lon_max, lat_min, lat_max, target_ref):
    wgs84 = spatial_ref_epsg(4326)
    wgs84.SetDataAxisToSRSAxisMapping([2, 1])
    target = target_ref.Clone()
    target.SetDataAxisToSRSAxisMapping([1, 2])
    transform = osr.CoordinateTransformation(wgs84, target)
    points = []
    for lon in np.linspace(lon_min, lon_max, 80):
        points.extend([transform.TransformPoint(float(lon), lat_min),
                       transform.TransformPoint(float(lon), lat_max)])
    for lat in np.linspace(lat_min, lat_max, 80):
        points.extend([transform.TransformPoint(lon_min, float(lat)),
                       transform.TransformPoint(lon_max, float(lat))])
    xs, ys = [p[0] for p in points], [p[1] for p in points]
    return min(xs), max(xs), min(ys), max(ys)


def load_official_features(target_ref):
    province_geometries = []
    dataset = ogr.Open(str(PROVINCES), 0)
    layer = dataset.GetLayer(0)
    source_ref = layer.GetSpatialRef()
    for feature in layer:
        province_geometries.append(transformed_geometry(feature.GetGeometryRef(), source_ref, target_ref))
    dataset = None
    dash_geometries = []
    dataset = ogr.Open(str(TEN_DASH), 0)
    layer = dataset.GetLayer(0)
    source_ref = layer.GetSpatialRef()
    for feature in layer:
        dash_geometries.append(transformed_geometry(feature.GetGeometryRef(), source_ref, target_ref))
    dataset = None
    return province_geometries, dash_geometries


def raster(path):
    dataset = gdal.Open(str(path), gdal.GA_ReadOnly)
    if dataset is None:
        raise RuntimeError(f"GDAL could not open {path}")
    array = dataset.GetRasterBand(1).ReadAsArray()
    nodata = dataset.GetRasterBand(1).GetNoDataValue()
    transform = dataset.GetGeoTransform()
    projection = dataset.GetProjection()
    width, height = dataset.RasterXSize, dataset.RasterYSize
    extent = [transform[0], transform[0] + width * transform[1],
              transform[3] + height * transform[5], transform[3]]
    metadata = {"width": width, "height": height, "geotransform": list(transform),
                "projection_wkt": projection, "nodata": nodata}
    dataset = None
    return array, extent, metadata


def validate_inputs():
    required = [STATUS, REPRESENTATIVE, BOUNDARIES, WC, DW, GEE_MANIFEST,
                PROVINCES, TEN_DASH, S1_MANIFEST]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing Figure 1 inputs:\n" + "\n".join(missing))
    status = pd.read_csv(STATUS)
    if status.map_status.value_counts().to_dict() != {
        "complete_8": 285, "partial_1_to_7": 9, "unrepresented_0": 2}:
        raise ValueError("Figure 1 map-status counts changed")
    representative = json.loads(REPRESENTATIVE.read_text(encoding="utf-8"))
    if representative["selected_city_id"] != "CN-611000" or representative["selection_uses_outcomes"]:
        raise ValueError("Representative-city audit changed")
    gee = pd.read_csv(GEE_MANIFEST)
    if len(gee) != 1 or gee.city_id.item() != "CN-611000":
        raise ValueError("GEE manifest does not identify Shangluo")
    wc, wc_extent, wc_meta = raster(WC)
    dw, dw_extent, dw_meta = raster(DW)
    if wc.shape != dw.shape or not np.allclose(wc_extent, dw_extent, atol=0.01):
        raise ValueError("The two land-cover rasters are not co-registered")
    if not np.isclose(abs(wc_meta["geotransform"][1]), 10, atol=0.01):
        raise ValueError("WorldCover export is not on the frozen 10 m grid")
    valid_dw = dw[np.isfinite(dw) & (dw > -9990)]
    if valid_dw.size == 0 or valid_dw.min() < -1e-6 or valid_dw.max() > 1 + 1e-6:
        raise ValueError("Dynamic World probability is outside 0..1")
    boundary_geometries = load_representative_boundaries(spatial_ref_epsg(32649))
    union = list(boundary_geometries.values())[0].Clone()
    for geometry in list(boundary_geometries.values())[1:]:
        union = union.Union(geometry)
    expected = union.Buffer(5000).GetEnvelope()
    expected_extent = [expected[0], expected[1], expected[2], expected[3]]
    extent_edge_difference_m = [wc_extent[i] - expected_extent[i] for i in range(4)]
    if max(abs(value) for value in extent_edge_difference_m) > 30:
        raise ValueError("GEE export extent does not match the frozen local boundaries within 30 m")
    raster_rectangle_area = (wc_extent[1] - wc_extent[0]) * (wc_extent[3] - wc_extent[2])
    gee_area = float(gee.export_region_area_m2.item())
    if abs(raster_rectangle_area - gee_area) / gee_area > 0.001:
        raise ValueError("GEE manifest area and realized raster rectangle differ by more than 0.1%")
    extent_audit = {
        "local_buffered_boundary_extent_epsg32649": expected_extent,
        "realized_raster_extent_epsg32649": wc_extent,
        "raster_minus_local_extent_edge_m": extent_edge_difference_m,
        "raster_rectangle_area_m2": raster_rectangle_area,
        "gee_manifest_region_area_m2": gee_area,
        "boundary_asset_recorded_by_gee": gee.boundary_asset.item(),
        "frozen_local_boundary_source": rel(BOUNDARIES),
        "interpretation": "asset ID differs from the protocol ID, but the realized extent matches the hashed local frozen geometries within one to three 10 m pixels",
    }
    return status, representative, gee, wc, dw, wc_extent, wc_meta, dw_meta, extent_audit


def draw_panel_a(ax, status):
    aea = osr.SpatialReference()
    dataset = ogr.Open(str(BOUNDARIES), 0)
    layer = dataset.GetLayerByName("measurement_boundaries_aea")
    aea.ImportFromWkt(layer.GetSpatialRef().ExportToWkt())
    aea.SetDataAxisToSRSAxisMapping([1, 2])
    dataset = None
    provinces, dashes = load_official_features(aea)
    for geometry in provinces:
        draw_polygon_geometry(ax, geometry, PALE, "#B7C0C5", linewidth=MAP_PROVINCE_LINEWIDTH, zorder=0)
    for map_status, marker, size in [
        ("complete_8", "o", 6), ("partial_1_to_7", "^", 18), ("unrepresented_0", "X", 24)]:
        part = status.loc[status.map_status.eq(map_status)]
        ax.scatter(part.map_x_aea_m, part.map_y_aea_m, s=size, marker=marker,
                   facecolor=STATUS_COLORS[map_status], edgecolor=WHITE if marker != "X" else TEXT,
                   linewidth=MAP_DASH_LINEWIDTH, zorder=4)
    haidong = status.loc[status.city_id.eq("CN-630200")].iloc[0]
    ax.scatter([haidong.map_x_aea_m], [haidong.map_y_aea_m], s=14, marker="o",
               facecolor="none", edgecolor="#CC79A7", linewidth=MAP_DASH_LINEWIDTH, zorder=5)
    xmin, xmax, ymin, ymax = projected_rectangle_bounds(72, 136, 17, 55, aea)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")
    ax.axis("off")
    handles = [
        Line2D([], [], marker="o", linestyle="", markerfacecolor=STATUS_COLORS["complete_8"],
               markeredgecolor=WHITE, markersize=4.2, label="All 8 settings (285)"),
        Line2D([], [], marker="^", linestyle="", markerfacecolor=STATUS_COLORS["partial_1_to_7"],
               markeredgecolor=WHITE, markersize=5.0, label="Partial (9)"),
        Line2D([], [], marker="X", linestyle="", color=STATUS_COLORS["unrepresented_0"],
               markersize=4.8, label="Unrepresented (2)"),
        Line2D([], [], marker="o", linestyle="", markerfacecolor="none", markeredgecolor="#CC79A7",
               markeredgewidth=0.7, markersize=4.5, label="Haidong: model covariate missing"),
    ]
    ax.legend(handles=handles, loc="lower left", frameon=False, fontsize=LEGEND_SIZE,
              handletextpad=0.4, borderaxespad=0.2)
    inset = ax.inset_axes([0.77, 0.015, 0.21, 0.25])
    for geometry in provinces:
        draw_polygon_geometry(inset, geometry, PALE, "#B7C0C5", linewidth=MAP_PROVINCE_LINEWIDTH, zorder=0)
    for geometry in dashes:
        draw_line_geometry(inset, geometry, OUTLINE, linewidth=MAP_DASH_LINEWIDTH)
    ixmin, ixmax, iymin, iymax = projected_rectangle_bounds(105, 125, 3, 25, aea)
    inset.set_xlim(ixmin, ixmax)
    inset.set_ylim(iymin, iymax)
    inset.set_aspect("equal")
    inset.set_xticks([]); inset.set_yticks([])
    for spine in inset.spines.values():
        spine.set_color("#9AA5AA"); spine.set_linewidth(MAP_DASH_LINEWIDTH)
    ax.text(0.01, 0.975, f"Map approval no. {MAP_APPROVAL_NUMBER}", transform=ax.transAxes,
            fontsize=ANNOTATION_SIZE, color=MUTED, va="top")


def load_representative_boundaries(target_ref):
    dataset = ogr.Open(str(BOUNDARIES), 0)
    layer = dataset.GetLayerByName("measurement_boundaries_aea")
    source_ref = layer.GetSpatialRef()
    records = {}
    for feature in layer:
        if (feature.GetField("city_id") == "CN-611000" and
                feature.GetField("geom_var") == "raw_product_geometry"):
            scenario = feature.GetField("bnd_scn")
            records[scenario] = transformed_geometry(feature.GetGeometryRef(), source_ref, target_ref)
    dataset = None
    if set(records) != set(BOUNDARY_LABELS):
        raise ValueError(f"Expected four representative boundaries, got {sorted(records)}")
    return records


def worldcover_five_classes(raw):
    result = np.full(raw.shape, np.nan)
    mapping = {
        80: 0, 10: 1, 20: 1, 30: 1, 90: 1, 95: 1,
        50: 2, 40: 3, 60: 4, 70: 4, 100: 4,
    }
    for original, target in mapping.items():
        result[raw == original] = target
    return result


def build_figure(status, wc, dw, extent):
    apply_style(mpl)
    fig = plt.figure(figsize=(WIDTH_MM / 25.4, HEIGHT_MM / 25.4), constrained_layout=True)
    outer = fig.add_gridspec(2, 2, width_ratios=[0.88, 1.12], height_ratios=[1.05, 0.95])
    ax_a = fig.add_subplot(outer[0, 0])
    draw_panel_a(ax_a, status)

    bgrid = outer[0, 1].subgridspec(2, 2, wspace=-0.32, hspace=0.02)
    target = spatial_ref_epsg(32649)
    boundaries = load_representative_boundaries(target)
    boundary_union = list(boundaries.values())[0].Clone()
    for geometry in list(boundaries.values())[1:]:
        boundary_union = boundary_union.Union(geometry)
    bxmin, bxmax, bymin, bymax = boundary_union.GetEnvelope()
    # Use one shared, modestly padded envelope for all four maps.  This zooms
    # the boundary content while preserving a like-for-like spatial frame.
    bpad_x = 0.08 * (bxmax - bxmin)
    bpad_y = 0.08 * (bymax - bymin)
    boundary_extent = [bxmin - bpad_x, bxmax + bpad_x,
                       bymin - bpad_y, bymax + bpad_y]
    b_axes = []
    for index, scenario in enumerate(BOUNDARY_LEVELS):
        ax = fig.add_subplot(bgrid[index // 2, index % 2])
        b_axes.append(ax)
        draw_polygon_geometry(ax, boundaries[scenario], BOUNDARY_COLORS[scenario], TEXT,
                              linewidth=MAP_DASH_LINEWIDTH, alpha=0.78)
        ax.set_xlim(boundary_extent[0], boundary_extent[1])
        ax.set_ylim(boundary_extent[2], boundary_extent[3])
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("#9AA5AA"); spine.set_linewidth(MAP_DASH_LINEWIDTH)
        ax.set_title(BOUNDARY_LABELS[scenario], pad=TITLE_PAD, fontsize=SUBPANEL_TITLE_SIZE)
        if index in (0, 1):
            ax.text(0.5, 0.96, "shared GCTB lineage", transform=ax.transAxes,
                    ha="center", va="top", fontsize=ANNOTATION_SIZE, color=MUTED)
    cgrid = outer[1, :].subgridspec(1, 2, wspace=0.10)
    ax_wc, ax_dw = fig.add_subplot(cgrid[0, 0]), fig.add_subplot(cgrid[0, 1])
    wc_five = np.ma.masked_invalid(worldcover_five_classes(wc))
    wc_cmap = ListedColormap(WC_COLORS)
    wc_cmap.set_bad("#FFFFFF")
    ax_wc.imshow(wc_five, extent=extent, origin="upper", interpolation="nearest",
                 cmap=wc_cmap, norm=BoundaryNorm(np.arange(-0.5, 5.5), 5))
    ax_wc.set_title("WorldCover 2021 · categorical native labels", pad=TITLE_PAD,
                    fontsize=SUBPANEL_TITLE_SIZE)
    legend = [Patch(facecolor=color, edgecolor="none", label=label)
              for color, label in zip(WC_COLORS, WC_LABELS)]
    ax_wc.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.5, -0.065),
                 ncol=5, frameon=False, fontsize=LEGEND_SIZE, handlelength=1.0, columnspacing=0.7)

    dw_mask = np.ma.masked_where(~np.isfinite(dw) | (dw < -100), dw)
    probability_cmap = mpl.cm.get_cmap("GnBu").copy()
    probability_cmap.set_bad("#D9D9D9")
    image = ax_dw.imshow(dw_mask, extent=extent, origin="upper", interpolation="nearest",
                         cmap=probability_cmap, norm=Normalize(0, 1))
    ax_dw.set_title("Dynamic World 2021 · annual-mean blue–green probability", pad=TITLE_PAD,
                    fontsize=SUBPANEL_TITLE_SIZE)
    colorbar_ax = ax_dw.inset_axes([0.0, -0.10, 1.0, 0.035])
    colorbar = fig.colorbar(image, cax=colorbar_ax, orientation="horizontal")
    colorbar.set_label("Blue–green probability contribution (0–1)", fontsize=AXIS_LABEL_SIZE)
    colorbar.ax.tick_params(labelsize=TICK_LABEL_SIZE, length=2)
    for ax in (ax_wc, ax_dw):
        ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3])
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("#9AA5AA"); spine.set_linewidth(MAP_DASH_LINEWIDTH)
    # Freeze the constrained-layout solution before applying the final optical
    # alignment.  The boundary maps are scaled as one compact 2 x 2 group so
    # their internal geometry stays unchanged while the panel gains clear space
    # above panel c.  The group's right edge is then aligned to the right
    # land-cover map.
    fig.canvas.draw()
    # Align panel a's left edge with the left edge of panel c's left map.
    # Move the inset together with the main panel so their relative geometry
    # remains unchanged; the figure-level panel-a title is positioned later
    # from ax_a's updated x-coordinate.
    a_position = ax_a.get_position()
    c_left_edge = ax_wc.get_position().x0
    a_shift_right = c_left_edge - a_position.x0
    if abs(a_shift_right) > 1e-6:
        ax_a.set_position([a_position.x0 + a_shift_right, a_position.y0,
                           a_position.width, a_position.height])
        for child in ax_a.child_axes:
            child_position = child.get_position()
            child.set_position([child_position.x0 + a_shift_right,
                                child_position.y0,
                                child_position.width, child_position.height])

    b_left = min(ax.get_position().x0 for ax in b_axes)
    b_right = max(ax.get_position().x1 for ax in b_axes)
    b_bottom = min(ax.get_position().y0 for ax in b_axes)
    b_top = max(ax.get_position().y1 for ax in b_axes)
    b_center_x = (b_left + b_right) / 2
    b_center_y = (b_bottom + b_top) / 2
    b_half_width = (b_right - b_left) / 2
    b_half_height = (b_top - b_bottom) / 2
    b_scale = 0.78
    target_right = ax_dw.get_position().x1
    compact_center_x = target_right - b_scale * b_half_width
    compact_center_y = b_top - b_scale * b_half_height
    for ax in b_axes:
        position = ax.get_position()
        old_center_x = (position.x0 + position.x1) / 2
        old_center_y = (position.y0 + position.y1) / 2
        new_center_x = compact_center_x + b_scale * (old_center_x - b_center_x)
        new_center_y = compact_center_y + b_scale * (old_center_y - b_center_y)
        new_width = position.width * b_scale
        new_height = position.height * b_scale
        ax.set_position([new_center_x - new_width / 2,
                         new_center_y - new_height / 2,
                         new_width, new_height])

    top_row_shift = -0.040
    for ax in [ax_a, *b_axes]:
        position = ax.get_position()
        ax.set_position([position.x0, position.y0 + top_row_shift,
                         position.width, position.height])
        ax.set_in_layout(False)

    # Lift panel a until the South China Sea inset's lower frame is exactly
    # level with the lower row of panel b.
    a_inset = ax_a.child_axes[0]
    b_lower_edge = min(ax.get_position().y0 for ax in b_axes[2:4])
    # Add a small optical lift to panel a while leaving its figure-level title
    # fixed at ``top_y`` below.  Move the inset with the map so the panel stays
    # internally aligned.
    a_lift = b_lower_edge - a_inset.get_position().y0 + 0.033
    if abs(a_lift) > 1e-6:
        a_position = ax_a.get_position()
        ax_a.set_position([a_position.x0, a_position.y0 + a_lift,
                           a_position.width, a_position.height])
        inset_position = a_inset.get_position()
        a_inset.set_position([inset_position.x0, inset_position.y0 + a_lift,
                              inset_position.width, inset_position.height])

    # Move the lower panel as one unit, including the Dynamic World colorbar.
    c_shift_up = 0.012
    for ax in (ax_wc, ax_dw):
        position = ax.get_position()
        ax.set_position([position.x0, position.y0 + c_shift_up,
                         position.width, position.height])
        ax.set_in_layout(False)
        for child in ax.child_axes:
            child_position = child.get_position()
            child.set_position([child_position.x0, child_position.y0 + c_shift_up,
                                child_position.width, child_position.height])

    fig.set_constrained_layout(False)
    top_y = 0.992
    fig.text(ax_a.get_position().x0, top_y, "a   Registered study population",
             fontsize=PANEL_TITLE_SIZE, fontweight="bold", ha="left", va="top")
    fig.text(min(ax.get_position().x0 for ax in b_axes), top_y,
             "b   Four boundary definitions · Shangluo",
             fontsize=PANEL_TITLE_SIZE, fontweight="bold", ha="left", va="top")

    c_left = min(ax_wc.get_position().x0, ax_dw.get_position().x0)
    c_right = max(ax_wc.get_position().x1, ax_dw.get_position().x1)
    c_title_y = min(0.992, max(ax_wc.get_position().y1, ax_dw.get_position().y1) + 0.030)
    fig.text((c_left + c_right) / 2, c_title_y,
             "c   Two land-cover representations",
             fontsize=PANEL_TITLE_SIZE, fontweight="bold", ha="center", va="bottom")
    return fig


def export(fig):
    OUT.mkdir(parents=True, exist_ok=True)
    outputs = {
        "pdf": STEM.with_suffix(".pdf"),
        "svg": STEM.with_suffix(".svg"),
        "png": STEM.with_suffix(".png"),
        "tiff": STEM.with_suffix(".tiff"),
    }
    fig.savefig(outputs["pdf"], bbox_inches=None)
    fig.savefig(outputs["svg"], bbox_inches=None)
    fig.savefig(outputs["png"], dpi=DPI, bbox_inches=None)
    fig.savefig(outputs["tiff"], dpi=DPI, bbox_inches=None,
                pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)
    with Image.open(outputs["png"]) as image:
        image.convert("RGB").save(outputs["png"], dpi=(DPI, DPI), optimize=True)
    with Image.open(outputs["tiff"]) as image:
        image.convert("RGB").save(outputs["tiff"], dpi=(DPI, DPI), compression="tiff_lzw")
    return outputs


def main():
    status, representative, gee, wc, dw, extent, wc_meta, dw_meta, extent_audit = validate_inputs()
    figure = build_figure(status, wc, dw, extent)
    main_outputs = export(figure)
    s1 = {suffix: OUT / f"Figure_S1_measurement_workflow.{suffix}"
          for suffix in ["pdf", "svg", "png", "tiff"]}
    if any(not path.exists() for path in s1.values()):
        raise FileNotFoundError("Run scripts/build_task7c_figure1.py before the combined manifest")
    all_outputs = list(main_outputs.values()) + list(s1.values())
    manifest = {
        "status": "pass_human_gate_r9_3_pending", "generated_on": "2026-08-22",
        "shared_style": {"module": "scripts/unified_figure_style.py", "version": "2026-08-28"},
        "representative_city_id": representative["selected_city_id"],
        "representative_city_name": representative["selected_city_name"],
        "representative_selection_uses_outcomes": False,
        "representative_selection_protocol": representative["protocol_decision"],
        "representative_eligible_city_count": representative["eligible_city_count"],
        "caption_detail_location": (
            "manuscript/supplement_r2.md#outcome-blind-selection-of-the-boundary-example"),
        "population_denominators": {
            "registered": 296, "at_least_one_setting": 294,
            "all_eight_settings": 285, "association_complete_cases": 284,
        },
        "main_figure_width_mm": WIDTH_MM, "main_figure_height_mm": HEIGHT_MM,
        "raster_dpi": DPI, "main_caption": CAPTION, "main_alt_text": ALT_TEXT,
        "categorical_probability_distinction": (
            "WorldCover categorical native labels versus Dynamic World continuous annual-mean "
            "blue-green probability contribution; neither is redrawn as the other estimator."),
        "map_source": "National Geomatics Center of China 2024 administrative data",
        "map_approval_number": MAP_APPROVAL_NUMBER,
        "source_hashes": [{"path": rel(path), "sha256": sha256(path)} for path in
                          [STATUS, REPRESENTATIVE, BOUNDARIES, WC, DW, GEE_MANIFEST,
                           Path(__file__).resolve(), ROOT / "scripts" / "build_task7c_figure1.py"]],
        "external_map_source_hashes": [
            {"path": str(PROVINCES), "sha256": sha256(PROVINCES)},
            {"path": str(TEN_DASH), "sha256": sha256(TEN_DASH)},
        ],
        "gee_manifest": gee.iloc[0].to_dict(),
        "raster_audit": {"worldcover": wc_meta, "dynamic_world": dw_meta},
        "export_extent_lineage_audit": extent_audit,
        "outputs": [{"path": rel(path), "sha256": sha256(path), "bytes": path.stat().st_size}
                    for path in all_outputs],
    }
    temp = COMBINED_MANIFEST.with_suffix(".json.tmp")
    temp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(COMBINED_MANIFEST)
    print(json.dumps({"status": manifest["status"], "outputs": len(all_outputs),
                      "representative_city": representative["selected_city_name"]},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
