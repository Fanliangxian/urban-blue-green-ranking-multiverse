"""Shared publication styling for the Urban Blue–Green Ranking figures.

The figure-specific scripts retain their own data, geometry and scientific
encodings, but import this module for the common physical width, typography,
axis treatment, line weights and panel-title convention.
"""

from __future__ import annotations


FIGURE_WIDTH_MM = 180
DPI = 600

# These sizes are calibrated for a 180-mm final figure and are intentionally
# larger than the earlier 5–7 pt settings so labels remain readable at print
# scale.  DejaVu Sans is the deterministic fallback when Arial is unavailable.
FONT_SIZE = 8.2
PANEL_TITLE_SIZE = 9.2
AXIS_LABEL_SIZE = 8.0
TICK_LABEL_SIZE = 7.2
LEGEND_SIZE = 7.0
ANNOTATION_SIZE = 6.8
PANEL_LABEL_SIZE = 10.0
SUBPANEL_TITLE_SIZE = 7.8
TITLE_PAD = 1.0

AXES_LINEWIDTH = 0.8
TICK_LINEWIDTH = 0.65
DATA_LINEWIDTH = 1.0
GRID_LINEWIDTH = 0.55
MAP_PROVINCE_LINEWIDTH = 0.32
MAP_NATIONAL_LINEWIDTH = 0.65
MAP_DASH_LINEWIDTH = 0.48

TEXT = "#17252E"
MUTED = "#59666D"
GRID = "#E6E6E6"
BASE_FILL = "#F5F6F7"
PROVINCE_EDGE = "#B7C0C5"
NATIONAL_EDGE = "#7A858B"
BLUE = "#3F67C6"
MAGENTA = "#B82E6B"
INTERACTION_GREY = "#7A7A7A"


def rcparams() -> dict[str, object]:
    """Return the Matplotlib rcParams shared by every active figure script."""

    return {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": FONT_SIZE,
        "axes.titlesize": PANEL_TITLE_SIZE,
        "axes.titleweight": "bold",
        "axes.labelsize": AXIS_LABEL_SIZE,
        "xtick.labelsize": TICK_LABEL_SIZE,
        "ytick.labelsize": TICK_LABEL_SIZE,
        "legend.fontsize": LEGEND_SIZE,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": AXES_LINEWIDTH,
        "xtick.major.width": TICK_LINEWIDTH,
        "ytick.major.width": TICK_LINEWIDTH,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    }


def apply_style(mpl) -> None:
    """Apply the shared style to an imported Matplotlib module."""

    mpl.rcParams.update(rcparams())


def panel_title(ax, label: str, title: str, *, pad: float = TITLE_PAD,
                x: float = 0.0, fontsize: float = PANEL_TITLE_SIZE) -> None:
    """Set a lower-case bold panel title with a consistent label prefix."""

    ax.set_title(
        f"{label}   {title}", loc="left", x=x,
        fontweight="bold", fontsize=fontsize, pad=pad,
    )


def figure_panel_title(fig, x: float, y: float, label: str, title: str,
                       *, ha: str = "left", va: str = "top",
                       fontsize: float = PANEL_TITLE_SIZE) -> None:
    """Place a panel title in figure coordinates for custom map layouts."""

    fig.text(
        x, y, f"{label}   {title}",
        fontsize=fontsize, fontweight="bold", ha=ha, va=va,
        color=TEXT,
    )
