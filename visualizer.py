"""
Visualization functions for the Multi-Well Viewer application.

Builds matplotlib figures for:
- Multi-well cross-section with configurable log tracks
- Formation tops overlay
- Flattened section
"""

from __future__ import annotations

import re
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

plt.rcParams["font.family"] = "Times New Roman"


# ---------------------------------------------------------------------------
# Colour maps and defaults
# ---------------------------------------------------------------------------

CURVE_STYLES: dict[str, dict] = {
    "GR":   {"color": "#4caf50", "lw": 1.2, "fill": True, "fill_alpha": 0.15},
    "RHOB": {"color": "#e53935", "lw": 1.2, "fill": False},
    "NPHI": {"color": "#1e88e5", "lw": 1.2, "fill": False},
    "RT":   {"color": "#8e24aa", "lw": 1.2, "fill": False, "log_scale": True},
    "ILD":  {"color": "#8e24aa", "lw": 1.2, "fill": False, "log_scale": True},
    "LLD":  {"color": "#7b1fa2", "lw": 1.2, "fill": False, "log_scale": True},
    "DT":   {"color": "#fb8c00", "lw": 1.2, "fill": False},
    "PHIE": {"color": "#00acc1", "lw": 1.2, "fill": False},
    "SW":   {"color": "#3949ab", "lw": 1.2, "fill": False},
    "VSH":  {"color": "#8d6e63", "lw": 1.2, "fill": True, "fill_alpha": 0.2},
}

_DEFAULT_STYLE = {"color": "#546e7a", "lw": 1.0, "fill": False}


def _get_style(curve_name: str) -> dict:
    upper = curve_name.upper()
    for key, style in CURVE_STYLES.items():
        if key in upper:
            return style
    return _DEFAULT_STYLE


def _normalize_well_name(name: str) -> str:
    """Return a canonical key for matching well names across files."""
    return re.sub(r"[^a-z0-9]+", "", str(name).strip().lower())


# ---------------------------------------------------------------------------
# Main cross-section builder
# ---------------------------------------------------------------------------

def plot_cross_section(
    wells: dict,
    curves_to_plot: list[str],
    depth_min: float,
    depth_max: float,
    formation_tops: Optional[pd.DataFrame] = None,
    show_tops: bool = True,
    shade_formations: bool = False,
    formation_shade_alpha: float = 0.14,
    smoothing_window: Optional[int] = None,
    figsize_per_well: tuple[float, float] = (3.0, 9.0),
    depth_label: str = "Depth (m)",
    title: str = "Multi-Well Display",
) -> plt.Figure:
    """Build and return a matplotlib Figure showing a multi-well cross-section.

    Parameters
    ----------
    wells : dict
        Well data dict (name → data) as produced by
        :func:`data_loader.load_las_file` (or the flattened version).
    curves_to_plot : list[str]
        Ordered list of curve mnemonics to display. Each mnemonic gets its
        own track column per well.
    depth_min, depth_max : float
        Depth window to display.
    formation_tops : pd.DataFrame or None
        Formation tops table to overlay.  Pass ``None`` to skip.
    show_tops : bool
        Whether to draw formation top markers.
    shade_formations : bool
        Whether to shade intervals between consecutive formation tops.
    formation_shade_alpha : float
        Transparency used for formation interval shading.
    smoothing_window : int or None
        Optional moving-average window (samples) for display-time smoothing.
    figsize_per_well : (float, float)
        (width, height) in inches per well × track combination.
    depth_label : str
        Label for the Y-axis.
    title : str
        Figure title.

    Returns
    -------
    matplotlib.figure.Figure
    """
    n_wells = len(wells)
    n_curves = len(curves_to_plot)

    if n_wells == 0 or n_curves == 0:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No data to display", ha="center", va="center")
        return fig

    n_cols = n_wells * n_curves
    fig_w = figsize_per_well[0] * n_cols
    fig_h = figsize_per_well[1]

    fig, axes = plt.subplots(
        1, n_cols,
        figsize=(max(fig_w, 6), fig_h),
        sharey=True,
    )
    if n_cols == 1:
        axes = [axes]

    fig.suptitle(title, fontsize=15, fontweight="bold", y=1.01)

    well_names = list(wells.keys())
    tops_colors: dict[str, str] = {}
    tops_for_plot = formation_tops
    if formation_tops is not None and not formation_tops.empty:
        tops_for_plot = formation_tops.copy()
        tops_for_plot["_well_key"] = tops_for_plot["well"].map(_normalize_well_name)
        for _, row in formation_tops.iterrows():
            tops_colors[row["formation"]] = row["color"]

    for wi, well_name in enumerate(well_names):
        well = wells[well_name]
        df = well["df"]
        depth_col = df.columns[0]
        well_key = _normalize_well_name(well_name)

        mask = (df[depth_col] >= depth_min) & (df[depth_col] <= depth_max)
        df_win = df[mask]

        well_tops = pd.DataFrame()
        shade_intervals: list[tuple[float, float, str]] = []
        if show_tops and tops_for_plot is not None and not tops_for_plot.empty:
            well_tops = tops_for_plot[tops_for_plot["_well_key"] == well_key].copy()
            if not well_tops.empty:
                well_tops["depth"] = pd.to_numeric(well_tops["depth"], errors="coerce")
                well_tops = well_tops.dropna(subset=["depth"]).sort_values("depth")

                if shade_formations and len(well_tops) > 1:
                    top_rows = well_tops[["depth", "color"]].to_dict("records")
                    for idx in range(len(top_rows) - 1):
                        top_depth = float(top_rows[idx]["depth"])
                        base_depth = float(top_rows[idx + 1]["depth"])
                        if base_depth <= depth_min or top_depth >= depth_max:
                            continue
                        y0 = max(top_depth, depth_min)
                        y1 = min(base_depth, depth_max)
                        if y1 > y0:
                            shade_intervals.append((y0, y1, str(top_rows[idx]["color"])))

        for ci, curve in enumerate(curves_to_plot):
            col_idx = wi * n_curves + ci
            ax = axes[col_idx]

            if shade_intervals:
                for y0, y1, fill_color in shade_intervals:
                    ax.axhspan(y0, y1, color=fill_color, alpha=formation_shade_alpha, zorder=0)

            style = _get_style(curve)
            use_log = style.get("log_scale", False)

            if curve in df_win.columns:
                vals = df_win[curve].values.astype(float)
                depth = df_win[depth_col].values

                if smoothing_window is not None and int(smoothing_window) > 1:
                    vals = (
                        pd.Series(vals)
                        .rolling(window=int(smoothing_window), center=True, min_periods=1)
                        .mean()
                        .to_numpy()
                    )

                valid = ~np.isnan(vals)
                if valid.any():
                    if use_log:
                        pos = vals > 0
                        ax.semilogx(
                            np.where(pos, vals, np.nan),
                            depth,
                            color=style["color"],
                            linewidth=style["lw"],
                        )
                    else:
                        ax.plot(
                            vals, depth,
                            color=style["color"],
                            linewidth=style["lw"],
                        )
                        if style.get("fill"):
                            ax.fill_betweenx(
                                depth, vals, 0,
                                alpha=style.get("fill_alpha", 0.1),
                                color=style["color"],
                            )
            else:
                ax.text(
                    0.5, 0.5, f"{curve}\nN/A",
                    ha="center", va="center",
                    transform=ax.transAxes,
                    fontsize=9, color="grey",
                )

            # Formation tops overlay
            if show_tops and not well_tops.empty:
                for _, top_row in well_tops.iterrows():
                    td = top_row["depth"]
                    if depth_min <= td <= depth_max:
                        ax.axhline(
                            td,
                            color=top_row["color"],
                            linewidth=1.5,
                            linestyle="--",
                            zorder=3,
                        )
                        if ci == 0:
                            x_left = ax.get_xlim()[0]
                            ax.text(
                                x_left if x_left != 0 else 0,
                                td,
                                f" {top_row['formation']}",
                                fontsize=8,
                                va="bottom",
                                color=top_row["color"],
                                clip_on=True,
                            )

            # Axis decoration
            unit = well.get("units", {}).get(curve, "")
            header = f"{curve}"
            if unit:
                header += f"\n({unit})"
            ax.set_title(header, fontsize=10, pad=4)
            ax.tick_params(axis="x", labelsize=9, rotation=45)
            ax.tick_params(axis="y", labelsize=9)
            ax.invert_yaxis()

            if col_idx == 0:
                ax.set_ylabel(depth_label, fontsize=11)

            if ci == 0:
                ax.set_xlabel(well_name, fontsize=10, labelpad=4)

    # Legend for formation tops
    if show_tops and tops_colors:
        patches = [
            mpatches.Patch(color=c, label=f)
            for f, c in tops_colors.items()
        ]
        fig.legend(
            handles=patches,
            loc="lower center",
            ncol=min(len(patches), 6),
            fontsize=9,
            title="Formation Tops",
            bbox_to_anchor=(0.5, -0.04),
        )

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Depth histogram / statistics helper
# ---------------------------------------------------------------------------

def plot_curve_histogram(
    wells: dict,
    curve: str,
    depth_min: float,
    depth_max: float,
    bins: int = 40,
) -> plt.Figure:
    """Return a histogram figure for *curve* across all wells in the window."""
    fig, ax = plt.subplots(figsize=(5, 4))

    for well_name, well in wells.items():
        df = well["df"]
        depth_col = df.columns[0]
        mask = (df[depth_col] >= depth_min) & (df[depth_col] <= depth_max)
        if curve in df.columns:
            vals = df.loc[mask, curve].dropna().values
            if len(vals):
                ax.hist(vals, bins=bins, alpha=0.5, label=well_name)

    ax.set_xlabel(curve, fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title(f"Distribution of {curve}", fontsize=13)
    ax.tick_params(axis="both", labelsize=9)
    ax.legend(fontsize=9)
    plt.tight_layout()
    return fig
