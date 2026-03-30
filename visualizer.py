"""
Visualization functions for the Multi-Well Viewer application.

Builds matplotlib figures for:
- Multi-well cross-section with configurable log tracks
- Formation tops overlay
- Flattened section
"""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd


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
    figsize_per_well: tuple[float, float] = (3.0, 9.0),
    depth_label: str = "Depth (m)",
    title: str = "Multi-Well Cross-Section",
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

    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.01)

    well_names = list(wells.keys())
    tops_colors: dict[str, str] = {}
    if formation_tops is not None and not formation_tops.empty:
        for _, row in formation_tops.iterrows():
            tops_colors[row["formation"]] = row["color"]

    for wi, well_name in enumerate(well_names):
        well = wells[well_name]
        df = well["df"]
        depth_col = df.columns[0]

        mask = (df[depth_col] >= depth_min) & (df[depth_col] <= depth_max)
        df_win = df[mask]

        for ci, curve in enumerate(curves_to_plot):
            col_idx = wi * n_curves + ci
            ax = axes[col_idx]

            style = _get_style(curve)
            use_log = style.get("log_scale", False)

            if curve in df_win.columns:
                vals = df_win[curve].values
                depth = df_win[depth_col].values

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
                    fontsize=8, color="grey",
                )

            # Formation tops overlay
            if show_tops and formation_tops is not None and not formation_tops.empty:
                well_tops = formation_tops[
                    formation_tops["well"].str.strip().str.lower()
                    == well_name.strip().lower()
                ]
                for _, top_row in well_tops.iterrows():
                    td = top_row["depth"]
                    if depth_min <= td <= depth_max:
                        ax.axhline(
                            td,
                            color=top_row["color"],
                            linewidth=1.5,
                            linestyle="--",
                        )
                        if ci == 0:
                            x_left = ax.get_xlim()[0]
                            ax.text(
                                x_left if x_left != 0 else 0,
                                td,
                                f" {top_row['formation']}",
                                fontsize=7,
                                va="bottom",
                                color=top_row["color"],
                                clip_on=True,
                            )

            # Axis decoration
            unit = well.get("units", {}).get(curve, "")
            header = f"{curve}"
            if unit:
                header += f"\n({unit})"
            ax.set_title(header, fontsize=8, pad=3)
            ax.tick_params(axis="x", labelsize=7, rotation=45)
            ax.tick_params(axis="y", labelsize=7)
            ax.invert_yaxis()

            if col_idx == 0:
                ax.set_ylabel(depth_label, fontsize=9)

            if ci == 0:
                ax.set_xlabel(well_name, fontsize=9, labelpad=4)

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
            fontsize=8,
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

    ax.set_xlabel(curve)
    ax.set_ylabel("Count")
    ax.set_title(f"Distribution of {curve}")
    ax.legend(fontsize=8)
    plt.tight_layout()
    return fig
