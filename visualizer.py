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
from matplotlib.patches import ConnectionPatch
import numpy as np
import pandas as pd
from scipy.interpolate import griddata

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

THEME_PRESETS: dict[str, dict[str, str]] = {
    "Geo Light": {
        "facecolor": "#ffffff",
        "grid_major": "#c9d5d2",
        "grid_minor": "#e6edeb",
        "separator": "#97aaa6",
        "connector": "#566a68",
    },
    "Mono Print": {
        "facecolor": "#ffffff",
        "grid_major": "#b0b0b0",
        "grid_minor": "#dfdfdf",
        "separator": "#8b8b8b",
        "connector": "#676767",
    },
    "Earth Warm": {
        "facecolor": "#fffdf8",
        "grid_major": "#d8cec0",
        "grid_minor": "#efe6da",
        "separator": "#b6aa9b",
        "connector": "#8d7660",
    },
}


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
    font_scale: float = 1.0,
    line_thickness_scale: float = 1.0,
    theme_preset: str = "Geo Light",
    show_connectors: bool = False,
    connector_formations: Optional[list[str]] = None,
    depth_grid_style: str = "Major",
    track_separators: bool = True,
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
    font_scale : float
        Global font scaling factor for publication output.
    line_thickness_scale : float
        Multiplier for all curve and overlay line widths.
    theme_preset : str
        One of ``Geo Light``, ``Mono Print`` or ``Earth Warm``.
    show_connectors : bool
        Whether to draw connectors between the same marker top across wells.
    connector_formations : list[str] or None
        Specific formations to connect. If ``None`` or empty, all common tops
        between neighboring wells are connected.
    depth_grid_style : str
        Depth grid mode: ``None``, ``Major`` or ``Major + Minor``.
    track_separators : bool
        Whether to draw visible separators at well boundaries.

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

    theme = THEME_PRESETS.get(theme_preset, THEME_PRESETS["Geo Light"])
    fig.patch.set_facecolor(theme["facecolor"])

    title_fs = max(11, int(round(15 * font_scale)))
    track_title_fs = max(8, int(round(10 * font_scale)))
    axis_label_fs = max(9, int(round(11 * font_scale)))
    tick_fs = max(7, int(round(9 * font_scale)))
    formation_label_fs = max(7, int(round(8 * font_scale)))

    fig.suptitle(title, fontsize=title_fs, fontweight="bold", y=1.01)

    well_names = list(wells.keys())
    tops_colors: dict[str, str] = {}
    tops_for_plot = formation_tops
    marker_depths_by_well: dict[str, dict[str, float]] = {}
    selected_connector_formations = {
        str(f).strip() for f in (connector_formations or []) if str(f).strip()
    }
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

                # Keep one depth per formation for optional connector overlays.
                dedup_tops = well_tops.drop_duplicates(subset=["formation"], keep="first")
                marker_depths_by_well[well_name] = {
                    str(row["formation"]): float(row["depth"])
                    for _, row in dedup_tops.iterrows()
                }

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
            lw = float(style.get("lw", 1.0)) * float(line_thickness_scale)

            ax.set_facecolor(theme["facecolor"])

            if depth_grid_style != "None":
                ax.grid(
                    axis="y",
                    which="major",
                    linestyle="--",
                    linewidth=0.7,
                    color=theme["grid_major"],
                    alpha=0.85,
                )
                if depth_grid_style == "Major + Minor":
                    ax.minorticks_on()
                    ax.grid(
                        axis="y",
                        which="minor",
                        linestyle=":",
                        linewidth=0.5,
                        color=theme["grid_minor"],
                        alpha=0.8,
                    )

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
                            linewidth=lw,
                        )
                    else:
                        ax.plot(
                            vals, depth,
                            color=style["color"],
                            linewidth=lw,
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
                            linewidth=1.5 * float(line_thickness_scale),
                            linestyle="--",
                            zorder=3,
                        )
                        if ci == 0:
                            x_left = ax.get_xlim()[0]
                            ax.text(
                                x_left if x_left != 0 else 0,
                                td,
                                f" {top_row['formation']}",
                                fontsize=formation_label_fs,
                                va="bottom",
                                color=top_row["color"],
                                clip_on=True,
                            )

            # Axis decoration
            unit = well.get("units", {}).get(curve, "")
            header = f"{curve}"
            if unit:
                header += f"\n({unit})"
            ax.set_title(header, fontsize=track_title_fs, pad=4)
            ax.tick_params(axis="x", labelsize=tick_fs, rotation=45)
            ax.tick_params(axis="y", labelsize=tick_fs)
            ax.invert_yaxis()

            if track_separators and ci == 0 and wi > 0:
                ax.spines["left"].set_visible(True)
                ax.spines["left"].set_color(theme["separator"])
                ax.spines["left"].set_linewidth(1.8)

            if col_idx == 0:
                ax.set_ylabel(depth_label, fontsize=axis_label_fs)

            if ci == 0:
                ax.set_xlabel(well_name, fontsize=track_title_fs, labelpad=4)

    # Optional cross-well connector lines for selected/common formation tops.
    if show_connectors and n_wells > 1 and marker_depths_by_well:
        for wi in range(n_wells - 1):
            left_well = well_names[wi]
            right_well = well_names[wi + 1]
            left_tops = marker_depths_by_well.get(left_well, {})
            right_tops = marker_depths_by_well.get(right_well, {})
            if not left_tops or not right_tops:
                continue

            common_forms = set(left_tops.keys()) & set(right_tops.keys())
            if selected_connector_formations:
                common_forms = common_forms & selected_connector_formations
            if not common_forms:
                continue

            left_ax = axes[wi * n_curves]
            right_ax = axes[(wi + 1) * n_curves]

            for formation in sorted(common_forms):
                y_left = float(left_tops[formation])
                y_right = float(right_tops[formation])
                if not (depth_min <= y_left <= depth_max and depth_min <= y_right <= depth_max):
                    continue

                connector_color = tops_colors.get(formation, theme["connector"])
                connector = ConnectionPatch(
                    xyA=(1.0, y_left),
                    coordsA=left_ax.get_yaxis_transform(),
                    xyB=(0.0, y_right),
                    coordsB=right_ax.get_yaxis_transform(),
                    color=connector_color,
                    linewidth=max(0.8, 1.2 * float(line_thickness_scale)),
                    alpha=0.65,
                    zorder=2,
                )
                fig.add_artist(connector)

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
            fontsize=max(8, int(round(9 * font_scale))),
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


# ---------------------------------------------------------------------------
# Well correlation diagram (stratigraphic column view)
# ---------------------------------------------------------------------------

def plot_well_correlation(
    wells: dict,
    formation_tops: Optional[pd.DataFrame] = None,
    depth_min: Optional[float] = None,
    depth_max: Optional[float] = None,
    figsize_per_well: tuple[float, float] = (1.2, 10.0),
    depth_label: str = "Depth (m)",
    title: str = "Well Correlation Diagram",
    font_scale: float = 1.0,
    line_thickness_scale: float = 1.0,
    theme_preset: str = "Geo Light",
    show_gr_curve: bool = True,
) -> plt.Figure:
    """Build and return a well correlation diagram showing formation intervals.

    This creates a stratigraphic column view where each well is displayed as
    a vertical strip, and formation intervals are shown as colored horizontal
    bands. Connector lines link the same formations across adjacent wells.

    Parameters
    ----------
    wells : dict
        Well data dict (name → data) as produced by
        :func:`data_loader.load_las_file` (or the flattened version).
    formation_tops : pd.DataFrame or None
        Formation tops table with columns well, formation, depth, color.
    depth_min, depth_max : float or None
        Depth window to display. If None, computed from well data.
    figsize_per_well : (float, float)
        (width, height) in inches per well.
    depth_label : str
        Label for the Y-axis.
    title : str
        Figure title.
    font_scale : float
        Global font scaling factor.
    line_thickness_scale : float
        Multiplier for connector line widths.
    theme_preset : str
        One of ``Geo Light``, ``Mono Print`` or ``Earth Warm``.
    show_gr_curve : bool
        Whether to overlay a thin GR curve for reference.

    Returns
    -------
    matplotlib.figure.Figure
    """
    n_wells = len(wells)
    if n_wells == 0:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No wells to display", ha="center", va="center")
        return fig

    # Determine depth range
    if depth_min is None or depth_max is None:
        all_depths = np.concatenate([w["depth"] for w in wells.values()])
        if depth_min is None:
            depth_min = float(np.nanmin(all_depths))
        if depth_max is None:
            depth_max = float(np.nanmax(all_depths))

    # Setup figure
    fig_w = figsize_per_well[0] * n_wells
    fig, axes = plt.subplots(
        1, n_wells,
        figsize=(max(fig_w, 8), figsize_per_well[1]),
        sharey=True,
    )
    if n_wells == 1:
        axes = [axes]

    theme = THEME_PRESETS.get(theme_preset, THEME_PRESETS["Geo Light"])
    fig.patch.set_facecolor(theme["facecolor"])

    title_fs = max(11, int(round(15 * font_scale)))
    axis_label_fs = max(9, int(round(11 * font_scale)))
    tick_fs = max(7, int(round(9 * font_scale)))
    formation_label_fs = max(7, int(round(8 * font_scale)))
    well_label_fs = max(8, int(round(10 * font_scale)))

    fig.suptitle(title, fontsize=title_fs, fontweight="bold", y=0.99)

    well_names = list(wells.keys())
    tops_colors: dict[str, str] = {}
    marker_depths_by_well: dict[str, dict[str, float]] = {}

    # Build metadata from formation tops
    if formation_tops is not None and not formation_tops.empty:
        formation_tops = formation_tops.copy()
        formation_tops["_well_key"] = formation_tops["well"].map(_normalize_well_name)
        for _, row in formation_tops.iterrows():
            tops_colors[row["formation"]] = row["color"]

    # Plot each well as a column
    for wi, well_name in enumerate(well_names):
        ax = axes[wi]
        well = wells[well_name]
        df = well["df"]
        depth_col = df.columns[0]
        well_key = _normalize_well_name(well_name)

        # Mask to depth window
        mask = (df[depth_col] >= depth_min) & (df[depth_col] <= depth_max)
        df_win = df[mask]

        # Set background
        ax.set_facecolor(theme["facecolor"])

        # Grid
        ax.grid(
            axis="y",
            which="major",
            linestyle="--",
            linewidth=0.5,
            color=theme["grid_major"],
            alpha=0.6,
        )

        # Collect formation intervals for this well
        well_tops = pd.DataFrame()
        formation_intervals: list[tuple[float, float, str, str]] = []  # (top, base, color, name)
        
        if formation_tops is not None and not formation_tops.empty:
            well_tops = formation_tops[formation_tops["_well_key"] == well_key].copy()
            if not well_tops.empty:
                well_tops["depth"] = pd.to_numeric(well_tops["depth"], errors="coerce")
                well_tops = well_tops.dropna(subset=["depth"]).sort_values("depth")

                # Build interval pairs
                top_rows = well_tops[["depth", "color", "formation"]].to_dict("records")
                for idx in range(len(top_rows) - 1):
                    top_depth = float(top_rows[idx]["depth"])
                    base_depth = float(top_rows[idx + 1]["depth"])
                    if base_depth <= depth_min or top_depth >= depth_max:
                        continue
                    y0 = max(top_depth, depth_min)
                    y1 = min(base_depth, depth_max)
                    if y1 > y0:
                        formation_intervals.append(
                            (y0, y1, str(top_rows[idx]["color"]), str(top_rows[idx]["formation"]))
                        )

                # Store depths for connectors
                dedup_tops = well_tops.drop_duplicates(subset=["formation"], keep="first")
                marker_depths_by_well[well_name] = {
                    str(row["formation"]): float(row["depth"])
                    for _, row in dedup_tops.iterrows()
                }

        # Draw formation intervals
        for y_top, y_base, color, formation_name in formation_intervals:
            ax.axhspan(y_top, y_base, xmin=0, xmax=1, color=color, alpha=0.25, zorder=1)
            ax.axhline(y_top, color=color, linewidth=1.0, linestyle="-", alpha=0.8, zorder=2)

        # Draw formation tops as lines and labels
        if not well_tops.empty:
            for _, row in well_tops.iterrows():
                td = float(row["depth"])
                if depth_min <= td <= depth_max:
                    ax.axhline(
                        td,
                        color=row["color"],
                        linewidth=1.2 * float(line_thickness_scale),
                        linestyle="-",
                        alpha=0.9,
                        zorder=3,
                    )
                    # Formation label on left side
                    ax.text(
                        0.02,
                        td,
                        f"{row['formation']}",
                        fontsize=formation_label_fs,
                        va="center",
                        ha="left",
                        color=row["color"],
                        fontweight="bold",
                        transform=ax.get_yaxis_transform(),
                        bbox=dict(
                            boxstyle="round,pad=0.3",
                            facecolor="white",
                            alpha=0.7,
                            edgecolor="none",
                        ),
                        clip_on=True,
                    )

        # Optional: overlay thin GR curve as reference
        if show_gr_curve and "GR" in df.columns:
            gr_vals = df_win["GR"].values
            if len(gr_vals) > 0 and not np.isnan(gr_vals).all():
                depth_vals = df_win[depth_col].values
                # Normalize GR to 0-1 for positioning
                gr_min, gr_max = np.nanmin(gr_vals), np.nanmax(gr_vals)
                if gr_max > gr_min:
                    gr_norm = (gr_vals - gr_min) / (gr_max - gr_min)
                else:
                    gr_norm = 0.5 * np.ones_like(gr_vals)
                ax.plot(
                    gr_norm,
                    depth_vals,
                    color="#4caf50",
                    linewidth=0.8,
                    alpha=0.4,
                    zorder=0,
                    label="GR (normalized)",
                )

        # Axis decoration
        ax.set_xlim(0, 1)
        ax.set_ylim(depth_max, depth_min)  # Inverted depth axis
        ax.set_xlabel(well_name, fontsize=well_label_fs, fontweight="bold", labelpad=6)
        ax.tick_params(axis="x", labelbottom=False)
        ax.tick_params(axis="y", labelsize=tick_fs)

        if wi == 0:
            ax.set_ylabel(depth_label, fontsize=axis_label_fs, fontweight="bold")
        else:
            ax.set_ylabel("")

        ax.spines["top"].set_visible(True)
        ax.spines["bottom"].set_visible(True)
        ax.spines["left"].set_linewidth(1.5)
        ax.spines["right"].set_visible(False)

    # Draw connector lines between adjacent wells
    if n_wells > 1 and marker_depths_by_well:
        for wi in range(n_wells - 1):
            left_well = well_names[wi]
            right_well = well_names[wi + 1]
            left_tops = marker_depths_by_well.get(left_well, {})
            right_tops = marker_depths_by_well.get(right_well, {})
            
            if not left_tops or not right_tops:
                continue

            common_forms = set(left_tops.keys()) & set(right_tops.keys())
            if not common_forms:
                continue

            left_ax = axes[wi]
            right_ax = axes[wi + 1]

            for formation in sorted(common_forms):
                y_left = float(left_tops[formation])
                y_right = float(right_tops[formation])
                
                if not (depth_min <= y_left <= depth_max and depth_min <= y_right <= depth_max):
                    continue

                connector_color = tops_colors.get(formation, theme["connector"])
                connector = ConnectionPatch(
                    xyA=(1.0, y_left),
                    coordsA=left_ax.transData,
                    xyB=(0.0, y_right),
                    coordsB=right_ax.transData,
                    color=connector_color,
                    linewidth=max(0.8, 1.0 * float(line_thickness_scale)),
                    linestyle="--",
                    alpha=0.6,
                    zorder=1,
                )
                fig.add_artist(connector)

    # Legend
    if tops_colors:
        patches = [
            mpatches.Patch(color=c, label=f)
            for f, c in tops_colors.items()
        ]
        fig.legend(
            handles=patches,
            loc="lower center",
            ncol=min(len(patches), 8),
            fontsize=max(7, int(round(8 * font_scale))),
            title="Formations",
            bbox_to_anchor=(0.5, -0.05),
            frameon=True,
            fancybox=True,
            shadow=False,
        )

    plt.tight_layout(rect=[0, 0.05, 1, 0.96])
    return fig


# ---------------------------------------------------------------------------
# Isopach (Thickness) Contour Maps
# ---------------------------------------------------------------------------

def plot_isopach_map(
    formation_tops: pd.DataFrame,
    formation_name: str,
    well_order: Optional[list[str]] = None,
    figsize: tuple[float, float] = (12, 8),
    title: str = "Isopach Map",
    font_scale: float = 1.0,
    theme_preset: str = "Geo Light",
    cmap: str = "RdYlGn_r",
    n_contours: int = 12,
) -> plt.Figure:
    """Generate an isopach (thickness) contour map for a formation.
    
    Shows formation thickness variations across wells, identifying depocenters
    and structural trends through color-coded thickness intervals.
    
    Parameters
    ----------
    formation_tops : pd.DataFrame
        Formation tops table with columns: well, formation, depth
    formation_name : str
        Formation to generate isopach map for
    well_order : list[str] or None
        Ordered list of well names. If None, uses natural order from data.
    figsize : (float, float)
        Figure size in inches
    title : str
        Figure title
    font_scale : float
        Font scaling factor
    theme_preset : str
        One of ``Geo Light``, ``Mono Print`` or ``Earth Warm``
    cmap : str
        Matplotlib colormap name
    n_contours : int
        Number of contour levels
        
    Returns
    -------
    matplotlib.figure.Figure
    """
    if formation_tops is None or formation_tops.empty:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No formation tops data", ha="center", va="center")
        return fig
    
    theme = THEME_PRESETS.get(theme_preset, THEME_PRESETS["Geo Light"])
    
    # Calculate formation thickness per well
    pts_data = []
    thickness_data = []
    well_labels = []
    
    formations = formation_tops["formation"].unique()
    matching_formations = [f for f in formations 
                          if str(formation_name).strip().lower() in str(f).strip().lower()]
    
    if not matching_formations:
        matching_formation = formation_name
    else:
        matching_formation = matching_formations[0]
    
    # Group by well
    well_groups = formation_tops.groupby(
        formation_tops["well"].map(_normalize_well_name)
    )
    
    for well_idx, (well_key, group) in enumerate(well_groups):
        well_name = group.iloc[0]["well"]
        group_sorted = group.sort_values("depth")
        
        # Find matching formation and next formation for thickness
        for idx in range(len(group_sorted) - 1):
            top_row = group_sorted.iloc[idx]
            base_row = group_sorted.iloc[idx + 1]
            
            if str(matching_formation).strip().lower() in str(top_row["formation"]).strip().lower():
                top_depth = float(top_row["depth"])
                base_depth = float(base_row["depth"])
                thickness = base_depth - top_depth
                
                # Use well index as x position, average depth as y position
                pts_data.append([well_idx, (top_depth + base_depth) / 2])
                thickness_data.append(thickness)
                well_labels.append(well_name)
                break
    
    if not pts_data:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, f"No data for {formation_name}", ha="center", va="center")
        return fig
    
    pts = np.array(pts_data)
    thickness_vals = np.array(thickness_data)
    
    # Create grid for interpolation
    grid_x = np.linspace(pts[:, 0].min() - 0.5, pts[:, 0].max() + 0.5, 100)
    grid_y = np.linspace(pts[:, 1].min() - 50, pts[:, 1].max() + 50, 100)
    grid_X, grid_Y = np.meshgrid(grid_x, grid_y)
    
    # Interpolate thickness values - choose method based on number of points
    n_points = len(pts_data)
    if n_points >= 4:
        method = "cubic"
    elif n_points >= 3:
        method = "linear"
    else:
        method = "nearest"
    
    grid_Z = griddata(pts, thickness_vals, (grid_X, grid_Y), method=method)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(theme["facecolor"])
    ax.set_facecolor(theme["facecolor"])
    
    # Plot contours
    contour = ax.contourf(grid_X, grid_Y, grid_Z, levels=n_contours, cmap=cmap, alpha=0.8)
    contour_lines = ax.contour(grid_X, grid_Y, grid_Z, levels=n_contours // 2, 
                               colors="black", linewidths=0.5, alpha=0.3)
    ax.clabel(contour_lines, inline=True, fontsize=8, fmt="%.0f m")
    
    # Plot well locations
    ax.scatter(pts[:, 0], pts[:, 1], s=100, c="black", zorder=5, edgecolor="white", linewidth=2)
    
    # Add well labels
    for idx, (pt, thick, label) in enumerate(zip(pts, thickness_vals, well_labels)):
        ax.text(pt[0], pt[1], f"{label}\n{thick:.1f}m", 
               ha="center", va="center", fontsize=8, fontweight="bold",
               bbox=dict(boxstyle="round,pad=0.4", facecolor="white", 
                        edgecolor="black", alpha=0.8))
    
    # Formatting
    title_fs = max(14, int(round(18 * font_scale)))
    label_fs = max(11, int(round(12 * font_scale)))
    
    ax.set_xlabel("Well Index (West → East)", fontsize=label_fs, fontweight="bold")
    ax.set_ylabel("Depth / Structural Position (m)", fontsize=label_fs, fontweight="bold")
    ax.set_title(f"Isopach Map: {matching_formation}", fontsize=title_fs, fontweight="bold", pad=20)
    
    # Colorbar
    cbar = plt.colorbar(contour, ax=ax, label="Thickness (m)", pad=0.02)
    cbar.ax.tick_params(labelsize=10)
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle="--", color=theme["grid_major"])
    
    # Add statistics box
    stats_text = f"Mean: {thickness_vals.mean():.1f} m\nMin: {thickness_vals.min():.1f} m\nMax: {thickness_vals.max():.1f} m\nStd: {thickness_vals.std():.1f} m"
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
           fontsize=9, verticalalignment="top",
           bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))
    
    plt.tight_layout()
    return fig
