"""
Multi-Well Data Viewer — Streamlit Application
===============================================
Main entry point.  Run with:
    streamlit run app.py
"""

from __future__ import annotations

import io
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from data_loader import (
    get_all_curves,
    get_common_curves,
    load_formation_tops,
    load_las_file,
    validate_wells,
)
from flattening import (
    flatten_formation_tops,
    flatten_wells,
    get_flattened_depth_range,
)
from visualizer import plot_cross_section, plot_curve_histogram

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------

CENTENARY_IMAGE_PATH = r"C:\Users\anshu\Desktop\CENTENARY.png"
IIT_ISM_LOGO_PATH = (
    r"C:\Users\anshu\Documents\IIT ISM- Geophysics"
    r"\Indian_Institute_of_Technology_(Indian_School_of_Mines),_Dhanbad_Logo.png"
)
if os.path.exists(CENTENARY_IMAGE_PATH):
    page_icon_value = CENTENARY_IMAGE_PATH
elif os.path.exists(IIT_ISM_LOGO_PATH):
    page_icon_value = IIT_ISM_LOGO_PATH
else:
    page_icon_value = "🛢️"

st.set_page_config(
    page_title="Multi-well data loading, display, formation tops overlay and flattening",
    page_icon=page_icon_value,
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
html, body, [class*="css"] {
    font-family: "Times New Roman", Times, serif;
}

.stApp {
    background-color: #ffffff;
}

[data-testid="stSidebar"] {
    background-color: #f5f7fb;
}

h1 { font-size: 2.0rem !important; }
h2 { font-size: 1.45rem !important; }
h3 { font-size: 1.2rem !important; }

p, li, label, .stMarkdown, .stAlert, .stCaption {
    font-size: 1.05rem !important;
}

.stButton > button,
.stDownloadButton > button {
    border-radius: 8px;
    font-weight: 600;
}

[data-testid="stMetricValue"] {
    font-size: 1.3rem !important;
}
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------

if "wells" not in st.session_state:
    st.session_state.wells = {}

if "formation_tops" not in st.session_state:
    st.session_state.formation_tops = None


def _las_header_section_df(las_obj, section_name: str) -> pd.DataFrame:
    """Convert a LAS header section into a tidy DataFrame for display."""
    section = getattr(las_obj, section_name, None)
    if section is None:
        return pd.DataFrame()

    rows = []
    try:
        items = list(section)
    except Exception:
        items = []

    for item in items:
        rows.append(
            {
                "Mnemonic": getattr(item, "mnemonic", ""),
                "Unit": getattr(item, "unit", ""),
                "Value": getattr(item, "value", ""),
                "Description": getattr(item, "descr", ""),
            }
        )

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def _normalize_name(value: str) -> str:
    """Return a canonical key for robust string matching."""
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())


def _get_depth_unit(well_data: dict) -> str:
    """Get the depth unit from LAS curve metadata for a well."""
    las_obj = well_data.get("las")
    df = well_data.get("df")
    if las_obj is None or df is None or len(df.columns) == 0:
        return ""

    depth_mnem = str(df.columns[0]).strip().upper()
    for item in getattr(las_obj, "curves", []):
        mnem = str(getattr(item, "mnemonic", "")).strip().upper()
        if mnem == depth_mnem:
            return str(getattr(item, "unit", "")).strip()
    return ""


def _build_data_health(wells: dict, formation_tops: pd.DataFrame | None) -> dict:
    """Build data-health checks used by the UI panel."""
    well_names = list(wells.keys())

    # Missing tops per loaded well
    missing_tops_rows = []
    if formation_tops is None or formation_tops.empty:
        for wname in well_names:
            missing_tops_rows.append({"Well": wname, "Issue": "No formation tops loaded"})
    else:
        tops_keys = {
            _normalize_name(w)
            for w in formation_tops["well"].astype(str).tolist()
        }
        for wname in well_names:
            if _normalize_name(wname) not in tops_keys:
                missing_tops_rows.append({"Well": wname, "Issue": "No tops matched"})

    # Curves missing by well (against global union)
    all_curves = sorted({c for well in wells.values() for c in well.get("curves", [])})
    missing_curve_rows = []
    for wname, well in wells.items():
        present = set(well.get("curves", []))
        missing = [c for c in all_curves if c not in present]
        if missing:
            missing_curve_rows.append(
                {
                    "Well": wname,
                    "Missing curves count": len(missing),
                    "Missing curves": ", ".join(missing),
                }
            )

    # Depth unit mismatch warnings
    depth_unit_rows = []
    for wname, well in wells.items():
        depth_unit_rows.append({"Well": wname, "Depth unit": _get_depth_unit(well) or "Unknown"})
    depth_units_df = pd.DataFrame(depth_unit_rows)
    non_unknown_units = {
        str(u).strip().lower()
        for u in depth_units_df["Depth unit"].tolist()
        if str(u).strip().lower() not in ("", "unknown")
    }
    has_depth_unit_mismatch = len(non_unknown_units) > 1

    # Duplicate formation picks: same well + formation more than once
    duplicate_picks_df = pd.DataFrame()
    if formation_tops is not None and not formation_tops.empty:
        tops_df = formation_tops.copy()
        tops_df["well_key"] = tops_df["well"].map(_normalize_name)
        tops_df["formation_key"] = tops_df["formation"].astype(str).str.strip().str.lower()
        dup_mask = tops_df.duplicated(subset=["well_key", "formation_key"], keep=False)
        if dup_mask.any():
            duplicate_picks_df = tops_df.loc[
                dup_mask,
                ["well", "formation", "depth", "color"],
            ].sort_values(["well", "formation", "depth"])

    return {
        "missing_tops_df": pd.DataFrame(missing_tops_rows),
        "missing_curves_df": pd.DataFrame(missing_curve_rows),
        "depth_units_df": depth_units_df,
        "has_depth_unit_mismatch": has_depth_unit_mismatch,
        "duplicate_picks_df": duplicate_picks_df,
    }

# ---------------------------------------------------------------------------
# Sidebar — Data loading
# ---------------------------------------------------------------------------

with st.sidebar:
    brand_cols = st.columns(2)
    if os.path.exists(CENTENARY_IMAGE_PATH):
        brand_cols[0].image(CENTENARY_IMAGE_PATH, width=95)
    if os.path.exists(IIT_ISM_LOGO_PATH):
        brand_cols[1].image(IIT_ISM_LOGO_PATH, width=95)
    st.title("Multi-well data loading, display, formation tops overlay and flattening")
    st.markdown("---")

    # ── LAS file upload ──────────────────────────────────────────────────
    st.header("📂 Load Well Data")
    las_files = st.file_uploader(
        "Upload LAS files",
        type=["las", "LAS"],
        accept_multiple_files=True,
        help="Select one or more LAS files to load.",
    )

    if las_files:
        loaded_filenames = {v.get("_filename") for v in st.session_state.wells.values()}
        for f in las_files:
            if f.name not in loaded_filenames:
                well_data = load_las_file(f)
                if well_data is not None:
                    # Make name unique if needed
                    base_name = well_data["name"]
                    name = base_name
                    counter = 1
                    while name in st.session_state.wells:
                        name = f"{base_name}_{counter}"
                        counter += 1
                    well_data["_filename"] = f.name
                    st.session_state.wells[name] = well_data
                    st.success(f"Loaded: **{name}**")
                else:
                    st.error(f"Failed to parse: {f.name}")

    # ── Well management ──────────────────────────────────────────────────
    if st.session_state.wells:
        st.markdown("**Loaded wells:**")
        wells_to_remove = []
        for wname in list(st.session_state.wells.keys()):
            col1, col2 = st.columns([4, 1])
            col1.markdown(f"- {wname}")
            if col2.button("✕", key=f"rm_{wname}", help=f"Remove {wname}"):
                wells_to_remove.append(wname)
        for wname in wells_to_remove:
            del st.session_state.wells[wname]
            st.rerun()

    # ── Formation tops upload ─────────────────────────────────────────────
    st.markdown("---")
    st.header("🗺️ Formation Tops")
    tops_file = st.file_uploader(
        "Upload formation tops (CSV/Excel)",
        type=["csv", "xlsx", "xls"],
        help=(
            "CSV/Excel with columns like well, formation, depth [, color] "
            "or borehole name, formation name, top depth (m)."
        ),
    )
    if tops_file is not None:
        tops_df = load_formation_tops(tops_file)
        if tops_df is not None:
            st.session_state.formation_tops = tops_df
            st.success(
                f"Loaded {len(tops_df)} top entries across "
                f"{tops_df['well'].nunique()} well(s)."
            )
        else:
            st.error(
                "Failed to load formation tops. "
                "Ensure file has columns: well, formation, depth."
            )

    if st.session_state.formation_tops is not None:
        if st.button("🗑️ Clear tops"):
            st.session_state.formation_tops = None
            st.rerun()

    # ── Visualisation controls ─────────────────────────────────────────────
    st.markdown("---")
    st.header("⚙️ Display Settings")

    all_curves = get_all_curves(st.session_state.wells)
    common_curves = get_common_curves(st.session_state.wells)

    track_pool_mode = st.radio(
        "Track source",
        ["All available tracks", "Common tracks only"],
        horizontal=True,
        help="Choose whether to pick tracks from all loaded LAS curves or only curves present in every well.",
    )
    available_track_options = (
        all_curves if track_pool_mode == "All available tracks" else common_curves
    )

    # Use a focused default set (4-5 tracks) so the data visualization remains readable.
    preferred_order = ["GR", "ILD", "LLD", "RT", "RHOB", "NPHI", "DT"]
    default_curves = [c for c in preferred_order if c in available_track_options][:5]

    # Backfill from the active track pool, if fewer than 4 were found.
    if len(default_curves) < 4:
        for c in available_track_options:
            if c not in default_curves:
                default_curves.append(c)
            if len(default_curves) >= 5:
                break

    if len(default_curves) < 4:
        for c in available_track_options:
            if c not in default_curves:
                default_curves.append(c)
            if len(default_curves) >= min(5, len(available_track_options)):
                break

    selected_curves = st.multiselect(
        "Tracks to display",
        options=available_track_options,
        default=default_curves,
        help="Choose exactly which tracks you want to see in the data visualization.",
    )

    depth_label = st.selectbox(
        "Depth axis label",
        ["Depth (m)", "MD (m)", "TVD (m)", "TVDSS (m)"],
        index=0,
    )

    # Depth range
    if st.session_state.wells:
        all_depths = np.concatenate(
            [w["depth"] for w in st.session_state.wells.values()]
        )
        global_min = float(np.nanmin(all_depths))
        global_max = float(np.nanmax(all_depths))
    else:
        global_min, global_max = 0.0, 3000.0

    depth_range = st.slider(
        "Depth window",
        min_value=global_min,
        max_value=global_max,
        value=(global_min, global_max),
        step=max(1.0, (global_max - global_min) / 500),
        format="%.1f",
    )

    # ── Quality check (smoothing) ───────────────────────────────────────
    quality_check_enabled = st.checkbox(
        "Quality check (moving average)",
        value=False,
        help="Smooth displayed tracks using a moving average filter for quick QC.",
    )
    average_filter_window = st.selectbox(
        "Average filter window",
        options=[5, 10, 15],
        index=0,
        disabled=not quality_check_enabled,
    )

    # ── Formation tops overlay toggle ─────────────────────────────────────
    show_tops = st.checkbox(
        "Show formation tops",
        value=True,
        disabled=st.session_state.formation_tops is None,
    )

    shade_formations = st.checkbox(
        "Shade formation intervals",
        value=True,
        disabled=st.session_state.formation_tops is None,
        help="Fill intervals between consecutive formation tops with translucent formation colors.",
    )
    formation_shade_alpha = st.slider(
        "Formation shade intensity",
        min_value=0.05,
        max_value=0.40,
        value=0.14,
        step=0.01,
        disabled=(st.session_state.formation_tops is None) or (not shade_formations),
        help="Higher values make formation color shading stronger.",
    )

    # ── Flattening ────────────────────────────────────────────────────────
    st.markdown("---")
    st.header("🔄 Flattening")

    enable_flattening = st.checkbox("Enable flattening", value=False)
    reference_formation = None
    if enable_flattening and st.session_state.formation_tops is not None:
        formations = sorted(
            st.session_state.formation_tops["formation"].unique().tolist()
        )
        reference_formation = st.selectbox(
            "Reference formation",
            options=formations,
            help="All well logs will be shifted so this top aligns at depth 0.",
        )
    elif enable_flattening:
        st.info("Load formation tops to enable flattening.")

    # ── Export ────────────────────────────────────────────────────────────
    st.markdown("---")
    st.header("💾 Export")
    export_format = st.radio("Format", ["PNG", "PDF"], horizontal=True)

    # ── Sample data ───────────────────────────────────────────────────────
    st.markdown("---")
    with st.expander("ℹ️ Sample Data / Help"):
        st.markdown(
            """
**Accepted file formats**
- **LAS files** (`.las`) — well log data
- **CSV or Excel file** (`.csv`, `.xlsx`, `.xls`) — formation tops with columns:
  `well`, `formation`, `depth` *(and optionally `color`)*

**Quick start**
1. Upload one or more LAS files.
2. *(Optional)* Upload a formation tops CSV or Excel file.
3. Choose curves and depth window from the sidebar.
4. Enable flattening to align wells on a key horizon.
5. Export the figure as PNG or PDF.

**Sample data** is available in the `sample_data/` folder of this repository.
            """
        )

# ---------------------------------------------------------------------------
# Main display area
# ---------------------------------------------------------------------------

st.title("Multi-well data loading, display, formation tops overlay and flattening")
st.caption("Interactive well-log comparison with formation-aware overlays")

tabs = st.tabs(["📊 Data Visualization", "📋 Data Summary", "📈 Statistics", "📖 Help"])

# ── Tab 1: Data Visualization ─────────────────────────────────────────────
with tabs[0]:
    if not st.session_state.wells:
        st.info(
            "👈 Upload LAS files from the sidebar to get started.  "
            "Sample data is available in the `sample_data/` directory."
        )
    elif not selected_curves:
        st.warning("Select at least one curve in the sidebar to display.")
    else:
        # Determine working data (flattened or original)
        working_wells = st.session_state.wells
        working_tops = st.session_state.formation_tops
        flat_depth_min, flat_depth_max = depth_range

        if (
            enable_flattening
            and reference_formation is not None
            and st.session_state.formation_tops is not None
        ):
            working_wells = flatten_wells(
                st.session_state.wells,
                st.session_state.formation_tops,
                reference_formation,
                reference_depth=0.0,
            )
            working_tops = flatten_formation_tops(
                st.session_state.formation_tops,
                st.session_state.wells,
                reference_formation,
                reference_depth=0.0,
            )
            flat_depth_min, flat_depth_max = get_flattened_depth_range(working_wells)
            st.info(
                f"🔄 Flattened on **{reference_formation}**. "
                "Depth axis is relative offset from the reference top."
            )

        # Validation warnings
        warnings_list = validate_wells(working_wells)
        if warnings_list:
            with st.expander("⚠️ Data quality warnings", expanded=False):
                for w in warnings_list:
                    st.warning(w)

        st.markdown("---")
        st.subheader("Data Health")
        st.caption("Quick integrity checks for wells, tops, curves, and depth units.")

        health = _build_data_health(working_wells, st.session_state.formation_tops)
        missing_tops_df = health["missing_tops_df"]
        missing_curves_df = health["missing_curves_df"]
        depth_units_df = health["depth_units_df"]
        has_depth_unit_mismatch = health["has_depth_unit_mismatch"]
        duplicate_picks_df = health["duplicate_picks_df"]

        total_wells = max(1, len(working_wells))
        missing_tops_pct = 100.0 * len(missing_tops_df) / total_wells
        missing_curves_pct = 100.0 * len(missing_curves_df) / total_wells

        mismatch_pct = 0.0
        if has_depth_unit_mismatch and not depth_units_df.empty:
            units_series = (
                depth_units_df["Depth unit"]
                .astype(str)
                .str.strip()
                .replace("", "Unknown")
            )
            common_unit = units_series.mode().iloc[0]
            mismatch_wells = int((units_series != common_unit).sum())
            mismatch_pct = 100.0 * mismatch_wells / total_wells

        duplicate_pct = 0.0
        if st.session_state.formation_tops is not None and len(st.session_state.formation_tops) > 0:
            duplicate_pct = 100.0 * len(duplicate_picks_df) / len(st.session_state.formation_tops)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Missing tops", f"{missing_tops_pct:.1f}%")
        col2.metric("Missing curves", f"{missing_curves_pct:.1f}%")
        col3.metric("Depth unit mismatch", f"{mismatch_pct:.1f}%")
        col4.metric("Duplicate formation picks", f"{duplicate_pct:.1f}%")

        st.markdown("---")
        st.subheader("~CURVE INFORMATION")
        st.caption("LAS curve header details shown before track visualization.")

        for wname, well in working_wells.items():
            with st.expander(f"{wname} curve header", expanded=False):
                las_obj = well.get("las")
                curve_header_rows = []
                if las_obj is not None:
                    for curve_item in getattr(las_obj, "curves", []):
                        curve_header_rows.append(
                            {
                                "MNEM": str(getattr(curve_item, "mnemonic", "")),
                                "UNIT": str(getattr(curve_item, "unit", "")),
                                "API CODE": str(getattr(curve_item, "value", "")),
                                "DESCRIPTION": str(getattr(curve_item, "descr", "")),
                            }
                        )

                if curve_header_rows:
                    st.dataframe(
                        pd.DataFrame(curve_header_rows),
                        use_container_width=True,
                        hide_index=True,
                    )
                else:
                    st.info("No LAS curve header information found for this well.")

        # Show well names in the title for quick context.
        well_names_label = ", ".join(list(working_wells.keys()))
        if len(well_names_label) > 90:
            well_names_label = f"{well_names_label[:87]}..."

        base_title = (
            f"Flattened on: {reference_formation}"
            if enable_flattening and reference_formation
            else "Multi-Well Data Visualization"
        )
        figure_title = f"{base_title} ({well_names_label})"

        with st.spinner("Rendering data visualization…"):
            fig = plot_cross_section(
                wells=working_wells,
                curves_to_plot=selected_curves,
                depth_min=flat_depth_min,
                depth_max=flat_depth_max,
                formation_tops=working_tops if show_tops else None,
                show_tops=show_tops,
                shade_formations=shade_formations,
                formation_shade_alpha=formation_shade_alpha,
                smoothing_window=(average_filter_window if quality_check_enabled else None),
                depth_label=depth_label,
                title=figure_title,
            )

        st.pyplot(fig, use_container_width=True)

        st.markdown("---")
        st.subheader("LAS File Headers")
        st.caption("Header metadata from uploaded LAS files.")

        for wname, raw_well in st.session_state.wells.items():
            with st.expander(f"{wname} header", expanded=False):
                las_obj = raw_well.get("las")
                if las_obj is None:
                    st.info("Header data is not available for this well.")
                    continue

                for label, section_name in [
                    ("Version", "version"),
                    ("Well", "well"),
                    ("Parameters", "params"),
                ]:
                    header_df = _las_header_section_df(las_obj, section_name)
                    if not header_df.empty:
                        st.markdown(f"**{label} section**")
                        st.dataframe(header_df, use_container_width=True, hide_index=True)

        # Export button
        buf = io.BytesIO()
        fmt = export_format.lower()
        fig.savefig(buf, format=fmt, bbox_inches="tight", dpi=150)
        buf.seek(0)
        st.download_button(
            label=f"⬇️ Download {export_format}",
            data=buf,
            file_name=f"data_visualization.{fmt}",
            mime=f"image/{fmt}" if fmt == "png" else "application/pdf",
        )
        plt.close(fig)

# ── Tab 2: Data Summary ───────────────────────────────────────────────────
with tabs[1]:
    if not st.session_state.wells:
        st.info("No wells loaded yet.")
    else:
        for wname, well in st.session_state.wells.items():
            with st.expander(f"**{wname}**", expanded=False):
                info = well["info"]
                col1, col2, col3 = st.columns(3)
                col1.metric("Top depth", f"{info['strt']:.1f} m")
                col2.metric("Base depth", f"{info['stop']:.1f} m")
                col3.metric("Step", f"{info['step']:.3f} m")

                meta_cols = st.columns(3)
                meta_cols[0].markdown(f"**Company:** {info['company'] or '—'}")
                meta_cols[1].markdown(f"**Field:** {info['field'] or '—'}")
                meta_cols[2].markdown(f"**KB:** {info['kb']} m")

                st.markdown("**Available curves:**")
                curve_info = []
                for curve in well["curves"]:
                    col_data = well["df"][curve]
                    curve_info.append(
                        {
                            "Curve": curve,
                            "Unit": well["units"].get(curve, ""),
                            "Min": f"{col_data.min():.3g}",
                            "Max": f"{col_data.max():.3g}",
                            "Mean": f"{col_data.mean():.3g}",
                            "Null %": f"{col_data.isna().mean()*100:.1f}%",
                        }
                    )
                st.dataframe(
                    pd.DataFrame(curve_info),
                    use_container_width=True,
                    hide_index=True,
                )

    if st.session_state.formation_tops is not None:
        st.markdown("---")
        st.subheader("Formation Tops")
        st.dataframe(
            st.session_state.formation_tops,
            use_container_width=True,
            hide_index=True,
        )

# ── Tab 3: Statistics ─────────────────────────────────────────────────────
with tabs[2]:
    if not st.session_state.wells:
        st.info("No wells loaded yet.")
    elif not selected_curves:
        st.warning("Select curves in the sidebar to view statistics.")
    else:
        hist_curve = st.selectbox(
            "Select curve for histogram",
            options=selected_curves,
        )
        if hist_curve:
            hfig = plot_curve_histogram(
                st.session_state.wells,
                hist_curve,
                depth_min=depth_range[0],
                depth_max=depth_range[1],
            )
            st.pyplot(hfig, use_container_width=True)
            plt.close(hfig)

        # Descriptive stats table
        st.subheader("Descriptive Statistics")
        for wname, well in st.session_state.wells.items():
            df = well["df"]
            depth_col = df.columns[0]
            mask = (df[depth_col] >= depth_range[0]) & (df[depth_col] <= depth_range[1])
            df_win = df[mask]

            available = [c for c in selected_curves if c in df_win.columns]
            if available:
                stats_df = df_win[available].describe().T
                stats_df.index.name = "Curve"
                st.markdown(f"**{wname}**")
                st.dataframe(stats_df.round(4), use_container_width=True)

# ── Tab 4: Help ────────────────────────────────────────────────────────────
with tabs[3]:
    st.markdown(
        """
## How to Use Multi-well data loading, display, formation tops overlay and flattening

### 1. Load Well Data
- Click **Upload LAS files** in the sidebar.
- You can upload **multiple LAS files** at once.
- Each file will appear as a separate well in the viewer.

### 2. Load Formation Tops *(optional)*
- Prepare a CSV/Excel file with columns: `well`, `formation`, `depth`.
- Optionally add a `color` column (hex colour, e.g. `#e6194b`).
- Upload via **Upload formation tops (CSV/Excel)** in the sidebar.

### 3. Configure Display
- Select which **log curves** to show in the data visualization.
- Adjust the **depth window** slider to zoom in/out.
- Toggle **Show formation tops** to overlay picks on the tracks.

### 4. Flatten on a Formation Top
- Enable **Flattening** in the sidebar.
- Choose the **reference formation** from the dropdown.
- All wells will be depth-shifted so the reference top aligns at 0.

### 5. Export
- Use the **Download PNG / PDF** button below the data visualization figure.

---

### Sample Formation Tops CSV
```
well,formation,depth
WELL_A,Top Sand,1200.5
WELL_A,Base Shale,1450.0
WELL_B,Top Sand,1185.0
WELL_B,Base Shale,1432.5
```

### Supported Curves
Common LAS mnemonics that receive predefined colour/scale styling:
`GR`, `RHOB`, `NPHI`, `RT`, `ILD`, `LLD`, `DT`, `PHIE`, `SW`, `VSH`.
Any other curve will be plotted with a default grey style on a linear scale.
        """
    )
