"""
Multi-Well Data Viewer — Streamlit Application
===============================================
Main entry point.  Run with:
    streamlit run app.py
"""

from __future__ import annotations

import io
from typing import Optional

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

st.set_page_config(
    page_title="Multi-Well Viewer",
    page_icon="🛢️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------

if "wells" not in st.session_state:
    st.session_state.wells: dict = {}

if "formation_tops" not in st.session_state:
    st.session_state.formation_tops: Optional[pd.DataFrame] = None

# ---------------------------------------------------------------------------
# Sidebar — Data loading
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("🛢️ Multi-Well Viewer")
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
        "Upload formation tops (CSV)",
        type=["csv"],
        help="CSV with columns: well, formation, depth [, color]",
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
                "Ensure CSV has columns: well, formation, depth."
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

    default_curves = [c for c in ["GR", "ILD", "RHOB", "NPHI", "DT"] if c in all_curves]
    if not default_curves:
        default_curves = all_curves[:3] if len(all_curves) >= 3 else all_curves

    selected_curves = st.multiselect(
        "Log curves to display",
        options=all_curves,
        default=default_curves,
        help="Select curves to show in the cross-section.",
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

    # ── Formation tops overlay toggle ─────────────────────────────────────
    show_tops = st.checkbox(
        "Show formation tops",
        value=True,
        disabled=st.session_state.formation_tops is None,
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
- **CSV file** — formation tops with columns:
  `well`, `formation`, `depth` *(and optionally `color`)*

**Quick start**
1. Upload one or more LAS files.
2. *(Optional)* Upload a formation tops CSV.
3. Choose curves and depth window from the sidebar.
4. Enable flattening to align wells on a key horizon.
5. Export the figure as PNG or PDF.

**Sample data** is available in the `sample_data/` folder of this repository.
            """
        )

# ---------------------------------------------------------------------------
# Main display area
# ---------------------------------------------------------------------------

st.markdown("# Multi-Well Cross-Section Viewer")

tabs = st.tabs(["📊 Cross-Section", "📋 Data Summary", "📈 Statistics", "📖 Help"])

# ── Tab 1: Cross-Section ──────────────────────────────────────────────────
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

        with st.spinner("Rendering cross-section…"):
            fig = plot_cross_section(
                wells=working_wells,
                curves_to_plot=selected_curves,
                depth_min=flat_depth_min,
                depth_max=flat_depth_max,
                formation_tops=working_tops if show_tops else None,
                show_tops=show_tops,
                depth_label=depth_label,
                title=(
                    f"Flattened on: {reference_formation}"
                    if enable_flattening and reference_formation
                    else "Multi-Well Cross-Section"
                ),
            )

        st.pyplot(fig, use_container_width=True)

        # Export button
        buf = io.BytesIO()
        fmt = export_format.lower()
        fig.savefig(buf, format=fmt, bbox_inches="tight", dpi=150)
        buf.seek(0)
        st.download_button(
            label=f"⬇️ Download {export_format}",
            data=buf,
            file_name=f"cross_section.{fmt}",
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
## How to Use the Multi-Well Viewer

### 1. Load Well Data
- Click **Upload LAS files** in the sidebar.
- You can upload **multiple LAS files** at once.
- Each file will appear as a separate well in the viewer.

### 2. Load Formation Tops *(optional)*
- Prepare a CSV with columns: `well`, `formation`, `depth`.
- Optionally add a `color` column (hex colour, e.g. `#e6194b`).
- Upload via **Upload formation tops (CSV)** in the sidebar.

### 3. Configure Display
- Select which **log curves** to show in the cross-section.
- Adjust the **depth window** slider to zoom in/out.
- Toggle **Show formation tops** to overlay picks on the tracks.

### 4. Flatten on a Formation Top
- Enable **Flattening** in the sidebar.
- Choose the **reference formation** from the dropdown.
- All wells will be depth-shifted so the reference top aligns at 0.

### 5. Export
- Use the **Download PNG / PDF** button below the cross-section figure.

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
