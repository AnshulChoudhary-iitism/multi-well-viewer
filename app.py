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
import hashlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from data_loader import (
    get_all_curves,
    get_common_curves,
    load_formation_tops,
    load_las_file,
)
from flattening import (
    flatten_formation_tops,
    flatten_wells,
    get_flattened_depth_range,
)
from merge_logs import merge_well_logs
from visualizer import plot_cross_section, plot_curve_histogram, plot_well_correlation
from report_generator import generate_pdf_report

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
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,700&family=Source+Sans+3:wght@400;500;600;700&display=swap');

:root {
    --bg-main-1: #f8f5ee;
    --bg-main-2: #eef3f1;
    --panel: #ffffff;
    --panel-soft: #f4f7f6;
    --text-main: #223233;
    --text-muted: #536867;
    --accent: #0d6b64;
    --accent-soft: #dff2ef;
    --line: #d5dfdc;
    --shadow: 0 12px 32px rgba(27, 54, 52, 0.10);
}

html, body, [class*="css"] {
    font-family: "Source Sans 3", "Segoe UI", sans-serif;
    color: var(--text-main);
}

/* Enhanced color contrast for WCAG AA compliance */
:root {
    --text-main: #1a2f2e;
    --text-muted: #3d5354;
}

/* Improved alert styling for better contrast */
.stAlert {
    border-radius: 10px;
    border-left: 4px solid;
    padding: 1rem;
    animation: slideIn 0.3s ease;
}

@keyframes slideIn {
    from { opacity: 0; transform: translateX(-10px); }
    to { opacity: 1; transform: translateX(0); }
}

.stAlert > div > div:first-child {
    font-weight: 600;
}

/* Success alert */
.stAlert {
    --alert-color: #065e5b;
}

div[data-testid="stAlert"] {
    border-radius: 10px;
    border-left: 5px solid;
}

.stApp {
    background:
        radial-gradient(1300px 420px at -10% -8%, #ffffff 0%, rgba(255,255,255,0) 60%),
        radial-gradient(900px 360px at 110% -5%, #edf6f4 0%, rgba(237,246,244,0) 55%),
        linear-gradient(165deg, var(--bg-main-1) 0%, var(--bg-main-2) 100%);
}

[data-testid="stAppViewContainer"] > .main {
    animation: pageFade 0.45s ease-out;
}

@keyframes pageFade {
    from { opacity: 0; transform: translateY(8px); }
    to { opacity: 1; transform: translateY(0); }
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #f4f6f3 0%, #ecf1ef 100%);
    border-right: 1px solid var(--line);
}

[data-testid="stSidebar"] .stMarkdown h1,
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3,
h1, h2, h3 {
    font-family: "Fraunces", Georgia, serif !important;
    letter-spacing: 0.2px;
    color: #1a2f2e !important;
    font-weight: 700 !important;
}

/* Consistent icon styling */
.stMarkdown h1, .stMarkdown h2, .stMarkdown h3,
[data-testid="stSidebar"] .stMarkdown h1,
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3 {
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

/* Icon styling for better accessibility */
.stMarkdown h1 span:first-child,
.stMarkdown h2 span:first-child,
.stMarkdown h3 span:first-child {
    font-size: 1.2em;
    display: inline-block;
    margin-right: 0.3rem;
}

h1 { font-size: 2.1rem !important; }
h2 { font-size: 1.45rem !important; }
h3 { font-size: 1.17rem !important; }

p, li, label, .stMarkdown, .stAlert, .stCaption {
    font-size: 1.03rem !important;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 0.45rem;
}

.stTabs [data-baseweb="tab"] {
    border: 1px solid var(--line);
    border-radius: 999px;
    background: #fbfcfc;
    color: var(--text-muted);
    padding: 0.4rem 0.9rem;
}

.stTabs [aria-selected="true"] {
    background: var(--accent-soft) !important;
    border-color: #9cc9c3 !important;
    color: #144e4a !important;
    font-weight: 700;
}

div[data-testid="stMetric"] {
    background: var(--panel);
    border: 2px solid var(--line);
    border-radius: 14px;
    padding: 0.75rem 0.85rem;
    box-shadow: var(--shadow);
    transition: all 0.2s ease;
}

div[data-testid="stMetric"]:hover {
    border-color: #0d6b64;
    box-shadow: 0 12px 32px rgba(13, 107, 100, 0.15);
}

[data-testid="stMetricValue"] {
    font-size: 1.45rem !important;
    font-weight: 700 !important;
    color: #0f6f68 !important;
}

[data-testid="stMetricLabel"] {
    color: #3d5354 !important;
    font-weight: 600 !important;
}

div[data-testid="stExpander"] {
    border: 1px solid var(--line);
    border-radius: 12px;
    background: var(--panel);
    box-shadow: 0 8px 22px rgba(30, 56, 55, 0.06);
    transition: all 0.2s ease;
}

div[data-testid="stExpander"]:hover {
    border-color: #0d6b64;
    box-shadow: 0 8px 24px rgba(13, 107, 100, 0.12);
}

div[data-testid="stExpander"] > details summary {
    font-weight: 600;
    cursor: pointer;
    padding: 0.75rem;
    transition: background-color 0.2s ease;
    color: #1a2f2e;
}

div[data-testid="stExpander"] > details summary:hover {
    background-color: #f0f5f4;
}

div[data-testid="stExpander"] > details summary:focus {
    outline: 2px solid #0d6b64;
    outline-offset: -2px;
}

div[data-testid="stExpander"] > details[open] {
    border-color: #0d6b64;
}

.stButton > button,
.stDownloadButton > button {
    border-radius: 10px;
    border: 2px solid #0f6f68;
    background: linear-gradient(180deg, #1f8a82 0%, #0f6f68 100%);
    color: #ffffff;
    font-weight: 700;
    box-shadow: 0 8px 18px rgba(17, 91, 85, 0.25);
    transition: all 0.2s ease;
    cursor: pointer;
    padding: 0.5rem 1rem;
}

.stButton > button:hover,
.stDownloadButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 14px 28px rgba(17, 91, 85, 0.35);
    background: linear-gradient(180deg, #2a9a94 0%, #1a7f78 100%);
    border-color: #0a5a54;
}

.stButton > button:focus,
.stDownloadButton > button:focus {
    outline: 3px solid #0d6b64;
    outline-offset: 2px;
}

.stButton > button:active,
.stDownloadButton > button:active {
    transform: translateY(0);
    box-shadow: 0 4px 12px rgba(17, 91, 85, 0.2);
}

.stButton > button:disabled,
.stDownloadButton > button:disabled {
    background: linear-gradient(180deg, #b0b0b0 0%, #8a8a8a 100%);
    color: #ffffff;
    border-color: #707070;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    cursor: not-allowed;
    opacity: 0.6;
    transform: none;
}

.stSelectbox, .stMultiSelect, .stSlider, .stRadio, .stCheckbox, .stNumberInput {
    background: rgba(255,255,255,0.56);
    border-radius: 10px;
    padding: 0.1rem 0.35rem;
}

/* Improved form element focus states for accessibility */
.stSelectbox > div > div > div,
.stMultiSelect > div > div > div,
.stNumberInput input {
    transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
}

.stSelectbox > div > div > div:focus,
.stMultiSelect > div > div > div:focus,
.stNumberInput input:focus {
    border-color: #0d6b64 !important;
    box-shadow: 0 0 0 3px rgba(13, 107, 100, 0.1) !important;
}

/* Checkbox and Radio button improvements */
.stCheckbox > label,
.stRadio > label {
    cursor: pointer;
    font-weight: 500;
    color: #223233;
    transition: color 0.2s ease;
}

.stCheckbox > label:hover,
.stRadio > label:hover {
    color: #0d6b64;
}

.stCheckbox > label > span:first-child,
.stRadio > label > span:first-child {
    display: inline-block;
    margin-right: 0.5rem;
}

/* Better contrast for labels */
label, .stLabel {
    color: #1a2f2e !important;
    font-weight: 600 !important;
}

[data-testid="stDataFrame"] {
    border: 1px solid var(--line);
    border-radius: 12px;
    overflow: hidden;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
}

/* Table styling for better readability */
table {
    color: #1a2f2e !important;
}

table th {
    background-color: #f0f5f4 !important;
    color: #1a2f2e !important;
    font-weight: 700 !important;
}

table td {
    color: #3d5354 !important;
}

table tr:hover td {
    background-color: #fafcfb !important;
}

.hero-card {
    border: 1px solid #c9d9d6;
    border-radius: 16px;
    padding: 1.1rem 1.2rem 1rem;
    background:
        linear-gradient(110deg, rgba(255,255,255,0.95) 0%, rgba(247,252,251,0.90) 55%, rgba(238,248,246,0.95) 100%);
    box-shadow: var(--shadow);
    margin-bottom: 0.65rem;
}

.hero-kicker {
    display: inline-block;
    font-size: 0.78rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    font-weight: 700;
    color: #11615c;
    background: #e0f3f0;
    border: 1px solid #b8ddd7;
    border-radius: 999px;
    padding: 0.22rem 0.58rem;
    margin-bottom: 0.55rem;
}

.hero-title {
    font-family: "Fraunces", Georgia, serif;
    margin: 0;
    color: #1e3735;
    font-size: 1.65rem;
    line-height: 1.2;
}

.hero-sub {
    margin-top: 0.45rem;
    margin-bottom: 0;
    color: var(--text-muted);
}

.wf-stepper {
    display: flex;
    align-items: center;
    gap: 0.45rem;
    margin: 0.1rem 0 0.9rem;
    overflow-x: auto;
    padding-bottom: 0.1rem;
}

.wf-chip {
    border: 1px solid #cad8d5;
    border-radius: 999px;
    padding: 0.26rem 0.65rem;
    font-size: 0.86rem;
    white-space: nowrap;
    background: #ffffff;
    color: #55706f;
}

.wf-chip.done {
    border-color: #9ccdc7;
    background: #e6f7f4;
    color: #165d57;
    font-weight: 700;
}

.wf-chip.active {
    border-color: #b1c8c4;
    background: #f4faf9;
    color: #234948;
    font-weight: 700;
}

.wf-arrow {
    color: #78928f;
    font-size: 0.95rem;
}

@media (max-width: 900px) {
    h1 { font-size: 1.7rem !important; }
    .hero-title { font-size: 1.34rem; }
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

if "formation_tops_upload_sig" not in st.session_state:
    st.session_state.formation_tops_upload_sig = None

if "formation_tops_upload_name" not in st.session_state:
    st.session_state.formation_tops_upload_name = None

if "last_render_success" not in st.session_state:
    st.session_state.last_render_success = False

if "last_export_success" not in st.session_state:
    st.session_state.last_export_success = False


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
    logo_paths = []
    if os.path.exists(IIT_ISM_LOGO_PATH):
        logo_paths.append(IIT_ISM_LOGO_PATH)
    if logo_paths:
        st.image(logo_paths, width=92)
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

    # ── Merge logs technique ────────────────────────────────────────────────
    if len(st.session_state.wells) >= 2:
        st.markdown("---")
        st.header("🔗 Merge Logs")
        st.caption("Combine curves from two wells using formation tops alignment.")
        
        well_names = list(st.session_state.wells.keys())
        
        col1, col2 = st.columns(2)
        with col1:
            well_a = st.selectbox(
                "Well 1",
                options=well_names,
                index=0,
                key="merge_well_a"
            )
        with col2:
            # Filter out well_a from well_b options
            well_b_options = [w for w in well_names if w != well_a]
            if well_b_options:
                well_b = st.selectbox(
                    "Well 2",
                    options=well_b_options,
                    index=0,
                    key="merge_well_b"
                )
            else:
                st.warning("Need at least 2 different wells to merge")
                well_b = None
        
        if well_b is not None:
            alignment_formation = None
            if st.session_state.formation_tops is not None and not st.session_state.formation_tops.empty:
                formations = sorted(st.session_state.formation_tops["formation"].unique())
                alignment_formation = st.selectbox(
                    "Alignment formation",
                    options=[None] + list(formations),
                    format_func=lambda x: "No alignment (depth merge)" if x is None else x,
                    key="merge_alignment"
                )
            
            if st.button("🔗 Merge Wells", key="merge_button"):
                merged_well = merge_well_logs(
                    st.session_state.wells[well_a],
                    st.session_state.wells[well_b],
                    formation_tops=st.session_state.formation_tops,
                    alignment_formation=alignment_formation,
                )
                
                if merged_well is not None:
                    # Add merged well to session state
                    merged_name = merged_well["name"]
                    st.session_state.wells[merged_name] = merged_well
                    st.success(
                        f"✅ Merged: **{well_a}** + **{well_b}** → **{merged_name}**"
                    )
                    if alignment_formation:
                        st.info(
                            f"Aligned on formation: **{alignment_formation}** "
                            f"(depth offset: {merged_well['alignment_depth_offset']:.2f} m)"
                        )
                    st.rerun()
                else:
                    st.error("Failed to merge wells. Check data compatibility.")

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
        tops_bytes = tops_file.getvalue()
        tops_sig = hashlib.md5(tops_bytes).hexdigest() if tops_bytes else None

        # Process tops only when a newly selected file is different from the
        # last successfully loaded one to avoid stale read-pointer behaviour
        # across Streamlit reruns.
        if tops_sig and tops_sig != st.session_state.formation_tops_upload_sig:
            tops_df = load_formation_tops(tops_file)
            if tops_df is not None:
                st.session_state.formation_tops = tops_df
                st.session_state.formation_tops_upload_sig = tops_sig
                st.session_state.formation_tops_upload_name = tops_file.name
                st.success(
                    f"Loaded {len(tops_df)} top entries across "
                    f"{tops_df['well'].nunique()} well(s)."
                )
            else:
                st.session_state.formation_tops = None
                st.session_state.formation_tops_upload_sig = None
                st.session_state.formation_tops_upload_name = None
                st.error(
                    "Failed to load formation tops. "
                    "Ensure file has columns: well, formation, depth."
                )

    if st.session_state.formation_tops is not None:
        active_name = st.session_state.formation_tops_upload_name or "Unknown"
        active_sig = st.session_state.formation_tops_upload_sig or ""
        short_sig = active_sig[:8] if active_sig else "n/a"
        st.caption(f"Active tops file: {active_name} | id: {short_sig}")
        if st.button("🗑️ Clear tops"):
            st.session_state.formation_tops = None
            st.session_state.formation_tops_upload_sig = None
            st.session_state.formation_tops_upload_name = None
            st.rerun()

    # ── Visualisation controls ─────────────────────────────────────────────
    st.markdown("---")
    st.header("⚙️ Display Settings")

    has_wells = bool(st.session_state.wells)
    has_tops = (
        st.session_state.formation_tops is not None
        and not st.session_state.formation_tops.empty
    )

    all_curves = get_all_curves(st.session_state.wells)
    common_curves = get_common_curves(st.session_state.wells)

    track_pool_mode = st.radio(
        "Track source",
        ["All available tracks", "Common tracks only"],
        horizontal=True,
        help="Choose whether to pick tracks from all loaded LAS curves or only curves present in every well.",
        disabled=not has_wells,
    )
    available_track_options = (
        all_curves if track_pool_mode == "All available tracks" else common_curves
    )

    # Use a focused default set (GR, NPHI, RHOB) for clarity and readability
    # Curves are now normalized at load time (GR/HCGR → GR, RHOB variants → RHOB, etc.)
    preferred_order = ["GR", "NPHI", "RHOB"]
    default_curves = [c for c in preferred_order if c in available_track_options]

    # Backfill if fewer than 3 were found
    if len(default_curves) < 3:
        for c in available_track_options:
            if c not in default_curves:
                default_curves.append(c)
            if len(default_curves) >= 3:
                break

    selected_curves = st.multiselect(
        "Tracks to display",
        options=available_track_options,
        default=default_curves,
        help="Choose exactly which tracks you want to see in the data visualization.",
        disabled=not has_wells,
    )

    depth_label = st.selectbox(
        "Depth axis label",
        ["Depth (m)", "MD (m)", "TVD (m)", "TVDSS (m)"],
        index=0,
        disabled=not has_wells,
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
        disabled=not has_wells,
    )

    # ── Quality check (smoothing) ───────────────────────────────────────
    quality_check_enabled = st.checkbox(
        "Quality check (moving average)",
        value=False,
        help="Smooth displayed tracks using a moving average filter for quick QC.",
        disabled=not has_wells,
    )
    average_filter_window = st.selectbox(
        "Average filter window",
        options=[5, 10, 15],
        index=0,
        disabled=(not quality_check_enabled) or (not has_wells),
    )

    # ── Formation tops overlay toggle ─────────────────────────────────────
    show_tops = st.checkbox(
        "Show formation tops",
        value=True,
        disabled=(not has_tops) or (not has_wells),
    )

    shade_formations = st.checkbox(
        "Shade formation intervals",
        value=True,
        disabled=(not has_tops) or (not has_wells),
        help="Fill intervals between consecutive formation tops with translucent formation colors.",
    )
    formation_shade_alpha = st.slider(
        "Formation shade intensity",
        min_value=0.05,
        max_value=0.40,
        value=0.14,
        step=0.01,
        disabled=(not has_tops) or (not shade_formations) or (not has_wells),
        help="Higher values make formation color shading stronger.",
    )

    # ── Flattening ────────────────────────────────────────────────────────
    st.markdown("---")
    st.header("🔄 Flattening")

    enable_flattening = st.checkbox(
        "Enable flattening",
        value=False,
        disabled=(not has_tops) or (not has_wells),
    )
    reference_formation = None
    if enable_flattening and has_tops:
        # Predefined common flattening surfaces
        predefined_surfaces = [
            "Maximum Flooding Surface (MFS)",
            "Sequence Boundary",
        ]
        
        formations = sorted(
            st.session_state.formation_tops["formation"].unique().tolist()
        )
        
        # Combine predefined surfaces with loaded formations, avoiding duplicates
        all_options = predefined_surfaces + [f for f in formations if f not in predefined_surfaces]
        
        reference_formation = st.selectbox(
            "Flatten on:",
            options=all_options,
            help="Choose a reference surface to align all wells. Can be a predefined surface or a loaded formation top.",
        )
    elif enable_flattening:
        st.info("Load formation tops to enable flattening.")

    # ── Visualization style ───────────────────────────────────────────────
    st.markdown("---")
    st.header("📊 Visualization")
    vis_mode = st.radio(
        "Display mode",
        ["Cross-section with curves", "Well correlation diagram"],
        help="Choose between traditional log-curve cross-section or stratigraphic formation correlation view.",
        disabled=not has_wells,
    )

    show_gr_in_correlation = st.checkbox(
        "Show GR curve overlay",
        value=True,
        disabled=(vis_mode != "Well correlation diagram") or (not has_wells),
        help="Overlay a thin, normalized GR profile on each column in the correlation diagram.",
    )

    # ── PDF Report ────────────────────────────────────────────────────────
    st.markdown("---")
    st.header("📄 PDF Report")
    generate_report = st.checkbox(
        "Generate comprehensive report",
        value=False,
        disabled=(not has_wells) or (not has_tops),
        help="One-click PDF report with well summary, formation tops, and thickness analysis.",
    )

    # ── Export ────────────────────────────────────────────────────────────
    st.markdown("---")
    st.header("💾 Export")
    ready_to_render = has_wells and bool(selected_curves)
    export_format = st.radio(
        "Format",
        ["PNG", "PDF"],
        horizontal=True,
        disabled=not ready_to_render,
    )

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

st.markdown(
        """
<div class="hero-card">
    <span class="hero-kicker">Subsurface Correlation Studio</span>
    <h1 class="hero-title">Multi-Well Correlation and Flattening Tool</h1>
    <p class="hero-sub">Upload LAS files, overlay picks, and flatten against a reference horizon.</p>
</div>
""",
        unsafe_allow_html=True,
)

step_load_wells = bool(st.session_state.wells)
step_load_tops = (
    st.session_state.formation_tops is not None
    and not st.session_state.formation_tops.empty
)
step_choose_tracks = step_load_wells and bool(selected_curves)
step_render = bool(st.session_state.last_render_success)
step_export = bool(st.session_state.last_export_success)

def _step_chip(label: str, done: bool, active: bool = False) -> str:
    css = "wf-chip"
    if done:
        css += " done"
    elif active:
        css += " active"
    icon = "✓" if done else "○"
    return f'<span class="{css}">{icon} {label}</span>'

stepper_html = "".join(
    [
        _step_chip("Load wells", step_load_wells, not step_load_wells),
        '<span class="wf-arrow">→</span>',
        _step_chip("Load tops", step_load_tops, step_load_wells and not step_load_tops),
        '<span class="wf-arrow">→</span>',
        _step_chip("Choose tracks", step_choose_tracks, step_load_wells and not step_choose_tracks),
        '<span class="wf-arrow">→</span>',
        _step_chip("Render", step_render, step_choose_tracks and not step_render),
        '<span class="wf-arrow">→</span>',
        _step_chip("Export", step_export, step_render and not step_export),
    ]
)
st.markdown(f'<div class="wf-stepper">{stepper_html}</div>', unsafe_allow_html=True)

tabs = st.tabs(["📊 Data Visualization", "📋 Data Summary", "📈 Statistics", "📖 Help"])

# ── Tab 1: Data Visualization ─────────────────────────────────────────────
with tabs[0]:
    if not st.session_state.wells:
        st.session_state.last_render_success = False
        st.session_state.last_export_success = False
        st.info(
            "👈 Upload LAS files from the sidebar to get started.  "
            "Sample data is available in the `sample_data/` directory."
        )
    elif not selected_curves:
        st.session_state.last_render_success = False
        st.session_state.last_export_success = False
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

        st.session_state.last_export_success = False
        with st.spinner("Rendering data visualization…"):
            if vis_mode == "Well correlation diagram":
                # Render well correlation diagram
                fig = plot_well_correlation(
                    wells=working_wells,
                    formation_tops=working_tops if show_tops else None,
                    depth_min=flat_depth_min,
                    depth_max=flat_depth_max,
                    depth_label=depth_label,
                    title=figure_title,
                    show_gr_curve=show_gr_in_correlation,
                )
            else:
                # Render traditional cross-section with curves
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
            st.session_state.last_render_success = True

        st.pyplot(fig, use_container_width=True)

        # Export button
        buf = io.BytesIO()
        fmt = export_format.lower()
        fig.savefig(buf, format=fmt, bbox_inches="tight", dpi=150)
        buf.seek(0)
        download_clicked = st.download_button(
            label=f"⬇️ Download {export_format}",
            data=buf,
            file_name=f"data_visualization.{fmt}",
            mime=f"image/{fmt}" if fmt == "png" else "application/pdf",
        )
        if download_clicked:
            st.session_state.last_export_success = True
        plt.close(fig)

        # PDF Report button
        st.markdown("---")
        if generate_report and has_tops:
            with st.spinner("Generating PDF report…"):
                try:
                    report_buf = generate_pdf_report(
                        wells=st.session_state.wells,
                        formation_tops=st.session_state.formation_tops,
                        title="Multi-Well Analysis Report",
                    )
                    st.download_button(
                        label="📥 Download PDF Report",
                        data=report_buf,
                        file_name="multi_well_report.pdf",
                        mime="application/pdf",
                    )
                    st.success("✅ PDF report generated successfully!")
                except ImportError:
                    st.error(
                        "❌ reportlab library not found. Install with: `pip install reportlab`"
                    )
                except Exception as e:
                    st.error(f"❌ Error generating report: {str(e)}")
        elif generate_report:
            st.warning("Load formation tops to generate the PDF report.")

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

# ── Tab 4: Help ───────────────────────────────────────────────────────────
with tabs[3]:
    st.markdown("""
    ### 📖 Quick Start Guide
    
    **1. Load Wells:**
    - Upload one or more LAS well-log files using the file uploader
    
    **2. Configure Display:**
    - Select which curves to display (GR, Resistivity, Density, etc.)
    - Choose smoothing and depth range as needed
    
    **3. Add Formation Tops:**
    - Import a CSV file with formation picks to overlay on the cross-section
    
    **4. Flatten (Optional):**
    - Select a reference formation to flatten all wells to that horizon
    
    **5. Export Results:**
    - Download the cross-section as PNG or PDF
    
    ### 📊 Features
    - **Multi-well cross-section** with configurable log tracks
    - **Formation tops overlay** with color-coded markers
    - **Depth flattening** to align wells on a reference formation
    - **Statistics panel** with per-well curve statistics
    - **PDF reports** for documentation and sharing
    """)
