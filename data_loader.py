"""
Data loading utilities for the Multi-Well Viewer application.
Handles LAS file parsing, CSV formation tops import, and data validation.
"""

import io
import warnings
from typing import Optional

import lasio
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# LAS file loading
# ---------------------------------------------------------------------------

def load_las_file(uploaded_file) -> Optional[dict]:
    """Parse a LAS file uploaded via Streamlit and return a well data dict.

    Parameters
    ----------
    uploaded_file:
        A file-like object returned by ``st.file_uploader``.

    Returns
    -------
    dict or None
        A dictionary with keys:
        - ``name``   : well name (str)
        - ``las``    : the raw ``lasio.LASFile`` object
        - ``df``     : ``pd.DataFrame`` of log curves indexed by depth
        - ``curves`` : list of available curve mnemonics (excluding depth)
        - ``depth``  : the depth array as a 1-D numpy array
        - ``units``  : dict mapping mnemonic → unit string
        - ``info``   : dict of well header information
        Returns ``None`` on failure.
    """
    try:
        raw_bytes = uploaded_file.read()
        las = lasio.read(io.StringIO(raw_bytes.decode("utf-8", errors="replace")))
    except Exception as exc:
        warnings.warn(f"Could not parse LAS file: {exc}")
        return None

    # Build DataFrame
    try:
        df = las.df()
    except Exception:
        return None

    df.index.name = "DEPTH"
    df.reset_index(inplace=True)

    # Well name from header or filename
    well_name = _extract_well_name(las, uploaded_file.name)

    depth_col = df.columns[0]  # first column after reset_index is depth
    depth_array = df[depth_col].values
    curve_mnemonics = [c for c in df.columns if c != depth_col]

    units = {c.mnemonic: c.unit for c in las.curves if c.mnemonic != depth_col}

    info = {
        "well": str(las.well.get("WELL", {}).value if las.well.get("WELL") else ""),
        "field": str(las.well.get("FLD", {}).value if las.well.get("FLD") else ""),
        "company": str(las.well.get("COMP", {}).value if las.well.get("COMP") else ""),
        "kb": float(las.well.get("KB", {}).value) if las.well.get("KB") else 0.0,
        "strt": float(las.well.get("STRT", {}).value) if las.well.get("STRT") else depth_array.min(),
        "stop": float(las.well.get("STOP", {}).value) if las.well.get("STOP") else depth_array.max(),
        "step": float(las.well.get("STEP", {}).value) if las.well.get("STEP") else np.diff(depth_array[:2])[0] if len(depth_array) > 1 else 0.5,
    }

    return {
        "name": well_name,
        "las": las,
        "df": df,
        "curves": curve_mnemonics,
        "depth": depth_array,
        "units": units,
        "info": info,
    }


def _extract_well_name(las: lasio.LASFile, filename: str) -> str:
    """Attempt to get a meaningful well name from LAS header or filename."""
    try:
        name = str(las.well["WELL"].value).strip()
        if name and name not in ("", "UNKNOWN", "unknown"):
            return name
    except Exception:
        pass
    # Fall back to filename without extension
    return filename.rsplit(".", 1)[0]


# ---------------------------------------------------------------------------
# Formation tops loading
# ---------------------------------------------------------------------------

REQUIRED_TOPS_COLS = {"well", "formation", "depth"}


def load_formation_tops(uploaded_file) -> Optional[pd.DataFrame]:
    """Parse a CSV file of formation tops.

    Expected CSV columns (case-insensitive):
        well, formation, depth

    An optional ``color`` column may be included.

    Returns
    -------
    pd.DataFrame or None
        Standardised DataFrame with columns:
        ``well``, ``formation``, ``depth``, ``color``.
        Returns ``None`` if parsing fails.
    """
    try:
        raw_bytes = uploaded_file.read()
        df = pd.read_csv(io.BytesIO(raw_bytes))
    except Exception as exc:
        warnings.warn(f"Could not parse formation tops CSV: {exc}")
        return None

    df.columns = [c.strip().lower() for c in df.columns]

    missing = REQUIRED_TOPS_COLS - set(df.columns)
    if missing:
        warnings.warn(f"Formation tops CSV missing columns: {missing}")
        return None

    if "color" not in df.columns:
        formations = df["formation"].unique()
        color_cycle = _default_colors(len(formations))
        color_map = {f: color_cycle[i] for i, f in enumerate(formations)}
        df["color"] = df["formation"].map(color_map)

    df["depth"] = pd.to_numeric(df["depth"], errors="coerce")
    df.dropna(subset=["depth"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df[["well", "formation", "depth", "color"]]


def _default_colors(n: int) -> list:
    """Generate a list of ``n`` distinct hex colours."""
    palette = [
        "#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231",
        "#911eb4", "#42d4f4", "#f032e6", "#bfef45", "#fabebe",
        "#469990", "#dcbeff", "#9a6324", "#fffac8", "#800000",
        "#aaffc3", "#808000", "#ffd8b1", "#000075", "#a9a9a9",
    ]
    return [palette[i % len(palette)] for i in range(n)]


# ---------------------------------------------------------------------------
# Data validation
# ---------------------------------------------------------------------------

def validate_wells(wells: dict) -> list:
    """Return a list of warning strings for common data quality issues.

    Parameters
    ----------
    wells : dict
        Mapping of well name → well data dict (as returned by
        :func:`load_las_file`).
    """
    warnings_list = []

    for name, well in wells.items():
        df = well["df"]
        depth_col = df.columns[0]

        # Check for null depth values
        null_depth = df[depth_col].isna().sum()
        if null_depth:
            warnings_list.append(f"{name}: {null_depth} null depth values.")

        # Check depth monotonicity
        depth = df[depth_col].values
        if not np.all(np.diff(depth) > 0):
            warnings_list.append(f"{name}: Depth is not strictly increasing.")

        # Check for all-null curve columns
        for curve in well["curves"]:
            null_frac = df[curve].isna().mean()
            if null_frac == 1.0:
                warnings_list.append(f"{name}/{curve}: All values are null.")
            elif null_frac > 0.5:
                warnings_list.append(
                    f"{name}/{curve}: {null_frac*100:.0f}% null values."
                )

    return warnings_list


def get_common_curves(wells: dict) -> list:
    """Return curve mnemonics present in ALL loaded wells."""
    if not wells:
        return []
    sets = [set(w["curves"]) for w in wells.values()]
    common = sets[0]
    for s in sets[1:]:
        common &= s
    return sorted(common)


def get_all_curves(wells: dict) -> list:
    """Return curve mnemonics present in ANY loaded well."""
    if not wells:
        return []
    all_curves = set()
    for w in wells.values():
        all_curves |= set(w["curves"])
    return sorted(all_curves)
