"""
Formation flattening logic for the Multi-Well Viewer application.

Flattening shifts each well's depth axis so that a chosen reference
formation top is aligned at a common datum (zero offset by default).
"""

import copy
import re
import numpy as np
import pandas as pd
from typing import Optional


def _normalize_well_name(name: str) -> str:
    """Return a canonical key for matching well names across files."""
    return re.sub(r"[^a-z0-9]+", "", str(name).strip().lower())


def flatten_wells(
    wells: dict,
    formation_tops: pd.DataFrame,
    reference_formation: str,
    reference_depth: float = 0.0,
) -> dict:
    """Return a copy of *wells* with depth axes shifted by flattening.

    For each well, the depth shift is computed as::

        shift = reference_depth - top_depth_in_well

    where *top_depth_in_well* is the depth of *reference_formation* in that
    well's formation tops table.  Wells that do not have the reference
    formation are included unchanged (no shift applied).

    Parameters
    ----------
    wells : dict
        Mapping ``well_name → well_data`` as produced by
        :func:`data_loader.load_las_file`.
    formation_tops : pd.DataFrame
        Formation tops table as produced by
        :func:`data_loader.load_formation_tops`.
    reference_formation : str
        Name of the formation top to use as the flattening datum.
    reference_depth : float, optional
        The depth value at which the reference top will be placed in the
        flattened view.  Defaults to ``0.0``.

    Returns
    -------
    dict
        A new dict with the same structure as *wells* but each well's
        ``"df"`` DataFrame has its depth column shifted, and an extra key
        ``"shift"`` records the applied offset.
    """
    # Build lookup: well_name → depth of reference formation
    ref_rows = formation_tops[
        formation_tops["formation"].str.strip() == reference_formation.strip()
    ]
    ref_tops = {
        _normalize_well_name(row["well"]): float(row["depth"])
        for _, row in ref_rows.iterrows()
    }

    flattened = {}
    for well_name, well_data in wells.items():
        flat_data = copy.deepcopy(well_data)
        df = flat_data["df"].copy()
        depth_col = df.columns[0]

        top_depth = ref_tops.get(_normalize_well_name(well_name))

        if top_depth is not None:
            shift = reference_depth - float(top_depth)
        else:
            shift = 0.0  # no shift if top not available for this well

        df[depth_col] = df[depth_col] + shift
        flat_data["df"] = df
        flat_data["depth"] = df[depth_col].values
        flat_data["shift"] = shift
        flattened[well_name] = flat_data

    return flattened


def flatten_formation_tops(
    formation_tops: pd.DataFrame,
    wells: dict,
    reference_formation: str,
    reference_depth: float = 0.0,
) -> pd.DataFrame:
    """Return a copy of *formation_tops* with depth values shifted.

    Uses the same shifts that :func:`flatten_wells` would compute so that
    formation top markers remain aligned with the flattened well logs.

    Parameters
    ----------
    formation_tops : pd.DataFrame
        As produced by :func:`data_loader.load_formation_tops`.
    wells : dict
        Original (un-flattened) well data used to compute shifts.
    reference_formation : str
        Name of the reference formation.
    reference_depth : float, optional
        Target depth for the reference top.

    Returns
    -------
    pd.DataFrame
        Copy of *formation_tops* with adjusted ``depth`` values.
    """
    ref_rows = formation_tops[
        formation_tops["formation"].str.strip() == reference_formation.strip()
    ]
    ref_tops = {
        _normalize_well_name(row["well"]): float(row["depth"])
        for _, row in ref_rows.iterrows()
    }

    flat_tops = formation_tops.copy()
    shifts = {}

    for well_name in formation_tops["well"].unique():
        top_depth = ref_tops.get(_normalize_well_name(well_name))
        if top_depth is not None:
            shifts[well_name] = reference_depth - float(top_depth)
        else:
            shifts[well_name] = 0.0

    flat_tops["depth"] = flat_tops.apply(
        lambda row: row["depth"] + shifts.get(row["well"], 0.0), axis=1
    )
    return flat_tops


def get_flattened_depth_range(flattened_wells: dict) -> tuple:
    """Return the (min, max) depth range across all flattened wells."""
    all_depths = []
    for well in flattened_wells.values():
        all_depths.extend(well["depth"].tolist())
    if not all_depths:
        return (0.0, 1000.0)
    return (float(np.min(all_depths)), float(np.max(all_depths)))
