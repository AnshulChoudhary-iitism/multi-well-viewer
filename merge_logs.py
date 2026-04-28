"""
Merge log technique for combining data from two wells.
Aligns wells using formation tops and creates a composite log.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Optional, Tuple


def merge_well_logs(
    well_a: dict,
    well_b: dict,
    formation_tops: Optional[pd.DataFrame] = None,
    alignment_formation: Optional[str] = None,
) -> Optional[dict]:
    """Merge logs from two wells using formation tops alignment.
    
    Parameters
    ----------
    well_a : dict
        First well data dictionary (from load_las_file)
    well_b : dict
        Second well data dictionary (from load_las_file)
    formation_tops : pd.DataFrame or None
        Formation tops with columns: well, formation, depth, color
    alignment_formation : str or None
        Formation name to use for depth alignment. If None, merges without alignment.
    
    Returns
    -------
    dict or None
        Merged well data with keys:
        - name: merged well name
        - df: merged dataframe
        - curves: list of available curves
        - depth: depth array
        - wells_merged: tuple of (well_a_name, well_b_name)
        - alignment_depth_offset: depth offset applied
        Or None if merge fails.
    """
    try:
        df_a = well_a["df"].copy()
        df_b = well_b["df"].copy()
        depth_col_a = df_a.columns[0]
        depth_col_b = df_b.columns[0]
        
        # Calculate depth offset based on formation alignment
        depth_offset = 0.0
        if formation_tops is not None and alignment_formation is not None:
            tops_a = formation_tops[
                (formation_tops["well"].str.lower() == well_a["name"].lower()) &
                (formation_tops["formation"].str.lower() == alignment_formation.lower())
            ]
            tops_b = formation_tops[
                (formation_tops["well"].str.lower() == well_b["name"].lower()) &
                (formation_tops["formation"].str.lower() == alignment_formation.lower())
            ]
            
            if not tops_a.empty and not tops_b.empty:
                depth_a = float(tops_a["depth"].iloc[0])
                depth_b = float(tops_b["depth"].iloc[0])
                depth_offset = depth_a - depth_b  # Offset to apply to well_b
        
        # Apply depth offset to well_b
        if depth_offset != 0:
            df_b[depth_col_b] = df_b[depth_col_b] + depth_offset
        
        # Rename depth columns to match
        df_b = df_b.rename(columns={depth_col_b: depth_col_a})
        
        # Get depth range that covers both wells
        depth_min = min(df_a[depth_col_a].min(), df_b[depth_col_a].min())
        depth_max = max(df_a[depth_col_a].max(), df_b[depth_col_a].max())
        
        # Create common depth array
        depth_step = max(
            np.diff(df_a[depth_col_a].values).mean(),
            np.diff(df_b[depth_col_a].values).mean()
        )
        common_depth = np.arange(depth_min, depth_max + depth_step, depth_step)
        
        # Merge dataframes on depth
        merged_df = pd.DataFrame({depth_col_a: common_depth})
        
        # Get all curves from both wells with suffixes to track origin
        curves_a = [c for c in df_a.columns if c != depth_col_a]
        curves_b = [c for c in df_b.columns if c != depth_col_b]
        
        # Interpolate curves from well A
        for curve in curves_a:
            curve_name_a = f"{curve}_({well_a['name']})"
            # Interpolate and use nearest value extrapolation for edges
            interpolated = np.interp(
                common_depth,
                df_a[depth_col_a].values,
                df_a[curve].values,
                left=np.nan,
                right=np.nan
            )
            merged_df[curve_name_a] = interpolated
        
        # Interpolate curves from well B
        for curve in curves_b:
            if curve != depth_col_a:  # Skip depth column
                curve_name_b = f"{curve}_({well_b['name']})"
                interpolated = np.interp(
                    common_depth,
                    df_b[depth_col_a].values,
                    df_b[curve].values,
                    left=np.nan,
                    right=np.nan
                )
                merged_df[curve_name_b] = interpolated
        
        # Create merged well name
        merged_name = f"{well_a['name']}+{well_b['name']}"
        
        # Get all curves
        all_curves = [c for c in merged_df.columns if c != depth_col_a]
        
        # Create complete info dictionary
        merged_info = {
            "well": merged_name,
            "field": well_a["info"].get("field", "") or well_b["info"].get("field", ""),
            "company": well_a["info"].get("company", "") or well_b["info"].get("company", ""),
            "kb": max(well_a["info"].get("kb", 0), well_b["info"].get("kb", 0)),
            "strt": float(depth_min),
            "stop": float(depth_max),
            "step": float(depth_step),
            "merged": True,
            "parent_wells": (well_a["name"], well_b["name"]),
        }
        
        return {
            "name": merged_name,
            "df": merged_df,
            "curves": all_curves,
            "depth": common_depth,
            "wells_merged": (well_a["name"], well_b["name"]),
            "alignment_formation": alignment_formation,
            "alignment_depth_offset": depth_offset,
            "units": {},
            "info": merged_info,
        }
    
    except Exception as e:
        print(f"Error merging wells: {e}")
        return None
