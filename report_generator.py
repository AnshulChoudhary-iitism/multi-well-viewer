"""
PDF Report generation for the Multi-Well Viewer application.

Generates comprehensive PDF reports including well summaries,
formation tops tables, and thickness maps.
"""

from __future__ import annotations

import io
import re
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from datetime import datetime

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter, A4, landscape
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import (
        SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer,
        PageBreak, Image as RLImage, KeepTogether
    )
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
    HAS_REPORTLAB = True
except ImportError:
    HAS_REPORTLAB = False


def _normalize_well_name(name: str) -> str:
    """Return a canonical key for matching well names across files."""
    return re.sub(r"[^a-z0-9]+", "", str(name).strip().lower())


def calculate_thicknesses(
    formation_tops: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate thickness of each formation interval per well.
    
    Parameters
    ----------
    formation_tops : pd.DataFrame
        Formation tops table with columns: well, formation, depth
        
    Returns
    -------
    pd.DataFrame
        Thickness table with columns: well, formation, top_depth, base_depth, thickness
    """
    if formation_tops is None or formation_tops.empty:
        return pd.DataFrame()
    
    well_groups = formation_tops.groupby(
        formation_tops["well"].map(_normalize_well_name)
    )
    
    thickness_rows = []
    for well_key, group in well_groups:
        group_sorted = group.sort_values("depth")
        well_name = group_sorted.iloc[0]["well"]
        
        for idx in range(len(group_sorted) - 1):
            top_rec = group_sorted.iloc[idx]
            base_rec = group_sorted.iloc[idx + 1]
            
            top_depth = float(top_rec["depth"])
            base_depth = float(base_rec["depth"])
            thickness = base_depth - top_depth
            
            thickness_rows.append({
                "Well": well_name,
                "Formation": str(top_rec["formation"]),
                "Top (m)": f"{top_depth:.2f}",
                "Base (m)": f"{base_depth:.2f}",
                "Thickness (m)": f"{thickness:.2f}",
            })
    
    return pd.DataFrame(thickness_rows)


def generate_pdf_report(
    wells: dict,
    formation_tops: Optional[pd.DataFrame] = None,
    title: str = "Multi-Well Analysis Report",
    output_path: Optional[str] = None,
) -> io.BytesIO:
    """Generate a comprehensive PDF report.
    
    Parameters
    ----------
    wells : dict
        Well data dict as produced by load_las_file
    formation_tops : pd.DataFrame or None
        Formation tops table
    title : str
        Report title
    output_path : str or None
        If provided, save PDF to this file path
        
    Returns
    -------
    io.BytesIO
        PDF content as bytes buffer
    """
    if not HAS_REPORTLAB:
        raise ImportError(
            "reportlab is required for PDF generation. "
            "Install it with: pip install reportlab"
        )
    
    # Create PDF document
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        topMargin=0.5 * inch,
        bottomMargin=0.5 * inch,
        leftMargin=0.5 * inch,
        rightMargin=0.5 * inch,
    )
    
    story = []
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        "CustomTitle",
        parent=styles["Heading1"],
        fontSize=20,
        textColor=colors.HexColor("#1f3735"),
        spaceAfter=12,
        alignment=TA_CENTER,
        fontName="Helvetica-Bold",
    )
    
    heading_style = ParagraphStyle(
        "CustomHeading",
        parent=styles["Heading2"],
        fontSize=14,
        textColor=colors.HexColor("#0d6b64"),
        spaceAfter=10,
        spaceBefore=6,
        fontName="Helvetica-Bold",
    )
    
    normal_style = ParagraphStyle(
        "CustomNormal",
        parent=styles["Normal"],
        fontSize=10,
        textColor=colors.HexColor("#223233"),
    )
    
    # Title page
    story.append(Paragraph(title, title_style))
    story.append(Spacer(1, 0.2 * inch))
    story.append(
        Paragraph(
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            normal_style,
        )
    )
    story.append(Spacer(1, 0.3 * inch))
    
    # --- Well Summary Section ---
    story.append(Paragraph("1. Well Summary", heading_style))
    
    well_summary_data = [["Well Name", "Top (m)", "Base (m)", "Curves", "Null %"]]
    for wname, well in wells.items():
        info = well["info"]
        curves = len(well.get("curves", []))
        df = well.get("df")
        null_pct = 0.0
        if df is not None and len(df) > 0:
            null_pct = df.isna().sum().sum() / (len(df) * len(df.columns)) * 100
        
        well_summary_data.append([
            str(wname),
            f"{info['strt']:.1f}",
            f"{info['stop']:.1f}",
            str(curves),
            f"{null_pct:.1f}%",
        ])
    
    well_summary_table = Table(well_summary_data, colWidths=[1.5*inch, 1*inch, 1*inch, 1*inch, 1*inch])
    well_summary_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0d6b64")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 10),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 8),
        ("BACKGROUND", (0, 1), (-1, -1), colors.beige),
        ("GRID", (0, 0), (-1, -1), 1, colors.grey),
        ("FONTSIZE", (0, 1), (-1, -1), 9),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f4f7f6")]),
    ]))
    story.append(well_summary_table)
    story.append(Spacer(1, 0.3 * inch))
    
    # --- Formation Tops Section ---
    if formation_tops is not None and not formation_tops.empty:
        story.append(PageBreak())
        story.append(Paragraph("2. Formation Tops", heading_style))
        
        tops_display = formation_tops.copy()
        display_cols = ["well", "formation", "depth"]
        if "color" in tops_display.columns:
            display_cols.append("color")
        
        tops_data = [["Well", "Formation", "Depth (m)"] + (["Color"] if "color" in tops_display.columns else [])]
        for _, row in tops_display.iterrows():
            row_data = [
                str(row["well"]),
                str(row["formation"]),
                f"{float(row['depth']):.2f}",
            ]
            if "color" in tops_display.columns:
                row_data.append(str(row["color"]))
            tops_data.append(row_data)
        
        tops_table = Table(tops_data, colWidths=[1.2*inch, 1.5*inch, 1.2*inch] + ([0.8*inch] if "color" in tops_display.columns else []))
        tops_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0d6b64")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, 0), 10),
            ("BOTTOMPADDING", (0, 0), (-1, 0), 8),
            ("GRID", (0, 0), (-1, -1), 1, colors.grey),
            ("FONTSIZE", (0, 1), (-1, -1), 8),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f4f7f6")]),
        ]))
        story.append(tops_table)
        story.append(Spacer(1, 0.3 * inch))
        
        # --- Thickness Maps Section ---
        story.append(PageBreak())
        story.append(Paragraph("3. Formation Thickness Analysis", heading_style))
        
        thickness_df = calculate_thicknesses(formation_tops)
        if not thickness_df.empty:
            thickness_data = [thickness_df.columns.tolist()]
            for _, row in thickness_df.iterrows():
                thickness_data.append([str(v) for v in row.tolist()])
            
            thickness_table = Table(
                thickness_data,
                colWidths=[1.2*inch, 1.5*inch, 1*inch, 1*inch, 1*inch]
            )
            thickness_table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0d6b64")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, 0), 9),
                ("BOTTOMPADDING", (0, 0), (-1, 0), 8),
                ("GRID", (0, 0), (-1, -1), 1, colors.grey),
                ("FONTSIZE", (0, 1), (-1, -1), 8),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f4f7f6")]),
            ]))
            story.append(KeepTogether([
                Paragraph("Formation Thickness Summary", 
                         ParagraphStyle("SubHeading", parent=styles["Normal"], 
                                      fontSize=11, fontName="Helvetica-Bold")),
                Spacer(1, 0.1 * inch),
                thickness_table,
            ]))
    
    # Build PDF
    doc.build(story)
    buffer.seek(0)
    
    if output_path:
        with open(output_path, "wb") as f:
            f.write(buffer.getvalue())
    
    return buffer
