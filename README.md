# 🛢️ Multi-Well Viewer

A comprehensive Streamlit application for multi-well data loading, display, formation tops overlay, and depth flattening.

## Features

- **LAS file import** — load one or more LAS well-log files simultaneously
- **Multi-well cross-section** — configurable log tracks (GR, Resistivity, Density, Neutron, Sonic, …)
- **Formation tops overlay** — import picks from CSV, overlay colour-coded horizontal markers with labels
- **Flattening** — align all wells on a chosen reference formation top
- **Statistics panel** — per-well curve statistics and histograms
- **Export** — download the cross-section figure as PNG or PDF

## Project Structure

```
multi-well-viewer/
├── app.py              # Main Streamlit application
├── data_loader.py      # LAS / CSV parsing and validation helpers
├── visualizer.py       # Matplotlib cross-section and histogram builders
├── flattening.py       # Depth-shift / flattening logic
├── requirements.txt    # Python dependencies
└── sample_data/
    ├── WELL_A.las      # Sample LAS files (synthetic)
    ├── WELL_B.las
    ├── WELL_C.las
    └── formation_tops.csv
```

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/AnshulChoudhary-iitism/multi-well-viewer.git
cd multi-well-viewer
```

### 2. Install dependencies

Python 3.9+ is recommended.

```bash
pip install -r requirements.txt
```

### 3. Run the application

```bash
streamlit run app.py
```

The app opens at <http://localhost:8501>.

## Quick Start with Sample Data

1. Click **Upload LAS files** in the sidebar and select all three files from `sample_data/`.
2. Click **Upload formation tops (CSV)** and select `sample_data/formation_tops.csv`.
3. The cross-section will render automatically with GR, ILD, RHOB, NPHI, and DT tracks.
4. Enable **Flattening** and choose *Top Sand* to align wells on the sand horizon.

## Formation Tops CSV Format

```
well,formation,depth[,color]
WELL_A,Top Sand,1200.0,#3cb44b
WELL_B,Top Sand,1185.0,#3cb44b
```

The `color` column is optional (hex colour string).  If omitted, colours are assigned automatically.

## Technology Stack

| Library | Purpose |
|---------|---------|
| [Streamlit](https://streamlit.io) | Web UI framework |
| [lasio](https://lasio.readthedocs.io) | LAS file parsing |
| [Pandas](https://pandas.pydata.org) | Data handling |
| [NumPy](https://numpy.org) | Numerical operations |
| [Matplotlib](https://matplotlib.org) | Visualization |

## Contributing

Pull requests are welcome. Please open an issue first to discuss significant changes.