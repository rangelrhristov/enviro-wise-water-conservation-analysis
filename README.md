# Enviro Wise — Water Conservation Analysis

**Destination Data Consultancy** | HND Case Study

A desktop data-science pipeline that ingests [FAO AQUASTAT](https://data.apps.fao.org/aquastat/)
water data (2020–2022) across 183 countries, generates six charts, runs K-Means
clustering, and projects water stress forward to 2025.

Two entry points:

- **`EnviroWise.exe`** — single-file Windows desktop app (customtkinter GUI).
- **`water_analysis.py`** — the same pipeline from the command line.

## Project structure

```
├── data/
│   └── aquastat_water_data.csv       # FAO AQUASTAT raw dataset
├── outputs/                           # Generated charts (PNG)
│   ├── 01_top15_water_stress.png
│   ├── 02_sector_breakdown_top10.png
│   ├── 03_trend_over_time.png        # actuals + linear projection to 2025
│   ├── 04_correlation_heatmap.png
│   ├── 05_kmeans_clusters.png
│   └── 06_projected_stress_2025.png  # top 15 projected stress (new)
├── water_analysis.py                  # core pipeline (CLI + library)
├── predictions.py                     # linear projection module
├── app.py                             # customtkinter desktop GUI
├── requirements.txt                   # runtime dependencies
├── requirements-build.txt             # runtime + PyInstaller
└── README.md
```

## Desktop app (GUI)

Run from source:

```bash
pip install -r requirements.txt
python app.py
```

1. The **CSV file** field is pre-populated with the bundled AQUASTAT dataset, so
   the app runs out of the box.
2. Pick your own CSV if you want — any AQUASTAT export with the same schema works.
3. Hit **Run Analysis**. Progress streams into the log; charts land in the
   chosen output folder.
4. **Open Output Folder** opens the charts in Explorer / Finder / xdg-open.

## Command line

```bash
pip install -r requirements.txt
python water_analysis.py --csv data/aquastat_water_data.csv --output outputs
```

Arguments are optional — defaults point at the bundled dataset and `outputs/`.

## Building the Windows .exe

```bash
pip install -r requirements-build.txt
pyinstaller --noconfirm --onefile --windowed \
    --name EnviroWise \
    --add-data "data/aquastat_water_data.csv;data" \
    --collect-all customtkinter \
    --collect-submodules sklearn \
    app.py
```

Output: `dist/EnviroWise.exe` (~90 MB). Ship that file on its own — everything
is bundled inside.

## Pipeline overview

1. **Data cleaning** — load the CSV, drop aggregate regions, keep individual countries.
2. **Summary statistics** — descriptives across the nine SDG 6.4 indicators.
3. **Forecast** — linear projection of 2020–22 trends to 2023, 2024, 2025
   (see *projection caveat* below).
4. **Visualisations** — six charts saved as PNGs.
5. **K-Means clustering** — three country clusters by water-use profile (risk tiers).
6. **Recommendations** — printed action list for Enviro Wise.

## Key findings

- **Kuwait, UAE and Saudi Arabia** face the highest water stress globally
  (>800% of renewable resources).
- **Agriculture** is the dominant contributor in nearly every high-stress country.
- **Industrial water-use efficiency** spans several orders of magnitude — a
  prime intervention opportunity.
- Three distinct country clusters emerge: low-stress, moderate-stress and
  critical-stress.

## Projection caveat

The 2023–2025 projections are a **naive linear extrapolation from a three-year
window**. Three data points is too short for any real forecasting technique;
a linear fit is the only honest option at that sample size.

Treat the projections as *"if the current slope held, where would we be in 2025?"* —
a directional trend indicator, not a forecast. Every chart that uses the
projection says so on its title. Rerun the pipeline annually against fresh
AQUASTAT releases for a proper update cycle.

## Data source

FAO AQUASTAT Dissemination System — SDG indicators 6.4.1 (Water Use Efficiency)
and 6.4.2 (Water Stress).

## Tech stack

`pandas`, `scikit-learn`, `matplotlib`, `seaborn`, `customtkinter`, `PyInstaller`.

## Author

Rangel Hristov
