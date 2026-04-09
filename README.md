# Enviro Wise — Water Conservation Analysis

**Destination Data Consultancy** | HND Case Study

Analysis of global water withdrawals, water use efficiency, and water stress using [FAO AQUASTAT](https://data.apps.fao.org/aquastat/) data (2020–2022) across 183 countries.

## Project Structure

```
├── data/
│   └── aquastat_water_data.csv      # FAO AQUASTAT raw dataset
├── outputs/                          # Generated visualisations (PNG)
│   ├── 01_top15_water_stress.png
│   ├── 02_sector_breakdown_top10.png
│   ├── 03_trend_over_time.png
│   ├── 04_correlation_heatmap.png
│   └── 05_kmeans_clusters.png
├── water_analysis.py                 # Main analysis script
├── requirements.txt
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
python water_analysis.py
```

## Analysis Overview

1. **Data Cleaning** — Load, inspect, and filter FAO AQUASTAT data (remove aggregate regions)
2. **Summary Statistics** — Descriptive stats for all 9 water-use indicators
3. **Visualisations**
   - Top 15 countries by total water stress (horizontal bar chart)
   - Sector breakdown: agricultural vs industrial vs municipal (stacked bar)
   - Trend over time 2020–2022 (line chart)
   - Correlation heatmap of all numeric indicators
4. **K-Means Clustering** — 3-cluster grouping of countries by water-use profile
5. **Recommendations** — Actionable insights for Enviro Wise clients

## Key Findings

- **Kuwait, UAE, and Saudi Arabia** face the highest water stress globally (>800% of renewable resources)
- **Agriculture** is the dominant contributor to water stress in nearly all high-stress countries
- **Industrial water use efficiency** varies by orders of magnitude, representing a key intervention opportunity
- Three distinct country clusters emerge: low-stress, moderate-stress, and critical-stress nations

## Data Source

FAO AQUASTAT Dissemination System — SDG indicators 6.4.1 (Water Use Efficiency) and 6.4.2 (Water Stress)

## Author

Rangel Hristov
