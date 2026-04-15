"""
Enviro Wise Water Conservation Analysis — Core Pipeline
========================================================
Destination Data Consultancy — Junior Analyst Case Study

Analyses FAO AQUASTAT data on global water withdrawals, water use
efficiency, and water stress by country and sector (2020–2022).

This module is callable both from the desktop GUI (``app.py``) and
directly from the command line. The heavy lifting lives in
``run_pipeline`` — everything above that function is support code
(loaders, chart builders, the clustering step).

Variables analysed:
  - Agricultural water withdrawal as % of total renewable water resources
  - SDG 6.4.1 Water Use Efficiency (overall, industrial, agriculture, services)
  - SDG 6.4.2 Water Stress (overall, agricultural, industrial, municipal contributions)

Data source: FAO AQUASTAT Dissemination System

Author: Rangel Hristov
Date: April 2026
"""

import argparse
import os
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

import predictions

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Plot styling — kept at module level so every chart function inherits it.
# ---------------------------------------------------------------------------
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "font.size": 10,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
    }
)

# Short, chart-friendly names for the AQUASTAT indicator strings.
SHORT_NAMES = {
    "Agricultural water withdrawal as % of total renewable water resources": "Agri_Withdrawal_%",
    "SDG 6.4.1. Industrial Water Use Efficiency": "Industrial_WUE",
    "SDG 6.4.1. Irrigated Agriculture Water Use Efficiency": "Agri_WUE",
    "SDG 6.4.1. Services Water Use Efficiency": "Services_WUE",
    "SDG 6.4.1. Water Use Efficiency": "Overall_WUE",
    "SDG 6.4.2. Agricultural Sector Contribution to Water Stress": "Agri_Stress_%",
    "SDG 6.4.2. Industrial Sector Contribution to Water Stress": "Industrial_Stress_%",
    "SDG 6.4.2. Municipal Sector Contribution to Water Stress": "Municipal_Stress_%",
    "SDG 6.4.2. Water Stress": "Total_Stress_%",
}


def _default_log(msg):
    print(msg)


def resource_path(relative):
    """
    Resolve a file path whether the script runs from source or from a
    PyInstaller one-file bundle. PyInstaller extracts bundled data into
    ``sys._MEIPASS`` at runtime.
    """
    if getattr(sys, "frozen", False):
        base = Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
    else:
        base = Path(__file__).parent
    return base / relative


# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------
def load_and_clean(csv_path, log=_default_log):
    """Load the raw AQUASTAT CSV, drop aggregates, return a tidy long frame."""
    df = pd.read_csv(csv_path)
    log(f"\nDataset shape: {df.shape[0]} rows x {df.shape[1]} columns")

    missing = df.isnull().sum().sum()
    if missing:
        log(f"--> {missing} missing values found. Dropping.")
        df = df.dropna()
    else:
        log("--> No missing values detected. Data is complete.")

    df = df[df["IsAggregate"] == False].copy()  # noqa: E712 — pandas idiom
    log(
        f"After removing aggregate regions: {df.shape[0]} rows, "
        f"{df['Area'].nunique()} unique countries"
    )

    years = sorted(df["Year"].unique())
    log(f"Years covered: {years}")
    return df


def build_wide_table(df, log=_default_log):
    """Pivot the long-format frame to one row per country for the latest year."""
    latest_year = int(df["Year"].max())
    log(f"\nUsing most recent year ({latest_year}) for summary and clustering.")

    df_latest = df[df["Year"] == latest_year].copy()
    df_wide = (
        df_latest.pivot_table(
            index="Area", columns="Variable", values="Value", aggfunc="first"
        )
        .reset_index()
        .rename(columns=SHORT_NAMES)
    )

    numeric_cols = [c for c in df_wide.columns if c != "Area"]
    return df_wide, numeric_cols, latest_year


def summary_stats(df_wide, numeric_cols, log=_default_log):
    """Print descriptive stats for all numeric indicators."""
    log("\nDescriptive statistics (latest year):")
    log(df_wide[numeric_cols].describe().round(2).to_string())


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------
def chart_top15(df_wide, latest_year, output_dir, log=_default_log):
    log("\n[1/6] Top 15 countries by total water stress...")
    if "Total_Stress_%" not in df_wide.columns:
        log("   Skipped — Total_Stress_% column missing.")
        return

    top15 = df_wide.nlargest(15, "Total_Stress_%")[["Area", "Total_Stress_%"]]
    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.barh(
        top15["Area"][::-1],
        top15["Total_Stress_%"][::-1],
        color=sns.color_palette("Reds_r", 15),
        edgecolor="white",
        linewidth=0.5,
    )
    ax.set_xlabel("Water Stress (%)")
    ax.set_title(
        f"Top 15 Countries by Total Water Stress ({latest_year})\n"
        "SDG 6.4.2 — Freshwater Withdrawal as % of Renewable Resources",
        fontweight="bold",
    )
    for bar in bars:
        width = bar.get_width()
        ax.text(
            width + 10,
            bar.get_y() + bar.get_height() / 2,
            f"{width:.0f}%",
            va="center",
            fontsize=9,
        )
    plt.tight_layout()
    plt.savefig(output_dir / "01_top15_water_stress.png")
    plt.close()
    log("   Saved: 01_top15_water_stress.png")


def chart_sector_breakdown(df_wide, latest_year, output_dir, log=_default_log):
    log("\n[2/6] Sector breakdown for top 10 water-stressed countries...")
    sector_cols = ["Agri_Stress_%", "Industrial_Stress_%", "Municipal_Stress_%"]
    if not all(c in df_wide.columns for c in sector_cols):
        log("   Skipped — sector columns missing.")
        return

    top10 = (
        df_wide.nlargest(10, "Total_Stress_%")[["Area"] + sector_cols]
        .copy()
        .set_index("Area")
    )

    fig, ax = plt.subplots(figsize=(12, 7))
    top10.plot(
        kind="barh",
        stacked=True,
        ax=ax,
        color=["#2ca02c", "#1f77b4", "#ff7f0e"],
        edgecolor="white",
        linewidth=0.5,
    )
    ax.set_xlabel("Contribution to Water Stress (%)")
    ax.set_title(
        f"Water Stress by Sector — Top 10 Countries ({latest_year})\n"
        "Agricultural vs Industrial vs Municipal Contributions",
        fontweight="bold",
    )
    ax.legend(["Agricultural", "Industrial", "Municipal"], loc="lower right", framealpha=0.9)
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(output_dir / "02_sector_breakdown_top10.png")
    plt.close()
    log("   Saved: 02_sector_breakdown_top10.png")


def chart_trend_over_time(df, output_dir, projections=None, log=_default_log):
    """
    Trend line of the four main stress indicators 2020–2022, extended with
    a dashed linear projection to 2025 if ``projections`` is supplied.
    """
    log("\n[3/6] Trend over time (2020–2022) with projection to 2025...")
    trend_vars = {
        "Total_Stress_%": "SDG 6.4.2. Water Stress",
        "Agri_Stress_%": "SDG 6.4.2. Agricultural Sector Contribution to Water Stress",
        "Industrial_Stress_%": "SDG 6.4.2. Industrial Sector Contribution to Water Stress",
        "Municipal_Stress_%": "SDG 6.4.2. Municipal Sector Contribution to Water Stress",
    }
    colours = ["#d62728", "#2ca02c", "#1f77b4", "#ff7f0e"]

    fig, ax = plt.subplots(figsize=(11, 6))
    for (short, full), colour in zip(trend_vars.items(), colours):
        yearly = df[df["Variable"] == full].groupby("Year")["Value"].mean().sort_index()
        ax.plot(yearly.index, yearly.values, marker="o", linewidth=2, label=short, color=colour)

        if projections is None:
            continue
        proj = projections[
            (projections["Variable"] == full) & (projections["IsPredicted"])
        ].sort_values("Year")
        if proj.empty:
            continue
        x = [int(yearly.index[-1])] + proj["Year"].tolist()
        y = [float(yearly.values[-1])] + proj["Value"].tolist()
        ax.plot(x, y, linestyle="--", linewidth=1.8, color=colour, alpha=0.75)
        ax.scatter(proj["Year"], proj["Value"], marker="x", color=colour, s=40, alpha=0.75)

    ax.axvline(x=2022.5, color="grey", linestyle=":", alpha=0.6)
    ymin, ymax = ax.get_ylim()
    ax.text(2022.6, ymin + (ymax - ymin) * 0.92, "← actual  |  projected →", fontsize=9, color="grey")

    ax.set_xlabel("Year")
    ax.set_ylabel("Average Stress (%)")
    ax.set_title(
        "Global Average Water Stress — Actual (2020–22) & Projected (2023–25)\n"
        "Linear extrapolation; illustrative only — see README caveat",
        fontweight="bold",
    )
    ax.set_xticks([2020, 2021, 2022, 2023, 2024, 2025])
    ax.legend(framealpha=0.9, loc="best")
    plt.tight_layout()
    plt.savefig(output_dir / "03_trend_over_time.png")
    plt.close()
    log("   Saved: 03_trend_over_time.png")


def chart_correlation_heatmap(df_wide, latest_year, output_dir, numeric_cols, log=_default_log):
    log("\n[4/6] Correlation heatmap...")
    corr_data = df_wide[numeric_cols].dropna()
    if corr_data.empty:
        log("   Skipped — no complete rows for correlation.")
        return

    corr_matrix = corr_data.corr().round(2)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        square=True,
        linewidths=0.5,
        ax=ax,
        cbar_kws={"shrink": 0.8},
    )
    ax.set_title(
        "Correlation Heatmap of Water Use Indicators\n"
        f"Based on {latest_year} cross-country data",
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(output_dir / "04_correlation_heatmap.png")
    plt.close()
    log("   Saved: 04_correlation_heatmap.png")


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------
def run_clustering(df_wide, numeric_cols, log=_default_log):
    """
    Fit a 3-cluster K-Means over the wide indicator table. Returns the
    labelled dataframe, the cluster means summary, and the list of
    countries in the highest-stress cluster.
    """
    cluster_cols = [c for c in numeric_cols if c in df_wide.columns]
    df_cluster = df_wide[["Area"] + cluster_cols].dropna().copy()
    log(f"\nCountries with complete data for clustering: {len(df_cluster)}")

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(df_cluster[cluster_cols])

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df_cluster["Cluster"] = kmeans.fit_predict(x_scaled)

    cluster_summary = (
        df_cluster.groupby("Cluster")[cluster_cols].mean().round(2)
    )
    log("\nCluster profiles (mean values):")
    log(cluster_summary.to_string())

    for c in sorted(df_cluster["Cluster"].unique()):
        countries = df_cluster[df_cluster["Cluster"] == c]["Area"].tolist()
        preview = ", ".join(countries[:8])
        extra = f" (+ {len(countries) - 8} more)" if len(countries) > 8 else ""
        log(f"  Cluster {c} — {len(countries)} countries: {preview}{extra}")

    high_cluster = cluster_summary["Total_Stress_%"].idxmax()
    high_countries = df_cluster[df_cluster["Cluster"] == high_cluster]["Area"].tolist()
    return df_cluster, cluster_summary, high_cluster, high_countries


def chart_clusters(df_cluster, output_dir, log=_default_log):
    log("\n[5/6] K-Means cluster scatter...")
    fig, ax = plt.subplots(figsize=(10, 7))

    y_col = "Overall_WUE" if "Overall_WUE" in df_cluster.columns else "Agri_Withdrawal_%"

    scatter = ax.scatter(
        df_cluster["Total_Stress_%"],
        df_cluster[y_col],
        c=df_cluster["Cluster"],
        cmap="Set1",
        s=60,
        alpha=0.75,
        edgecolors="white",
        linewidth=0.5,
    )
    ax.set_xlabel("Total Water Stress (%)")
    ax.set_ylabel("Overall Water Use Efficiency (US$/m³)" if y_col == "Overall_WUE" else "Agri Withdrawal (%)")
    ax.set_title(
        "K-Means Clustering of Countries by Water-Use Profile\n"
        "3 clusters based on stress, efficiency, and sector breakdown",
        fontweight="bold",
    )
    ax.legend(*scatter.legend_elements(), title="Cluster", loc="upper right")

    for _, row in df_cluster.nlargest(5, "Total_Stress_%").iterrows():
        ax.annotate(
            row["Area"],
            (row["Total_Stress_%"], row[y_col]),
            fontsize=8,
            alpha=0.8,
            xytext=(5, 5),
            textcoords="offset points",
        )

    plt.tight_layout()
    plt.savefig(output_dir / "05_kmeans_clusters.png")
    plt.close()
    log("   Saved: 05_kmeans_clusters.png")


def chart_projected_top15(country_projections, output_dir, log=_default_log):
    """Horizontal bar chart of the top-15 countries by projected 2025 stress."""
    log("\n[6/6] Projected top-15 countries by water stress (2025)...")
    if country_projections.empty:
        log("   Skipped — no projections available.")
        return

    top = country_projections.head(15)
    fig, ax = plt.subplots(figsize=(11, 7))
    bars = ax.barh(
        top["Area"][::-1],
        top["Projected_2025"][::-1],
        color=sns.color_palette("Oranges_r", 15),
        edgecolor="white",
        linewidth=0.5,
    )
    ax.scatter(
        top["Actual_2022"][::-1],
        range(len(top)),
        color="black",
        s=24,
        zorder=5,
        label="2022 actual",
    )
    ax.set_xlabel("Water Stress (%)")
    ax.set_title(
        "Projected Top 15 Countries by Water Stress — 2025\n"
        "Linear extrapolation from 2020–2022 AQUASTAT data (illustrative)",
        fontweight="bold",
    )
    ax.legend(loc="lower right", framealpha=0.9)

    for bar, val in zip(bars, top["Projected_2025"][::-1]):
        ax.text(
            val + max(top["Projected_2025"]) * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.0f}%",
            va="center",
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(output_dir / "06_projected_stress_2025.png")
    plt.close()
    log("   Saved: 06_projected_stress_2025.png")


# ---------------------------------------------------------------------------
# Recommendations
# ---------------------------------------------------------------------------
def build_recommendations(cluster_summary, high_cluster, high_countries, log=_default_log):
    agri_pct = cluster_summary.loc[high_cluster, "Agri_Stress_%"]
    n = len(high_countries)
    example_countries = ", ".join(high_countries[:3]) if high_countries else "—"
    cluster_plural = "country" if n == 1 else "countries"

    log(
        "\n" + "=" * 65
        + f"\nRECOMMENDATIONS FOR ENVIRO WISE\n"
        + "=" * 65
        + f"""

1. PRIORITISE AGRICULTURAL WATER EFFICIENCY IN HIGH-STRESS REGIONS
   Agriculture is by far the largest contributor to water stress globally.
   {example_countries} leads the critical cluster, with agricultural stress
   averaging {agri_pct:.0f}%. Drip irrigation, drought-resistant crops and
   precision farming would yield the biggest savings.

2. TARGET THE "CRITICAL" CLUSTER FOR IMMEDIATE INTERVENTION
   Clustering flagged {n} {cluster_plural} in the highest-stress group.
   Tailoring interventions to that profile beats one-size-fits-all
   programmes.

3. LEVERAGE INDUSTRIAL WATER USE EFFICIENCY GAINS
   Industrial WUE varies by orders of magnitude. Low-WUE countries are
   targets for technology transfer and process optimisation with rapid
   return on investment.

4. MONITOR TRENDS AND PROJECT FORWARD
   Linear projection of 2020–22 trends to 2025 suggests water stress is
   not declining. Treat the projection as indicative — rerun the pipeline
   annually against fresh AQUASTAT releases for a defensible update cycle.
"""
    )


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def run_pipeline(csv_path, output_dir="outputs", log=_default_log):
    """
    End-to-end run. Consumers (CLI, GUI) pass a ``log`` callback so the
    pipeline can stream status updates into whatever UI is hosting it.
    Returns the resolved output directory path.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log("=" * 65)
    log("STEP 1: DATA LOADING & CLEANING")
    log("=" * 65)
    df = load_and_clean(csv_path, log=log)

    df_wide, numeric_cols, latest_year = build_wide_table(df, log=log)

    log("\n" + "=" * 65)
    log("STEP 2: SUMMARY STATISTICS")
    log("=" * 65)
    summary_stats(df_wide, numeric_cols, log=log)

    log("\n" + "=" * 65)
    log("STEP 3: FORECAST (LINEAR PROJECTION 2023–2025)")
    log("=" * 65)
    global_proj = predictions.project_all_indicators(df, log=log)
    country_proj = predictions.per_country_stress_projection(df, log=log)

    log("\n" + "=" * 65)
    log("STEP 4: VISUALISATIONS")
    log("=" * 65)
    chart_top15(df_wide, latest_year, output_dir, log=log)
    chart_sector_breakdown(df_wide, latest_year, output_dir, log=log)
    chart_trend_over_time(df, output_dir, projections=global_proj, log=log)
    chart_correlation_heatmap(df_wide, latest_year, output_dir, numeric_cols, log=log)

    log("\n" + "=" * 65)
    log("STEP 5: K-MEANS CLUSTERING (3 CLUSTERS)")
    log("=" * 65)
    df_cluster, cluster_summary, high_cluster, high_countries = run_clustering(
        df_wide, numeric_cols, log=log
    )
    chart_clusters(df_cluster, output_dir, log=log)
    chart_projected_top15(country_proj, output_dir, log=log)

    build_recommendations(cluster_summary, high_cluster, high_countries, log=log)

    log("\n" + "=" * 65)
    log(f"ANALYSIS COMPLETE — 6 charts saved to: {output_dir.resolve()}")
    log("=" * 65)
    return output_dir


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def _cli():
    parser = argparse.ArgumentParser(description="Enviro Wise water analysis CLI")
    parser.add_argument(
        "--csv",
        default=str(resource_path("data/aquastat_water_data.csv")),
        help="Path to AQUASTAT CSV file",
    )
    parser.add_argument(
        "--output",
        default="outputs",
        help="Output directory for generated charts",
    )
    args = parser.parse_args()
    run_pipeline(args.csv, args.output)


if __name__ == "__main__":
    _cli()
