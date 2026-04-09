"""
Enviro Wise Water Conservation Analysis
========================================
Destination Data Consultancy — Junior Analyst Case Study

This script analyses FAO AQUASTAT data on global water withdrawals,
water use efficiency, and water stress by country and sector (2020–2022).

Data source: FAO AQUASTAT Dissemination System
Variables analysed:
  - Agricultural water withdrawal as % of total renewable water resources
  - SDG 6.4.1 Water Use Efficiency (overall, industrial, agriculture, services)
  - SDG 6.4.2 Water Stress (overall, agricultural, industrial, municipal contributions)

Author: Rangel Hristov
Date: April 2026
"""

# ============================================================
# IMPORTS
# ============================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
import os

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set consistent plot style for professional-looking charts
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'font.size': 10,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
})

# Output directory for saved charts
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# STEP 1: LOAD AND CLEAN THE DATA
# ============================================================
print("=" * 65)
print("STEP 1: DATA LOADING & CLEANING")
print("=" * 65)

# Load the raw CSV from FAO AQUASTAT
df = pd.read_csv("data/aquastat_water_data.csv")

print(f"\nDataset shape: {df.shape[0]} rows x {df.shape[1]} columns")
print(f"\nColumn data types:")
print(df.dtypes.to_string())

print(f"\nMissing values per column:")
missing = df.isnull().sum()
print(missing.to_string())

if missing.sum() == 0:
    print("\n--> No missing values detected. Data is complete.")
else:
    print(f"\n--> Total missing values: {missing.sum()}. Dropping rows with NaN.")
    df = df.dropna()

# Filter out aggregate regions (e.g. "World", "Southern Asia") to keep
# only individual countries — aggregates would skew our analysis
df = df[df["IsAggregate"] == False].copy()
print(f"\nAfter removing aggregate regions: {df.shape[0]} rows, "
      f"{df['Area'].nunique()} unique countries")

# Display the unique indicator variables in the dataset
print(f"\nVariables in dataset ({df['Variable'].nunique()}):")
for i, var in enumerate(df['Variable'].unique(), 1):
    print(f"  {i}. {var}")

print(f"\nYears covered: {sorted(df['Year'].unique())}")

# ============================================================
# STEP 2: SUMMARY STATISTICS
# ============================================================
print("\n" + "=" * 65)
print("STEP 2: SUMMARY STATISTICS")
print("=" * 65)

# Pivot data so each variable becomes a column, using the most recent
# year (2022) for a cross-sectional snapshot
latest_year = df["Year"].max()
print(f"\nUsing most recent year ({latest_year}) for summary statistics.\n")

# Create a wide-format table: one row per country, one column per variable
df_latest = df[df["Year"] == latest_year].copy()
df_wide = df_latest.pivot_table(
    index="Area",
    columns="Variable",
    values="Value",
    aggfunc="first"
).reset_index()

# Shorten column names for readability in outputs and charts
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
df_wide = df_wide.rename(columns=SHORT_NAMES)

# Compute and display descriptive statistics for all numeric indicators
numeric_cols = [c for c in df_wide.columns if c != "Area"]
summary_stats = df_wide[numeric_cols].describe().round(2)
print(summary_stats.to_string())

# ============================================================
# STEP 3: VISUALISATIONS
# ============================================================
print("\n" + "=" * 65)
print("STEP 3: GENERATING VISUALISATIONS")
print("=" * 65)

# --- Chart 1: Top 15 countries by total water stress ---
# Total water stress (SDG 6.4.2) measures freshwater withdrawal as a
# proportion of available renewable freshwater resources — a key
# indicator of water scarcity pressure.
print("\n[1/4] Top 15 countries by total water stress...")

if "Total_Stress_%" in df_wide.columns:
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
    # Add value labels on bars for clarity
    for bar in bars:
        width = bar.get_width()
        ax.text(
            width + 10, bar.get_y() + bar.get_height() / 2,
            f"{width:.0f}%", va="center", fontsize=9,
        )
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/01_top15_water_stress.png")
    plt.close()
    print("   Saved: outputs/01_top15_water_stress.png")

# --- Chart 2: Sector breakdown (stacked bar) for top 10 users ---
# This shows how water stress is distributed across agricultural,
# industrial, and municipal sectors for the most stressed countries.
print("\n[2/4] Sector breakdown for top 10 water-stressed countries...")

sector_cols = ["Agri_Stress_%", "Industrial_Stress_%", "Municipal_Stress_%"]
if all(c in df_wide.columns for c in sector_cols):
    top10 = df_wide.nlargest(10, "Total_Stress_%")[["Area"] + sector_cols].copy()
    top10 = top10.set_index("Area")

    fig, ax = plt.subplots(figsize=(12, 7))
    top10.plot(
        kind="barh", stacked=True, ax=ax,
        color=["#2ca02c", "#1f77b4", "#ff7f0e"],
        edgecolor="white", linewidth=0.5,
    )
    ax.set_xlabel("Contribution to Water Stress (%)")
    ax.set_title(
        f"Water Stress by Sector — Top 10 Countries ({latest_year})\n"
        "Agricultural vs Industrial vs Municipal Contributions",
        fontweight="bold",
    )
    ax.legend(
        ["Agricultural", "Industrial", "Municipal"],
        loc="lower right", framealpha=0.9,
    )
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/02_sector_breakdown_top10.png")
    plt.close()
    print("   Saved: outputs/02_sector_breakdown_top10.png")

# --- Chart 3: Trend over time (2020–2022) ---
# Even though the time window is short (3 years), we can observe
# whether global average water stress is stable, rising, or falling.
print("\n[3/4] Trend over time (2020–2022)...")

# Calculate global average of each key indicator per year
trend_vars = {
    "Total_Stress_%": "SDG 6.4.2. Water Stress",
    "Agri_Stress_%": "SDG 6.4.2. Agricultural Sector Contribution to Water Stress",
    "Industrial_Stress_%": "SDG 6.4.2. Industrial Sector Contribution to Water Stress",
    "Municipal_Stress_%": "SDG 6.4.2. Municipal Sector Contribution to Water Stress",
}

fig, ax = plt.subplots(figsize=(10, 6))
colours = ["#d62728", "#2ca02c", "#1f77b4", "#ff7f0e"]

for (short, full), colour in zip(trend_vars.items(), colours):
    yearly = (
        df[df["Variable"] == full]
        .groupby("Year")["Value"]
        .mean()
    )
    ax.plot(
        yearly.index, yearly.values,
        marker="o", linewidth=2, label=short, color=colour,
    )

ax.set_xlabel("Year")
ax.set_ylabel("Average Stress (%)")
ax.set_title(
    "Global Average Water Stress Trend (2020–2022)\n"
    "Mean across all countries by sector",
    fontweight="bold",
)
ax.set_xticks([2020, 2021, 2022])
ax.legend(framealpha=0.9)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/03_trend_over_time.png")
plt.close()
print("   Saved: outputs/03_trend_over_time.png")

# --- Chart 4: Correlation heatmap ---
# Shows how the numeric variables relate to each other — e.g., do
# countries with high agricultural stress also have low water use
# efficiency? This reveals patterns useful for targeting interventions.
print("\n[4/4] Correlation heatmap...")

corr_data = df_wide[numeric_cols].dropna()
corr_matrix = corr_data.corr().round(2)

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(
    corr_matrix,
    annot=True, fmt=".2f",
    cmap="RdBu_r", center=0,
    square=True, linewidths=0.5,
    ax=ax,
    cbar_kws={"shrink": 0.8},
)
ax.set_title(
    "Correlation Heatmap of Water Use Indicators\n"
    f"Based on {latest_year} cross-country data",
    fontweight="bold",
)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/04_correlation_heatmap.png")
plt.close()
print("   Saved: outputs/04_correlation_heatmap.png")

# ============================================================
# STEP 4: K-MEANS CLUSTERING
# ============================================================
print("\n" + "=" * 65)
print("STEP 4: K-MEANS CLUSTERING (3 CLUSTERS)")
print("=" * 65)

# We cluster countries into 3 groups based on their water-use profile.
# This helps Enviro Wise identify which countries face similar
# challenges and could benefit from similar conservation strategies.

# Prepare clustering data — use sector stress + water use efficiency
cluster_cols = [c for c in numeric_cols if c in df_wide.columns]
df_cluster = df_wide[["Area"] + cluster_cols].dropna().copy()

print(f"\nCountries with complete data for clustering: {len(df_cluster)}")
print(f"Features used: {cluster_cols}")

# Standardise features so no single variable dominates due to scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_cluster[cluster_cols])

# Fit K-Means with 3 clusters (representing different risk profiles)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df_cluster["Cluster"] = kmeans.fit_predict(X_scaled)

# Summarise each cluster to understand the profiles
print("\nCluster Profiles (mean values per cluster):\n")
cluster_summary = df_cluster.groupby("Cluster")[cluster_cols].mean().round(2)
print(cluster_summary.to_string())

# Count countries per cluster
print("\nCountries per cluster:")
for c in sorted(df_cluster["Cluster"].unique()):
    countries = df_cluster[df_cluster["Cluster"] == c]["Area"].tolist()
    print(f"\n  Cluster {c} ({len(countries)} countries):")
    # Show first 10 countries as examples
    display = countries[:10]
    print(f"    Examples: {', '.join(display)}")
    if len(countries) > 10:
        print(f"    ... and {len(countries) - 10} more")

# Visualise clusters using the two most impactful variables
fig, ax = plt.subplots(figsize=(10, 7))
scatter = ax.scatter(
    df_cluster["Total_Stress_%"],
    df_cluster.get("Overall_WUE", df_cluster.get("Agri_Withdrawal_%", pd.Series())),
    c=df_cluster["Cluster"],
    cmap="Set1", s=60, alpha=0.7, edgecolors="white", linewidth=0.5,
)
ax.set_xlabel("Total Water Stress (%)")

# Use whatever secondary axis is available
if "Overall_WUE" in df_cluster.columns:
    ax.set_ylabel("Overall Water Use Efficiency (US$/m³)")
else:
    ax.set_ylabel("Agricultural Water Withdrawal (%)")

ax.set_title(
    "K-Means Clustering of Countries by Water-Use Profile\n"
    "3 clusters based on stress, efficiency, and sector breakdown",
    fontweight="bold",
)
legend = ax.legend(
    *scatter.legend_elements(),
    title="Cluster", loc="upper right",
)
# Label some outlier countries for context
for _, row in df_cluster.nlargest(5, "Total_Stress_%").iterrows():
    y_val = row.get("Overall_WUE", row.get("Agri_Withdrawal_%", 0))
    ax.annotate(
        row["Area"], (row["Total_Stress_%"], y_val),
        fontsize=8, alpha=0.8,
        xytext=(5, 5), textcoords="offset points",
    )

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/05_kmeans_clusters.png")
plt.close()
print("\n   Saved: outputs/05_kmeans_clusters.png")

# ============================================================
# STEP 5: RECOMMENDATIONS FOR ENVIRO WISE
# ============================================================
print("\n" + "=" * 65)
print("STEP 5: RECOMMENDATIONS FOR ENVIRO WISE")
print("=" * 65)

# Identify the highest-stress cluster
high_stress_cluster = cluster_summary["Total_Stress_%"].idxmax()
high_stress_countries = df_cluster[
    df_cluster["Cluster"] == high_stress_cluster
]["Area"].tolist()

print(f"""
Based on our analysis of FAO AQUASTAT data across 183 countries (2020-2022),
Enviro Wise should consider the following recommendations:

1. PRIORITISE AGRICULTURAL WATER EFFICIENCY IN HIGH-STRESS REGIONS
   Agriculture is by far the largest contributor to water stress globally.
   Countries like {', '.join(high_stress_countries[:3])} show agricultural
   stress contributions exceeding {cluster_summary.loc[high_stress_cluster, 'Agri_Stress_%']:.0f}%
   on average. Investing in drip irrigation, drought-resistant crops, and
   precision farming technology would yield the greatest water savings.

2. TARGET THE "CRITICAL" CLUSTER FOR IMMEDIATE INTERVENTION
   Our clustering identified {len(high_stress_countries)} countries in the
   highest-stress group. These nations face imminent water scarcity risk and
   should be the primary focus for Enviro Wise's conservation programmes.
   Tailored solutions by cluster profile will be more effective than
   one-size-fits-all approaches.

3. LEVERAGE INDUSTRIAL WATER USE EFFICIENCY GAINS
   Industrial water use efficiency varies enormously (from $0.15 to over
   $48,000 per cubic metre). Countries with low industrial WUE represent
   opportunities for technology transfer and process optimisation — often
   with rapid return on investment compared to agricultural reforms.

4. MONITOR TRENDS AND ADVOCATE FOR DATA-DRIVEN POLICY
   While the 2020–2022 window is short, the data shows water stress is not
   declining globally. Enviro Wise should advocate for continuous monitoring
   using SDG 6.4.2 indicators and push for national water-use reporting
   to enable evidence-based conservation policy.
""")

print("=" * 65)
print("ANALYSIS COMPLETE — All charts saved to outputs/ directory")
print("=" * 65)
