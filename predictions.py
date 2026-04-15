"""
Linear projection module for the Enviro Wise water analysis pipeline.

NOTE ON STATISTICAL VALIDITY
----------------------------
This module extrapolates water-stress indicators forward from a three-year
data window (2020–2022). Three data points is too short for any serious
forecasting technique; a naive linear fit is the only honest option given
that sample size. The output is therefore an *indicative trend projection*,
not a forecast. Treat it as "if the current slope held, where would we be
in 2025?" — nothing more.

Why this is still useful:
  - It gives Enviro Wise a consistent, auditable way to rank countries by
    direction of travel, not just current stress level.
  - Clients care about trajectory as much as absolute values; the chart
    communicates that in one glance.
  - The caveat is front-and-centre on every chart it feeds.

Technique: scikit-learn ``LinearRegression`` fit on (year, value) pairs,
projected to [2023, 2024, 2025]. Negative predictions are clipped to zero
because water stress percentages cannot go below it.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


PROJECTION_YEARS = [2023, 2024, 2025]
STRESS_VARIABLE = "SDG 6.4.2. Water Stress"


def _default_log(msg):
    print(msg)


def project_series(years, values, target_years=None):
    """
    Fit y = mx + b on (years, values) and return predictions for
    ``target_years``. Returns None if there aren't enough usable points.
    """
    if target_years is None:
        target_years = PROJECTION_YEARS

    years = np.asarray(years, dtype=float).reshape(-1, 1)
    values = np.asarray(values, dtype=float)

    mask = ~np.isnan(values)
    if mask.sum() < 2:
        return None

    model = LinearRegression()
    model.fit(years[mask], values[mask])

    predictions = model.predict(np.asarray(target_years, dtype=float).reshape(-1, 1))
    return np.maximum(predictions, 0.0)


def project_all_indicators(df, log=_default_log):
    """
    For every indicator (Variable) compute the global mean per year,
    then project 2023–2025. Returns a long-format DataFrame with an
    ``IsPredicted`` flag so downstream charts can style actual vs
    projected points differently.
    """
    rows = []
    for variable, group in df.groupby("Variable"):
        yearly = group.groupby("Year")["Value"].mean().sort_index()

        for year, value in yearly.items():
            rows.append(
                {"Variable": variable, "Year": int(year), "Value": float(value), "IsPredicted": False}
            )

        projected = project_series(yearly.index.values, yearly.values)
        if projected is None:
            continue
        for year, value in zip(PROJECTION_YEARS, projected):
            rows.append(
                {"Variable": variable, "Year": int(year), "Value": float(value), "IsPredicted": True}
            )

    out = pd.DataFrame(rows)
    log(f"  Projected {out['Variable'].nunique()} indicators forward to 2025")
    return out


def per_country_stress_projection(df, target_year=2025, log=_default_log):
    """
    Fit a linear trend on total water stress for each country over the
    observed years, project to ``target_year``, and return a dataframe
    sorted descending by projected value.

    Columns: Area, Actual_2022, Projected_<target>, Delta
    """
    subset = df[df["Variable"] == STRESS_VARIABLE]
    proj_col = f"Projected_{target_year}"

    rows = []
    for country, group in subset.groupby("Area"):
        yearly = group.groupby("Year")["Value"].mean().sort_index()
        projected = project_series(
            yearly.index.values, yearly.values, target_years=[target_year]
        )
        if projected is None:
            continue

        last_actual = float(yearly.iloc[-1]) if len(yearly) else float("nan")
        proj_val = float(projected[0])
        rows.append(
            {
                "Area": country,
                "Actual_2022": last_actual,
                proj_col: proj_val,
                "Delta": proj_val - last_actual,
            }
        )

    out = (
        pd.DataFrame(rows)
        .sort_values(proj_col, ascending=False)
        .reset_index(drop=True)
    )
    # Rename the dynamic column to a stable name for downstream consumers.
    out = out.rename(columns={proj_col: "Projected_2025"})
    log(f"  Per-country projections computed for {len(out)} countries")
    return out
