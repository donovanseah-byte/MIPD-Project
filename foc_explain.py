"""Display-only illustrations of the existing FOC assessment; never refits it."""

from __future__ import annotations

from html import escape

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


FUEL_UNIT = "VLSFO-equivalent MT"
AXES = {"STW": "Speed through water (kn)", "DisplacementMT": "Displacement / total vessel mass (MT)"}


def training_rows(eligible):
    """Return positive, finite pre-DD observations without touching the source."""
    eligible = eligible if isinstance(eligible, pd.DataFrame) else pd.DataFrame()
    if "Period" not in eligible:
        return pd.DataFrame(columns=["STW", "DisplacementMT", "FOC_MT_Day"])
    frame = eligible.loc[eligible["Period"].eq("Before DD")].copy()
    for column in ["STW", "DisplacementMT", "FOC_MT_Day"]:
        frame[column] = pd.to_numeric(frame.get(column, pd.Series(np.nan, index=frame.index)), errors="coerce")
    numeric = frame[["STW", "DisplacementMT", "FOC_MT_Day"]]
    return frame.loc[np.isfinite(numeric).all(axis=1) & numeric.gt(0).all(axis=1)].copy()


def raw_correlation(frame, feature):
    """Pearson correlation of original units, not a model coefficient."""
    values = frame[[feature, "FOC_MT_Day"]].apply(pd.to_numeric, errors="coerce")
    values = values.replace([np.inf, -np.inf], np.nan).dropna()
    if len(values) < 3 or values[feature].nunique() < 2 or values["FOC_MT_Day"].nunique() < 2:
        return None
    value = float(values[feature].corr(values["FOC_MT_Day"]))
    return value if np.isfinite(value) else None


def relationship_figure(frame, feature):
    """Raw observations only. Do not overlay a fixed-covariate curve on raw dots."""
    hover = []
    for _, row in frame.iterrows():
        date = pd.to_datetime(row.get("Date"), errors="coerce")
        when = date.strftime("%d %b %Y") if pd.notna(date) else "Date unavailable"
        hover.append(
            f"{when}<br>Voyage: {escape(str(row.get('Voyage', 'Unknown')))}"
            f"<br>STW: {row['STW']:.2f} kn<br>Displacement: {row['DisplacementMT']:,.0f} MT"
            f"<br>FOC: {row['FOC_MT_Day']:.2f} VLSFO-eq. MT/day"
        )
    figure = go.Figure(go.Scatter(
        x=frame[feature], y=frame["FOC_MT_Day"], mode="markers",
        marker={"size": 8, "opacity": 0.7, "color": "#3b82f6"},
        text=hover, hovertemplate="%{text}<extra></extra>", name="Pre-DD training reports",
    ))
    figure.update_layout(
        height=360, xaxis_title=AXES[feature], yaxis_title="M/E FOC (VLSFO-eq. MT/day)",
        margin={"l": 10, "r": 10, "t": 12, "b": 12}, showlegend=False,
    )
    figure.update_yaxes(rangemode="tozero")
    return figure


def conditional_slice(frame, details, feature):
    """A slice through the existing fitted model, not a new fit or support test."""
    required = ["transformed_intercept", "speed_coefficient", "displacement_coefficient",
                "displacement_reference_mt", "smearing_factor"]
    try:
        params = {key: float(details[key]) for key in required}
    except (KeyError, TypeError, ValueError):
        return None
    if frame.empty or not all(np.isfinite(list(params.values()))):
        return None
    if params["displacement_reference_mt"] <= 0 or params["smearing_factor"] <= 0:
        return None
    lo, hi = float(frame[feature].min()), float(frame[feature].max())
    if not np.isfinite([lo, hi]).all() or lo >= hi:
        return None
    x = np.linspace(lo, hi, 80)
    other = "DisplacementMT" if feature == "STW" else "STW"
    held = float(frame[other].median())
    speed = x if feature == "STW" else np.full_like(x, held)
    displacement = x if feature == "DisplacementMT" else np.full_like(x, held)
    with np.errstate(over="ignore", invalid="ignore"):
        y = params["smearing_factor"] * np.exp(
            params["transformed_intercept"] + params["speed_coefficient"] * np.log(speed)
            + params["displacement_coefficient"] * np.log(displacement / params["displacement_reference_mt"])
        )
    if not np.isfinite(y).all():
        return None
    return {"x": x, "y": y, "held_feature": other, "held_value": held}


def fuel_summary(result):
    """Use exactly the supported, finite interval pairs used by aggregation."""
    if not result.get("valid"):
        return None
    assessed = result.get("assessed_rows", pd.DataFrame())
    required = ["InsideSupport", "FOC_MT_Day", "ExpectedPreDDCondition_FOC_MT_Day", "PropellingHours"]
    if assessed.empty or not set(required).issubset(assessed.columns):
        return None
    rows = assessed.loc[assessed["InsideSupport"].eq(True)].copy()
    numeric = rows[required[1:]].apply(pd.to_numeric, errors="coerce")
    mask = np.isfinite(numeric).all(axis=1) & numeric.gt(0).all(axis=1)
    rows = rows.loc[mask].copy()
    numeric = numeric.loc[mask]
    if rows.empty:
        return None
    rows["ActualForIllustrationMT"] = numeric["FOC_MT_Day"] * numeric["PropellingHours"] / 24
    rows["ExpectedForIllustrationMT"] = numeric["ExpectedPreDDCondition_FOC_MT_Day"] * numeric["PropellingHours"] / 24
    actual = float(rows["ActualForIllustrationMT"].sum())
    expected = float(rows["ExpectedForIllustrationMT"].sum())
    if not np.isfinite([actual, expected]).all() or expected <= 0:
        return None
    return {"actual": actual, "expected": expected, "difference": expected - actual,
            "saving_pct": (1 - actual / expected) * 100, "rows": rows,
            "hours": float(numeric["PropellingHours"].sum())}


def selection_counts(period_data, eligible, result):
    """Distinguish support rejection from reports never assessed."""
    def count(frame, period):
        return int(frame["Period"].eq(period).sum()) if "Period" in frame else 0
    assessed = result.get("assessed_rows", pd.DataFrame())
    assessed_count = len(assessed)
    supported = int(assessed["InsideSupport"].eq(True).sum()) if "InsideSupport" in assessed else 0
    post_eligible = count(eligible, "After DD")
    return {"pre_raw": count(period_data, "Before DD"), "pre_eligible": count(eligible, "Before DD"),
            "post_raw": count(period_data, "After DD"), "post_eligible": post_eligible,
            "supported": supported, "outside": assessed_count - supported,
            "unassessed": max(0, post_eligible - assessed_count)}


