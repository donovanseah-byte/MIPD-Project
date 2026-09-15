"""Plain-language, calculation-backed comments for engineering evidence charts."""
from __future__ import annotations

import numpy as np
import pandas as pd


def _range_text(frame: pd.DataFrame, column: str, digits: int = 1) -> str | None:
    if column not in frame:
        return None
    values = pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if values.empty:
        return None
    if column == "DisplacementMT":
        return f"{values.min():,.0f}-{values.max():,.0f} MT"
    return f"{values.min():,.{digits}f}-{values.max():,.{digits}f} kn"


def operating_support_comment(eligible: pd.DataFrame, result: dict) -> dict:
    """Explain the speed/loading support graph without treating it as proof."""
    assessed = result.get("assessed_rows", pd.DataFrame())
    if not isinstance(assessed, pd.DataFrame) or assessed.empty or "InsideSupport" not in assessed:
        return {"summary": "No post-DD operating-support interpretation is available.", "warnings": []}
    supported = assessed.loc[assessed["InsideSupport"].fillna(False).astype(bool)].copy()
    post_count = len(assessed)
    used_count = len(supported)
    report_share = used_count / post_count * 100.0 if post_count else 0.0
    coverage = float(result.get("coverage_pct") or 0.0)
    basis = str(result.get("comparison_basis") or "Selected")
    summary = (
        f"{basis}: {used_count} of {post_count} eligible post-DD reports ({report_share:.1f}%) were "
        f"inside vessel-specific pre-DD speed/loading support. They represent {coverage:.1f}% of "
        "eligible post-DD fuel."
    )
    pre = eligible.loc[eligible.get("Period", pd.Series(index=eligible.index)).eq("Before DD")].copy()
    post_speed = _range_text(supported, "STW")
    post_loading = _range_text(supported, "DisplacementMT")
    pre_speed = _range_text(pre, "STW")
    pre_loading = _range_text(pre, "DisplacementMT")
    ranges = []
    if post_speed and post_loading:
        ranges.append(f"Comparable post-DD conditions span {post_speed} and {post_loading}")
    if pre_speed and pre_loading:
        ranges.append(f"the displayed pre-DD training conditions span {pre_speed} and {pre_loading}")
    if ranges:
        summary += " " + "; ".join(ranges) + "."

    warnings = []
    outside = post_count - used_count
    if outside:
        warnings.append(
            f"{outside} eligible post-DD report{'s were' if outside != 1 else ' was'} outside pre-DD support and do not contribute to the saving estimate."
        )
    scope = result.get("service_leg_summary", pd.DataFrame())
    if isinstance(scope, pd.DataFrame) and not scope.empty and "FuelSharePct" in scope:
        dominant = scope.iloc[0]
        share = float(dominant["FuelSharePct"])
        if np.isfinite(share) and share >= 80.0:
            warnings.append(
                f"The supported comparison is dominated by {dominant.get('AnalysisRoute', 'Unknown route')} / "
                f"{dominant.get('ServiceLeg', 'Unknown leg')} ({share:.1f}% of supported fuel)."
            )
    warnings.append(
        "This graph shows operating-condition coverage, not the vessel's sailed route, and visual overlap alone does not prove comparability or saving."
    )
    return {"summary": summary, "warnings": warnings}


def actual_expected_comment(result: dict, tolerance_pct: float = 1.0) -> dict:
    """Explain the actual-versus-expected chart using supported rows only."""
    assessed = result.get("assessed_rows", pd.DataFrame())
    required = {"InsideSupport", "FOC_MT_Day", "ExpectedPreDDCondition_FOC_MT_Day", "PropellingHours"}
    if not isinstance(assessed, pd.DataFrame) or assessed.empty or not required.issubset(assessed.columns):
        return {"summary": "No report-level actual-versus-expected interpretation is available.", "warnings": []}
    supported = assessed.loc[assessed["InsideSupport"].fillna(False).astype(bool)].copy()
    actual = pd.to_numeric(supported["FOC_MT_Day"], errors="coerce")
    expected = pd.to_numeric(supported["ExpectedPreDDCondition_FOC_MT_Day"], errors="coerce")
    hours = pd.to_numeric(supported["PropellingHours"], errors="coerce")
    valid = actual.gt(0) & expected.gt(0) & hours.gt(0) & np.isfinite(actual) & np.isfinite(expected) & np.isfinite(hours)
    supported, actual, expected, hours = supported.loc[valid], actual.loc[valid], expected.loc[valid], hours.loc[valid]
    if supported.empty:
        return {"summary": "No valid comparable reports are available for this interpretation.", "warnings": []}
    deviation = (actual / expected - 1.0) * 100.0
    below = int(deviation.lt(-tolerance_pct).sum())
    similar = int(deviation.abs().le(tolerance_pct).sum())
    above = int(deviation.gt(tolerance_pct).sum())
    count_phrase = lambda count: f"{count} report was" if count == 1 else f"{count} reports were"
    saving = result.get("improvement_pct")
    saving_text = (
        f"The interval-weighted package result is {float(saving):.2f}% saving."
        if saving is not None and np.isfinite(float(saving)) and float(saving) >= 0
        else f"The interval-weighted package result is {abs(float(saving)):.2f}% higher FOC."
        if saving is not None and np.isfinite(float(saving))
        else "No valid aggregate saving is available."
    )
    summary = (
        f"Among {len(supported)} comparable post-DD reports, {count_phrase(below)} more than {tolerance_pct:.0f}% below "
        f"model-expected FOC, {count_phrase(similar)} within +/-{tolerance_pct:.0f}%, and {count_phrase(above)} more than "
        f"{tolerance_pct:.0f}% above it. {saving_text}"
    )
    warnings = [
        "The overall percentage uses interval fuel weighted by propelling hours; it is not the percentage of points below the line and it does not average report-level percentages."
    ]
    interval_difference = (expected - actual).abs() * hours / 24.0
    total_difference = float(interval_difference.sum())
    if len(interval_difference) >= 6 and total_difference > 0:
        top_share = float(interval_difference.nlargest(min(3, len(interval_difference))).sum() / total_difference * 100.0)
        if top_share >= 50.0:
            warnings.append(
                f"The three largest report-level differences account for {top_share:.1f}% of the total absolute difference, so the aggregate is relatively concentrated."
            )
    if below and above:
        warnings.append(
            "Comparable reports appear on both sides of the no-change line, showing report-to-report variation despite the aggregate result."
        )
    warnings.append(
        "Position below the line indicates lower reported FOC than the model expected; it does not by itself prove that dry docking caused the difference."
    )
    return {"summary": summary, "warnings": warnings}
