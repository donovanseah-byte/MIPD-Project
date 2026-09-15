"""Read-only report ledger: original operating rules plus assessment membership."""
from __future__ import annotations

import numpy as np
import pandas as pd


def build_report_ledger(period_data, eligible, result, settings):
    ledger = period_data.copy(deep=True)
    if "ReportID" not in ledger or ledger["ReportID"].duplicated().any():
        raise ValueError("The report ledger needs a unique ReportID assigned before filtering. Run the assessment again.")
    number = lambda field: pd.to_numeric(ledger[field], errors="coerce")
    checks = [
        (ledger["Date"].notna(), "Missing report timestamp"),
        (number("PropellingHours").ge(settings.get("min_hours", 18)), "Propelling hours missing or below minimum"),
        (
            number("STW").between(
                settings.get("min_plausible_stw", 5),
                settings.get("max_plausible_stw", 35),
            ),
            "STW missing or physically implausible",
        ),
        (number("DisplacementMT").gt(0), "Displacement missing or non-positive"),
        (number("FOC_MT_Day").gt(0), "FOC missing or non-positive"),
        (number("Beaufort").le(settings.get("max_beaufort", 4)), "Beaufort missing or above maximum"),
        (
            ~ledger.get(
                "IntervalAlignmentStatus", pd.Series("Internally consistent", index=ledger.index)
            ).isin(["Duplicate report timestamp", "Propelling hours exceed elapsed interval"]),
            "Fuel/distance/propelling interval is internally inconsistent",
        ),
    ]
    passing = pd.Series(True, index=ledger.index)
    reasons = pd.Series("", index=ledger.index, dtype=object)
    for check, label in checks:
        check = check.fillna(False)
        passing &= check
        reasons.loc[~check] += label + "; "
    in_period = ledger["Period"].isin(["Before DD", "After DD"])
    expected_ids = set(ledger.loc[in_period & passing, "ReportID"])
    if expected_ids != set(eligible["ReportID"]):
        raise ValueError("Report-ledger rules do not reconcile with the operating filter. No ledger is shown.")
    ledger["Operating checks passed"] = passing
    ledger["Assessment use"] = "Outside selected dates"
    ledger["Reason"] = "Outside the selected pre-DD and post-DD windows"
    docked = ledger["Period"].eq("Dry dock - excluded")
    ledger.loc[docked, ["Assessment use", "Reason"]] = ["Dry dock - excluded", "Within the declared dry-dock interval"]
    ledger.loc[ledger["Date"].isna(), ["Assessment use", "Reason"]] = ["Missing timestamp", "Cannot assign an analysis period"]
    rejected = in_period & ~passing
    ledger.loc[rejected, "Assessment use"] = "Excluded by operating filters"
    ledger.loc[rejected, "Reason"] = reasons.loc[rejected].str.rstrip("; ")
    before = in_period & passing & ledger["Period"].eq("Before DD")
    fitted = bool(result.get("model_details"))
    ledger.loc[before, "Assessment use"] = "Pre-DD training" if fitted else "Eligible pre-DD; model not fitted"
    ledger.loc[before, "Reason"] = "Passed operating filters"
    after = in_period & passing & ledger["Period"].eq("After DD")
    ledger.loc[after, ["Assessment use", "Reason"]] = ["Post-DD not assessed", "No comparability decision available"]
    assessed = result.get("assessed_rows", pd.DataFrame())
    if not assessed.empty:
        if assessed["ReportID"].duplicated().any() or not set(assessed["ReportID"]).issubset(expected_ids):
            raise ValueError("Assessment row identifiers do not reconcile with the report ledger.")
        by_id = assessed.set_index("ReportID")
        for i, row in ledger.loc[after].iterrows():
            if row["ReportID"] not in by_id.index:
                continue
            decision = by_id.loc[row["ReportID"]]
            if bool(decision["InsideSupport"]):
                ledger.at[i, "Assessment use"] = "Post-DD comparison" if result.get("valid") else "Comparable post-DD; no valid estimate"
                ledger.at[i, "Reason"] = "Within the selected " + str(result.get("comparison_basis", "pre-DD")) + " speed/loading support test"
            else:
                ledger.at[i, "Assessment use"] = "Outside pre-DD support"
                threshold = decision.get("SupportThreshold", np.nan)
                ledger.at[i, "Reason"] = (
                    "Too few pre-DD reports in the selected comparison group for the support test"
                    if pd.isna(threshold) else "Speed/loading combination outside the group-specific support threshold"
                )
        for column in [
            "ExpectedPreDDCondition_FOC_MT_Day", "SavingEquivalentMT", "InsideSupport",
            "StrictInsideSupport", "ExpandedInsideSupport", "SupportMode",
            "SupportDistance", "SupportThreshold", "SupportGroupRows",
        ]:
            if column in by_id:
                ledger[column] = ledger["ReportID"].map(by_id[column])
    # Gross FOC anomalies are review flags, not automatic exclusions. Keeping the
    # original assessment membership makes the primary estimate reproducible,
    # while the separate sensitivity result shows whether these reports matter.
    ledger["FOC anomaly flag"] = "No"
    ledger["FOC anomaly review"] = ""
    anomalies = result.get("foc_anomalies", pd.DataFrame())
    if isinstance(anomalies, pd.DataFrame) and not anomalies.empty and "ReportID" in anomalies:
        anomaly_ids = set(anomalies["ReportID"].dropna())
        anomaly_rows = ledger["ReportID"].isin(anomaly_ids)
        ledger.loc[anomaly_rows, "FOC anomaly flag"] = "Review"
        review_column = next(
            (name for name in ["FlagReason", "ReviewReason", "AnomalyReason", "Reason"] if name in anomalies),
            None,
        )
        if review_column:
            review_by_id = (
                anomalies.drop_duplicates("ReportID", keep="first")
                .set_index("ReportID")[review_column]
                .astype(str)
            )
            ledger.loc[anomaly_rows, "FOC anomaly review"] = ledger.loc[
                anomaly_rows, "ReportID"
            ].map(review_by_id).fillna("Review unusually large difference from expected FOC")
        else:
            ledger.loc[anomaly_rows, "FOC anomaly review"] = (
                "Review unusually large difference from expected FOC"
            )
    return ledger


def report_comparison(assessed):
    """Lead with the outcome and units, not identifiers that hide the fuel columns."""
    if assessed.empty:
        return pd.DataFrame()
    view = assessed.copy()
    view["Used in comparison"] = view["InsideSupport"].map({True: "Yes", False: "No"})
    columns = {"Date": "Date", "Used in comparison": "Used in comparison",
               "FOC_MT_Day": "Reported FOC (MT/day)",
               "ExpectedPreDDCondition_FOC_MT_Day": "Expected FOC (MT/day)",
               "SavingEquivalentMT": "Interval difference (MT)",
               "PropellingHours": "Propelling hours", "STW": "STW (kn)",
               "DisplacementMT": "Displacement (MT)", "ReportID": "Report ID",
               "Voyage": "Voyage", "AnalysisRoute": "Assessment route", "ServiceLeg": "Operating leg"}
    # Unsupported differences are not part of the estimate; keep them out of this summary.
    view.loc[~view["InsideSupport"], "SavingEquivalentMT"] = np.nan
    return view[[c for c in columns if c in view]].rename(columns=columns)
