"""V9.14: sign-neutral supported FOC assessment grading."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from foc_quality import assess_reliability, quality_level, placebo_signal
from foc_audit import build_report_ledger, report_comparison
from foc_explain import fuel_summary, selection_counts
from foc_exports import (_actual_expected_scatter, _excel_export, _fmt,
                         _support_scatter, _validation_table)
from foc_interpret import actual_expected_comment, operating_support_comment
from foc_report import printable_report
from foc_setup import render_setup
from foc_visuals import assessment_story_figure, figure_png

st.set_page_config(
    page_title="MIPD | Package-level post-dry-dock FOC assessment",
    page_icon="M",
    layout="wide",
)


CONCLUSION_GUIDE = pd.DataFrame(
    [
        {
            "Evidence category": "Supported Assessment",
            "Plain-language meaning": "The data, unseen pre-DD prediction checks and operating-condition comparability support interpreting the package-level FOC change, whether positive or negative.",
            "Appropriate use": "Prototype engineering screening; not proof that dry docking or a particular retrofit caused the change.",
        },
        {
            "Evidence category": "Evidence-Limited",
            "Plain-language meaning": "An interpretable estimate exists, but important non-critical evidence or sensitivity conditions limit confidence.",
            "Appropriate use": "Treat as an indication only; collect more comparable evidence before stronger conclusions.",
        },
        {
            "Evidence category": "Inconclusive",
            "Plain-language meaning": "Critical data integrity, quantity, operating support or predictive-model validation failed.",
            "Appropriate use": "Do not treat the calculated FOC change as a reliable result.",
        },
    ]
)

DATA_QUALITY_GUIDE = pd.DataFrame(
    [
        {
            "Data quality": "High",
            "Required conditions": "At least 60 valid pre-DD reports, 20 comparable post-DD reports, and 60% comparable post-DD fuel coverage; intervals and monitoring periods verified; operating-leg/route mapping confirmed.",
            "Effect on evidence category": "May qualify as Supported Assessment if the predictive model and robustness checks also pass.",
        },
        {
            "Data quality": "Limited",
            "Required conditions": "At least 60 pre-DD reports, 10–19 comparable post-DD reports or 40–59% coverage, or another non-critical evidence limitation.",
            "Effect on evidence category": "Can support only an Evidence-Limited assessment.",
        },
        {
            "Data quality": "Insufficient",
            "Required conditions": "Fewer than 60 pre-DD reports, fewer than 10 comparable post-DD reports, below 40% fuel coverage, or failed reporting-interval integrity.",
            "Effect on evidence category": "Inconclusive.",
        },
    ]
)

ASSESSMENT_FLOW_GUIDE = pd.DataFrame(
    [
        {"Step": "1. Data quality", "Question": "Is there enough trustworthy and comparable pre-/post-DD evidence?", "Output": "High, Limited or Insufficient"},
        {"Step": "2. Model validity", "Question": "Does the pre-DD model predict unseen data within the declared MAPE, bias and engineering limits?", "Output": "Pass or critical failure"},
        {"Step": "3. Evidence category", "Question": "Can the calculated percentage be interpreted?", "Output": "Supported Assessment, Evidence-Limited or Inconclusive"},
        {"Step": "4. Robustness", "Question": "Do reasonable alternative analyses retain the same direction?", "Output": "Stable or Sensitive"},
    ]
)


def selected_comparable_rows(result):
    """Return the report count for the comparison basis used in the headline result."""
    if result.get("comparison_basis") == "Strict same-route":
        return int(result.get("strict_supported_after_rows") or 0)
    if result.get("comparison_basis") == "Expanded cross-route":
        return int(result.get("expanded_supported_after_rows") or 0)
    return max(
        int(result.get("strict_supported_after_rows") or 0),
        int(result.get("expanded_supported_after_rows") or 0),
    )


def robustness_status(result):
    """Keep robustness separate from evidence strength."""
    raw = str(result.get("stability_status") or "").strip().lower()
    if raw in {"unstable", "sensitive"} or result.get("evidence_tier") == "Unstable":
        return "Sensitive"
    if raw == "stable":
        return "Stable"
    return "Not assessable"


def evidence_grade(result):
    """Classify evidence based on reliability, independently of saving direction."""
    return assess_reliability(result, result.get("_quality_settings", {}))["tier"]


def data_quality_grade(result, settings):
    """Grade data quantity and integrity separately from model performance."""
    return quality_level(result, settings)


def assessment_verdict(result):
    grade = evidence_grade(result)
    verdict = {
        "Supported Assessment": "Supported FOC assessment for prototype engineering screening",
        "Evidence-Limited": "Evidence-Limited - interpret cautiously",
        "Inconclusive": "Inconclusive - do not interpret the percentage",
    }[grade]
    if robustness_status(result) == "Sensitive":
        verdict += " | Sensitive to reasonable alternative analyses"
    return verdict


def business_decision(result):
    """Translate the prototype evidence tier into an honest user action."""
    grade = evidence_grade(result)
    guidance = {
        "Supported Assessment": (
            "Use the estimate as engineering screening evidence for the stated operating scope. "
            "Commercial or contractual verification still requires an agreed measurement protocol."
        ),
        "Evidence-Limited": (
            "Treat this as a screening indication only. Do not use it alone for an investment, "
            "guarantee or contractual claim; collect the missing comparable evidence and reassess."
        ),
        "Inconclusive": (
            "Do not interpret the calculated percentage. Correct the failed model, data or "
            "operating-comparability requirement before reassessment."
        ),
    }[grade]
    if robustness_status(result) == "Sensitive":
        guidance += " Investigate why reasonable alternative analyses disagree before making any performance claim."
    if placebo_signal(result)["status"] == "Not separated":
        guidance += " The estimated effect magnitude is not separated from fake-date effects; avoid asserting a detected dry-dock benefit or penalty."
    return guidance


def plain_language_conclusion(result):
    """Translate the numeric estimate and evidence tier for a non-specialist."""
    effect = result.get("improvement_pct")
    tier = evidence_grade(result)
    if effect is None or not np.isfinite(effect):
        effect_text = "The app could not calculate a dependable package-level fuel difference."
    elif tier == "Inconclusive":
        effect_text = (
            f"The app calculated a numerical difference of {effect:.2f}%, but at least one critical "
            "model, data or operating-comparability requirement failed. The percentage is shown only "
            "for diagnostic review and must not be interpreted as saving or deterioration."
        )
    elif effect >= 0:
        effect_text = (
            f"Across the comparable reports used, the vessel recorded approximately {effect:.2f}% "
            "less fuel than the pre-dry-dock model expected."
        )
    else:
        effect_text = (
            f"Across the comparable reports used, the vessel recorded approximately {abs(effect):.2f}% "
            "more fuel than the pre-dry-dock model expected. This does not by itself prove that dry "
            "docking caused poorer performance."
        )
    if tier == "Evidence-Limited" and effect is not None and np.isfinite(effect):
        effect_text += (
            " This is an Evidence-Limited estimate, not a confirmed saving or deterioration, "
            "because one or more supporting requirements were not strong enough for Supported evidence."
        )
    reasons = []
    if result.get("comparison_basis") == "Expanded cross-route":
        reasons.append("the same-route comparison was insufficient, so different routes were included")
    if result.get("flagged_foc_rows", 0):
        reasons.append(f"{result['flagged_foc_rows']} gross FOC report(s) require engineering review")
    if robustness_status(result) == "Sensitive":
        reasons.append("the estimated direction changed during sensitivity checks")
    if result.get("coverage_pct", 0) < 60:
        reasons.append("less than the 60% supported-assessment fuel-coverage target was represented")
    reason_text = (
        " The conclusion is limited because " + "; ".join(reasons) + "."
        if reasons else ""
    )
    return tier, effect_text + reason_text


def tier_assignment_explanation(result):
    """Explain why this vessel received its displayed evidence tier."""
    tier = evidence_grade(result)
    explanations = {
        "Supported Assessment": (
            "The critical model-validation, data-quantity and operating-comparability "
            "requirements passed for prototype engineering screening."
        ),
        "Evidence-Limited": (
            "A numerical estimate was calculated, but one or more non-critical supporting "
            "requirements were not strong enough for Supported evidence."
        ),
        "Inconclusive": (
            "A critical model, data or operating-comparability requirement failed, so no "
            "reliable FOC conclusion can be made."
        ),
    }

    reasons = list(assess_reliability(result, result.get("_quality_settings", {}))["reasons"])
    if tier == "Supported Assessment":
        reasons = ["Sufficient matched pre/post data and fuel coverage, predictive-model validation passed, and sensitivity checks retained a stable direction."]


    if robustness_status(result) == "Sensitive":
        stability_summary = str(result.get("stability_summary") or "").strip()
        if stability_summary and stability_summary not in reasons:
            reasons.insert(0, stability_summary)
        elif not reasons:
            reasons.append(
                "At least one reasonable sensitivity analysis changed the result from "
                "saving to increased FOC, or vice versa."
            )
    if tier == "Inconclusive":
        failure_reason = str(result.get("reason") or "").strip()
        if failure_reason and failure_reason not in reasons:
            reasons.insert(0, failure_reason)
        elif not reasons:
            reasons.append(
                "The available model or comparison evidence did not pass the minimum "
                "requirements for interpretation."
            )

    return explanations.get(tier, "The assessment status could not be explained."), reasons


def show_assessment_guide(result, settings):
    st.markdown(
        "The app grades analysis reliability regardless of whether post-DD FOC increased or decreased. Data quality, evidence strength "
        "and robustness are deliberately not combined into one ambiguous label."
    )
    q1, q2, q3 = st.columns(3)
    q1.metric("Current data quality", data_quality_grade(result, settings))
    q2.metric("Current evidence category", evidence_grade(result))
    q3.metric("Current robustness", robustness_status(result))
    st.markdown("**Assessment sequence**")
    st.dataframe(ASSESSMENT_FLOW_GUIDE, hide_index=True, width="stretch")
    st.markdown("**Data-quality conditions**")
    st.dataframe(DATA_QUALITY_GUIDE, hide_index=True, width="stretch")
    st.markdown("**Evidence-category conditions**")
    st.dataframe(CONCLUSION_GUIDE, hide_index=True, width="stretch")
    st.caption(
        "Stable means that reasonable alternative analyses retained the same direction. "
        "It does not mean that the estimate is accurate. Sensitive means that the direction changed."
    )


def comparison_method_label(result):
    return {
        "Strict same-route": "Same-route comparison (preferred)",
        "Expanded cross-route": "Cross-route comparison (fallback)",
    }.get(result.get("comparison_basis"), "No usable comparison")


def comparison_selection_table(result, settings):
    minimum_reports = int(settings.get("minimum_supported_after", 10))
    strict_rows = int(result.get("strict_supported_after_rows") or 0)
    expanded_rows = int(result.get("expanded_supported_after_rows") or 0)
    strict_coverage = float(result.get("strict_coverage_pct") or 0.0)
    expanded_coverage = float(result.get("expanded_coverage_pct") or 0.0)
    selected = result.get("comparison_basis")
    strict_ready = strict_rows >= minimum_reports and strict_coverage >= 40.0
    expanded_ready = expanded_rows >= minimum_reports and expanded_coverage >= 40.0
    return pd.DataFrame(
        [
            {
                "Comparison option": "Same-route comparison",
                "Condition-matching rule": "Same route and operating leg; comparable STW and displacement",
                "Comparable post-DD reports": strict_rows,
                "Eligible post-DD fuel represented": _fmt(strict_coverage, 1, "%"),
                "Outcome": (
                    "Selected"
                    if selected == "Strict same-route"
                    else f"Insufficient: below {minimum_reports} reports or 40% coverage"
                ),
            },
            {
                "Comparison option": "Cross-route fallback",
                "Condition-matching rule": "Different routes allowed; same operating leg and comparable STW/displacement",
                "Comparable post-DD reports": expanded_rows,
                "Eligible post-DD fuel represented": _fmt(expanded_coverage, 1, "%"),
                "Outcome": (
                    "Selected fallback"
                    if selected == "Expanded cross-route" and expanded_ready
                    else "Fallback also insufficient"
                    if selected == "Expanded cross-route"
                    else "Not required"
                ),
            },
        ]
    )


def show_comparison_selection(result, settings):
    minimum_reports = int(settings.get("minimum_supported_after", 10))
    selected = result.get("comparison_basis")
    strict_rows = int(result.get("strict_supported_after_rows") or 0)
    strict_coverage = float(result.get("strict_coverage_pct") or 0.0)
    st.dataframe(
        comparison_selection_table(result, settings),
        hide_index=True,
        width="stretch",
    )
    if selected == "Strict same-route":
        st.info(
            f"The same-route comparison met the minimum calculation floor of {minimum_reports} "
            "comparable post-DD reports and 40% fuel coverage."
        )
    elif selected == "Expanded cross-route":
        st.warning(
            f"The same-route comparison produced {strict_rows} comparable reports and "
            f"{strict_coverage:.1f}% fuel coverage, below the minimum floor of {minimum_reports} "
            "reports and 40% coverage. The app therefore used different routes while retaining "
            "the same operating leg and vessel-specific STW/displacement support test. "
            "Cross-route matching can qualify for Supported Assessment only when pre-DD operating support, evidence checks and the same operating leg are confirmed."
        )
    st.caption(
        "Both options first require Beaufort <=4 and at least 18 propelling hours. The support "
        "test then checks whether each post-DD STW/displacement condition was represented in "
        "the relevant pre-DD data."
    )


def methods_used_table(result):
    features = result.get("numeric_features", [])
    predictors = "STW and displacement" if "LogDisplacementRatio" in features else "STW"
    return pd.DataFrame(
        [
            {
                "Assessment step": "Calculate each noon report",
                "Data used": "M/E fuel by grade, LCV, LOG distance and propelling hours",
                "Method used": "VLSFO-equivalent 24-hour FOC and STW calculations",
                "Output": "FOC and STW for each report",
            },
            {
                "Assessment step": "Predict expected post-DD FOC",
                "Data used": f"Valid pre-DD {predictors}",
                "Method used": result.get("selected_ml_model") or "Huber ML",
                "Output": "Expected FOC at each comparable post-DD condition",
            },
            {
                "Assessment step": "Compare with a public benchmark",
                "Data used": "The same pre-DD validation and comparable post-DD reports",
                "Method used": "Cubic-speed benchmark: FOC = a x STW^3",
                "Output": "Benchmark prediction error and benchmark saving",
            },
            {
                "Assessment step": "Select comparable post-DD data",
                "Data used": "Route, operating leg, STW and displacement",
                "Method used": comparison_method_label(result),
                "Output": "Post-DD reports included in the fuel comparison",
            },
            {
                "Assessment step": "Calculate package saving",
                "Data used": "Expected FOC, reported FOC and propelling hours",
                "Method used": "Propelling-hour-weighted interval fuel aggregation",
                "Output": "Package-level FOC saving percentage",
            },
        ]
    )


def assessment_data_table(saved):
    result = saved["result"]
    counts = selection_counts(saved["period_data"], saved["eligible"], result)
    return pd.DataFrame(
        [
            {
                "Data stage": "Within selected assessment dates",
                "Pre-DD reports": counts["pre_raw"],
                "Post-DD reports": counts["post_raw"],
                "Purpose": "Available before common operating filters",
            },
            {
                "Data stage": "Passed common operating filters",
                "Pre-DD reports": counts["pre_eligible"],
                "Post-DD reports": counts["post_eligible"],
                "Purpose": "Valid for ML training or comparability testing",
            },
            {
                "Data stage": "Used in the assessment",
                "Pre-DD reports": result.get("before_rows", 0),
                "Post-DD reports": counts["supported"],
                "Purpose": "Trained the ML model or contributed to the saving",
            },
        ]
    )


def main_filter_reasons(saved, limit=3):
    ledger = build_report_ledger(
        saved["period_data"], saved["eligible"], saved["result"], saved["settings"]
    )
    reasons = (
        ledger.loc[
            ledger["Assessment use"].eq("Excluded by operating filters"), "Reason"
        ]
        .fillna("")
        .astype(str)
        .str.split("; ")
        .explode()
        .str.strip()
    )
    counts = reasons.loc[reasons.ne("")].value_counts().head(limit)
    return [f"{reason} ({count})" for reason, count in counts.items()]


def evidence_checklist(saved):
    result, settings = saved["result"], saved["settings"]
    validation = result.get("validation", {})
    mape = validation.get("huber", {}).get("mape_pct")
    bias = validation.get("huber", {}).get("bias_pct")
    placebo = result.get("placebo", {})
    dominant = result.get("service_leg_summary", pd.DataFrame())
    dominant_share = (
        float(dominant.iloc[0]["FuelSharePct"])
        if isinstance(dominant, pd.DataFrame) and not dominant.empty
        else None
    )
    documentation_complete = bool(settings.get("documentation_complete"))
    return pd.DataFrame(
        [
            {
                "Evidence check": "Overall data-quality grade",
                "Outcome": data_quality_grade(result, settings),
                "What it means": "Assessed separately from model validity and sensitivity",
            },
            {
                "Evidence check": "Pre-DD training data",
                "Outcome": "Pass" if result.get("before_rows", 0) >= settings.get("minimum_train_rows", 60) else "Review",
                "What it means": f"{result.get('before_rows', 0)} reports used",
            },
            {
                "Evidence check": "Chronological prediction error",
                "Outcome": "Pass" if mape is not None and mape <= 12 and bias is not None and abs(bias) <= 8 else "Fail - result is Inconclusive",
                "What it means": f"MAPE {_fmt(mape, 2, '%')} (limit 12%); bias {_fmt(bias, 2, '%')} (absolute limit 8%)",
            },
            {
                "Evidence check": "Comparable post-DD fuel coverage",
                "Outcome": (
                    "Pass"
                    if result.get("coverage_pct", 0) >= 60
                    else "Calculation floor only"
                    if result.get("coverage_pct", 0) >= 40
                    else "Fail - below calculation floor"
                ),
                "What it means": _fmt(result.get("coverage_pct"), 1, "%") + " of eligible post-DD fuel; 40% minimum and 60% supported target",
            },
            {
                "Evidence check": "Complete normal service cycle",
                "Outcome": "Confirmed" if settings.get("service_cycle_confirmed") else "Not confirmed",
                "What it means": "Based on the user's operational-record confirmation",
            },
            {
                "Evidence check": "ML specification review",
                "Outcome": "Selected before post-DD prediction",
                "What it means": result.get("selected_ml_model") or "Unavailable",
            },
            {
                "Evidence check": "ML comparison with public cubic-speed benchmark",
                "Outcome": "Benchmark lower MAPE" if validation.get("huber_outperforms_cubic") is False else "Selected Huber lower or equal MAPE",
                "What it means": "ML superiority not demonstrated" if validation.get("huber_outperforms_cubic") is False else "Benchmark comparison passed",
            },
            {
                "Evidence check": "Fake-date effect-signal diagnostic (not a reliability gate)",
                "Outcome": placebo_signal(result)["status"],
                "What it means": placebo_signal(result)["explanation"],
            },
            {
                "Evidence check": "Operating-scope representation",
                "Outcome": "Limited" if dominant_share is not None and dominant_share >= 80 else "Pass",
                "What it means": result.get("scope_statement", "Unavailable"),
            },
            {
                "Evidence check": "Source references",
                "Outcome": "Recorded" if documentation_complete else "Incomplete",
                "What it means": "Dry-dock, route, service-cycle and LCV references",
            },
        ]
    )


def show_limits(result):
    limits = result.get("limitations", [])
    if limits:
        st.warning("Limits on interpretation:\n\n- " + "\n- ".join(limits))


def explain_report(fuel):
    rows = fuel["rows"].reset_index(drop=True)
    selected = st.selectbox("Comparable report", list(range(len(rows))), key="example_report",
        format_func=lambda i: f"{pd.Timestamp(rows.iloc[i]['Date']):%d %b %Y} | Voyage {rows.iloc[i].get('Voyage', 'Unknown')}")
    row = rows.iloc[selected]
    c1, c2, c3 = st.columns(3)
    c1.metric("Recorded STW", f"{row['STW']:.2f} kn")
    c2.metric("Recorded displacement", f"{row['DisplacementMT']:,.0f} MT")
    c3.metric("Propelling time", f"{row['PropellingHours']:.2f} h")
    st.table(pd.DataFrame({
        "Fuel basis": ["Expected from pre-DD model", "Reported after DD"],
        "FOC (VLSFO-eq. MT/day)": [f"{row['ExpectedPreDDCondition_FOC_MT_Day']:.2f}", f"{row['FOC_MT_Day']:.2f}"],
        "Interval fuel (VLSFO-eq. MT)": [f"{row['ExpectedForIllustrationMT']:.2f}", f"{row['ActualForIllustrationMT']:.2f}"],
    }).set_index("Fuel basis"))
    st.write(f"Interval difference: **{row['ExpectedForIllustrationMT'] - row['ActualForIllustrationMT']:+.2f} VLSFO-equivalent MT**.")
    st.caption("Interval fuel = FOC x propelling hours / 24. Add all comparable interval amounts before calculating the overall percentage; do not average individual report percentages.")


def show_results(saved):
    result, settings = saved["result"], saved["settings"]
    st.title(f"{settings['vessel']}: Package-level post-dry-dock FOC assessment")
    st.caption(
        f"V9.14 | Dock-in: {settings['dock_in']} | Dock-out: {settings['dock_out']} | "
        f"Post-DD endpoint: {settings['required_post_end']} | "
        + settings.get("monitoring_basis", "Selected post-DD period")
    )
    fuel = fuel_summary(result)
    if not result.get("valid") or fuel is None:
        st.error(result.get("reason") or "No valid estimate is available.")
        show_limits(result)
        st.info("Open Supporting analysis / Reports used and excluded to review the available reports and exclusion reasons.")
        return
    if not np.isclose(fuel["saving_pct"], result["improvement_pct"], atol=1e-7, rtol=0):
        st.error("The fuel comparison does not reconcile with the primary estimate. No result is displayed.")
        return
    if settings.get("is_demo"):
        st.warning("Synthetic demonstration - this illustrates the workflow and is not evidence from an actual vessel.")

    st.subheader("1. Assessment conclusion")
    tier, conclusion_text = plain_language_conclusion(result)
    conclusion_message = f"**{assessment_verdict(result)}**\n\n{conclusion_text}"
    if tier == "Supported Assessment":
        st.success(conclusion_message)
    elif tier == "Evidence-Limited":
        st.warning(conclusion_message)
    else:
        st.error(conclusion_message)

    assignment_summary, assignment_reasons = tier_assignment_explanation(result)
    st.markdown("**Why this conclusion was assigned**")
    st.write(assignment_summary)
    if assignment_reasons:
        st.markdown("\n".join(f"- {reason}" for reason in assignment_reasons))
    st.caption("V9.14 reliability grading uses revised, declared prototype thresholds. "
               "Previous model classification: " + str(result.get("original_model_evidence_tier", result.get("evidence_tier", "Unavailable"))) + ". "
               "The fuel estimate and underlying validation metrics have not been changed.")

    st.info("**Recommended action:** " + business_decision(result))
    d1, d2, d3 = st.columns(3)
    d1.metric("Data quality", data_quality_grade(result, settings))
    d2.metric("Evidence category", evidence_grade(result))
    d3.metric("Robustness", robustness_status(result))
    st.caption(
        "Data quality describes the available comparison evidence; evidence category determines "
        "whether the percentage can be interpreted; robustness describes sensitivity to alternative analyses."
    )
    with st.expander("How this result was assessed", expanded=False):
        show_assessment_guide(result, settings)

    m1, m2, m3 = st.columns(3)
    m1.metric("ML-estimated package FOC change", _fmt(result["improvement_pct"], 2, "%"))
    m2.metric("Post-DD comparison basis", comparison_method_label(result))
    m3.metric("Eligible post-DD fuel represented", _fmt(result.get("coverage_pct"), 1, "%"))
    st.caption(
        "The displayed decimal precision reflects the calculation output and does not represent measurement certainty."
    )

    st.subheader("2. Reports selected for the comparison")
    st.markdown("**How comparable post-DD reports were selected**")
    show_comparison_selection(result, settings)
    st.markdown("**Reports available, eligible and used**")
    st.dataframe(assessment_data_table(saved), hide_index=True, width="stretch")
    st.caption(
        f"The uploaded workbook contained {saved['filter_report']['input_rows']:,} raw reports. "
        "Reports failing common filters remain available in the Technical Evidence report audit."
    )
    common_reasons = main_filter_reasons(saved)
    if common_reasons:
        st.caption("Most common operating-filter exclusions: " + "; ".join(common_reasons) + ".")

    st.subheader("3. How the package-level FOC result was calculated")
    st.image(
        figure_png(assessment_story_figure(fuel, result, assessment_verdict(result))),
        width="stretch",
    )
    st.markdown("### Technical basis of this calculation")

    selected_features = result.get("numeric_features", [])
    if "LogDisplacementRatio" in selected_features:
        prediction_variables = "Speed through water (STW) and displacement"
    else:
        prediction_variables = (
            "Speed through water (STW) only. Displacement is still "
            "checked when determining operating comparability."
        )

    comparison_basis = result.get("comparison_basis")
    if comparison_basis == "Strict same-route":
        comparison_description = (
            "Same-route comparison: post-DD reports are compared with "
            "pre-DD data from the same route and operating leg."
        )
    elif comparison_basis == "Expanded cross-route":
        comparison_description = (
            "Expanded cross-route comparison: different routes are used "
            "because same-route evidence was insufficient. The app retains "
            "the same operating leg and comparable STW and displacement."
        )
    else:
        comparison_description = "No usable operating comparison was available."

    technical_basis = pd.DataFrame(
        {
            "Technical item": [
                "Baseline model",
                "Prediction variables",
                "Model selection",
                "Operating comparison",
                "Eligibility filters",
                "Final calculation",
            ],
            "Method used": [
                (
                    "Huber regression trained using pre-DD noon reports. "
                    "Huber regression reduces the influence of abnormal reports "
                    "without automatically deleting them."
                ),
                prediction_variables,
                (
                    "Candidate ML models are tested on later pre-DD reports "
                    "that were not used for training. Post-DD results are not "
                    "used to select the model."
                ),
                comparison_description,
                (
                    "At least 18 propelling hours, Beaufort 4 or below, "
                    "valid interval-aligned fuel and LOG distance, and "
                    "physically plausible STW."
                ),
                (
                    "Expected and reported fuel are calculated over the same "
                    "comparable post-DD intervals and weighted by propelling hours."
                ),
            ],
        }
    )

    st.dataframe(
        technical_basis,
        hide_index=True,
        width="stretch",
    )

    st.latex(
        r"FOC\ saving(\%)="
        r"\frac{Expected\ fuel-Reported\ fuel}"
        r"{Expected\ fuel}\times100"
    )

    st.caption(
        "A positive result means reported post-DD fuel was lower than expected. "
        "A negative result means reported post-DD fuel was higher than expected."
    )

    st.subheader("4. Checks affecting how the result can be used")
    validation = result.get("validation", {})
    sensitivity_low = result.get("stability_min_pct")
    sensitivity_high = result.get("stability_max_pct")
    sensitivity_text = (
        f"{sensitivity_low:.2f}% to {sensitivity_high:.2f}%"
        if sensitivity_low is not None and sensitivity_high is not None
        else "Unavailable"
    )
    r1, r2, r3, r4 = st.columns(4)
    r1.metric(
        "Robustness",
        robustness_status(result),
        help="Whether the tested alternatives retain the same saving-or-higher-FOC direction; this does not mean the percentage is unchanged.",
    )
    r2.metric("Unseen pre-DD prediction error", _fmt(validation.get("huber", {}).get("mape_pct"), 2, "%"), help="Average percentage error when predicting later pre-dry-dock reports not used for fitting.")
    r3.metric(
        "Sensitivity range",
        sensitivity_text,
        help="Range across model, route, anomaly and individual-report rechecks; not a confidence interval.",
    )
    r4.metric("Gross FOC reports flagged", f"{result.get('flagged_foc_rows', 0)}")
    st.write("**Operating scope:** " + result.get("scope_statement", "Comparable post-DD reports only."))
    st.caption("The estimate is package-level. It does not prove causation or assign saving to individual dry-dock work items.")

    with st.expander("Limits behind this conclusion"):
        show_limits(result)
        st.caption("The complete decision checklist is available under Supporting analysis / Evidence checks and model reliability.")
    with st.expander("How individual reports contribute to the overall estimate"):
        explain_report(fuel)
        st.code(f"Overall: ({fuel['expected']:,.2f} - {fuel['actual']:,.2f}) / {fuel['expected']:,.2f} x 100 = {fuel['saving_pct']:.2f}%", language=None)
        st.caption("Displayed totals are rounded. The calculation uses full precision.")
    # Original report generator may still use historical tier wording; expose revised grade in separate CSV.
    grade_report = assess_reliability(result, settings)
    grade_csv = pd.DataFrame([{
        "Vessel": settings.get("vessel", ""),
        "Reliability tier (V9.14)": grade_report["tier"],
        "FOC saving (+) / increase (-), %": result.get("improvement_pct"),
        "Pre-DD reports": result.get("before_rows"),
        "Comparable post-DD reports": selected_comparable_rows(result),
        "Eligible post-DD fuel coverage, %": result.get("coverage_pct"),
        "Chronological MAPE, %": result.get("validation", {}).get("huber", {}).get("mape_pct"),
        "Absolute bias, %": abs(result.get("validation", {}).get("huber", {}).get("bias_pct") or 0),
        "Robustness": robustness_status(result),
        "Effect signal vs pre-DD fake dates (exploratory)": placebo_signal(result)["status"],
        "Prior model tier": result.get("original_model_evidence_tier", result.get("evidence_tier")),
        "Reasons": "; ".join(grade_report["reasons"]),
    }]).to_csv(index=False)
    st.download_button("Download V9.14 reliability assessment (CSV)", data=grade_csv,
                       file_name="mipd_foc_reliability_v9_14.csv", mime="text/csv")
    st.caption("The printable HTML and technical workbook may include the prior model-tier nomenclature; "
               "the V9.14 reliability CSV and on-screen conclusion use the revised tier.")
    st.download_button("Download printable assessment summary (HTML)", data=printable_report(saved),
                       file_name="mipd_foc_summary_v9_14.html", mime="text/html")
    st.caption("Open the downloaded summary in a browser and choose Print / Save as PDF.")


def show_checks(saved):
    result = saved["result"]
    st.subheader("Evidence checks behind the conclusion")
    st.write("These project-defined checks determine whether the calculated percentage is supported, limited or not interpretable. They are not certification requirements.")
    st.dataframe(evidence_checklist(saved), hide_index=True, width="stretch")
    st.subheader("ML model selected from pre-DD data")
    st.write(result.get("model_selection_reason") or "No model-selection explanation is available.")
    comparison = result.get("ml_candidate_comparison", pd.DataFrame())
    if isinstance(comparison, pd.DataFrame) and not comparison.empty:
        st.dataframe(comparison.round(2), hide_index=True, width="stretch")
    sensitivity = result.get("model_sensitivity", pd.DataFrame())
    if isinstance(sensitivity, pd.DataFrame) and not sensitivity.empty:
        st.dataframe(sensitivity.round(2), hide_index=True, width="stretch")
        st.caption("These are alternative method results, not additive savings or a statistical confidence interval.")
    st.subheader("Direction check across alternative analyses")
    st.metric("Robustness", robustness_status(result))
    st.write(result.get("stability_summary", "No stability explanation is available."))
    if result.get("stability_min_pct") is not None and result.get("stability_max_pct") is not None:
        st.caption(
            f"Available sensitivity range: {result['stability_min_pct']:.2f}% to "
            f"{result['stability_max_pct']:.2f}%. This range is not a statistical confidence interval."
        )
    anomalies = result.get("foc_anomalies", pd.DataFrame())
    if isinstance(anomalies, pd.DataFrame) and not anomalies.empty:
        st.warning(
            "Gross FOC reports were flagged for review. They remain in the source audit and were not "
            "silently deleted from the primary calculation."
        )
        st.dataframe(anomalies.round(3), hide_index=True, width="stretch")
    st.subheader("Prediction check on unseen pre-DD reports")
    st.write("Earlier pre-DD reports train each model and later pre-DD reports test it. This checks performance on observations that were not used to fit the model.")
    validation = result["validation"]
    left, right = st.columns(2)
    left.metric("Selected Huber MAPE", _fmt(validation["huber"].get("mape_pct"), 2, "%"))
    right.metric("Cubic-speed benchmark MAPE", _fmt(validation["cubic"].get("mape_pct"), 2, "%"))
    st.caption("Lower MAPE indicates lower average percentage prediction error. MAPE is not an uncertainty interval around the saving estimate.")
    better = validation.get("huber_outperforms_cubic")
    if better is False:
        difference = validation["huber"].get("mape_pct") - validation["cubic"].get("mape_pct")
        st.warning(f"ML superiority was not demonstrated. The selected Huber MAPE was {difference:.2f} percentage points higher than the public cubic-speed benchmark on these unseen reports.")
    elif better is True:
        st.info("The selected Huber model had lower or equal MAPE on these unseen reports. This does not guarantee better performance for another vessel or operating period.")
    else:
        st.warning("There were insufficient valid test results to compare prediction errors.")
    with st.expander("Validation metrics and report-level predictions"):
        st.dataframe(_validation_table(validation).round(2), hide_index=True, width="stretch")
        st.dataframe(validation["predictions"], hide_index=True, width="stretch", height=260)
    st.subheader("False-effect check using pre-DD dates")
    placebo = result.get("placebo", {})
    count = placebo.get("valid_count", 0)
    signal = placebo_signal(result)
    if signal["status"] == "Not assessable":
        st.info(signal["explanation"])
    elif signal["status"] == "Magnitude exceeds placebos":
        st.info(signal["explanation"])
    else:
        st.warning(signal["explanation"])
    st.caption("This two-sided diagnostic compares the absolute estimated effect with absolute fake-date effects. It does not determine the quality tier, and it is not evidence of causality. A Supported Assessment can still have an effect that is not distinguishable from normal variation.")
    with st.expander("Fake-date results and exploratory statistic"):
        st.metric("Exploratory one-sided p-value", _fmt(placebo.get("empirical_p_one_sided"), 3))
        st.caption("This statistic is not the probability that the saving is real. No 100% confidence claim is made.")
        st.dataframe(placebo.get("results", pd.DataFrame()), hide_index=True, width="stretch")
    with st.expander("Primary ML estimate and public cubic-speed benchmark"):
        st.table(pd.DataFrame({"Method": [str(result.get("selected_ml_model") or "Huber ML") + " - primary", "Public cubic-speed benchmark (V^3)"],
            "Estimated saving (%)": [_fmt(result.get("improvement_pct"), 2), _fmt(result.get("cubic_benchmark_saving_pct"), 2)]}).set_index("Method"))
        st.caption("Both methods use the same comparable post-DD reports. The cubic-speed rule is a simple public physics benchmark, not an ISO-prescribed FOC-saving method or a replacement for the primary ML method. The estimates are not additive.")


def show_operating_evidence(saved):
    result = saved["result"]
    assessed = result.get("assessed_rows", pd.DataFrame())
    st.subheader("Reported post-DD FOC compared with model-expected FOC")
    figure = _actual_expected_scatter(assessed)
    if figure is None:
        st.info("No assessed post-DD reports are available for this comparison.")
    else:
        st.plotly_chart(figure, width="stretch")
        st.caption("The dashed diagonal represents reported FOC equal to model-expected FOC. Comparable reports below the line used less fuel than expected; reports above the line used more.")
        comparison_comment = actual_expected_comment(result)
        st.markdown("**What this chart shows for the selected vessel**")
        st.write(comparison_comment["summary"])
        if comparison_comment["warnings"]:
            st.warning("Important context:\n\n- " + "\n- ".join(comparison_comment["warnings"]))

    st.subheader("Same-route and cross-route comparison options")
    st.table(pd.DataFrame({
        "Comparison": ["Same-route comparison (preferred)", "Cross-route comparison (fallback)"],
        "Comparable reports": [
            result.get("strict_supported_after_rows", 0),
            result.get("expanded_supported_after_rows", 0),
        ],
        "Comparable fuel coverage": [
            _fmt(result.get("strict_coverage_pct"), 1, "%"),
            _fmt(result.get("expanded_coverage_pct"), 1, "%"),
        ],
        "Estimated effect": [
            _fmt(result.get("strict_saving_pct"), 2, "%"),
            _fmt(result.get("expanded_saving_pct"), 2, "%"),
        ],
        "Role": [
            "Preferred" if result.get("comparison_basis") == "Strict same-route" else "Insufficient for headline",
            "Selected fallback" if result.get("comparison_basis") == "Expanded cross-route" else "Not required",
        ],
    }).set_index("Comparison"))

    st.caption("Automatic interpretations are recalculated from the current assessment. They describe the displayed evidence and do not establish causation.")

    with st.expander("Technical diagnostic: were post-DD speed and loading represented before dry docking?", expanded=False):
        st.plotly_chart(_support_scatter(saved["eligible"], assessed), width="stretch")
        st.caption(
            "Blue points are relevant pre-DD training reports. Green points are comparable post-DD reports used. "
            "Amber points are post-DD reports outside learned support. This diagnostic shows the report-level "
            "STW and displacement classification; it is not a trend graph and does not calculate the saving."
        )
        support_comment = operating_support_comment(saved["eligible"], result)
        st.write(support_comment["summary"])
        profile = result.get("pre_dd_speed_profile", {})
        if profile:
            st.caption(
                f"Valid pre-DD STW observed: {profile['minimum']:.1f}-{profile['maximum']:.1f} kn; "
                f"central 90%: {profile['p05']:.1f}-{profile['p95']:.1f} kn. These values describe "
                "the available data and are not a fixed speed filter."
            )

    with st.expander("Report-level expected and reported FOC"):
        st.dataframe(report_comparison(assessed).round(2), hide_index=True, width="stretch")


def show_reports(saved):
    result, settings = saved["result"], saved["settings"]
    counts = selection_counts(saved["period_data"], saved["eligible"], result)
    st.subheader("Noon-report inclusion and exclusion log")
    st.table(pd.DataFrame({"Stage": ["Within selected dates", "Excluded by operating filters", "Passed operating filters"],
        "Pre-DD": [counts["pre_raw"], counts["pre_raw"] - counts["pre_eligible"], counts["pre_eligible"]],
        "Post-DD": [counts["post_raw"], counts["post_raw"] - counts["post_eligible"], counts["post_eligible"]]}).set_index("Stage"))
    st.write(f"Post-DD support assessment: **{counts['supported']} used**, **{counts['outside']} outside pre-DD support**, **{counts['unassessed']} not assessed**.")
    st.caption(f"Raw workbook: {saved['filter_report']['input_rows']} reports. Pre-DD and post-DD counts are separate groups. Passing operating filters is not the same as passing the operating-support test.")
    ledger = build_report_ledger(saved["period_data"], saved["eligible"], result, settings)
    status = st.selectbox("Report category to display", ["All"] + sorted(ledger["Assessment use"].unique().tolist()))
    display = ledger if status == "All" else ledger.loc[ledger["Assessment use"].eq(status)]
    cols = [
        "ReportID", "Date", "Assessment use", "Reason", "FOC anomaly flag",
        "FOC anomaly review", "Period", "Voyage", "AnalysisRoute", "ServiceLeg",
        "STW", "DisplacementMT", "FOC_MT_Day", "PropellingHours", "Beaufort",
    ]
    display_table = display[[c for c in cols if c in display]].rename(
        columns={
            "ReportID": "Report ID",
            "AnalysisRoute": "Assessment route",
            "ServiceLeg": "Operating leg",
            "DisplacementMT": "Displacement (MT)",
            "FOC_MT_Day": "FOC (VLSFO-eq. MT/day)",
            "PropellingHours": "Propelling hours",
        }
    )
    st.dataframe(display_table, hide_index=True, width="stretch", height=360)
    st.caption("Report ID is the parsed report sequence, not an Excel row number. A report may fail multiple rules; reasons are listed together and it is counted once. No raw rows are deleted.")
    st.download_button("Download complete report audit (CSV)", data=ledger.to_csv(index=False).encode("utf-8-sig"),
                       file_name="mipd_report_audit_v9_14.csv", mime="text/csv")
    with st.expander("Expected and reported FOC for assessed post-DD reports"):
        st.caption("Fuel is VLSFO-equivalent. Unsupported rows are marked No and do not contribute to the estimate.")
        st.dataframe(report_comparison(result.get("assessed_rows", pd.DataFrame())).round(2), hide_index=True, width="stretch")


def show_method(saved):
    result, settings = saved["result"], saved["settings"]
    st.subheader("Calculation method, assumptions and downloads")
    st.info("V9.14 grades input/model assessment quality independently of FOC direction and independently "
            "of the size of fake-date effects. A Supported Assessment does not imply statistically "
            "distinct savings or higher FOC. The underlying Huber prediction is unchanged.")
    st.markdown("**Methods applied in this assessment**")
    st.dataframe(methods_used_table(result), hide_index=True, width="stretch")
    st.markdown("**Calculation equations**")
    st.latex(r"FOC_{eq,24h}=\frac{\sum_f m_f LCV_f}{40.5}\frac{24}{H_p}")
    if "LogDisplacementRatio" in result.get("numeric_features", []):
        st.latex(r"\ln(FOC)=\beta_0+\beta_v\ln(STW)+\beta_\Delta\ln(\Delta/\Delta_{ref})")
    else:
        st.latex(r"\ln(FOC)=\beta_0+\beta_v\ln(STW)")
    st.latex(r"Saving(\%)=100\left(1-\frac{\sum_i FOC_{actual,i}H_i/24}{\sum_i\widehat{FOC}_iH_i/24}\right)")
    st.write("Both Huber specifications learn only from pre-DD reports. The app selects between STW-only and STW-and-displacement Huber using pre-DD chronological validation and coefficient review. Displacement always remains in the operating-support test. Beaufort <=4 and at least 18 propelling hours are fixed eligibility rules. The app first tests the same route and operating leg, then uses an expanded same-leg cross-route comparison only when strict evidence is insufficient. A training-only smearing factor returns log predictions to MT/day.")
    st.write(
        "For transparency, the selected Huber model is compared with the public cubic-speed rule of "
        "thumb, FOC = a x STW^3. The vessel-specific coefficient a is fitted using pre-DD reports only. "
        "This benchmark is not a confidential company method and is not an ISO-prescribed saving calculation."
    )
    st.info(result.get("model_selection_reason") or "No model-selection explanation is available.")
    details = result.get("model_details", {})
    if details:
        equation = f"ln(FOC) = {details['transformed_intercept']:.4f} {details['speed_coefficient']:+.4f} ln(STW)"
        if details.get("displacement_coefficient") is not None:
            equation += f" {details['displacement_coefficient']:+.4f} ln(displacement / {details['displacement_reference_mt']:.0f})"
        equation += f"\nPredicted FOC = {details['smearing_factor']:.4f} x exp(predicted ln(FOC))"
        st.code(equation, language=None)
        st.dataframe(result["coefficients"], hide_index=True, width="stretch")
    st.caption("Huber settings: epsilon 1.35 and alpha 0.0001. A result is Inconclusive when MAPE exceeds 12%, absolute bias exceeds 8%, validation is unavailable, comparable fuel coverage is below 40%, fewer than 10 comparable post-DD reports are available, or the learned speed exponent falls outside 1.5-4.5. Evidence-Limited is used when a result can be calculated but one or more non-critical supporting requirements are weaker than the Supported standard. At least 20 comparable reports and 60% fuel coverage are required for Supported Assessment; same-route is preferred but a confirmed same-leg cross-route fallback is permitted. These are declared project rules, not IMO, ISO or class requirements.")
    with st.expander("Method scope and standards reference", expanded=False):
        st.write(
            "This prototype applies the same-vessel and comparable-operating-condition principles commonly "
            "used in vessel performance assessment. It does not claim ISO 19030 conformity. The prototype "
            "uses noon-report data, vessel-specific Huber regression and a project-defined speed/loading "
            "comparability test rather than an ISO 19030 verification calculation. Results are intended for "
            "internal engineering screening. Its adjusted-baseline concept is consistent with general "
            "measurement-and-verification principles, but ISO 50015 conformity is not claimed."
        )
        st.table(
            pd.DataFrame(
                {
                    "Standards-related principle": [
                        "Same-vessel comparison",
                        "Comparable operating conditions",
                        "Weather restriction",
                        "Continuous sensor data",
                        "ISO reference-displacement correction",
                        "ISO 19030 performance indicator",
                        "ISO 19030 conformity claimed",
                        "ISO 50015 conformity claimed",
                        "Public speed benchmark",
                    ],
                    "Prototype treatment": [
                        "Applied",
                        "Applied using vessel-specific STW/displacement support",
                        "Beaufort 4 or below",
                        "Not available; noon reports are used",
                        "Not implemented",
                        "Not implemented",
                        "No",
                        "No",
                        "Cubic-speed rule of thumb, FOC = a x STW^3",
                    ],
                }
            ).set_index("Standards-related principle")
        )
        st.caption(
            "Public references: ISO 19030-2 overview (iso.org/standard/63775.html); "
            "ISO 50015 overview (iso.org/standard/60043.html); IMO GreenVoyage2050 speed management "
            "(greenvoyage2050.imo.org/technology/speed-management/)."
        )
    with st.expander("Saved inputs, source references and operating mappings"):
        st.json({key: str(value) for key, value in settings.items()})
        st.dataframe(saved["lcv_table"], hide_index=True, width="stretch")
        st.dataframe(saved["route_table"], hide_index=True, width="stretch")
        st.dataframe(saved["service_leg_table"], hide_index=True, width="stretch")
    export = _excel_export(result, settings, saved["eligible"], saved["lcv_table"], saved["route_table"], saved["service_leg_table"])
    st.download_button("Download assessment workbook", data=export,
        file_name="mipd_foc_assessment_v9_14.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    st.download_button("Download saved assessment inputs (JSON)", data=json.dumps(settings, default=str, indent=2),
        file_name="mipd_foc_inputs_v9_14.json", mime="application/json")
    st.download_button("Download full calculation methodology", data=Path(__file__).with_name("METHODOLOGY.md").read_text(encoding="utf-8"),
        file_name="MIPD_FOC_Methodology.md", mime="text/markdown")


if st.session_state.get("pending_page"):
    st.session_state["page"] = st.session_state.pop("pending_page")
if "page" not in st.session_state:
    st.session_state["page"] = "Result summary" if "mipd_v9_11" in st.session_state else "Setup"
st.sidebar.title("Package-level FOC assessment")
st.sidebar.caption("V9.14 | Last completed assessment is preserved")
page = st.sidebar.radio("View", ["Result summary", "Supporting analysis", "Setup"], key="page")
if page == "Setup":
    render_setup()
else:
    saved = st.session_state.get("mipd_v9_11")
    if saved is None:
        st.info("Open Setup to upload a workbook and run the assessment.")
        st.stop()
    # Re-grade the stored calculation; do not modify the Huber model or FOC values.
    saved["result"]["_quality_settings"] = saved.get("settings", {})
    saved["result"].setdefault("original_model_evidence_tier", saved["result"].get("evidence_tier"))
    if saved.get("setup_signature") != st.session_state.get("draft_signature"):
        st.warning("Setup has unrun changes. This is the last completed assessment, using its saved settings.")
    if page == "Result summary":
        show_results(saved)
    else:
        st.title("Supporting analysis and report audit")
        st.caption(f"{saved['settings']['vessel']} | Post-DD endpoint: {saved['settings']['required_post_end']}")
        topic = st.selectbox(
            "Supporting analysis section",
            ["Evidence checks and model reliability", "Comparable operating conditions", "Reports used and excluded", "Method and downloads"],
            key="technical_topic",
        )
        {
            "Evidence checks and model reliability": lambda: show_checks(saved),
            "Comparable operating conditions": lambda: show_operating_evidence(saved),
            "Reports used and excluded": lambda: show_reports(saved),
            "Method and downloads": lambda: show_method(saved),
        }[topic]()
