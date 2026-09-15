"""V9.11: guided package-level post-dry-dock FOC assessment."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

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
            "Conclusion": "Supported (prototype screening)",
            "Plain-language meaning": "The result passed the prototype's strongest data, model-validation, comparability and direction checks.",
            "Appropriate use": "Internal engineering screening; not certified or contractual evidence.",
        },
        {
            "Conclusion": "Indicative",
            "Plain-language meaning": "The validated model and same-route data support a limited engineering estimate.",
            "Appropriate use": "Early engineering estimate; continue monitoring.",
        },
        {
            "Conclusion": "Preliminary",
            "Plain-language meaning": "The model passed validation, but the operating comparison or amount of evidence remains limited.",
            "Appropriate use": "Do not use for a guarantee, investment approval or contractual claim.",
        },
        {
            "Conclusion": "Unstable",
            "Plain-language meaning": "Reasonable alternative analyses changed the result between saving and higher FOC.",
            "Appropriate use": "Do not claim saving or deterioration; review the data and operating conditions.",
        },
        {
            "Conclusion": "Inconclusive",
            "Plain-language meaning": "The model, data or operating support cannot support an interpretable percentage.",
            "Appropriate use": "Collect or correct the missing evidence before reassessment.",
        },
    ]
)


def assessment_verdict(result):
    return {
        "Supported (prototype screening)": "Supported for prototype engineering screening",
        "Indicative": "Indicative result - use with stated limitations",
        "Preliminary": "Preliminary result - more comparable evidence required",
        "Unstable": "Direction-sensitive result - do not interpret the percentage",
        "Inconclusive": "Inconclusive - model or data not adequate for interpretation",
    }.get(result.get("evidence_tier"), "Assessment status unavailable")


def business_decision(result):
    """Translate the prototype evidence tier into an honest user action."""
    return {
        "Supported (prototype screening)": (
            "Use the estimate as engineering screening evidence for the stated operating scope. "
            "Commercial or contractual verification still requires an agreed measurement protocol."
        ),
        "Indicative": (
            "Use the estimate as an indicative engineering result for the stated operating scope. "
            "Continue monitoring before using it for an investment, guarantee or contractual claim."
        ),
        "Preliminary": (
            "Do not use this estimate for a commercial decision yet. Collect more comparable post-DD "
            "reports or complete the missing operational evidence."
        ),
        "Unstable": (
            "Do not claim saving or deterioration. The estimated direction changed during the "
            "automatic stability checks; review the flagged data and operating conditions."
        ),
        "Inconclusive": (
            "Do not make a saving claim. The available data does not support a usable package-level estimate."
        ),
    }.get(result.get("evidence_tier"), "No decision guidance is available.")


def plain_language_conclusion(result):
    """Translate the numeric estimate and evidence tier for a non-specialist."""
    effect = result.get("improvement_pct")
    tier = str(result.get("evidence_tier") or "Inconclusive")
    if effect is None or not np.isfinite(effect):
        effect_text = "The app could not calculate a dependable package-level fuel difference."
    elif tier == "Inconclusive":
        effect_text = (
            f"The app calculated a numerical difference of {effect:.2f}%, but the pre-DD model or "
            "available data did not pass the minimum checks. Do not interpret this percentage as a "
            "supported saving or deterioration result."
        )
    elif tier == "Unstable":
        effect_text = (
            f"The primary calculation produced {effect:.2f}%, but reasonable alternative analyses "
            "changed the direction of the result. Do not interpret the percentage as a saving or "
            "deterioration conclusion."
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
    reasons = []
    if result.get("comparison_basis") == "Expanded cross-route":
        reasons.append("the same-route comparison was insufficient, so different routes were included")
    if result.get("flagged_foc_rows", 0):
        reasons.append(f"{result['flagged_foc_rows']} gross FOC report(s) require engineering review")
    if result.get("stability_status") == "Unstable":
        reasons.append("the estimated direction changed during sensitivity checks")
    if result.get("coverage_pct", 0) < 70:
        reasons.append("less than 70% of eligible post-dry-dock fuel was represented")
    reason_text = (
        " The conclusion is limited because " + "; ".join(reasons) + "."
        if reasons else ""
    )
    return tier, effect_text + reason_text


def show_conclusion_guide():
    st.markdown("**Use the conclusion status to decide whether the percentage can be interpreted.**")
    st.dataframe(CONCLUSION_GUIDE, hide_index=True, width="stretch")


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
            "This fallback cannot exceed Preliminary."
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
                    if result.get("coverage_pct", 0) >= 70
                    else "Calculation floor only"
                    if result.get("coverage_pct", 0) >= 40
                    else "Fail - below calculation floor"
                ),
                "What it means": _fmt(result.get("coverage_pct"), 1, "%") + " of eligible post-DD fuel; 40% minimum and 70% target",
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
                "Evidence check": "Fake-date stability screen",
                "Outcome": "Limited pass" if placebo.get("separated_from_placebos") is True else "Review",
                "What it means": f"{placebo.get('valid_count', 0)} valid fake-date tests",
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
        f"V9.11 | Dock-in: {settings['dock_in']} | Dock-out: {settings['dock_out']} | "
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
    if tier in {"Supported (prototype screening)", "Indicative"}:
        st.success(conclusion_message)
    elif tier in {"Preliminary", "Unstable"}:
        st.warning(conclusion_message)
    else:
        st.error(conclusion_message)
    st.info("**Recommended action:** " + business_decision(result))
    with st.expander("How to understand the five possible conclusions"):
        show_conclusion_guide()

    m1, m2, m3 = st.columns(3)
    m1.metric("ML-estimated package FOC saving", _fmt(result["improvement_pct"], 2, "%"))
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
        "Sensitivity direction",
        result.get("stability_status", "Not assessable"),
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
    st.download_button("Download printable assessment summary (HTML)", data=printable_report(saved),
                       file_name="mipd_foc_summary_v9_11.html", mime="text/html")
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
    st.metric("Sensitivity direction", result.get("stability_status", "Not assessable"))
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
    if not count:
        st.info(placebo.get("reason") or "No valid fake dry-dock tests were available.")
    elif placebo.get("separated_from_placebos") is True:
        st.info(f"The actual estimate exceeded all {count} valid fake-date effects. This is limited stability evidence, not proof of causation.")
    else:
        st.warning(f"Across {count} valid fake-date tests, at least one effect was as large as the actual estimate.")
    st.caption("The app applies fake intervention dates within the pre-DD period. Large apparent savings at those dates would indicate that the method can detect changes even without the real dry dock. Few valid tests limit the conclusion.")
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
                       file_name="mipd_report_audit_v9_11.csv", mime="text/csv")
    with st.expander("Expected and reported FOC for assessed post-DD reports"):
        st.caption("Fuel is VLSFO-equivalent. Unsupported rows are marked No and do not contribute to the estimate.")
        st.dataframe(report_comparison(result.get("assessed_rows", pd.DataFrame())).round(2), hide_index=True, width="stretch")


def show_method(saved):
    result, settings = saved["result"], saved["settings"]
    st.subheader("Calculation method, assumptions and downloads")
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
    st.caption("Huber settings: epsilon 1.35 and alpha 0.0001. A result is Inconclusive when MAPE exceeds 12%, absolute bias exceeds 8%, validation is unavailable, or the learned speed exponent falls outside 1.5-4.5. Indicative requires at least 20 comparable post-DD reports and 70% fuel coverage; 30 reports are preferred for Supported prototype screening. The fixed 13-25 kn filter is not used. These are declared project rules, not IMO, ISO or class requirements.")
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
        file_name="mipd_foc_assessment_v9_11.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    st.download_button("Download saved assessment inputs (JSON)", data=json.dumps(settings, default=str, indent=2),
        file_name="mipd_foc_inputs_v9_11.json", mime="application/json")
    st.download_button("Download full calculation methodology", data=Path(__file__).with_name("METHODOLOGY.md").read_text(encoding="utf-8"),
        file_name="MIPD_FOC_Methodology.md", mime="text/markdown")


if st.session_state.get("pending_page"):
    st.session_state["page"] = st.session_state.pop("pending_page")
if "page" not in st.session_state:
    st.session_state["page"] = "Result summary" if "mipd_v9_11" in st.session_state else "Setup"
st.sidebar.title("Package-level FOC assessment")
st.sidebar.caption("V9.11 | Last completed assessment is preserved")
page = st.sidebar.radio("View", ["Result summary", "Supporting analysis", "Setup"], key="page")
if page == "Setup":
    render_setup()
else:
    saved = st.session_state.get("mipd_v9_11")
    if saved is None:
        st.info("Open Setup to upload a workbook and run the assessment.")
        st.stop()
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
