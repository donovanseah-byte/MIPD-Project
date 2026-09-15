from __future__ import annotations
import io
import numpy as np
import pandas as pd
import plotly.express as px

def _fmt(value: object, digits: int = 1, suffix: str = "") -> str:
    try:
        number = float(value)
        return f"{number:,.{digits}f}{suffix}" if np.isfinite(number) else "Unavailable"
    except (TypeError, ValueError):
        return "Unavailable"


def _validation_table(validation: dict) -> pd.DataFrame:
    features = validation.get("numeric_features", [])
    huber_label = (
        "Huber ML (STW + displacement)"
        if "LogDisplacementRatio" in features
        else "Huber ML (STW)"
    )
    return pd.DataFrame(
        [
            {
                "Model": huber_label,
                "MAPE (%)": validation["huber"].get("mape_pct"),
                "Bias (%)": validation["huber"].get("bias_pct"),
                "RMSE (FOC MT/day)": validation["huber"].get("rmse"),
                "Test rounds": validation.get("folds", 0),
                "Comparable unseen reports": validation.get("supported_test_rows", 0),
            },
            {
                "Model": "Public cubic-speed benchmark (V^3)",
                "MAPE (%)": validation["cubic"].get("mape_pct"),
                "Bias (%)": validation["cubic"].get("bias_pct"),
                "RMSE (FOC MT/day)": validation["cubic"].get("rmse"),
                "Test rounds": validation.get("folds", 0),
                "Comparable unseen reports": validation.get("supported_test_rows", 0),
            },
        ]
    )


def _support_scatter(eligible: pd.DataFrame, assessed: pd.DataFrame):
    before = eligible[eligible["Period"] == "Before DD"].copy()
    if assessed.empty:
        display = before
    else:
        after = assessed.copy()
        mode = str(after.get("SupportMode", pd.Series("strict", index=after.index)).iloc[0])
        if mode == "expanded":
            represented_legs = set(after.get("ServiceLeg", pd.Series(dtype=str)).astype(str))
            before = before.loc[before.get("ServiceLeg", pd.Series(index=before.index)).astype(str).isin(represented_legs)]
        else:
            represented = set(
                zip(
                    after.get("AnalysisRoute", pd.Series(dtype=str)).astype(str),
                    after.get("ServiceLeg", pd.Series(dtype=str)).astype(str),
                )
            )
            before_keys = list(
                zip(
                    before.get("AnalysisRoute", pd.Series(index=before.index)).astype(str),
                    before.get("ServiceLeg", pd.Series(index=before.index)).astype(str),
                )
            )
            before = before.loc[[key in represented for key in before_keys]]
        before["Assessment group"] = "Relevant pre-DD training reports"
        after["Assessment group"] = np.where(
            after["InsideSupport"],
            "Comparable post-DD reports used",
            "Post-DD outside pre-DD support",
        )
        display = pd.concat([before, after], ignore_index=True)
    if "Assessment group" not in display:
        display["Assessment group"] = "Pre-DD training reports"
    figure = px.scatter(
        display,
        x="STW",
        y="DisplacementMT",
        color="Assessment group",
        hover_data={
            "Date": True,
            "Voyage": True,
            "AnalysisRoute": True,
            "ServiceLeg": True,
            "FOC_MT_Day": ":.2f",
            "STW": ":.2f",
            "DisplacementMT": ":,.0f",
        },
        color_discrete_map={
            "Pre-DD training reports": "#2563eb",
            "Relevant pre-DD training reports": "#2563eb",
            "Comparable post-DD reports used": "#16a34a",
            "Post-DD outside pre-DD support": "#f59e0b",
        },
        labels={
            "STW": "STW (kn)",
            "DisplacementMT": "Displacement (MT)",
            "ServiceLeg": "Operating leg",
            "AnalysisRoute": "Assessment route",
            "Assessment group": "Assessment use",
        },
    )
    figure.update_traces(marker={"size": 9, "opacity": 0.75, "line": {"width": 0.5, "color": "white"}})
    figure.update_layout(height=470, legend_title_text="", margin={"l": 10, "r": 10, "t": 20, "b": 10})
    return figure


def _actual_expected_scatter(assessed: pd.DataFrame):
    """Show whether supported post-DD reports fall above or below no-change."""
    frame = assessed.copy() if isinstance(assessed, pd.DataFrame) else pd.DataFrame()
    required = {
        "FOC_MT_Day",
        "ExpectedPreDDCondition_FOC_MT_Day",
        "InsideSupport",
    }
    if frame.empty or not required.issubset(frame.columns):
        return None
    frame["Assessment group"] = np.where(
        frame["InsideSupport"],
        "Comparable report used",
        "Outside pre-DD support",
    )
    figure = px.scatter(
        frame,
        x="ExpectedPreDDCondition_FOC_MT_Day",
        y="FOC_MT_Day",
        color="Assessment group",
        hover_data={
            "Date": True,
            "Voyage": True,
            "ServiceLeg": True,
            "STW": ":.2f",
            "DisplacementMT": ":,.0f",
            "ExpectedPreDDCondition_FOC_MT_Day": ":.2f",
            "FOC_MT_Day": ":.2f",
        },
        color_discrete_map={
            "Comparable report used": "#16856f",
            "Outside pre-DD support": "#d89214",
        },
        labels={
            "ExpectedPreDDCondition_FOC_MT_Day": "Model-expected FOC (VLSFO-eq. MT/day)",
            "FOC_MT_Day": "Reported post-DD FOC (VLSFO-eq. MT/day)",
            "ServiceLeg": "Operating leg",
            "Assessment group": "Assessment use",
        },
    )
    values = pd.concat(
        [
            pd.to_numeric(frame["ExpectedPreDDCondition_FOC_MT_Day"], errors="coerce"),
            pd.to_numeric(frame["FOC_MT_Day"], errors="coerce"),
        ],
        ignore_index=True,
    ).replace([np.inf, -np.inf], np.nan).dropna()
    if not values.empty:
        low, high = float(values.min()), float(values.max())
        padding = max((high - low) * 0.05, 0.5)
        figure.add_shape(
            type="line",
            x0=low - padding,
            y0=low - padding,
            x1=high + padding,
            y1=high + padding,
            line={"color": "#64748b", "dash": "dash", "width": 2},
        )
        figure.update_xaxes(range=[low - padding, high + padding])
        figure.update_yaxes(range=[low - padding, high + padding])
    figure.update_traces(marker={"size": 10, "opacity": 0.8, "line": {"width": 0.5, "color": "white"}})
    figure.update_layout(height=470, legend_title_text="", margin={"l": 10, "r": 10, "t": 20, "b": 10})
    return figure


def _overview_table(result: dict, settings: dict) -> pd.DataFrame:
    placebo = result.get("placebo", {})
    rows = [
        ("Data source", "Synthetic demonstration - not vessel evidence" if settings.get("is_demo") else "Uploaded vessel workbook"),
        ("Assessment scope", "Package-level main-engine FOC change associated with the dry-dock event"),
        ("Selected ML specification", result.get("selected_ml_model")),
        ("Selected comparison basis", result.get("comparison_basis")),
        ("ML-estimated package FOC saving (%)", result.get("improvement_pct")),
        ("Alternative ML specification saving (%)", result.get("alternative_ml_saving_pct")),
        ("Public cubic-speed benchmark saving (%)", result.get("cubic_benchmark_saving_pct")),
        ("Prototype evidence classification", result.get("evidence_tier")),
        ("Sensitivity direction", result.get("stability_status")),
        ("Stability sensitivity minimum (%)", result.get("stability_min_pct")),
        ("Stability sensitivity maximum (%)", result.get("stability_max_pct")),
        ("Gross FOC reports flagged", result.get("flagged_foc_rows")),
        ("Anomaly-excluded sensitivity (%)", result.get("anomaly_sensitivity_pct")),
        ("Pre-DD reports used to train ML", result.get("before_rows")),
        ("Post-DD reports passing filters", result.get("after_rows")),
        ("Comparable post-DD reports used", result.get("supported_after_rows")),
        ("Comparable post-DD fuel coverage (%)", result.get("coverage_pct")),
        ("Strict same-route comparable reports", result.get("strict_supported_after_rows")),
        ("Strict same-route fuel coverage (%)", result.get("strict_coverage_pct")),
        ("Expanded cross-route comparable reports", result.get("expanded_supported_after_rows")),
        ("Expanded cross-route fuel coverage (%)", result.get("expanded_coverage_pct")),
        ("Supported operating scope", result.get("scope_statement")),
        ("Post-DD assessment period", settings.get("monitoring_basis")),
        ("Complete service cycle confirmed", settings.get("service_cycle_confirmed")),
        ("Observed equivalent fuel saved (MT)", result.get("observed_equivalent_fuel_saved_mt")),
        ("Learned speed exponent", result.get("learned_speed_exponent")),
        ("Valid placebo dates", placebo.get("valid_count")),
        ("Actual-effect placebo percentile (%)", placebo.get("actual_percentile")),
        ("Dock-in", str(settings["dock_in"])),
        ("Dock-out", str(settings["dock_out"])),
        ("Dry-dock record reference", settings.get("dry_dock_record_reference")),
        ("Assessment-route evidence reference", settings.get("route_evidence_reference")),
        ("Operating-leg evidence reference", settings.get("service_leg_evidence_reference")),
        ("Complete-cycle evidence reference", settings.get("service_cycle_evidence_reference")),
        ("Assessment prepared by", settings.get("assessment_prepared_by")),
        ("Source documentation complete", settings.get("documentation_complete")),
    ]
    return pd.DataFrame(rows, columns=["Item", "Value"])


def _excel_export(
    result: dict,
    settings: dict,
    eligible: pd.DataFrame,
    lcv_table: pd.DataFrame,
    route_table: pd.DataFrame,
    service_leg_table: pd.DataFrame,
) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        _overview_table(result, settings).to_excel(writer, sheet_name="Overview", index=False)
        _validation_table(result["validation"]).to_excel(writer, sheet_name="Model Validation", index=False)
        ml_candidates = result.get("ml_candidate_comparison", pd.DataFrame())
        if isinstance(ml_candidates, pd.DataFrame) and not ml_candidates.empty:
            ml_candidates.to_excel(writer, sheet_name="ML Specification Selection", index=False)
        sensitivity = result.get("model_sensitivity", pd.DataFrame())
        if isinstance(sensitivity, pd.DataFrame) and not sensitivity.empty:
            sensitivity.to_excel(writer, sheet_name="Method Sensitivity", index=False)
        anomalies = result.get("foc_anomalies", pd.DataFrame())
        if isinstance(anomalies, pd.DataFrame) and not anomalies.empty:
            anomalies.to_excel(writer, sheet_name="Flagged FOC Review", index=False)
        eligible.to_excel(writer, sheet_name="Eligible Data", index=False)
        if not result["assessed_rows"].empty:
            result["assessed_rows"].to_excel(writer, sheet_name="After Assessment", index=False)
        if not result["validation"]["predictions"].empty:
            result["validation"]["predictions"].to_excel(
                writer, sheet_name="Validation Predictions", index=False
            )
        placebo_table = result.get("placebo", {}).get("results", pd.DataFrame())
        if not placebo_table.empty:
            placebo_table.to_excel(writer, sheet_name="Placebo Tests", index=False)
        if not result["coefficients"].empty:
            result["coefficients"].to_excel(writer, sheet_name="Huber Coefficients", index=False)
        if not result.get("service_leg_summary", pd.DataFrame()).empty:
            result["service_leg_summary"].to_excel(writer, sheet_name="Supported Scope", index=False)
        lcv_table.to_excel(writer, sheet_name="LCV Register", index=False)
        route_table.to_excel(writer, sheet_name="Assessment Route Mapping", index=False)
        service_leg_table.to_excel(writer, sheet_name="Operating Leg Mapping", index=False)
    return output.getvalue()
