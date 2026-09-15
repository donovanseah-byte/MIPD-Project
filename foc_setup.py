"""V9.11 guided setup and assessment pipeline."""
from __future__ import annotations
import re
import hashlib
import json
import numpy as np
import pandas as pd
import streamlit as st
from foc_demo import build_demo_assessment
from foc_model import run_package_assessment
from foc_processing import (
    DEFAULT_FUEL_LCV, REFERENCE_LCV_MJ_KG, apply_analysis_route_mapping,
    apply_service_leg_mapping, assign_periods, available_sheets,
    build_calculation_table, filter_operating_rows, largest_date_gap,
    read_two_row_excel,
)


class SetupWidgets:
    """Keep draft values when Streamlit cleans up off-screen widget keys."""
    def __getattr__(self, name):
        def render(*args, **kwargs):
            label = str(kwargs.get("key") or (args[0] if name != "data_editor" else name))
            key = "setup_" + re.sub(r"[^a-zA-Z0-9_]+", "_", label)
            kwargs["key"] = key
            values = st.session_state.setdefault("setup_values", {})
            if name == "data_editor":
                default = args[0]
                bases = st.session_state.setdefault("editor_bases", {})
                identifier = default.columns[0]
                signature = (tuple(default.columns), tuple(default[identifier].astype(str)))
                if key not in st.session_state or key not in bases or bases[key][0] != signature:
                    base = default.copy(deep=True)
                    previous = values.get(key)
                    if isinstance(previous, pd.DataFrame) and identifier in previous:
                        previous = previous.drop_duplicates(identifier).set_index(identifier)
                        for index, row in base.iterrows():
                            if row[identifier] in previous.index:
                                for column in base.columns[1:]:
                                    if column in previous:
                                        base.at[index, column] = previous.at[row[identifier], column]
                    bases[key] = (signature, base)
                # Keep the same input base for the editor's delta across reruns.
                args = (bases[key][1].copy(deep=True),) + args[1:]
            elif key not in st.session_state and key in values:
                value = values[key]
                if name in ("selectbox", "radio"):
                    options = args[1] if len(args) > 1 else kwargs["options"]
                    if value in options:
                        kwargs["index"] = list(options).index(value)
                elif name == "number_input":
                    arguments = list(args)
                    low = args[1] if len(args) > 1 else kwargs.get("min_value")
                    high = args[2] if len(args) > 2 else kwargs.get("max_value")
                    if low is not None:
                        value = max(low, value)
                    if high is not None:
                        value = min(high, value)
                    if len(arguments) > 3:
                        arguments[3] = value
                        args = tuple(arguments)
                    else:
                        kwargs["value"] = value
                else:
                    if name == "date_input":
                        value = max(kwargs.get("min_value", value), min(kwargs.get("max_value", value), value))
                    kwargs["value"] = value
            value = getattr(st, name)(*args, **kwargs)
            values[key] = value.copy(deep=True) if isinstance(value, pd.DataFrame) else value
            return value
        return render


def render_setup():
    st.title("Set up a vessel assessment")
    st.caption("Use the synthetic demonstration to learn the workflow, or upload a vessel workbook to perform an assessment.")
    with st.expander("New to vessel performance? Start here", expanded=False):
        st.markdown(
            "1. **Check the uploaded data** and how each noon report is calculated.\n"
            "2. **Confirm operational information** using official dry-dock, route and voyage records.\n"
            "3. **Review and run** the assessment, then interpret the percentage together with its conclusion status."
        )
        st.info(
            "The app estimates a package-level fuel difference. It does not prove that dry docking caused "
            "the difference or separate the effects of individual retrofit work items."
        )
    with st.expander("Plain-language maritime glossary", expanded=False):
        st.table(
            pd.DataFrame(
                {
                    "Term": [
                        "Before dry docking", "After dry docking", "STW", "Displacement",
                        "Propelling hours", "Beaufort scale", "Operating leg", "Comparable report",
                    ],
                    "Meaning": [
                        "Reports recorded before the vessel entered dry dock.",
                        "Reports recorded after the vessel returned to service.",
                        "Speed through the water, calculated from LOG distance divided by propelling hours.",
                        "Approximate total weight of the vessel, cargo, fuel and everything onboard.",
                        "Hours in the report interval when the main engine was propelling the vessel.",
                        "A standard description of wind strength; the main assessment uses scale 4 or below.",
                        "One operational direction or part of a service, such as outbound or return.",
                        "A post-dry-dock report whose speed and loading condition is represented by pre-dry-dock data.",
                    ],
                }
            ).set_index("Term")
        )
    if "mipd_v9_11" in st.session_state:
        st.info("Changes on this page are drafts. Result summary continues to show the last completed run until you run the assessment again.")
    ui = SetupWidgets()
    uploaded = st.file_uploader("Upload a vessel noon-report workbook", type=["xlsx", "xlsm"])
    demo = st.button("Open synthetic demonstration", help="Runs a generated example with a known approximately 10% post-DD reduction. No vessel data is used.")
    st.caption("The synthetic demonstration explains the POC only. It must not be presented as vessel evidence.")
    if demo:
        with st.spinner("Preparing the synthetic demonstration..."):
            saved = build_demo_assessment()
        st.session_state["mipd_v9_11"] = saved
        st.session_state["draft_signature"] = saved["setup_signature"]
        st.session_state["pending_page"] = "Result summary"
        st.rerun()
    if uploaded is None and "workbook_payload" not in st.session_state:
        st.markdown(
            "For a vessel assessment, upload the raw workbook containing the two-row IBIS header. "
            "The app calculates STW and LCV-normalised M/E FOC directly from the interval fields."
        )
        st.stop()

    payload = uploaded.getvalue() if uploaded is not None else st.session_state["workbook_payload"]
    if payload != st.session_state.get("workbook_payload"):
        st.session_state["setup_values"] = {}
        st.session_state["editor_bases"] = {}
        for key in list(st.session_state):
            if key.startswith("setup_") and key != "setup_values":
                del st.session_state[key]
        st.session_state.pop("mipd_v9_11", None)
    st.session_state["workbook_payload"] = payload
    st.session_state["workbook_name"] = getattr(uploaded, "name", st.session_state.get("workbook_name", "Uploaded workbook"))
    st.caption("Current workbook: " + st.session_state["workbook_name"])
    try:
        sheets = available_sheets(payload)
    except Exception as exc:
        st.error(f"The workbook could not be opened: {exc}")
        st.stop()

    default_sheet = sheets.index("Original") if "Original" in sheets else 0
    sheet = ui.selectbox("Raw-data sheet", sheets, index=default_sheet)
    try:
        raw = read_two_row_excel(payload, sheet_name=sheet)
    except Exception as exc:
        st.error(f"The selected sheet could not be read as a two-row-header IBIS sheet: {exc}")
        st.stop()

    st.subheader("1. Check workbook data and calculation inputs")
    with st.expander("Fuel and speed calculation definitions", expanded=False):
        st.markdown(
            "- **Fuel inputs:** M/E interval fuel masses from last report to noon, in metric tonnes by grade.\n"
            "- **Propelling hours:** hours propelling from last report to noon.\n"
            "- **LOG speed:** STW = log distance from last report to noon / propelling hours.\n"
            "- **FOC:** LCV-normalised M/E fuel rate in MT/day at a 40.5 MJ/kg reference.\n"
            "- **ML candidates:** the app compares a log-STW model with a log-STW-and-displacement model using pre-DD data only.\n"
            "- **Loading comparability:** displacement remains in the operating-support check even when the speed-only model is selected.\n"
            "- **Speed support:** no fixed 13-25 knot operating filter is used. Comparable speed/loading conditions are learned from valid pre-DD reports.\n"
            "- **Weather rule:** Beaufort <=4 is an eligibility filter; noon-report weather magnitudes are not ML predictors.\n"
            "- **Invalid values:** excluded as unavailable; missing fuel is never changed to zero."
        )
    lcv_default = pd.DataFrame(
        {
            "Fuel grade": list(DEFAULT_FUEL_LCV),
            "LCV (MJ/kg)": list(DEFAULT_FUEL_LCV.values()),
            "Source / reference": ["Document exact online, BDN or laboratory source" for _ in DEFAULT_FUEL_LCV],
        }
    )
    with st.expander("Fuel-conversion settings - advanced", expanded=False):
        st.caption(
            "The approved default LCV values are applied automatically. Edit them only when a BDN, "
            "laboratory result or controlled reference requires a different value."
        )
        lcv_table = ui.data_editor(
            lcv_default,
            hide_index=True,
            disabled=["Fuel grade"],
            num_rows="fixed",
            width="stretch",
            key="lcv_v9_11",
            column_config={"LCV (MJ/kg)": st.column_config.NumberColumn(min_value=1.0, format="%.2f")},
        )
    lcv_map = dict(
        zip(lcv_table["Fuel grade"], pd.to_numeric(lcv_table["LCV (MJ/kg)"], errors="coerce"))
    )
    try:
        preview = build_calculation_table(raw, lcv_map, REFERENCE_LCV_MJ_KG, "Unknown")
    except Exception as exc:
        st.error(f"Raw-data validation failed: {exc}")
        st.stop()

    voyage_text = preview.get("Voyage", pd.Series(dtype=str)).fillna("").astype(str).str.strip().str.upper()
    has_adhoc_c = voyage_text.str.endswith("C").any()
    c_mapping = "Unknown"
    if has_adhoc_c:
        with st.expander("Ad-hoc C-voyage mapping - action required when known", expanded=True):
            c_mapping = ui.selectbox(
                "Underlying rotation for voyage suffix C",
                ["Unknown", "E", "W", "N", "S"],
                help=(
                    "C is an ad-hoc call code, not a direction. Leave it as Unknown unless operational "
                    "records confirm the underlying rotation."
                ),
            )
        if c_mapping != "Unknown":
            try:
                preview = build_calculation_table(
                    raw, lcv_map, REFERENCE_LCV_MJ_KG, c_mapping
                )
            except Exception as exc:
                st.error(f"C-voyage mapping could not be applied: {exc}")
                st.stop()

    vessels = sorted(
        value
        for value in preview["Vessel"].dropna().astype(str).str.strip().unique().tolist()
        if value and value.casefold() != "unknown"
    )
    detected_vessel = vessels[0] if len(vessels) == 1 else ""
    detected_dates = pd.to_datetime(
        preview.get("Date", pd.Series(dtype="datetime64[ns]")), errors="coerce"
    ).dropna()
    detected_lanes = sorted(
        value for value in preview.get("ServiceLane", pd.Series(dtype=str)).fillna("").astype(str).str.strip().unique()
        if value
    )
    detected_rotations = sorted(
        value for value in preview.get("Rotation", pd.Series(dtype=str)).fillna("").astype(str).str.strip().unique()
        if value
    )
    st.markdown("**Detected workbook summary**")
    st.table(
        pd.DataFrame(
            {
                "Detected item": ["Vessel", "Parsed reports", "Report date range", "Service lanes", "Voyage rotation codes"],
                "Detected value": [
                    detected_vessel or "Unavailable or multiple names detected",
                    f"{len(preview):,}",
                    (
                        f"{detected_dates.min():%d %b %Y} to {detected_dates.max():%d %b %Y}"
                        if not detected_dates.empty else "Unavailable"
                    ),
                    ", ".join(detected_lanes) if detected_lanes else "Unavailable",
                    ", ".join(detected_rotations) if detected_rotations else "Unavailable",
                ],
            }
        ).set_index("Detected item")
    )
    if len(vessels) > 1:
        st.error("Multiple vessel names were detected: " + ", ".join(vessels))
    elif not vessels:
        st.error("No usable vessel name was detected in the workbook.")
    else:
        st.success(f"Vessel identity validated automatically: {vessels[0]}")
    vessel_match = len(vessels) == 1

    st.subheader("2. Confirm dry-dock dates and operating labels")
    st.caption(
        "The app prefers the same route and operating leg. If that evidence is insufficient, it can use "
        "an expanded cross-route comparison with the same operating leg and vessel-specific speed/loading support."
    )
    service_values = sorted(
        preview["ServiceLane"].fillna("Unknown").astype(str).str.strip().replace("", "Unknown").unique().tolist()
    )
    if all(value.casefold() == "unknown" for value in service_values):
        st.error("No usable Service Lane was detected; route comparability cannot be confirmed.")
    route_default = pd.DataFrame({"Service lane in workbook": service_values, "Assessment route": service_values})
    with st.expander("Assessment-route mapping - review only if lanes should be combined", expanded=False):
        st.caption(
            "Each detected service lane is kept as its own assessment route by default. Edit the second "
            "column only when operational records show that lanes should be assessed together."
        )
        route_table = ui.data_editor(
            route_default,
            hide_index=True,
            num_rows="fixed",
            width="stretch",
            disabled=["Service lane in workbook"],
            key="route_v9_11",
        )
    route_mapping = dict(
        zip(
            route_table["Service lane in workbook"].fillna("").astype(str).str.strip(),
            route_table["Assessment route"].fillna("").astype(str).str.strip(),
        )
    )
    blank_route = route_table["Assessment route"].fillna("").astype(str).str.strip().eq("").any()
    combined = (
        route_table.assign(
            **{
                "Service lane in workbook": route_table["Service lane in workbook"].astype(str).str.strip(),
                "Assessment route": route_table["Assessment route"].astype(str).str.strip(),
            }
        )
        .groupby("Assessment route")["Service lane in workbook"]
        .nunique()
    )
    combined_names = combined[combined > 1].index.tolist()
    route_justification = ""
    if combined_names:
        st.warning("Different workbook service lanes are combined into the assessment route: " + ", ".join(combined_names))
        route_justification = ui.text_area(
            "Operational reason for combining these service lanes",
            placeholder="State the physical corridor, major ports/passages and operating-role evidence checked.",
        )
    route_ready = (
        not blank_route
        and any(value.casefold() != "unknown" for value in service_values)
        and (
        not combined_names or bool(route_justification.strip())
        )
    )
    route_confirmed = route_ready

    st.markdown("**Confirm the operating leg represented by each voyage code**")
    st.caption(
        "The detected E/W/N/S/C suffix is only a voyage-rotation code. Map it to an operational label such as "
        "Outbound, Inbound, Singapore-Japan or Japan-Singapore. The app does not require every compass direction."
    )
    rotation_values = sorted(
        preview["Rotation"].fillna("Unknown").astype(str).str.strip().replace("", "Unknown").unique().tolist()
    )
    service_leg_default = pd.DataFrame(
        {"Voyage rotation code": rotation_values, "Confirmed operating leg": rotation_values}
    )
    with st.expander("Operating-leg mapping - review detected voyage codes", expanded=False):
        service_leg_table = ui.data_editor(
            service_leg_default,
            hide_index=True,
            num_rows="fixed",
            width="stretch",
            disabled=["Voyage rotation code"],
            key="service_leg_v9_11",
        )
    service_leg_mapping = dict(
        zip(
            service_leg_table["Voyage rotation code"].fillna("").astype(str).str.strip(),
            service_leg_table["Confirmed operating leg"].fillna("").astype(str).str.strip(),
        )
    )
    blank_service_leg = (
        service_leg_table["Confirmed operating leg"].fillna("").astype(str).str.strip().eq("").any()
    )
    service_leg_ready = not blank_service_leg

    date_values = pd.to_datetime(preview["Date"], errors="coerce").dropna()
    if date_values.empty:
        st.error("No valid noon-report dates were found.")
        st.stop()
    gap_before, gap_after = largest_date_gap(preview)
    minimum_date, maximum_date = date_values.min().date(), date_values.max().date()
    suggested_in = (gap_before + pd.Timedelta(days=1)).date() if gap_before is not None else minimum_date
    suggested_out = (gap_after - pd.Timedelta(days=1)).date() if gap_after is not None else suggested_in
    suggested_in = min(max(suggested_in, minimum_date), maximum_date)
    suggested_out = min(max(suggested_out, suggested_in), maximum_date)
    st.warning(
        "The dates below are suggested from the largest reporting gap. They are not official "
        "dry-dock dates. Replace them using the official dry-dock record before confirming."
    )

    d1, d2, d3, d4 = st.columns(4)
    with d1:
        dock_in = ui.date_input(
            "Official dock-in date", value=suggested_in, min_value=minimum_date, max_value=maximum_date
        )
    with d2:
        dock_out = ui.date_input(
            "Official dock-out date", value=suggested_out, min_value=minimum_date, max_value=maximum_date
        )
    with d3:
        st.metric("Preferred pre-DD data history", "12 months")
    with d4:
        st.metric("Primary post-DD assessment period", "3 months")
    if pd.Timestamp(dock_out) < pd.Timestamp(dock_in):
        st.error("Dock-out cannot be earlier than dock-in.")

    required_pre_start = pd.Timestamp(dock_in) - pd.DateOffset(months=12)
    three_month_post_end = pd.Timestamp(dock_out) + pd.DateOffset(months=3)
    six_month_post_end = pd.Timestamp(dock_out) + pd.DateOffset(months=6)
    monitoring_basis = ui.radio(
        "Post-DD assessment period",
        [
            "First 3 months (primary)",
            "Extend to the first complete service cycle (maximum 6 months)",
        ],
        help=(
            "Use the first three months as the primary assessment. Extend only when that period does not contain "
            "one complete normal service cycle, and record the cycle-completion date before viewing the result."
        ),
    )
    required_post_end = three_month_post_end
    if monitoring_basis.startswith("Extend"):
        latest_allowed_end = min(pd.Timestamp(maximum_date), six_month_post_end)
        if latest_allowed_end > three_month_post_end:
            required_post_end = pd.Timestamp(
                ui.date_input(
                    "Complete-cycle assessment endpoint",
                    value=latest_allowed_end.date(),
                    min_value=three_month_post_end.date(),
                    max_value=latest_allowed_end.date(),
                    help="Select the date on which the first complete normal post-DD service cycle was available.",
                )
            )
        else:
            st.warning("The workbook does not contain data beyond the primary three-month endpoint.")

    service_cycle_status = ui.selectbox(
        "Post-DD service-cycle evidence",
        [
            "Not confirmed",
            "Confirmed from voyage or port-sequence records",
        ],
        help=(
            "A complete cycle means the vessel completed its normal operating rotation. If it is not "
            "confirmed, the app can still calculate a result but will limit its interpretation."
        ),
    )
    service_cycle_confirmed = service_cycle_status.startswith("Confirmed")
    normal_dates = date_values.dt.normalize()
    pre_dates = normal_dates[normal_dates < pd.Timestamp(dock_in)]
    post_dates = normal_dates[normal_dates > pd.Timestamp(dock_out)]
    actual_pre_start = pre_dates.min() if not pre_dates.empty else pd.NaT
    actual_post_end = post_dates.max() if not post_dates.empty else pd.NaT
    pre_period_ok = bool(pd.notna(actual_pre_start) and actual_pre_start <= required_pre_start)
    post_period_ok = bool(pd.notna(actual_post_end) and actual_post_end >= required_post_end)
    period_coverage_ok = pre_period_ok and post_period_ok
    if period_coverage_ok:
        st.success(
            f"Calendar coverage passed: the workbook reaches the declared post-DD endpoint "
            f"of {required_post_end:%d %b %Y}."
        )
    else:
        issues = []
        if not pre_period_ok:
            issues.append(
                f"Pre-DD records should begin by {required_pre_start:%d %b %Y}; "
                f"available start: {actual_pre_start:%d %b %Y}" if pd.notna(actual_pre_start)
                else "No valid pre-DD records were found."
            )
        if not post_period_ok:
            issues.append(
                f"Post-DD records should continue through {required_post_end:%d %b %Y}; "
                f"available end: {actual_post_end:%d %b %Y}" if pd.notna(actual_post_end)
                else "No valid post-DD records were found."
            )
        st.warning("Insufficient calendar coverage:\n\n- " + "\n- ".join(issues))

    min_hours = 18.0
    min_plausible_stw = 5.0
    max_plausible_stw = 35.0
    max_beaufort = 4.0
    minimum_train_rows = 60
    minimum_supported_after = 10
    placebo_count = 5
    st.subheader("3. Review required inputs and run the assessment")
    with st.expander("Fixed assessment rules", expanded=False):
        st.markdown(
            "- At least 18 propelling hours\n"
            "- Beaufort 4 or below\n"
            "- Physically plausible steady-sea STW of 5-35 kn; this is only an error screen\n"
            "- Vessel-specific speed/loading support learned from valid pre-DD reports\n"
            "- Same route and operating leg preferred; controlled cross-route fallback when strict evidence is insufficient\n"
            "- At least 60 pre-DD reports to train and 10 comparable post-DD reports to calculate a preliminary result"
            "\n- Pre-DD validation must achieve MAPE <=12%, absolute bias <=8% and a learned speed exponent of 1.5-4.5; otherwise the result is Inconclusive"
        )
    interval_alignment_confirmed = ui.checkbox(
        "I confirm that M/E fuel, LOG distance and propelling hours cover the same reporting interval",
        help=(
            "This is the only mandatory human declaration because the workbook cannot prove that the "
            "three source fields refer to the same interval. Misalignment would produce an incorrect daily FOC."
        ),
    )

    with st.expander("Report documentation - optional", expanded=False):
        st.caption(
            "These references do not change the numerical calculation. Complete them when the result "
            "will be issued as a documented engineering assessment."
        )
        dry_dock_reference = ui.text_input(
            "Official dry-dock record reference",
            placeholder="For example: docking report number, work order or confirmed vessel record",
        )
        route_evidence = ui.text_input(
            "Assessment-route evidence reference",
            placeholder="For example: voyage schedule, service profile or port-rotation record",
        )
        service_leg_evidence = ui.text_input(
            "Operating-leg evidence reference",
            placeholder="For example: port sequence, voyage instruction or service rotation",
        )
        if service_cycle_confirmed:
            service_cycle_evidence = ui.text_input(
                "Complete service-cycle evidence reference",
                placeholder="For example: voyage numbers and port sequence covered by the selected period",
            )
        else:
            service_cycle_evidence = ""
        analyst_name = ui.text_input(
            "Assessment prepared by",
            placeholder="Name or initials",
        )

    st.table(
        pd.DataFrame(
            {
                "Review item": [
                    "Vessel identity",
                    "Official dry-dock dates",
                    "Fuel, LOG distance and propelling interval",
                    "Assessment-route mapping",
                    "Operating-leg mapping",
                    "Complete post-DD service cycle",
                    "Required calendar coverage available",
                ],
                "Status": [
                    "Validated automatically" if vessel_match else "Required",
                    f"Entered: {dock_in:%d %b %Y} to {dock_out:%d %b %Y}",
                    "Confirmed" if interval_alignment_confirmed else "Required",
                    "Ready" if route_ready else "Required",
                    "Ready" if service_leg_ready else "Required",
                    "Confirmed" if service_cycle_confirmed else "Not confirmed - result may be limited",
                    "Available" if period_coverage_ok else "Incomplete - result may be limited",
                ],
            }
        ).set_index("Review item")
    )
    draft_values = {key: value.to_json(date_format="iso") if isinstance(value, pd.DataFrame) else value
                    for key, value in st.session_state.get("setup_values", {}).items()}
    st.session_state["draft_signature"] = hashlib.sha256(
        json.dumps(draft_values, default=str, sort_keys=True).encode("utf-8")
    ).hexdigest()
    run = st.button(
        "Run vessel assessment",
        type="primary",
        width="stretch",
        disabled=not (
            interval_alignment_confirmed
            and route_ready
            and service_leg_ready
            and vessel_match
            and dock_out >= dock_in
        ),
    )

    if run:
        with st.status("Running the ML package assessment...", expanded=True) as process:
            try:
                live_bar = st.progress(0, text="Starting workbook calculations")

                def show_live_progress(percent: int, message: str) -> None:
                    live_bar.progress(percent, text=message)
                    process.write(message)

                process.write("Stage 1/6 - Calculating STW and LCV-normalised M/E FOC from the uploaded intervals.")
                calculated = build_calculation_table(raw, lcv_map, REFERENCE_LCV_MJ_KG, c_mapping)
                calculated["ReportID"] = np.arange(1, len(calculated) + 1)
                calculated = apply_analysis_route_mapping(calculated, route_mapping)
                calculated = apply_service_leg_mapping(calculated, service_leg_mapping)
                process.write(f"Calculated {len(calculated):,} raw noon-report rows.")
                live_bar.progress(8, text="STW and FOC calculations completed")

                process.write("Stage 2/6 - Assigning pre-DD/post-DD periods and applying common operating filters.")
                period_data = assign_periods(
                    calculated,
                    dock_in,
                    dock_out,
                    required_pre_start,
                    required_post_end,
                )
                eligible, filter_report = filter_operating_rows(
                    period_data,
                    min_propelling_hours=min_hours,
                    min_plausible_stw=min_plausible_stw,
                    max_plausible_stw=max_plausible_stw,
                    max_beaufort=max_beaufort,
                )
                process.write(
                    f"Retained {filter_report['before_rows']:,} eligible pre-DD and "
                    f"{filter_report['after_rows']:,} eligible post-DD rows."
                )
                live_bar.progress(15, text="Period assignment and common filtering completed")

                result = run_package_assessment(
                    eligible,
                    period_coverage_ok=period_coverage_ok,
                    route_confirmed=route_confirmed,
                    service_cycle_confirmed=service_cycle_confirmed,
                    minimum_train_rows=int(minimum_train_rows),
                    minimum_supported_after=int(minimum_supported_after),
                    placebo_count=int(placebo_count),
                    progress_callback=show_live_progress,
                )

                source_text = lcv_table["Source / reference"].fillna("").astype(str).str.strip()
                lcv_sources_documented = bool(
                    source_text.ne("").all()
                    and ~source_text.str.contains("Document exact", case=False, regex=False).any()
                )
                documentation_complete = bool(
                    dry_dock_reference.strip()
                    and route_evidence.strip()
                    and service_leg_evidence.strip()
                    and analyst_name.strip()
                    and lcv_sources_documented
                    and (not service_cycle_confirmed or service_cycle_evidence.strip())
                )
                st.session_state["mipd_v9_11"] = {
                    "result": result,
                    "setup_signature": st.session_state["draft_signature"],
                    "workbook_name": st.session_state["workbook_name"],
                    "eligible": eligible,
                    "period_data": period_data,
                    "filter_report": filter_report,
                    "settings": {
                        "vessel": vessels[0],
                        "sheet": sheet,
                        "dock_in": dock_in,
                        "dock_out": dock_out,
                        "required_pre_start": required_pre_start.date(),
                        "required_post_end": required_post_end.date(),
                        "monitoring_basis": monitoring_basis,
                        "service_cycle_confirmed": service_cycle_confirmed,
                        "period_coverage_ok": period_coverage_ok,
                        "interval_alignment_confirmed": interval_alignment_confirmed,
                        "min_hours": min_hours,
                        "min_plausible_stw": min_plausible_stw,
                        "max_plausible_stw": max_plausible_stw,
                        "max_beaufort": max_beaufort,
                        "minimum_train_rows": int(minimum_train_rows),
                        "minimum_supported_after": int(minimum_supported_after),
                        "placebo_count": int(placebo_count),
                        "route_mapping": route_mapping,
                        "route_justification": route_justification.strip(),
                        "c_mapping": c_mapping,
                        "service_leg_mapping": service_leg_mapping,
                        "dry_dock_record_reference": dry_dock_reference.strip(),
                        "route_evidence_reference": route_evidence.strip(),
                        "service_leg_evidence_reference": service_leg_evidence.strip(),
                        "service_cycle_evidence_reference": service_cycle_evidence.strip(),
                        "assessment_prepared_by": analyst_name.strip(),
                        "documentation_complete": documentation_complete,
                    },
                    "lcv_table": lcv_table,
                    "route_table": route_table.assign(Justification=route_justification.strip()),
                    "service_leg_table": service_leg_table,
                }
                process.update(label="ML package assessment completed", state="complete", expanded=False)
                st.session_state["pending_page"] = "Result summary"
            except Exception as exc:
                process.update(label="ML assessment failed", state="error", expanded=True)
                st.error(f"The assessment could not be completed: {exc}")


    if st.session_state.get("pending_page"):
        st.rerun()
