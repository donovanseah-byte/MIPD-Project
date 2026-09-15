"""Synthetic demonstration for explaining the POC without vessel data."""
from __future__ import annotations

import numpy as np
import pandas as pd

from foc_model import run_package_assessment
from foc_processing import DEFAULT_FUEL_LCV


def _synthetic_noon_reports() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    start = pd.Timestamp("2024-01-01")
    dock_in = pd.Timestamp("2024-12-31")
    dock_out = pd.Timestamp("2025-01-14")
    rows = []
    report_id = 1
    for day in range(470):
        date = start + pd.Timedelta(days=day)
        if dock_in <= date <= dock_out:
            continue
        stw = 13.2 + 0.12 * (day % 30)
        displacement = 84_000.0 + 1_500.0 * (day % 9)
        beaufort = float(day % 5)
        expected = np.exp(
            -4.25
            + 2.85 * np.log(stw)
            + 0.35 * np.log(displacement / 90_000.0)
        )
        foc = expected * np.exp(rng.normal(0.0, 0.012))
        period = "Before DD" if date < dock_in else "After DD"
        if period == "After DD":
            foc *= 0.90
        leg = "Outbound" if (day // 20) % 2 == 0 else "Inbound"
        rows.append(
            {
                "ReportID": report_id,
                "Date": date,
                "Vessel": "SYNTHETIC DEMO VESSEL",
                "Period": period,
                "STW": stw,
                "DisplacementMT": displacement,
                "Beaufort": beaufort,
                "FOC_MT_Day": foc,
                "PropellingHours": 24.0,
                "Voyage": f"D{day // 20:03d}",
                "ServiceLane": "DEMO",
                "AnalysisRoute": "Demonstration service",
                "Rotation": "OUT" if leg == "Outbound" else "IN",
                "ServiceLeg": leg,
            }
        )
        report_id += 1
    return pd.DataFrame(rows)


def build_demo_assessment() -> dict:
    """Return a fully calculated, clearly labelled synthetic app state."""
    eligible = _synthetic_noon_reports()
    result = run_package_assessment(
        eligible,
        period_coverage_ok=True,
        route_confirmed=True,
        service_cycle_confirmed=True,
        minimum_train_rows=60,
        minimum_supported_after=10,
        placebo_count=5,
    )
    if not result.get("valid"):
        raise RuntimeError(result.get("reason") or "Synthetic demonstration failed.")
    settings = {
        "vessel": "SYNTHETIC DEMO VESSEL",
        "sheet": "Generated demonstration",
        "dock_in": pd.Timestamp("2024-12-31").date(),
        "dock_out": pd.Timestamp("2025-01-14").date(),
        "required_pre_start": pd.Timestamp("2024-01-01").date(),
        "required_post_end": pd.Timestamp("2025-04-14").date(),
        "monitoring_basis": "First 3 months (synthetic demonstration)",
        "service_cycle_confirmed": True,
        "period_coverage_ok": True,
        "interval_alignment_confirmed": True,
        "min_hours": 18.0,
        "min_plausible_stw": 5.0,
        "max_plausible_stw": 35.0,
        "max_beaufort": 4.0,
        "minimum_train_rows": 60,
        "minimum_supported_after": 10,
        "placebo_count": 5,
        "route_mapping": {"DEMO": "Demonstration service"},
        "service_leg_mapping": {"OUT": "Outbound", "IN": "Inbound"},
        "dry_dock_record_reference": "Synthetic demonstration assumption",
        "route_evidence_reference": "Synthetic demonstration assumption",
        "service_leg_evidence_reference": "Synthetic demonstration assumption",
        "service_cycle_evidence_reference": "Synthetic demonstration assumption",
        "assessment_prepared_by": "Built-in demonstration",
        "documentation_complete": True,
        "is_demo": True,
    }
    lcv_table = pd.DataFrame(
        {
            "Fuel grade": list(DEFAULT_FUEL_LCV),
            "LCV (MJ/kg)": list(DEFAULT_FUEL_LCV.values()),
            "Source / reference": ["Synthetic demonstration assumption"] * len(DEFAULT_FUEL_LCV),
        }
    )
    return {
        "result": result,
        "eligible": eligible,
        "period_data": eligible.copy(deep=True),
        "filter_report": {
            "input_rows": len(eligible),
            "before_rows": int(eligible["Period"].eq("Before DD").sum()),
            "after_rows": int(eligible["Period"].eq("After DD").sum()),
        },
        "settings": settings,
        "lcv_table": lcv_table,
        "route_table": pd.DataFrame(
            {"Service lane in workbook": ["DEMO"], "Assessment route": ["Demonstration service"]}
        ),
        "service_leg_table": pd.DataFrame(
            {"Voyage rotation code": ["OUT", "IN"], "Confirmed operating leg": ["Outbound", "Inbound"]}
        ),
        "setup_signature": "synthetic-demo-v9.7",
    }
