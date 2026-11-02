"""Input parsing and auditable noon-report FOC preparation."""

from __future__ import annotations

import io
import re
from datetime import time as dt_time
from typing import BinaryIO, Mapping

import numpy as np
import pandas as pd


REFERENCE_LCV_MJ_KG = 40.5

DEFAULT_FUEL_LCV = {
    "VLSFO": 40.5,
    "ULSFO": 40.5,
    "HSFO": 40.2,
    "LSDO/LSGO": 42.7,
    "MDO/MGO": 42.7,
    "BIO-VLSFO": 39.5,
    "BIO-HSFO": 39.5,
    "BIO-LSDO/LSGO": 41.5,
}

RAW_COLUMNS = {
    "date": "Noon | Time",
    "latitude": "Noon | Position Lat.",
    "longitude": "Noon | Position Long.",
    "propelling_hours": "Data while Steaming | Hours Propelling from Last Report to Noon",
    "log_distance": "Data while Steaming | Dist. from Last Report to Noon (Log)",
    "displacement": "Next Port | Displacement",
    "beaufort": "Wind | Ave. Beaufort",
    "wind_speed": "Wind | Ave. Speed(Knot)",
    "wind_sea_height": "Wind | Ave. Sea Height(m)",
    "swell_height": "Swell | Ave. Sea Height(m)",
    "fore_draft": "Next Port | Fore",
    "aft_draft": "Next Port | Aft",
}

FUEL_COLUMNS = {
    fuel: f"Consumption while steaming from last report to Noon | M/E FOC {fuel}"
    for fuel in DEFAULT_FUEL_LCV
}


def _normalise_header(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return " ".join(str(value).replace("\n", " ").strip().split())


def _make_unique(names: list[str]) -> list[str]:
    counts: dict[str, int] = {}
    output: list[str] = []
    for name in names:
        base = name or "Unnamed"
        count = counts.get(base, 0)
        output.append(base if count == 0 else f"{base} [{count + 1}]")
        counts[base] = count + 1
    return output


def available_sheets(source: bytes | bytearray | BinaryIO) -> list[str]:
    payload = bytes(source) if isinstance(source, (bytes, bytearray)) else source.read()
    return pd.ExcelFile(io.BytesIO(payload), engine="openpyxl").sheet_names


def read_two_row_excel(
    source: bytes | bytearray | BinaryIO,
    sheet_name: str = "Original",
) -> pd.DataFrame:
    """Read the two-row IBIS group-header/field-header layout."""
    payload = bytes(source) if isinstance(source, (bytes, bytearray)) else source.read()
    header = pd.read_excel(
        io.BytesIO(payload), sheet_name=sheet_name, header=None, nrows=2, engine="openpyxl"
    )
    if len(header) < 2:
        raise ValueError("The selected sheet does not contain the expected two header rows.")
    top = header.iloc[0].map(_normalise_header).replace("", np.nan).ffill()
    bottom = header.iloc[1].map(_normalise_header)
    names: list[str] = []
    for group, field in zip(top, bottom):
        group_text = "" if pd.isna(group) else str(group)
        if field and group_text and field != group_text:
            names.append(f"{group_text} | {field}")
        else:
            names.append(field or group_text)
    data = pd.read_excel(
        io.BytesIO(payload), sheet_name=sheet_name, header=None, skiprows=2, engine="openpyxl"
    )
    data.columns = _make_unique(names[: len(data.columns)])
    return data.dropna(how="all").reset_index(drop=True)


def duration_to_hours(value: object) -> float:
    """Convert Excel durations, time values and HH:MM strings to decimal hours."""
    if value is None or pd.isna(value):
        return np.nan
    if isinstance(value, pd.Timedelta):
        return value.total_seconds() / 3600.0
    if isinstance(value, dt_time):
        return value.hour + value.minute / 60.0 + value.second / 3600.0
    if isinstance(value, (int, float, np.integer, np.floating)):
        number = float(value)
        return number * 24.0 if 0 <= number <= 1 else number
    text = str(value).strip()
    if not text or text.upper() in {"NA", "N/A", "NONE", "-"}:
        return np.nan
    parts = text.split(":")
    if len(parts) in (2, 3):
        try:
            hours = float(parts[0])
            minutes = float(parts[1])
            seconds = float(parts[2]) if len(parts) == 3 else 0.0
            return hours + minutes / 60.0 + seconds / 3600.0
        except ValueError:
            return np.nan
    try:
        return float(text)
    except ValueError:
        return np.nan


def coordinate_to_decimal(value: object, latitude: bool) -> float:
    """Convert IBIS degree-minute coordinates such as 1-16S to decimal degrees."""
    if value is None or pd.isna(value):
        return np.nan
    limit = 90.0 if latitude else 180.0
    if isinstance(value, (int, float, np.integer, np.floating)):
        number = float(value)
        return number if np.isfinite(number) and -limit <= number <= limit else np.nan

    text = str(value).strip().upper()
    if not text or text in {"NA", "N/A", "NONE", "-"}:
        return np.nan
    hemisphere_match = re.search(r"([NSEW])\s*$", text)
    hemisphere = hemisphere_match.group(1) if hemisphere_match else ""
    numeric_text = re.sub(r"[NSEW]", "", text)
    parts = re.findall(r"\d+(?:\.\d+)?", numeric_text)
    if not parts:
        return np.nan
    try:
        degrees = float(parts[0])
        minutes = float(parts[1]) if len(parts) >= 2 else 0.0
        seconds = float(parts[2]) if len(parts) >= 3 else 0.0
    except ValueError:
        return np.nan
    if minutes >= 60 or seconds >= 60:
        return np.nan
    number = degrees + minutes / 60.0 + seconds / 3600.0
    if hemisphere in {"S", "W"} or (not hemisphere and text.lstrip().startswith("-")):
        number = -number
    if not (-limit <= number <= limit):
        return np.nan
    if latitude and hemisphere in {"E", "W"}:
        return np.nan
    if not latitude and hemisphere in {"N", "S"}:
        return np.nan
    return number


def validate_raw_columns(raw: pd.DataFrame) -> list[str]:
    required = [
        RAW_COLUMNS["date"],
        RAW_COLUMNS["propelling_hours"],
        RAW_COLUMNS["log_distance"],
        RAW_COLUMNS["displacement"],
        RAW_COLUMNS["beaufort"],
    ]
    missing = [column for column in required if column not in raw.columns]
    if not any(column in raw.columns for column in FUEL_COLUMNS.values()):
        missing.append("At least one M/E FOC fuel-grade column")
    return missing


def _number(raw: pd.DataFrame, column: str) -> pd.Series:
    if column not in raw.columns:
        return pd.Series(np.nan, index=raw.index, dtype=float)
    return pd.to_numeric(raw[column], errors="coerce")


def voyage_rotation(value: object, c_mapping: str = "Unknown") -> str:
    """Read terminal E/W/N/S voyage suffix; map C only after user confirmation."""
    text = "" if value is None or pd.isna(value) else str(value).strip().upper()
    match = re.search(r"([EWNSC])$", text)
    if not match:
        return "Unknown"
    suffix = match.group(1)
    return c_mapping if suffix == "C" else suffix


def apply_analysis_route_mapping(
    calculated: pd.DataFrame,
    route_mapping: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    """Add a controlled modelling corridor while preserving ServiceLane for audit."""
    if "ServiceLane" not in calculated.columns:
        raise ValueError("ServiceLane is required before applying an analysis-route mapping.")
    result = calculated.copy()
    service = result["ServiceLane"].fillna("Unknown").astype(str).str.strip()
    service = service.mask(service.eq(""), "Unknown")
    clean_mapping = {
        str(source).strip(): str(target).strip()
        for source, target in dict(route_mapping or {}).items()
        if str(source).strip() and str(target).strip()
    }
    result["AnalysisRoute"] = service.map(clean_mapping).fillna(service).astype(str)
    return result


def apply_service_leg_mapping(
    calculated: pd.DataFrame,
    service_leg_mapping: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    """Map detected voyage rotations to user-confirmed operational service legs.

    The detected Rotation is preserved for audit. ServiceLeg is the support
    group used by the model and may be named Outbound/Inbound, Singapore-Japan,
    Japan-Singapore, or another operationally meaningful label.
    """
    if "Rotation" not in calculated.columns:
        raise ValueError("Rotation is required before applying a service-leg mapping.")
    result = calculated.copy()
    rotation = result["Rotation"].fillna("Unknown").astype(str).str.strip()
    rotation = rotation.mask(rotation.eq(""), "Unknown")
    clean_mapping = {
        str(source).strip(): str(target).strip()
        for source, target in dict(service_leg_mapping or {}).items()
        if str(source).strip() and str(target).strip()
    }
    result["ServiceLeg"] = rotation.map(clean_mapping).fillna(rotation).astype(str)
    return result


def build_calculation_table(
    raw: pd.DataFrame,
    fuel_lcv: Mapping[str, float] | None = None,
    reference_lcv: float = REFERENCE_LCV_MJ_KG,
    c_mapping: str = "Unknown",
) -> pd.DataFrame:
    """Calculate STW and 24-hour LCV-normalised main-engine FOC.

    Fuel cells represent interval masses in metric tonnes. Invalid rows are
    retained for audit and assigned NaN rather than a manufactured zero.
    """
    missing = validate_raw_columns(raw)
    if missing:
        raise ValueError("Required columns were not found: " + ", ".join(missing))
    lcv = dict(DEFAULT_FUEL_LCV if fuel_lcv is None else fuel_lcv)
    if reference_lcv <= 0:
        raise ValueError("Reference LCV must be positive.")
    absent = [fuel for fuel in DEFAULT_FUEL_LCV if fuel not in lcv]
    if absent:
        raise ValueError("LCV values are missing for: " + ", ".join(absent))
    if any(not np.isfinite(float(value)) or float(value) <= 0 for value in lcv.values()):
        raise ValueError("Every LCV value must be a positive number.")

    output = pd.DataFrame(index=raw.index)
    output["SourceRow"] = pd.Series(raw.index, index=raw.index).astype(int) + 3
    output["Date"] = pd.to_datetime(raw[RAW_COLUMNS["date"]], errors="coerce")
    latitude_source = raw.get(RAW_COLUMNS["latitude"], pd.Series(np.nan, index=raw.index))
    longitude_source = raw.get(RAW_COLUMNS["longitude"], pd.Series(np.nan, index=raw.index))
    output["Latitude"] = latitude_source.map(lambda value: coordinate_to_decimal(value, True))
    output["Longitude"] = longitude_source.map(lambda value: coordinate_to_decimal(value, False))
    output["PropellingHours"] = raw[RAW_COLUMNS["propelling_hours"]].map(duration_to_hours)
    output["LogDistanceNM"] = _number(raw, RAW_COLUMNS["log_distance"])
    output["STW"] = np.where(
        output["PropellingHours"] > 0,
        output["LogDistanceNM"] / output["PropellingHours"],
        np.nan,
    )
    output["DisplacementMT"] = _number(raw, RAW_COLUMNS["displacement"])
    output["Beaufort"] = _number(raw, RAW_COLUMNS["beaufort"])
    output["WindSpeedKn"] = _number(raw, RAW_COLUMNS["wind_speed"])
    output["WindSeaHeightM"] = _number(raw, RAW_COLUMNS["wind_sea_height"])
    output["SwellHeightM"] = _number(raw, RAW_COLUMNS["swell_height"])
    output["ForeDraftM"] = _number(raw, RAW_COLUMNS["fore_draft"])
    output["AftDraftM"] = _number(raw, RAW_COLUMNS["aft_draft"])
    output["TrimM"] = output["AftDraftM"] - output["ForeDraftM"]

    fuel_frame = pd.DataFrame(index=raw.index)
    energy = pd.Series(0.0, index=raw.index)
    for fuel in DEFAULT_FUEL_LCV:
        mass = _number(raw, FUEL_COLUMNS[fuel])
        fuel_frame[fuel] = mass
        output[f"MEFuel_{fuel}_MT"] = mass
        energy = energy.add(mass.fillna(0.0) * float(lcv[fuel]), fill_value=0.0)
    any_fuel_recorded = fuel_frame.notna().any(axis=1)
    negative_fuel = (fuel_frame < 0).any(axis=1)
    equivalent_mass = energy / float(reference_lcv)
    valid_fuel = any_fuel_recorded & ~negative_fuel & (equivalent_mass > 0)
    valid_hours = output["PropellingHours"] > 0
    output["IntervalFuelEquivalentMT"] = equivalent_mass.where(valid_fuel)
    output["FOC_MT_Day"] = (
        equivalent_mass * 24.0 / output["PropellingHours"]
    ).where(valid_fuel & valid_hours)
    output["FuelDataStatus"] = np.select(
        [negative_fuel, ~any_fuel_recorded, ~valid_hours, equivalent_mass <= 0],
        ["Negative fuel mass", "All fuel cells blank", "Invalid propelling hours", "Non-positive fuel"],
        default="Valid",
    )

    for source, target in [
        ("Report ID", "ReportID"),
        ("Vessel", "Vessel"),
        ("Voyage", "Voyage"),
        ("Service Lane", "ServiceLane"),
    ]:
        output[target] = raw[source] if source in raw.columns else "Unknown"
    output["Vessel"] = output["Vessel"].fillna("Unknown").astype(str).str.strip()
    output["Voyage"] = output["Voyage"].fillna("Unknown").astype(str)
    output["ServiceLane"] = output["ServiceLane"].fillna("Unknown").astype(str).str.strip()
    output["AnalysisRoute"] = output["ServiceLane"]
    output["Rotation"] = output["Voyage"].map(lambda value: voyage_rotation(value, c_mapping))
    output["ServiceLeg"] = output["Rotation"]

    # Check whether the recorded propelling time is possible within the elapsed
    # report interval.  A two-hour tolerance accommodates 23/25-hour reporting
    # days and small timestamp differences without accepting clearly impossible
    # intervals.  The first report for a vessel remains usable but is marked as
    # not independently verifiable from the workbook.
    ordered = output[["Vessel", "Date"]].sort_values(["Vessel", "Date"], kind="stable")
    ordered["PreviousReportDate"] = ordered.groupby("Vessel", dropna=False)["Date"].shift(1)
    ordered["ElapsedReportHours"] = (
        ordered["Date"] - ordered["PreviousReportDate"]
    ).dt.total_seconds() / 3600.0
    output["PreviousReportDate"] = ordered["PreviousReportDate"].reindex(output.index)
    output["ElapsedReportHours"] = ordered["ElapsedReportHours"].reindex(output.index)
    duplicate_timestamp = output.duplicated(["Vessel", "Date"], keep=False) & output["Date"].notna()
    impossible_interval = (
        output["ElapsedReportHours"].notna()
        & (
            (output["ElapsedReportHours"] <= 0)
            | (output["PropellingHours"] > output["ElapsedReportHours"] + 2.0)
        )
    )
    output["IntervalAlignmentStatus"] = np.select(
        [duplicate_timestamp, impossible_interval, output["ElapsedReportHours"].isna()],
        ["Duplicate report timestamp", "Propelling hours exceed elapsed interval", "Not verifiable - first report"],
        default="Internally consistent",
    )
    return output.replace([np.inf, -np.inf], np.nan)


def assign_periods(
    calculated: pd.DataFrame,
    dock_in: object,
    dock_out: object,
    before_start: object,
    after_end: object,
) -> pd.DataFrame:
    """Assign exact calendar periods and exclude the complete dry-dock interval."""
    start = pd.Timestamp(dock_in).normalize()
    end = pd.Timestamp(dock_out).normalize()
    pre_start = pd.Timestamp(before_start).normalize()
    post_end = pd.Timestamp(after_end).normalize()
    if end < start:
        raise ValueError("Dock-out cannot be earlier than dock-in.")
    if pre_start >= start:
        raise ValueError("The pre-DD start must be earlier than dock-in.")
    if post_end <= end:
        raise ValueError("The post-DD end must be later than dock-out.")
    result = calculated.copy()
    dates = pd.to_datetime(result["Date"], errors="coerce").dt.normalize()
    result["Period"] = "Outside analysis window"
    result.loc[(dates >= pre_start) & (dates < start), "Period"] = "Before DD"
    result.loc[(dates >= start) & (dates <= end), "Period"] = "Dry dock - excluded"
    result.loc[(dates > end) & (dates <= post_end), "Period"] = "After DD"
    return result


def filter_operating_rows(
    calculated: pd.DataFrame,
    min_propelling_hours: float = 18.0,
    min_plausible_stw: float = 5.0,
    max_plausible_stw: float = 35.0,
    max_beaufort: float = 4.0,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Apply hard data-quality rules before learned operating support.

    The 5-35 knot bounds reject physically implausible steady-sea records; they
    are not the vessel's operating range.  Actual comparability is learned from
    pre-DD STW and displacement density in ``assess_operating_support``.
    """
    df = calculated.copy()
    in_period = df["Period"].isin(["Before DD", "After DD"])
    interval_ok = ~df.get(
        "IntervalAlignmentStatus", pd.Series("Internally consistent", index=df.index)
    ).isin(["Duplicate report timestamp", "Propelling hours exceed elapsed interval"])
    valid = (
        df["Date"].notna()
        & (pd.to_numeric(df["PropellingHours"], errors="coerce") >= min_propelling_hours)
        & pd.to_numeric(df["STW"], errors="coerce").between(
            min_plausible_stw, max_plausible_stw
        )
        & (pd.to_numeric(df["DisplacementMT"], errors="coerce") > 0)
        & (pd.to_numeric(df["FOC_MT_Day"], errors="coerce") > 0)
        & (pd.to_numeric(df["Beaufort"], errors="coerce") <= max_beaufort)
        & interval_ok
    )
    eligible = df.loc[in_period & valid].sort_values("Date").reset_index(drop=True)
    report = {
        "input_rows": int(len(df)),
        "period_rows": int(in_period.sum()),
        "eligible_rows": int(len(eligible)),
        "before_rows": int((eligible["Period"] == "Before DD").sum()),
        "after_rows": int((eligible["Period"] == "After DD").sum()),
        "excluded_quality_rows": int((in_period & ~valid).sum()),
        "interval_mismatch_rows": int((in_period & ~interval_ok).sum()),
    }
    return eligible, report


def largest_date_gap(calculated: pd.DataFrame) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    """Return the largest reporting gap as a convenience, never as official DD evidence."""
    dates = pd.to_datetime(calculated["Date"], errors="coerce").dropna().drop_duplicates().sort_values()
    if len(dates) < 2:
        return None, None
    gaps = dates.diff()
    current_index = gaps.idxmax()
    position = dates.index.get_loc(current_index)
    return dates.iloc[position - 1].normalize(), dates.iloc[position].normalize()
