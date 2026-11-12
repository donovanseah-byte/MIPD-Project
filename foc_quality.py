"""V9.14 MIPD data/model reliability and independent operational verification.

This is an experimental *project-defined* grading framework, not an ISO
performance-verification standard. It does not change the Huber estimate or
post-dry-docking comparable-data selection.
"""
from __future__ import annotations

import math

MIN_PRE_DD = 60
MIN_POST_CALCULATION = 10
MIN_POST_SUPPORTED = 15
MIN_COVERAGE_CALCULATION = 40.0
MIN_COVERAGE_SUPPORTED = 60.0
MAX_MAPE = 12.0
MAX_ABSOLUTE_BIAS = 8.0
SPEED_EXPONENT_MIN = 1.5
SPEED_EXPONENT_MAX = 4.5


def _real(value):
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (ValueError, TypeError):
        return None


def _rows(result):
    basis = result.get("comparison_basis")
    if basis == "Strict same-route":
        return int(result.get("strict_supported_after_rows") or result.get("supported_after_rows") or 0)
    if basis == "Expanded cross-route":
        return int(result.get("expanded_supported_after_rows") or result.get("supported_after_rows") or 0)
    return int(result.get("supported_after_rows") or 0)


def quality_level(result, settings):
    """Data quantity and comparability; service-cycle proof is reported separately."""
    min_pre = max(MIN_PRE_DD, int(settings.get("minimum_train_rows") or MIN_PRE_DD))
    pre = int(result.get("before_rows") or 0)
    post = _rows(result)
    coverage = _real(result.get("coverage_pct")) or 0.0
    if (pre < min_pre or post < MIN_POST_CALCULATION
            or coverage < MIN_COVERAGE_CALCULATION
            or settings.get("interval_alignment_confirmed") is False):
        return "Insufficient"
    if (post >= MIN_POST_SUPPORTED
            and coverage >= MIN_COVERAGE_SUPPORTED
            and settings.get("interval_alignment_confirmed") is True
            and settings.get("period_coverage_ok") is True
            and settings.get("route_confirmed") is True):
        return "High"
    return "Limited"


def assess_reliability(result, settings):
    """Classify reliability independent of estimated saving sign.

    Returns the three-tier analytical grade plus independent service-cycle
    verification status and traceable reasons. Original model-tier gates stay available in result['evidence_tier'] for comparison.
    """
    reasons = []
    # A complete normal service cycle strengthens operational interpretation,
    # but its absence alone must not change the grade of the data/model test.
    # Never infer that a service cycle is verified from calendar coverage.
    cycle_confirmed = settings.get("service_cycle_confirmed") is True
    cycle_status = "Confirmed" if cycle_confirmed else "Pending"
    cycle_note = ("A normal post-DD service cycle was confirmed by the assessor."
                  if cycle_confirmed else
                  "Normal post-DD service-cycle completion has not been confirmed; "
                  "the analytical assessment is not an operationally verified dry-dock effect.")
    verification = {"operational_verification": cycle_status,
                    "operational_verification_note": cycle_note}
    validation = result.get("validation") or {}
    huber = validation.get("huber") or {}
    mape = _real(huber.get("mape_pct"))
    bias = _real(huber.get("bias_pct"))
    exponent = _real(result.get("learned_speed_exponent"))
    effect = _real(result.get("improvement_pct"))
    legacy = str(result.get("original_model_evidence_tier") or result.get("evidence_tier") or "")
    post = _rows(result)
    pre = int(result.get("before_rows") or 0)
    cov = _real(result.get("coverage_pct")) or 0.0

    critical = []
    if not result.get("valid") or effect is None:
        critical.append("A valid, finite package-level FOC estimate was not available.")
    if legacy == "Inconclusive":
        critical.append("The underlying model classified the estimate as Inconclusive.")
    if pre < max(MIN_PRE_DD, int(settings.get("minimum_train_rows") or MIN_PRE_DD)):
        critical.append(f"Only {pre} eligible pre-DD reports; at least {MIN_PRE_DD} required.")
    if post < MIN_POST_CALCULATION:
        critical.append(f"Only {post} comparable post-DD reports; at least {MIN_POST_CALCULATION} required.")
    if cov < MIN_COVERAGE_CALCULATION:
        critical.append(f"Comparable fuel coverage is {cov:.1f}%; at least {MIN_COVERAGE_CALCULATION:.0f}% required.")
    if mape is None or mape > MAX_MAPE:
        critical.append(f"Unseen pre-DD model MAPE is {mape if mape is not None else 'unavailable'}; must be <= {MAX_MAPE:.0f}%.")
    if bias is None or abs(bias) > MAX_ABSOLUTE_BIAS:
        critical.append(f"Absolute unseen pre-DD prediction bias is {abs(bias) if bias is not None else 'unavailable'}; must be <= {MAX_ABSOLUTE_BIAS:.0f}%.")
    if exponent is None or not SPEED_EXPONENT_MIN <= exponent <= SPEED_EXPONENT_MAX:
        critical.append("The learned speed exponent is unavailable or outside the 1.5–4.5 engineering band.")
    if settings.get("interval_alignment_confirmed") is not True:
        critical.append("Alignment of fuel, LOG distance and propelling-hour report intervals is not confirmed.")
    if critical:
        return {"tier": "Inconclusive", "reasons": critical, "data_quality": quality_level(result, settings), **verification}

    if post < MIN_POST_SUPPORTED:
        reasons.append(f"{post} comparable post-DD reports, below the Supported target of {MIN_POST_SUPPORTED}.")
    if cov < MIN_COVERAGE_SUPPORTED:
        reasons.append(f"{cov:.1f}% post-DD fuel coverage, below the Supported target of {MIN_COVERAGE_SUPPORTED:.0f}%.")
    if settings.get("period_coverage_ok") is not True:
        reasons.append("The declared monitoring-period coverage has not been confirmed.")
    if settings.get("route_confirmed") is not True:
        reasons.append("Route and operating-leg mappings are not confirmed.")
    if result.get("comparison_basis") not in {"Strict same-route", "Expanded cross-route"}:
        reasons.append("Neither a same-route nor a comparable same-leg cross-route assessment basis was established.")
    stability = str(result.get("stability_status") or "").lower()
    if stability != "stable" or legacy == "Unstable":
        reasons.append("Sensitivity and robustness checks did not establish a stable direction of change.")
    anomaly_count = int(result.get("flagged_foc_rows") or 0)
    if anomaly_count:
        reasons.append(f"{anomaly_count} gross FOC reports were flagged for engineering review.")
    # Placebo effect magnitude tests an *effect claim*, not whether the input
    # data/model analysis met declared adequacy checks. Keep it as a separate
    # visible diagnostic, with both positive and negative effects treated fairly.

    tier = "Evidence-Limited" if reasons else "Supported Assessment"
    if not reasons:
        reasons = ["Model validation, minimum matched-data coverage and stable-direction checks passed; the result may represent a saving or higher FOC."]
    return {"tier": tier, "reasons": reasons, "data_quality": quality_level(result, settings), **verification}


def placebo_signal(result):
    """Two-sided exploratory effect-signal diagnostic, separate from quality tier.

    This is not a statistical confidence interval, hypothesis test with a
    controlled false-positive rate, or evidence that dry docking caused the change.
    """
    placebo = result.get("placebo") or {}
    actual = _real(result.get("improvement_pct"))
    table = placebo.get("results")
    values = []
    if table is not None:
        try:
            for value in table["PlaceboSavingPct"]:
                value = _real(value)
                if value is not None:
                    values.append(value)
        except (KeyError, TypeError):
            pass
    if actual is None or len(values) < 3:
        return {"status": "Not assessable", "count": len(values), "max_abs_placebo": None,
                "explanation": "Too few valid fake-date effects for a two-sided comparison."}
    bound = max(abs(item) for item in values)
    if abs(actual) > bound:
        return {"status": "Magnitude exceeds placebos", "count": len(values), "max_abs_placebo": bound,
                "explanation": "The absolute estimated change exceeds all available fake-date changes. This is exploratory only."}
    return {"status": "Not separated", "count": len(values), "max_abs_placebo": bound,
            "explanation": "Fake-date changes of comparable or greater magnitude exist; do not claim a clearly detected dry-dock effect."}
