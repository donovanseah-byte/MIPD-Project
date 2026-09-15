"""Self-contained printable HTML, using the same PNG charts and saved result."""
from __future__ import annotations

from base64 import b64encode
from html import escape

import numpy as np
import pandas as pd

from foc_explain import fuel_summary
from foc_visuals import figure_png, fuel_figure


def printable_report(saved):
    result, settings = saved["result"], saved["settings"]
    fuel = fuel_summary(result)
    if fuel is None or not np.isclose(fuel["saving_pct"], result.get("improvement_pct", np.nan), atol=1e-7, rtol=0):
        raise ValueError("No reconciled estimate is available for a printable report.")
    def img(png, alt):
        return f'<img alt="{escape(alt)}" src="data:image/png;base64,{b64encode(png).decode()}" />'
    chart = img(figure_png(fuel_figure(fuel)), "Model-expected versus reported fuel over the same post-DD reports")
    limitations = "".join(f"<li>{escape(str(item))}</li>" for item in result.get("limitations", []))
    validation = result.get("validation", {})
    def metric(value):
        return "Unavailable" if value is None else f"{float(value):.2f}%"
    sensitivity = result.get("model_sensitivity", pd.DataFrame())
    sensitivity_html = ""
    if isinstance(sensitivity, pd.DataFrame) and not sensitivity.empty:
        sensitivity_html = "<section><h2>ML specification and method sensitivity</h2>"
        sensitivity_html += f"<p>{escape(str(result.get('model_selection_reason', '')))}</p>"
        sensitivity_html += sensitivity.to_html(index=False, border=0, float_format=lambda x: f"{x:,.2f}")
        sensitivity_html += "<p>These values show method sensitivity; they are not a statistical confidence interval and are not additive.</p></section>"
    method = f"""<section><h2>Chronological prediction validation</h2><p>Unseen-report mean absolute percentage error:
selected Huber {metric(validation.get('huber', {}).get('mape_pct'))}; public cubic-speed benchmark {metric(validation.get('cubic', {}).get('mape_pct'))}.
These are prediction errors, not confidence intervals around the saving estimate.</p></section>"""
    verdict = {
        "Supported (prototype screening)": "Supported for prototype engineering screening",
        "Indicative": "Indicative result - use with stated limitations",
        "Preliminary": "Preliminary result - more comparable evidence required",
        "Unstable": "Direction-sensitive result - do not interpret the percentage",
        "Inconclusive": "Inconclusive - model or data not adequate for interpretation",
    }.get(result.get("evidence_tier"), "Assessment status unavailable")
    headline = (
        f"{fuel['saving_pct']:.2f}% calculated difference - not interpretable"
        if result.get("evidence_tier") in {"Unstable", "Inconclusive"}
        else f"{fuel['saving_pct']:.2f}% ML-estimated package FOC saving"
    )
    comparison_label = {
        "Strict same-route": "Same-route comparison (preferred)",
        "Expanded cross-route": "Cross-route comparison (fallback)",
    }.get(result.get("comparison_basis"), "No usable comparison")
    sensitivity_range = (
        f"{result['stability_min_pct']:.2f}% to {result['stability_max_pct']:.2f}%"
        if result.get("stability_min_pct") is not None and result.get("stability_max_pct") is not None
        else "Unavailable"
    )
    demo_notice = (
        '<div class="demo"><strong>Synthetic demonstration:</strong> This report explains the POC workflow. '
        'It is not evidence from an actual vessel and must not support a vessel saving claim.</div>'
        if settings.get("is_demo") else ""
    )
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{escape(str(settings['vessel']))} - Package-level post-DD FOC assessment V9.11</title><style>
body{{font:15px/1.5 Arial,sans-serif;color:#172b46;background:#fff;max-width:1000px;margin:32px auto;padding:0 24px}}
h1{{font-size:28px;margin-bottom:4px}}h2{{font-size:19px}}.meta{{color:#56667a}}.estimate{{font-size:36px;font-weight:700}}
.scope{{background:#eef4f8;padding:14px;border-left:4px solid #376eaa}}.demo{{background:#fff4dc;padding:14px;border-left:4px solid #9a640e;margin:16px 0}}img{{display:block;width:100%;height:auto}}
table{{width:100%;border-collapse:collapse;font-size:13px}}th,td{{text-align:left;padding:7px;border-bottom:1px solid #d9e2eb}}
section{{margin-top:22px}}li{{margin:5px 0}}@media print{{body{{margin:0;max-width:none;padding:0;font-size:11pt}}
img,table,.scope{{break-inside:avoid}}h2{{break-after:avoid}}@page{{size:A4;margin:15mm}}}}
</style></head><body><h1>{escape(str(settings['vessel']))} | Package-level post-DD FOC assessment</h1>
<p class="meta">V9.11 | Dock-in: {escape(str(settings['dock_in']))} | Dock-out: {escape(str(settings['dock_out']))}<br>
Post-DD assessment endpoint: {escape(str(settings['required_post_end']))} | {escape(str(settings.get('monitoring_basis', '')))}</p>
{demo_notice}
<div class="estimate">{escape(headline)}</div>
<p>Assessment verdict: <strong>{escape(verdict)}</strong>. Positive means reported fuel was lower than model-expected; negative means it was higher.</p>
<p>Sensitivity direction: <strong>{escape(str(result.get('stability_status', 'Not assessable')))}</strong>. {escape(str(result.get('stability_summary', '')))}</p>
<p>Sensitivity range: <strong>{escape(sensitivity_range)}</strong>. This is not a statistical confidence interval.</p>
<p>Operating comparison method: <strong>{escape(comparison_label)}</strong>.</p>
<p>Selected ML specification: <strong>{escape(str(result.get('selected_ml_model', 'Unavailable')))}</strong>.</p>
<p>The selected model learned from {result['before_rows']} pre-DD reports. Displacement remains part of the operating-support check even when it is not selected as a prediction term.
Both totals below use the same {len(fuel['rows'])} comparable reports and {fuel['hours']:,.1f} propelling hours.</p>{chart}
<p>Calculation: ({fuel['expected']:,.2f} - {fuel['actual']:,.2f}) / {fuel['expected']:,.2f} x 100 = {fuel['saving_pct']:.2f}%.
Totals are rounded here; the calculation uses full precision.</p>
<div class="scope">{escape(str(result.get('scope_statement', 'Comparable post-DD reports only.')))}<br>
Comparable post-DD fuel coverage: {result['coverage_pct']:.1f}%. This is data coverage, not model accuracy.<br>
Complete service cycle: {'operationally confirmed' if settings.get('service_cycle_confirmed') else 'not confirmed'}.</div>
<p>This estimates the package-level difference associated with the dry-dock event. It does not isolate individual work items or prove that dry docking alone caused the difference.</p>
<p>The method follows the general same-vessel, comparable-condition principle associated with ISO 19030, but it does not implement the ISO 19030 default method. The speed/loading support check is not an ISO reference-displacement correction.</p>
<h2>Limits on interpretation</h2><ul>{limitations}</ul>{sensitivity_html}{method}
<p class="meta">Local report. No external scripts, images or online services are required. Open in a browser and use Print / Save as PDF.</p>
</body></html>"""
