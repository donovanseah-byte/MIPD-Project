"""Display-only, browser-independent charts for the existing assessment."""
from __future__ import annotations

from io import BytesIO
from threading import RLock

import numpy as np
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.ticker import StrMethodFormatter

INK = "#172b46"
BLUE = "#376eaa"
TEAL = "#197a69"
AMBER = "#9a640e"
RED = "#aa4435"
LOCK = RLock()


def figure_png(figure):
    # No HTML, SVG, JS, network, browser, or shared pyplot state is involved.
    with LOCK:
        data = BytesIO()
        FigureCanvasAgg(figure).print_png(data)
        return data.getvalue()


def assessment_story_figure(summary, result, verdict):
    """Explain the complete estimate as one evidence chain."""
    expected = float(summary["expected"])
    actual = float(summary["actual"])
    saving = float(summary["saving_pct"])
    if not np.isfinite([expected, actual, saving]).all():
        raise ValueError("Assessment story requires finite fuel totals and saving.")

    model = str(result.get("selected_ml_model") or "Huber ML")
    model_display = (
        "Huber ML\nSTW + displacement"
        if "displacement" in model.casefold()
        else "Huber ML\nSTW only"
    )

    tier = str(result.get("evidence_tier") or "Inconclusive")
    post_rows = int(result.get("supported_after_rows") or 0)
    coverage = float(result.get("coverage_pct") or 0.0)
    comparison_basis = str(result.get("comparison_basis") or "")
    outcome = (
        f"{saving:.2f}% less fuel"
        if saving >= 0
        else f"{abs(saving):.2f}% more fuel"
    )

    if comparison_basis == "Expanded cross-route":
        status_detail = (
            f"{tier}: cross-route fallback\n"
            "Same-route evidence was insufficient"
        )
    elif tier == "Preliminary" and post_rows < 20:
        status_detail = (
            f"Preliminary: only {post_rows} comparable reports\n"
            "At least 20 required for Indicative"
        )
    elif tier == "Preliminary" and coverage < 70:
        status_detail = (
            f"Preliminary: {coverage:.1f}% fuel coverage\n"
            "At least 70% required for Indicative"
        )
    elif tier == "Supported (prototype screening)":
        status_detail = (
            "Strongest prototype checks passed\n"
            "Not contractual verification"
        )
    elif tier == "Indicative":
        status_detail = (
            "Same-route evidence passed\n"
            "Continue post-DD monitoring"
        )
    elif tier == "Unstable":
        status_detail = (
            "Alternative checks changed direction\n"
            "Do not interpret the percentage"
        )
    elif tier == "Inconclusive":
        status_detail = (
            "Model or data checks failed\n"
            "Do not interpret the percentage"
        )
    else:
        status_detail = verdict

    panels = [
        (
            "1  SCREEN",
            "Reports used",
            f"{result.get('before_rows', 0):,} pre-DD reports trained the model\n"
            f"{post_rows:,} comparable post-DD reports assessed",
        ),
        (
            "2  SELECT",
            "ML model selected",
            f"{model_display}\nSelected using pre-DD validation",
        ),
        (
            "3  PREDICT",
            "Expected fuel",
            f"{expected:,.2f} MT\nIf pre-DD performance had continued",
        ),
        (
            "4  OBSERVE",
            "Reported fuel",
            f"{actual:,.2f} MT\nRecorded over the same intervals",
        ),
        (
            "5  CONCLUDE",
            outcome,
            status_detail,
        ),
    ]
    accent = [BLUE, AMBER, BLUE, INK, TEAL if saving >= 0 else RED]
    fig = Figure(figsize=(13.5, 4.0), dpi=160, facecolor="white")
    ax = fig.add_axes([0.015, 0.08, 0.97, 0.84])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    width, gap, y, height = 0.176, 0.025, 0.20, 0.64
    for index, ((step, heading, detail), colour) in enumerate(zip(panels, accent)):
        x = 0.005 + index * (width + gap)
        box = FancyBboxPatch(
            (x, y), width, height,
            boxstyle="round,pad=0.012,rounding_size=0.018",
            facecolor="#f7f9fc", edgecolor="#c8d3df", linewidth=1.0,
        )
        ax.add_patch(box)
        ax.add_patch(FancyBboxPatch(
            (x, y + height - 0.055), width, 0.055,
            boxstyle="round,pad=0.0,rounding_size=0.015",
            facecolor=colour, edgecolor=colour, linewidth=0,
        ))
        ax.text(x + 0.012, y + height - 0.095, step, color=colour,
                fontsize=9, weight="bold", va="top")
        ax.text(x + 0.012, y + height - 0.205, heading, color=INK,
                fontsize=13, weight="bold", va="top", wrap=True)
        ax.text(x + 0.012, y + height - 0.335, detail, color="#526176",
                fontsize=9.5, va="top", linespacing=1.5, wrap=True)
        if index < len(panels) - 1:
            arrow = FancyArrowPatch(
                (x + width + 0.003, 0.52), (x + width + gap - 0.003, 0.52),
                arrowstyle="-|>", mutation_scale=13, linewidth=1.2,
                color="#7e8da1",
            )
            ax.add_patch(arrow)
    # Keep this return outside the loop. Placing it inside the loop causes
    # Matplotlib to stop after drawing only the first card.
    return fig


def fuel_figure(summary):
    expected, actual = summary["expected"], summary["actual"]
    if not np.isfinite([expected, actual]).all() or expected <= 0 or actual < 0:
        raise ValueError("Fuel comparison requires positive expected and non-negative reported fuel.")
    scale = max(expected, actual)
    gap = expected - actual
    fig = Figure(figsize=(11, 3.8), dpi=160, facecolor="white")
    ax = fig.add_axes([0.08, 0.23, 0.86, 0.64])
    ax.set_facecolor("white")
    ax.barh(1.0, expected, height=0.3, color=BLUE)
    ax.barh(0.0, actual, height=0.3, color=INK)
    ax.text(0, 1.27, "Model-expected fuel", color=INK, fontsize=12, weight="bold")
    ax.text(0, 0.27, "Reported post-DD fuel", color=INK, fontsize=12, weight="bold")
    for y, value in [(1.0, expected), (0.0, actual)]:
        ax.text(scale * 1.02, y, f"{value:,.2f}", va="center", color=INK, fontsize=12)
    if abs(gap) > scale * 1e-10:
        y = 0.0 if gap > 0 else 1.0
        ax.barh(y, abs(gap), left=min(expected, actual), height=.3,
                color="#edf4f2" if gap > 0 else "#faece9",
                edgecolor=TEAL if gap > 0 else RED, hatch="////", linewidth=.8)
        # Put both labels above the axis instead of inside a narrow difference.
        ax.text(.0, -.35, f"Difference: {abs(gap):,.2f} MT {'less' if gap > 0 else 'more'} than expected",
                color=TEAL if gap > 0 else RED, fontsize=11, weight="bold")
    else:
        ax.text(0, -.35, "No difference between the two totals", color=INK, fontsize=11)
    ax.set_xlim(0, scale * 1.18)
    ax.set_ylim(-.55, 1.55)
    ax.set_yticks([])
    ax.xaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    ax.tick_params(axis="x", colors="#526176", labelsize=10)
    ax.set_xlabel("Main-engine fuel over the same reports (VLSFO-equivalent MT)",
                  color=INK, labelpad=10, fontsize=10)
    for side in ["top", "left", "right"]:
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#bcc9d7")
    fig.text(.08, .95, f"Same {len(summary['rows']):,} comparable reports | Same {summary['hours']:,.1f} propelling hours",
             fontsize=11, color="#526176")
    return fig


def monthly_table(result, settings):
    data = result.get("persistence", pd.DataFrame()).copy()
    columns = ["Month", "SavingPct", "SupportedRows", "PropellingHours",
               "ActualFuelMT", "ExpectedPreDDConditionFuelMT"]
    if data.empty:
        return pd.DataFrame(columns=columns + ["Month label", "Data status"])
    data["Month"] = pd.to_datetime(data["Month"], errors="coerce").dt.to_period("M")
    data = data.dropna(subset=["Month"]).set_index("Month").sort_index()
    if data.empty:
        return pd.DataFrame(columns=columns + ["Month label", "Data status"])
    if settings.get("dock_out") and settings.get("required_post_end"):
        first = (pd.Timestamp(settings["dock_out"]) + pd.Timedelta(days=1)).to_period("M")
        last = pd.Timestamp(settings["required_post_end"]).to_period("M")
        if first <= last:
            data = data.reindex(pd.period_range(first, last, freq="M"))
    data.index.name = "Month"
    data = data.reset_index()
    data["Month label"] = data["Month"].dt.strftime("%b %Y")
    data["SupportedRows"] = data["SupportedRows"].fillna(0).astype(int)
    data["PropellingHours"] = data["PropellingHours"].fillna(0)
    data["Data status"] = np.select(
        [data["SupportedRows"].eq(0), data["SupportedRows"].lt(5), data["SupportedRows"].lt(10)],
        ["No comparable reports", "Too few reports", "Limited reports"], default="10+ reports")
    # Missing months remain missing estimates, never zero saving.
    data.loc[data["SupportedRows"].eq(0), "SavingPct"] = np.nan
    return data


def monthly_figure(data, overall):
    fig = Figure(figsize=(11, 4.4), dpi=160, facecolor="white")
    ax = fig.add_axes([.08, .23, .86, .67])
    values = pd.to_numeric(data["SavingPct"], errors="coerce").to_numpy()
    finite = values[np.isfinite(values)]
    span = max(5., np.max(np.abs(finite)) if len(finite) else 5., abs(overall) if np.isfinite(overall) else 0.)
    for i, row in data.reset_index(drop=True).iterrows():
        value = row["SavingPct"]
        if not np.isfinite(value):
            ax.text(i, span * .08, "No comparable\nreports", ha="center", color="#69798b", fontsize=10)
            continue
        count = int(row["SupportedRows"])
        sparse = count < 5
        colour = "#aab6c4" if sparse else (AMBER if count < 10 else BLUE)
        ax.bar(i, value, width=.58, color=colour, edgecolor=INK,
               hatch="///" if sparse else None, linewidth=.6)
        ax.text(i, value + np.sign(value or 1) * span * .035, f"{value:.2f}%",
                ha="center", va="bottom" if value >= 0 else "top", color=INK, fontsize=11, weight="bold")
    if np.isfinite(overall):
        ax.axhline(overall, color=TEAL, linestyle="--", linewidth=1.2,
                   label=f"Overall estimate: {overall:.2f}% (reference, not a monthly target)")
        ax.legend(loc="upper left", bbox_to_anchor=(0, 1.16), frameon=False, fontsize=10)
    labels = [f"{row['Month label']}\n{row['SupportedRows']} reports | {row['PropellingHours']:,.0f} h"
              for _, row in data.iterrows()]
    ax.set_xticks(range(len(data)), labels, fontsize=10, color=INK)
    ax.set_xlim(-.6, len(data) - .4)
    extent = np.append(finite, [overall]) if np.isfinite(overall) else finite
    low = min(0., extent.min()) if len(extent) else 0.
    high = max(0., extent.max()) if len(extent) else span
    ax.set_ylim(low - span * .17, high + span * .3)
    ax.set_ylabel("Estimated FOC saving (%)", fontsize=11, color=INK)
    ax.axhline(0, color="#9cadbf", linewidth=.8)
    ax.yaxis.grid(True, color="#e7edf3")
    ax.set_axisbelow(True)
    for side in ["top", "right", "left"]:
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#bcc9d7")
    fig.text(.08, .035, "Hatched: fewer than 5 reports. Amber: 5-9 reports. These are count flags, not confidence intervals.",
             fontsize=9, color="#526176")
    return fig
