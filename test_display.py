"""Regression checks for presentation only; no confidential workbook required."""
import io
import unittest
from copy import deepcopy
from unittest.mock import patch

import numpy as np
import pandas as pd
from PIL import Image

from foc_audit import build_report_ledger, report_comparison
from foc_demo import build_demo_assessment
from foc_explain import conditional_slice, fuel_summary, raw_correlation, training_rows
from foc_exports import _actual_expected_scatter, _overview_table
from foc_interpret import actual_expected_comment, operating_support_comment
from foc_processing import filter_operating_rows
from foc_report import printable_report
from foc_visuals import (assessment_story_figure, figure_png, fuel_figure,
                         monthly_figure, monthly_table)
from foc_setup import SetupWidgets


def fixture():
    return {"valid": True, "assessed_rows": pd.DataFrame({
        "InsideSupport": [True, True, False], "FOC_MT_Day": [80., 110., 1.],
        "ExpectedPreDDCondition_FOC_MT_Day": [100., 100., 9999.],
        "PropellingHours": [12., 24., 24.],
    })}


class DisplayTests(unittest.TestCase):
    def test_interval_weighting_and_support(self):
        result = fixture()
        original = deepcopy(result)
        summary = fuel_summary(result)
        self.assertEqual((summary["actual"], summary["expected"], summary["saving_pct"]), (150., 150., 0.))
        self.assertEqual(summary["hours"], 36.)
        self.assertEqual(len(summary["rows"]), 2)
        pd.testing.assert_frame_equal(result["assessed_rows"], original["assessed_rows"])

    def test_failed_and_empty_support(self):
        result = fixture()
        result["valid"] = False
        self.assertIsNone(fuel_summary(result))
        result["valid"] = True
        result["assessed_rows"]["InsideSupport"] = False
        self.assertIsNone(fuel_summary(result))

    def test_nonfinite_pairs_excluded(self):
        result = fixture()
        result["assessed_rows"].loc[0, "FOC_MT_Day"] = np.inf
        self.assertEqual(len(fuel_summary(result)["rows"]), 1)

    def test_png_renders_actual_chart_pixels(self):
        summary = fuel_summary(fixture())
        payload = figure_png(fuel_figure(summary))
        self.assertTrue(payload.startswith(b"\x89PNG\r\n\x1a\n"))
        image = np.asarray(Image.open(io.BytesIO(payload)).convert("RGB"))
        self.assertGreater(np.all(image == [55, 110, 170], axis=2).sum(), 10000)
        self.assertGreater(np.all(image == [23, 43, 70], axis=2).sum(), 10000)

    def test_negative_result_has_more_fuel_and_common_zero_axis(self):
        result = fixture()
        result["assessed_rows"].loc[0, "FOC_MT_Day"] = 200
        summary = fuel_summary(result)
        fig = fuel_figure(summary)
        self.assertLess(summary["saving_pct"], 0)
        self.assertEqual(fig.axes[0].get_xlim()[0], 0)
        self.assertTrue(any("more than expected" in item.get_text() for item in fig.axes[0].texts))
        self.assertEqual(fig.axes[0].patches[0].get_width(), summary["expected"])
        self.assertEqual(fig.axes[0].patches[1].get_width(), summary["actual"])

    def test_months_ordered_and_empty_month_not_zero(self):
        frame = pd.DataFrame({"Month": ["2026-05-01", "2026-03-01", "2026-04-01"],
            "SavingPct": [12.12, 7.52, 5.01], "SupportedRows": [2, 17, 6], "PropellingHours": [48., 415., 145.]})
        original = frame.copy(deep=True)
        table = monthly_table({"persistence": frame}, {"dock_out": "2026-02-10", "required_post_end": "2026-05-10"})
        self.assertEqual(table["Month label"].tolist(), ["Feb 2026", "Mar 2026", "Apr 2026", "May 2026"])
        self.assertTrue(np.isnan(table.iloc[0]["SavingPct"]))
        self.assertEqual(table["Data status"].tolist(), ["No comparable reports", "10+ reports", "Limited reports", "Too few reports"])
        fig = monthly_figure(table, 7.32)
        self.assertEqual(len(fig.axes[0].patches), 3)
        self.assertEqual(fig.axes[0].patches[-1].get_hatch(), "///")
        pd.testing.assert_frame_equal(frame, original)

    def test_negative_months_not_clipped(self):
        table = monthly_table({"persistence": pd.DataFrame({"Month": ["2026-04-01"], "SavingPct": [-9.], "SupportedRows": [7], "PropellingHours": [165.]})}, {})
        fig = monthly_figure(table, -9)
        self.assertLess(fig.axes[0].get_ylim()[0], -9)

    def test_raw_relationships_only_pre_dd(self):
        frame = pd.DataFrame({"Period": ["Before DD", "Before DD", "After DD", "Before DD"],
            "STW": [15, 16, 25, 0], "DisplacementMT": [100, 110, 200, 130], "FOC_MT_Day": [30, 35, 1000, 20]})
        self.assertEqual(training_rows(frame).index.tolist(), [0, 1])
        self.assertIsNone(raw_correlation(training_rows(frame), "STW"))
        self.assertIsNone(conditional_slice(pd.DataFrame(), {}, "STW"))

    def test_actual_expected_chart_has_no_change_reference(self):
        assessed = pd.DataFrame({
            "Date": pd.to_datetime(["2026-03-01", "2026-03-02"]),
            "Voyage": ["001E", "001E"],
            "ServiceLeg": ["Outbound", "Outbound"],
            "STW": [15.0, 16.0],
            "DisplacementMT": [120000.0, 125000.0],
            "FOC_MT_Day": [60.0, 75.0],
            "ExpectedPreDDCondition_FOC_MT_Day": [65.0, 70.0],
            "InsideSupport": [True, False],
        })
        figure = _actual_expected_scatter(assessed)
        self.assertIsNotNone(figure)
        self.assertEqual(len(figure.layout.shapes), 1)
        self.assertEqual(len(figure.data), 2)

    def test_assessment_story_uses_run_values(self):
        summary = fuel_summary(fixture())
        result = {
            "before_rows": 100,
            "supported_after_rows": 20,
            "selected_ml_model": "Huber ML - STW",
        }
        figure = assessment_story_figure(summary, result, "Usable with stated limitations")
        labels = [item.get_text() for item in figure.axes[0].texts]
        self.assertTrue(any("100 pre-DD" in value for value in labels))
        self.assertTrue(any("Huber ML\nSTW only" in value for value in labels))
        self.assertTrue(any("0.00%" in value for value in labels))
        self.assertTrue(figure_png(figure).startswith(b"\x89PNG\r\n\x1a\n"))

    def test_synthetic_demo_is_labelled_and_recovers_declared_effect(self):
        saved = build_demo_assessment()
        self.assertTrue(saved["settings"]["is_demo"])
        self.assertEqual(saved["settings"]["vessel"], "SYNTHETIC DEMO VESSEL")
        self.assertTrue(saved["result"]["valid"])
        self.assertAlmostEqual(saved["result"]["improvement_pct"], 10.0, delta=1.0)
        self.assertIn("Synthetic demonstration:", printable_report(saved))
        overview = _overview_table(saved["result"], saved["settings"])
        self.assertIn("not vessel evidence", str(overview.iloc[0]["Value"]))

    def test_operating_support_comment_uses_assessment_counts_and_scope(self):
        eligible = pd.DataFrame({
            "Period": ["Before DD", "Before DD"],
            "STW": [14.0, 18.0],
            "DisplacementMT": [100000.0, 120000.0],
        })
        result = {
            "coverage_pct": 80.0,
            "assessed_rows": pd.DataFrame({
                "InsideSupport": [True, True, False],
                "STW": [15.0, 16.0, 20.0],
                "DisplacementMT": [105000.0, 110000.0, 140000.0],
            }),
            "service_leg_summary": pd.DataFrame({
                "AnalysisRoute": ["EC3"], "ServiceLeg": ["Outbound"], "FuelSharePct": [85.0]
            }),
        }
        comment = operating_support_comment(eligible, result)
        self.assertIn("2 of 3 eligible post-DD reports", comment["summary"])
        self.assertTrue(any("1 eligible post-DD report was" in item for item in comment["warnings"]))
        self.assertTrue(any("EC3 / Outbound" in item for item in comment["warnings"]))

    def test_actual_expected_comment_separates_counts_from_weighted_result(self):
        result = {
            "improvement_pct": 2.5,
            "assessed_rows": pd.DataFrame({
                "InsideSupport": [True, True, True, False],
                "FOC_MT_Day": [90.0, 100.0, 110.0, 1.0],
                "ExpectedPreDDCondition_FOC_MT_Day": [100.0, 100.0, 100.0, 999.0],
                "PropellingHours": [24.0, 24.0, 24.0, 24.0],
            }),
        }
        comment = actual_expected_comment(result)
        self.assertIn("1 report was more than 1% below", comment["summary"])
        self.assertIn("1 report was within +/-1%", comment["summary"])
        self.assertIn("1 report was more than 1% above", comment["summary"])
        self.assertTrue(any("weighted by propelling hours" in item for item in comment["warnings"]))


class AuditTests(unittest.TestCase):
    def setUp(self):
        self.raw = pd.DataFrame({"ReportID": range(1, 9), "Date": pd.to_datetime(["2026-01-01"] * 8),
            "Period": ["Before DD", "After DD", "After DD", "After DD", "After DD", "After DD", "Dry dock - excluded", "Outside analysis window"],
            "PropellingHours": [24, 24, 24, 24, 24, 10, 24, 24], "STW": [15, 15, 15, 15, 15, 5, 15, 15],
            "DisplacementMT": [100000] * 8, "FOC_MT_Day": [60] * 8, "Beaufort": [3] * 8})
        self.eligible, _ = filter_operating_rows(self.raw)
        self.result = {"valid": True, "model_details": {"fitted": True}, "assessed_rows": pd.DataFrame({
            "ReportID": [2, 3, 4], "InsideSupport": [True, False, False], "SupportThreshold": [1., 1., np.nan]})}

    def test_every_report_once_all_reasons(self):
        ledger = build_report_ledger(self.raw, self.eligible, self.result, {})
        self.assertEqual(len(ledger), 8)
        self.assertEqual(ledger["ReportID"].nunique(), 8)
        self.assertEqual(ledger.loc[0, "Assessment use"], "Pre-DD training")
        self.assertEqual(ledger.loc[1, "Assessment use"], "Post-DD comparison")
        self.assertIn("group-specific", ledger.loc[2, "Reason"])
        self.assertIn("Too few pre-DD", ledger.loc[3, "Reason"])
        self.assertEqual(ledger.loc[4, "Assessment use"], "Post-DD not assessed")
        self.assertIn("Propelling hours", ledger.loc[5, "Reason"])
        self.assertNotIn("STW", ledger.loc[5, "Reason"])
        self.assertEqual(ledger.loc[6, "Assessment use"], "Dry dock - excluded")

    def test_no_model_means_not_training(self):
        ledger = build_report_ledger(self.raw, self.eligible, {"valid": False}, {})
        self.assertEqual(ledger.loc[0, "Assessment use"], "Eligible pre-DD; model not fitted")
        self.assertEqual(ledger.loc[1, "Assessment use"], "Post-DD not assessed")

    def test_duplicate_dates_not_used_as_join_key(self):
        ledger = build_report_ledger(self.raw, self.eligible, self.result, {})
        self.assertEqual(ledger["Assessment use"].eq("Post-DD comparison").sum(), 1)

    def test_duplicate_ids_and_filter_mismatch_rejected(self):
        raw = self.raw.copy()
        raw.loc[0, "ReportID"] = 2
        with self.assertRaises(ValueError):
            build_report_ledger(raw, self.eligible, self.result, {})
        with self.assertRaises(ValueError):
            build_report_ledger(self.raw, self.eligible.iloc[1:], self.result, {})

    def test_unsupported_summary_does_not_show_contribution(self):
        assessed = pd.DataFrame({"Date": ["2026-03-01"], "InsideSupport": [False], "SavingEquivalentMT": [500.]})
        view = report_comparison(assessed)
        self.assertTrue(np.isnan(view.iloc[0]["Interval difference (MT)"]))
        self.assertEqual(assessed.iloc[0]["SavingEquivalentMT"], 500.)

    def test_foc_anomaly_is_flagged_without_changing_assessment_use(self):
        result = deepcopy(self.result)
        result["foc_anomalies"] = pd.DataFrame({
            "ReportID": [2],
            "FlagReason": ["Reported FOC is less than one quarter of expectation"],
        })
        ledger = build_report_ledger(self.raw, self.eligible, result, {})
        row = ledger.loc[ledger["ReportID"].eq(2)].iloc[0]
        self.assertEqual(row["Assessment use"], "Post-DD comparison")
        self.assertEqual(row["FOC anomaly flag"], "Review")
        self.assertIn("less than one quarter", row["FOC anomaly review"])


class SetupStateTests(unittest.TestCase):
    def test_editor_base_stays_restored_across_reruns(self):
        original = pd.DataFrame({"Fuel": ["VLSFO"], "LCV": [40.5]})
        edited = pd.DataFrame({"Fuel": ["VLSFO"], "LCV": [41.0]})
        state = {"setup_values": {"setup_lcv": edited}}
        inputs = []

        def editor(data, **kwargs):
            inputs.append(data.copy())
            state[kwargs["key"]] = {"edited_rows": {}}
            return data.copy()

        with patch("foc_setup.st.session_state", state), patch("foc_setup.st.data_editor", editor):
            ui = SetupWidgets()
            ui.data_editor(original, key="lcv")
            ui.data_editor(original, key="lcv")
        self.assertEqual([frame.iloc[0]["LCV"] for frame in inputs], [41.0, 41.0])
        self.assertEqual(original.iloc[0]["LCV"], 40.5)

    def test_editor_preserves_mapping_by_identifier_when_codes_change(self):
        state = {"setup_values": {"setup_legs": pd.DataFrame({"Rotation": ["E", "W"], "Leg": ["Outbound", "Inbound"]})}}
        current = pd.DataFrame({"Rotation": ["E", "N", "W"], "Leg": ["E", "N", "W"]})
        with patch("foc_setup.st.session_state", state), patch("foc_setup.st.data_editor", side_effect=lambda data, **kwargs: data):
            value = SetupWidgets().data_editor(current, key="legs")
        self.assertEqual(value["Leg"].tolist(), ["Outbound", "N", "Inbound"])


if __name__ == "__main__":
    unittest.main()
