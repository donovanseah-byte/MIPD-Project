import unittest

import numpy as np
import pandas as pd

from foc_model import (
    aggregate_improvement,
    assess_operating_support,
    classify_evidence,
    chronological_validation,
    detect_foc_anomalies,
    fit_cubic_speed_benchmark,
    fit_foc_model,
    run_package_assessment,
)
from foc_processing import (
    DEFAULT_FUEL_LCV,
    FUEL_COLUMNS,
    RAW_COLUMNS,
    apply_analysis_route_mapping,
    apply_service_leg_mapping,
    assign_periods,
    build_calculation_table,
    coordinate_to_decimal,
    duration_to_hours,
    filter_operating_rows,
)


class ProcessingTests(unittest.TestCase):
    def _raw_row(self):
        return {
            RAW_COLUMNS["date"]: pd.Timestamp("2026-01-01 12:00"),
            RAW_COLUMNS["latitude"]: "1-16S",
            RAW_COLUMNS["longitude"]: "12-47W",
            RAW_COLUMNS["propelling_hours"]: "25:00",
            RAW_COLUMNS["log_distance"]: 400.0,
            RAW_COLUMNS["displacement"]: 100_000.0,
            RAW_COLUMNS["beaufort"]: 2.0,
            RAW_COLUMNS["wind_speed"]: 8.0,
            RAW_COLUMNS["wind_sea_height"]: 0.5,
            RAW_COLUMNS["swell_height"]: 0.8,
            "Voyage": "0001W",
            "Service Lane": "EC3",
            "Vessel": "DEMO VESSEL",
            FUEL_COLUMNS["HSFO"]: 60.4,
        }

    def test_duration_to_hours(self):
        self.assertAlmostEqual(duration_to_hours("25:30"), 25.5)
        self.assertAlmostEqual(duration_to_hours(0.5), 12.0)

    def test_ibis_coordinates_are_converted_to_decimal_degrees(self):
        self.assertAlmostEqual(coordinate_to_decimal("1-16S", True), -(1 + 16 / 60))
        self.assertAlmostEqual(coordinate_to_decimal("12-47W", False), -(12 + 47 / 60))
        self.assertAlmostEqual(coordinate_to_decimal("35-30N", True), 35.5)
        self.assertTrue(np.isnan(coordinate_to_decimal("181-00E", False)))

    def test_lcv_formula_and_stw(self):
        result = build_calculation_table(pd.DataFrame([self._raw_row()]), DEFAULT_FUEL_LCV)
        expected = (60.4 * 40.2 / 40.5) * 24.0 / 25.0
        self.assertAlmostEqual(result.loc[0, "FOC_MT_Day"], expected, places=9)
        self.assertAlmostEqual(result.loc[0, "STW"], 16.0)
        self.assertAlmostEqual(result.loc[0, "Latitude"], -(1 + 16 / 60))
        self.assertAlmostEqual(result.loc[0, "Longitude"], -(12 + 47 / 60))

    def test_missing_unused_fuel_columns_are_allowed(self):
        result = build_calculation_table(pd.DataFrame([self._raw_row()]), DEFAULT_FUEL_LCV)
        self.assertEqual(result.loc[0, "FuelDataStatus"], "Valid")

    def test_missing_weather_columns_are_allowed(self):
        row = self._raw_row()
        del row[RAW_COLUMNS["wind_speed"]]
        del row[RAW_COLUMNS["wind_sea_height"]]
        del row[RAW_COLUMNS["swell_height"]]
        result = build_calculation_table(pd.DataFrame([row]), DEFAULT_FUEL_LCV)
        self.assertEqual(result.loc[0, "FuelDataStatus"], "Valid")
        self.assertTrue(np.isnan(result.loc[0, "WindSpeedKn"]))

    def test_dock_out_date_is_excluded(self):
        frame = pd.DataFrame(
            {"Date": [pd.Timestamp("2026-02-10 12:00"), pd.Timestamp("2026-02-11 12:00")]}
        )
        result = assign_periods(
            frame,
            "2026-01-21",
            "2026-02-10",
            "2025-01-21",
            "2026-05-10",
        )
        self.assertEqual(result.loc[0, "Period"], "Dry dock - excluded")
        self.assertEqual(result.loc[1, "Period"], "After DD")

    def test_analysis_route_preserves_service_lane(self):
        frame = pd.DataFrame({"ServiceLane": ["EC5", "EC2", "Other"]})
        result = apply_analysis_route_mapping(frame, {"EC5": "Asia-Europe", "EC2": "Asia-Europe"})
        self.assertEqual(result["ServiceLane"].tolist(), ["EC5", "EC2", "Other"])
        self.assertEqual(result["AnalysisRoute"].tolist(), ["Asia-Europe", "Asia-Europe", "Other"])

    def test_service_leg_mapping_preserves_detected_rotation(self):
        frame = pd.DataFrame({"Rotation": ["E", "W", "Unknown"]})
        result = apply_service_leg_mapping(
            frame, {"E": "Outbound", "W": "Inbound", "Unknown": "Ad-hoc"}
        )
        self.assertEqual(result["Rotation"].tolist(), ["E", "W", "Unknown"])
        self.assertEqual(result["ServiceLeg"].tolist(), ["Outbound", "Inbound", "Ad-hoc"])

    def test_old_13_to_25_knot_band_is_not_an_operating_filter(self):
        frame = pd.DataFrame(
            {
                "Date": pd.to_datetime(["2026-01-01", "2026-01-02"]),
                "Period": ["Before DD", "Before DD"],
                "PropellingHours": [24.0, 24.0],
                "STW": [12.0, 26.0],
                "DisplacementMT": [100_000.0, 100_000.0],
                "FOC_MT_Day": [50.0, 100.0],
                "Beaufort": [3.0, 3.0],
                "IntervalAlignmentStatus": ["Internally consistent", "Internally consistent"],
            }
        )
        eligible, report = filter_operating_rows(frame)
        self.assertEqual(eligible["STW"].tolist(), [12.0, 26.0])
        self.assertEqual(report["eligible_rows"], 2)

    def test_impossible_propelling_interval_is_excluded(self):
        first = self._raw_row()
        first[RAW_COLUMNS["date"]] = pd.Timestamp("2026-01-01 12:00")
        first[RAW_COLUMNS["propelling_hours"]] = 24.0
        second = dict(first)
        second[RAW_COLUMNS["date"]] = pd.Timestamp("2026-01-02 12:00")
        second[RAW_COLUMNS["propelling_hours"]] = 30.0
        calculated = build_calculation_table(pd.DataFrame([first, second]), DEFAULT_FUEL_LCV)
        calculated["Period"] = "Before DD"
        eligible, report = filter_operating_rows(calculated)
        self.assertEqual(calculated.loc[1, "IntervalAlignmentStatus"], "Propelling hours exceed elapsed interval")
        self.assertEqual(len(eligible), 1)
        self.assertEqual(report["interval_mismatch_rows"], 1)


class ModelTests(unittest.TestCase):
    @staticmethod
    def synthetic_data():
        rng = np.random.default_rng(42)
        start = pd.Timestamp("2024-01-01")
        dock_in = start + pd.Timedelta(days=365)
        dock_out = dock_in + pd.Timedelta(days=14)
        rows = []
        for i in range(470):
            date = start + pd.Timedelta(days=i)
            if dock_in <= date <= dock_out:
                continue
            stw = 13.2 + 0.12 * (i % 30)
            displacement = 84_000.0 + 1_500.0 * (i % 9)
            beaufort = float(i % 5)
            wind_speed = 5.0 + 2.0 * beaufort
            base = np.exp(
                -4.25
                + 2.85 * np.log(stw)
                + 0.35 * np.log(displacement / 90_000.0)
                + 0.003 * wind_speed
                + 0.018 * (0.4 + 0.1 * beaufort)
                + 0.012 * (0.7 + 0.1 * (i % 3))
            )
            noise = np.exp(rng.normal(0.0, 0.012))
            if date < dock_in:
                foc = base * noise
                period = "Before DD"
            else:
                foc = base * 0.90 * noise
                period = "After DD"
            rows.append(
                {
                    "Date": date,
                    "Period": period,
                    "STW": stw,
                    "DisplacementMT": displacement,
                    "Beaufort": beaufort,
                    "WindSpeedKn": wind_speed,
                    "WindSeaHeightM": 0.4 + 0.1 * beaufort,
                    "SwellHeightM": 0.7 + 0.1 * (i % 3),
                    "TrimM": 0.1 * (i % 2),
                    "FOC_MT_Day": foc,
                    "PropellingHours": 24.0,
                    "Voyage": f"{i // 20:04d}W",
                    "ServiceLane": "EC3",
                    "AnalysisRoute": "Asia-Europe",
                    "Rotation": "W",
                    "ServiceLeg": "Outbound" if (i // 20) % 2 == 0 else "Inbound",
                }
            )
        return pd.DataFrame(rows)

    def test_interval_weighted_improvement(self):
        self.assertAlmostEqual(aggregate_improvement([90, 90], [100, 100], [12, 24]), 10.0)

    def test_chronological_validation_has_no_future_leakage(self):
        data = self.synthetic_data()
        before = data[data["Period"] == "Before DD"]
        validation = chronological_validation(before)
        self.assertGreater(validation["folds"], 0)
        self.assertIsNotNone(validation["huber"]["mape_pct"])
        predictions = validation["predictions"]
        self.assertTrue((predictions["TrainingEnd"] < predictions["Date"]).all())
        self.assertIn("cubic", validation)
        self.assertIn("CubicSpeedPrediction", predictions.columns)

    def test_public_benchmark_uses_cubic_speed_relationship(self):
        training = pd.DataFrame(
            {"STW": [10.0, 11.0, 12.0, 13.0, 14.0],
             "FOC_MT_Day": [0.04 * speed ** 3 for speed in [10.0, 11.0, 12.0, 13.0, 14.0]]}
        )
        benchmark = fit_cubic_speed_benchmark(training)
        predicted = benchmark.predict(pd.DataFrame({"STW": [10.0, 20.0]}))
        self.assertAlmostEqual(predicted[1] / predicted[0], 8.0, places=10)

    def test_package_assessment_recovers_saving_and_placebos(self):
        progress_events = []
        result = run_package_assessment(
            self.synthetic_data(),
            period_coverage_ok=True,
            route_confirmed=True,
            placebo_count=5,
            progress_callback=lambda percent, message: progress_events.append((percent, message)),
        )
        self.assertTrue(result["valid"])
        self.assertAlmostEqual(result["improvement_pct"], 10.0, delta=1.0)
        self.assertGreaterEqual(result["supported_after_rows"], 10)
        self.assertEqual(result["comparison_basis"], "Strict same-route")
        self.assertGreaterEqual(result["placebo"]["valid_count"], 3)
        self.assertTrue(result["placebo"]["separated_from_placebos"])
        self.assertEqual(progress_events[-1][0], 100)
        self.assertTrue(any("Selected" in message for _, message in progress_events))
        self.assertTrue(any("Validation completed" in message for _, message in progress_events))
        self.assertEqual(result["selected_ml_model"], "Huber ML - STW and displacement")
        self.assertEqual(len(result["ml_candidate_comparison"]), 2)
        self.assertIsNotNone(result["alternative_ml_saving_pct"])
        self.assertEqual(result["stability_status"], "Stable")

    def test_negative_displacement_candidate_falls_back_to_speed_only_huber(self):
        data = self.synthetic_data()
        data["FOC_MT_Day"] *= np.power(data["DisplacementMT"] / 90_000.0, -0.8)
        result = run_package_assessment(
            data,
            period_coverage_ok=True,
            route_confirmed=True,
            placebo_count=3,
        )
        self.assertTrue(result["valid"])
        self.assertEqual(result["selected_ml_model"], "Huber ML - STW")
        self.assertEqual(result["numeric_features"], ["LogSTW"])
        self.assertIn("non-negative displacement effect", result["model_selection_reason"])

    def test_cross_route_fallback_keeps_same_operating_leg(self):
        frame = self.synthetic_data()
        frame.loc[frame["Period"].eq("After DD"), "AnalysisRoute"] = "NEW SERVICE"
        result = run_package_assessment(
            frame,
            period_coverage_ok=True,
            route_confirmed=True,
            service_cycle_confirmed=True,
            minimum_train_rows=60,
            minimum_supported_after=10,
            placebo_count=3,
        )
        self.assertTrue(result["valid"])
        self.assertEqual(result["strict_supported_after_rows"], 0)
        self.assertGreaterEqual(result["expanded_supported_after_rows"], 10)
        self.assertEqual(result["comparison_basis"], "Expanded cross-route")
        self.assertEqual(result["evidence_tier"], "Preliminary")

    def test_speed_only_prediction_still_checks_displacement_support(self):
        before = self.synthetic_data().query("Period == 'Before DD'").copy()
        model = fit_foc_model(before, numeric_features=["LogSTW"])
        target = before.iloc[[0]].copy()
        target["DisplacementMT"] = 500_000.0
        assessed = assess_operating_support(before, target, model)
        self.assertFalse(bool(assessed.iloc[0]["InsideSupport"]))

    def test_unconfirmed_service_cycle_cannot_receive_indicative_or_supported_tier(self):
        result = run_package_assessment(
            self.synthetic_data(),
            period_coverage_ok=True,
            route_confirmed=True,
            service_cycle_confirmed=False,
            placebo_count=5,
        )
        self.assertTrue(result["valid"])
        self.assertEqual(result["evidence_tier"], "Preliminary")
        self.assertTrue(any("complete normal service cycle" in item for item in result["limitations"]))

    def test_incomplete_period_or_route_confirmation_caps_result_at_preliminary(self):
        for period_ok, route_ok in [(False, True), (True, False)]:
            result = run_package_assessment(
                self.synthetic_data(),
                period_coverage_ok=period_ok,
                route_confirmed=route_ok,
                service_cycle_confirmed=True,
                placebo_count=3,
            )
            self.assertTrue(result["valid"])
            self.assertEqual(result["evidence_tier"], "Preliminary")

    def test_gross_foc_screen_flags_extreme_report_without_deleting_it(self):
        before = self.synthetic_data().query("Period == 'Before DD'").copy()
        model = fit_foc_model(before, numeric_features=["LogSTW"])
        target = before.iloc[[0]].copy()
        target["ReportID"] = 999
        target["FOC_MT_Day"] = target["FOC_MT_Day"] / 100.0
        flagged = detect_foc_anomalies(before, target, model)
        self.assertEqual(len(flagged), 1)
        self.assertEqual(flagged.iloc[0]["ReportID"], 999)
        self.assertLess(flagged.iloc[0]["ActualExpectedRatio"], 0.25)

    def test_unstable_direction_has_its_own_evidence_tier(self):
        result = {
            "valid": True,
            "validation": {
                "huber": {"mape_pct": 5.0, "bias_pct": 0.0},
                "huber_outperforms_cubic": True,
            },
            "coverage_pct": 90.0,
            "supported_after_rows": 30,
            "comparison_basis": "Strict same-route",
            "learned_speed_exponent": 2.8,
            "coefficients": pd.DataFrame(),
            "placebo": {},
            "stability_status": "Unstable",
            "flagged_foc_rows": 0,
        }
        tier, limitations = classify_evidence(result, True, True, True)
        self.assertEqual(tier, "Unstable")
        self.assertTrue(any("changed direction" in item for item in limitations))

    def test_failed_prediction_validation_is_inconclusive_not_preliminary(self):
        result = {
            "valid": True,
            "validation": {
                "huber": {"mape_pct": 125.6, "bias_pct": 0.7},
                "huber_outperforms_cubic": False,
            },
            "coverage_pct": 92.2,
            "supported_after_rows": 18,
            "comparison_basis": "Strict same-route",
            "learned_speed_exponent": 2.52,
            "coefficients": pd.DataFrame(),
            "placebo": {},
            "stability_status": "Sensitive but direction stable",
            "flagged_foc_rows": 1,
        }
        tier, limitations = classify_evidence(result, True, True, True)
        self.assertEqual(tier, "Inconclusive")
        self.assertTrue(any("MAPE" in item for item in limitations))
        self.assertTrue(any("not suitable for interpretation" in item for item in limitations))

    def test_all_five_evidence_tiers_remain_reachable(self):
        base = {
            "valid": True,
            "validation": {
                "huber": {"mape_pct": 5.0, "bias_pct": 0.0},
                "huber_outperforms_cubic": True,
            },
            "coverage_pct": 90.0,
            "supported_after_rows": 30,
            "comparison_basis": "Strict same-route",
            "learned_speed_exponent": 2.8,
            "coefficients": pd.DataFrame(),
            "placebo": {"valid_count": 3, "separated_from_placebos": True},
            "stability_status": "Stable",
            "flagged_foc_rows": 0,
        }
        self.assertEqual(classify_evidence(base, True, True, True)[0], "Supported (prototype screening)")
        indicative = dict(base, supported_after_rows=25)
        self.assertEqual(classify_evidence(indicative, True, True, True)[0], "Indicative")
        preliminary = dict(base, comparison_basis="Expanded cross-route")
        self.assertEqual(classify_evidence(preliminary, True, True, True)[0], "Preliminary")
        unstable = dict(base, stability_status="Unstable")
        self.assertEqual(classify_evidence(unstable, True, True, True)[0], "Unstable")
        inconclusive = dict(base, validation={
            "huber": {"mape_pct": 13.0, "bias_pct": 0.0},
            "huber_outperforms_cubic": True,
        })
        self.assertEqual(classify_evidence(inconclusive, True, True, True)[0], "Inconclusive")

    def test_model_uses_only_declared_ml_predictors(self):
        before = self.synthetic_data().query("Period == 'Before DD'")
        model = fit_foc_model(before)
        self.assertEqual(
            model.numeric_features,
            ["LogSTW", "LogDisplacementRatio"],
        )
        self.assertGreater(model.smearing_factor, 0.0)
        self.assertEqual(
            model.coefficient_table()["Feature"].tolist(),
            ["LogSTW", "LogDisplacementRatio"],
        )

    def test_weather_changes_do_not_change_primary_predictions(self):
        before = self.synthetic_data().query("Period == 'Before DD'").copy()
        model = fit_foc_model(before)
        sample = before.iloc[:8].copy()
        original = model.predict(sample)
        sample["WindSpeedKn"] = 99.0
        sample["WindSeaHeightM"] = 9.0
        sample["SwellHeightM"] = 9.0
        changed = model.predict(sample)
        np.testing.assert_allclose(original, changed)


if __name__ == "__main__":
    unittest.main()
