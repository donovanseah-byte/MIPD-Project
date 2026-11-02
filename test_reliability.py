"""Run with: python -m unittest test_reliability.py -v"""
import copy
import unittest
from foc_quality import assess_reliability, quality_level, placebo_signal

SETTINGS = {
    'minimum_train_rows': 60,
    'interval_alignment_confirmed': True,
    'period_coverage_ok': True,
    'route_confirmed': True,
    'service_cycle_confirmed': True,
}
RESULT = {
    'valid': True,
    'improvement_pct': 5.9,
    'evidence_tier': 'Indicative',
    'before_rows': 80,
    'supported_after_rows': 25,
    'strict_supported_after_rows': 25,
    'expanded_supported_after_rows': 30,
    'comparison_basis': 'Strict same-route',
    'coverage_pct': 65.0,
    'validation': {'huber': {'mape_pct': 9.5, 'bias_pct': 3.0}},
    'learned_speed_exponent': 2.8,
    'stability_status': 'Stable',
    'flagged_foc_rows': 0,
    'placebo': {'valid_count': 3, 'separated_from_placebos': True},
}

class TestReliability(unittest.TestCase):
    def grade(self, **updates):
        r = copy.deepcopy(RESULT)
        r.update(updates)
        return assess_reliability(r, SETTINGS)['tier']

    def test_supported_positive(self):
        self.assertEqual(self.grade(improvement_pct=5.9), 'Supported Assessment')

    def test_supported_negative(self):
        self.assertEqual(self.grade(improvement_pct=-15.0), 'Supported Assessment')

    def test_supported_crossroute(self):
        self.assertEqual(self.grade(comparison_basis='Expanded cross-route'), 'Supported Assessment')

    def test_old_supported_not_automatically_supported(self):
        self.assertEqual(self.grade(evidence_tier='Supported (prototype screening)', strict_supported_after_rows=12, supported_after_rows=12), 'Evidence-Limited')

    def test_mape_failure_is_inconclusive(self):
        self.assertEqual(self.grade(validation={'huber': {'mape_pct': 120.0, 'bias_pct': 3.0}}), 'Inconclusive')

    def test_low_coverage_is_inconclusive(self):
        self.assertEqual(self.grade(coverage_pct=39.9), 'Inconclusive')

    def test_middle_coverage_limited(self):
        self.assertEqual(self.grade(coverage_pct=59.9), 'Evidence-Limited')

    def test_inadequate_after_reports_limited(self):
        self.assertEqual(self.grade(supported_after_rows=15,strict_supported_after_rows=15), 'Evidence-Limited')

    def test_legacy_critical_is_preserved(self):
        self.assertEqual(self.grade(evidence_tier='Inconclusive'), 'Inconclusive')

    def test_unstable_not_supported(self):
        self.assertEqual(self.grade(stability_status='Unstable'), 'Evidence-Limited')

    def test_unconfirmed_route_not_supported(self):
        s={**SETTINGS,'route_confirmed':False}
        self.assertEqual(assess_reliability(RESULT,s)['tier'], 'Evidence-Limited')

    def test_unconfirmed_intervals_inconclusive(self):
        s={**SETTINGS,'interval_alignment_confirmed':False}
        self.assertEqual(assess_reliability(RESULT,s)['tier'], 'Inconclusive')

    def test_placebo_not_a_data_quality_gate(self):
        self.assertEqual(self.grade(placebo={'valid_count': 5, 'separated_from_placebos': False}), 'Supported Assessment')

    def test_two_sided_effect_signal_for_negative_change(self):
        import pandas as pd
        p={'valid_count': 5, 'results': pd.DataFrame({'PlaceboSavingPct': [-1.0, 1.0, 2.0, -2.5, 0.7]})}
        r={**RESULT,'improvement_pct': -3.0,'placebo': p}
        self.assertEqual(placebo_signal(r)['status'], 'Magnitude exceeds placebos')
        r['improvement_pct'] = -1.0
        self.assertEqual(placebo_signal(r)['status'], 'Not separated')

    def test_missing_cycle_does_not_demote_analytical_reliability(self):
        settings = {**SETTINGS, 'service_cycle_confirmed': False}
        assessment = assess_reliability(RESULT, settings)
        self.assertEqual(assessment['tier'], 'Supported Assessment')
        self.assertEqual(assessment['operational_verification'], 'Pending')
        self.assertIn('not been confirmed', assessment['operational_verification_note'])
        self.assertEqual(assessment['data_quality'], 'High')

    def test_inadequate_data_still_limited_with_cycle_pending(self):
        settings = {**SETTINGS, 'service_cycle_confirmed': False}
        result = {**RESULT, 'supported_after_rows': 11, 'strict_supported_after_rows': 11}
        self.assertEqual(assess_reliability(result, settings)['tier'], 'Evidence-Limited')

    def test_missing_period_remains_limited_even_if_cycle_pending(self):
        settings = {**SETTINGS, 'service_cycle_confirmed': False, 'period_coverage_ok': False}
        self.assertEqual(assess_reliability(RESULT, settings)['tier'], 'Evidence-Limited')

    def test_quality_level(self):
        self.assertEqual(quality_level(RESULT, SETTINGS), 'High')

if __name__ == '__main__':
    unittest.main()
