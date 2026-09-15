"""STW/displacement Huber counterfactual, validation, support and placebo methods."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np
import pandas as pd
from sklearn.linear_model import HuberRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.neighbors import NearestNeighbors


MODEL_NUMERIC = ["LogSTW", "LogDisplacementRatio"]
SPEED_ONLY_NUMERIC = ["LogSTW"]
SUPPORT_GROUPS = ["AnalysisRoute", "ServiceLeg"]


ProgressCallback = Callable[[int, str], None]


def _emit(callback: ProgressCallback | None, percent: int, message: str) -> None:
    if callback is not None:
        callback(percent, message)


def _display_metric(value: float | None) -> str:
    return f"{float(value):.2f}%" if value is not None and np.isfinite(value) else "unavailable"


@dataclass
class FittedFOCModel:
    displacement_reference: float
    numeric_features: list[str]
    medians: pd.Series
    means: pd.Series
    scales: pd.Series
    estimator: HuberRegressor
    smearing_factor: float

    def design(self, frame: pd.DataFrame) -> np.ndarray:
        work = _feature_frame(frame, self.displacement_reference)
        numeric = work[self.numeric_features].copy().fillna(self.medians)
        numeric = (numeric - self.means) / self.scales
        return numeric.to_numpy(dtype=float)

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return self.smearing_factor * np.exp(self.estimator.predict(self.design(frame)))

    @property
    def learned_speed_exponent(self) -> float:
        index = self.numeric_features.index("LogSTW")
        return float(self.estimator.coef_[index] / self.scales["LogSTW"])

    @property
    def transformed_intercept(self) -> float:
        raw_coefficients = self.estimator.coef_ / self.scales[self.numeric_features].to_numpy()
        centring = float(
            np.sum(raw_coefficients * self.means[self.numeric_features].to_numpy())
        )
        return float(self.estimator.intercept_ - centring)

    def coefficient_table(self) -> pd.DataFrame:
        rows: list[dict] = []
        for position, feature in enumerate(self.numeric_features):
            rows.append(
                {
                    "Feature": feature,
                    "Standardised coefficient": float(self.estimator.coef_[position]),
                    "Coefficient in transformed units": float(
                        self.estimator.coef_[position] / self.scales[feature]
                    ),
                }
            )
        return pd.DataFrame(rows)


def _feature_frame(frame: pd.DataFrame, displacement_reference: float) -> pd.DataFrame:
    result = frame.copy()
    if "AnalysisRoute" not in result:
        result["AnalysisRoute"] = result.get("ServiceLane", "Unknown")
    if "ServiceLeg" not in result:
        result["ServiceLeg"] = result.get("Rotation", "Unknown")
    result["LogSTW"] = np.log(pd.to_numeric(result["STW"], errors="coerce"))
    result["LogDisplacementRatio"] = np.log(
        pd.to_numeric(result["DisplacementMT"], errors="coerce") / displacement_reference
    )
    for column in SUPPORT_GROUPS:
        if column not in result:
            result[column] = "Unknown"
        result[column] = result[column].fillna("Unknown").astype(str)
    return result.replace([np.inf, -np.inf], np.nan)


def fit_foc_model(
    train: pd.DataFrame,
    minimum_rows: int = 20,
    numeric_features: list[str] | None = None,
) -> FittedFOCModel:
    """Fit log(FOC) with Huber loss using pre-DD observations only."""
    if len(train) < minimum_rows:
        raise ValueError(f"At least {minimum_rows} training rows are required; found {len(train)}.")
    displacement_reference = float(pd.to_numeric(train["DisplacementMT"], errors="coerce").median())
    if not np.isfinite(displacement_reference) or displacement_reference <= 0:
        raise ValueError("The training displacement reference is invalid.")
    work = _feature_frame(train, displacement_reference)
    numeric_features = list(MODEL_NUMERIC if numeric_features is None else numeric_features)
    unsupported = [feature for feature in numeric_features if feature not in MODEL_NUMERIC]
    if unsupported:
        raise ValueError("Unsupported model features: " + ", ".join(unsupported))
    numeric = work[numeric_features]
    medians = numeric.median(numeric_only=True)
    filled = numeric.fillna(medians)
    means = filled.mean()
    scales = filled.std(ddof=0).replace(0, 1.0).fillna(1.0)
    matrix = ((filled - means) / scales).to_numpy(dtype=float)
    target = np.log(pd.to_numeric(train["FOC_MT_Day"], errors="coerce").to_numpy(dtype=float))
    if not np.isfinite(matrix).all() or not np.isfinite(target).all():
        raise ValueError("The training matrix contains invalid values after preparation.")
    # The two-predictor specification is deliberately fixed before post-DD
    # evaluation. The small default L2 term is declared for reproducibility;
    # feature selection is not performed after observing the saving result.
    estimator = HuberRegressor(epsilon=1.35, alpha=0.0001, max_iter=3000)
    estimator.fit(matrix, target)
    # Duan's non-parametric smearing estimate corrects retransformation from
    # log(FOC) back to the arithmetic FOC scale used for interval totals.
    residuals = target - estimator.predict(matrix)
    finite_residuals = residuals[np.isfinite(residuals)]
    smearing_factor = float(np.mean(np.exp(finite_residuals)))
    if not np.isfinite(smearing_factor) or smearing_factor <= 0:
        raise ValueError("The log-scale retransformation factor is invalid.")
    return FittedFOCModel(
        displacement_reference=displacement_reference,
        numeric_features=numeric_features,
        medians=medians,
        means=means,
        scales=scales,
        estimator=estimator,
        smearing_factor=smearing_factor,
    )


@dataclass
class CubicSpeedBenchmarkModel:
    coefficient: float

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        speed = pd.to_numeric(frame["STW"], errors="coerce").to_numpy(dtype=float)
        return self.coefficient * np.power(speed, 3.0)


def fit_cubic_speed_benchmark(train: pd.DataFrame) -> CubicSpeedBenchmarkModel:
    """Fit the public cubic-speed rule-of-thumb coefficient on pre-DD data."""
    speed = pd.to_numeric(train["STW"], errors="coerce").to_numpy(dtype=float)
    foc = pd.to_numeric(train["FOC_MT_Day"], errors="coerce").to_numpy(dtype=float)
    x = np.power(speed, 3.0)
    mask = np.isfinite(x) & np.isfinite(foc) & (x > 0) & (foc > 0)
    if mask.sum() < 5:
        raise ValueError("At least five valid rows are required for the cubic-speed benchmark.")
    denominator = float(np.sum(x[mask] * x[mask]))
    if denominator <= 0:
        raise ValueError("The cubic-speed benchmark denominator is invalid.")
    return CubicSpeedBenchmarkModel(float(np.sum(x[mask] * foc[mask]) / denominator))


def _metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float | None]:
    mask = np.isfinite(actual) & np.isfinite(predicted) & (actual > 0)
    if not mask.any():
        return {"mape_pct": None, "bias_pct": None, "rmse": None}
    a = actual[mask]
    p = predicted[mask]
    return {
        "mape_pct": float(np.mean(np.abs((p - a) / a)) * 100.0),
        "bias_pct": float(np.sum(p - a) / np.sum(a) * 100.0),
        "rmse": float(np.sqrt(np.mean(np.square(p - a)))),
    }


def chronological_validation(
    frame: pd.DataFrame,
    minimum_train_rows: int = 20,
    numeric_features: list[str] | None = None,
) -> dict:
    """Compare the primary Huber model and public V^3 benchmark on later pre-DD records."""
    numeric_features = list(MODEL_NUMERIC if numeric_features is None else numeric_features)
    ordered = frame.sort_values("Date").reset_index(drop=True)
    empty = {
        "folds": 0,
        "huber": {"mape_pct": None, "bias_pct": None, "rmse": None},
        "cubic": {"mape_pct": None, "bias_pct": None, "rmse": None},
        "predictions": pd.DataFrame(),
        "huber_outperforms_cubic": None,
        "candidate_test_rows": 0,
        "supported_test_rows": 0,
        "numeric_features": numeric_features,
        "fold_coefficients": [],
    }
    if len(ordered) < minimum_train_rows + 8:
        return empty
    splits_count = min(4, max(2, (len(ordered) - minimum_train_rows) // 6))
    splitter = TimeSeriesSplit(n_splits=splits_count)
    records: list[pd.DataFrame] = []
    valid_folds = 0
    candidate_test_rows = 0
    supported_test_rows = 0
    fold_coefficients: list[dict[str, float]] = []
    for fold, (train_index, test_index) in enumerate(splitter.split(ordered), start=1):
        train = ordered.iloc[train_index]
        test = ordered.iloc[test_index]
        if len(train) < minimum_train_rows or len(test) < 2:
            continue
        try:
            huber = fit_foc_model(
                train,
                minimum_rows=minimum_train_rows,
                numeric_features=numeric_features,
            )
            cubic = fit_cubic_speed_benchmark(train)
            candidate_test_rows += int(len(test))
            assessed_test = assess_operating_support(
                train, test, huber, comparison_mode="expanded"
            )
            test = assessed_test[assessed_test["InsideSupport"]].copy()
            if len(test) < 2:
                continue
            supported_test_rows += int(len(test))
            huber_prediction = huber.predict(test)
            cubic_prediction = cubic.predict(test)
        except Exception:
            continue
        part = test[["Date", "Voyage", "FOC_MT_Day", "STW"]].copy()
        part["HuberPrediction"] = huber_prediction
        part["CubicSpeedPrediction"] = cubic_prediction
        part["Fold"] = fold
        part["TrainingEnd"] = pd.to_datetime(train["Date"]).max()
        records.append(part)
        fold_coefficients.append(
            huber.coefficient_table()
            .set_index("Feature")["Coefficient in transformed units"]
            .astype(float)
            .to_dict()
        )
        valid_folds += 1
    if not records:
        return empty
    predictions = pd.concat(records, ignore_index=True)
    actual = predictions["FOC_MT_Day"].to_numpy(dtype=float)
    huber_metrics = _metrics(actual, predictions["HuberPrediction"].to_numpy(dtype=float))
    cubic_metrics = _metrics(actual, predictions["CubicSpeedPrediction"].to_numpy(dtype=float))
    huber_mape = huber_metrics["mape_pct"]
    cubic_mape = cubic_metrics["mape_pct"]
    return {
        "folds": valid_folds,
        "huber": huber_metrics,
        "cubic": cubic_metrics,
        "predictions": predictions,
        "candidate_test_rows": candidate_test_rows,
        "supported_test_rows": supported_test_rows,
        "numeric_features": numeric_features,
        "fold_coefficients": fold_coefficients,
        "huber_outperforms_cubic": (
            bool(huber_mape <= cubic_mape)
            if huber_mape is not None and cubic_mape is not None
            else None
        ),
    }


def select_huber_specification(
    before: pd.DataFrame,
    minimum_train_rows: int,
) -> dict:
    """Select an ML specification using pre-DD evidence only.

    The speed-and-displacement model is selected only when its displacement
    effect is non-negative in the full pre-DD fit and every valid chronological
    fold, it improves both MAPE and RMSE over speed-only Huber, and its absolute
    bias remains inside the declared 8% project gate. Otherwise the simpler
    speed-only Huber is selected. The post-DD
    saving is never used in this decision.
    """
    speed_model = fit_foc_model(
        before,
        minimum_rows=minimum_train_rows,
        numeric_features=SPEED_ONLY_NUMERIC,
    )
    two_model = fit_foc_model(
        before,
        minimum_rows=minimum_train_rows,
        numeric_features=MODEL_NUMERIC,
    )
    speed_validation = chronological_validation(
        before,
        minimum_train_rows,
        numeric_features=SPEED_ONLY_NUMERIC,
    )
    two_validation = chronological_validation(
        before,
        minimum_train_rows,
        numeric_features=MODEL_NUMERIC,
    )

    two_coefficients = (
        two_model.coefficient_table()
        .set_index("Feature")["Coefficient in transformed units"]
        .astype(float)
    )
    full_displacement = float(two_coefficients.get("LogDisplacementRatio", np.nan))
    fold_displacements = [
        float(item["LogDisplacementRatio"])
        for item in two_validation.get("fold_coefficients", [])
        if item.get("LogDisplacementRatio") is not None
        and np.isfinite(item.get("LogDisplacementRatio"))
    ]
    engineering_ok = bool(
        np.isfinite(full_displacement)
        and full_displacement >= 0
        and fold_displacements
        and all(value >= 0 for value in fold_displacements)
    )

    def metric(validation: dict, name: str) -> float | None:
        value = validation.get("huber", {}).get(name)
        return float(value) if value is not None and np.isfinite(value) else None

    speed_mape, two_mape = metric(speed_validation, "mape_pct"), metric(two_validation, "mape_pct")
    speed_rmse, two_rmse = metric(speed_validation, "rmse"), metric(two_validation, "rmse")
    speed_bias, two_bias = metric(speed_validation, "bias_pct"), metric(two_validation, "bias_pct")
    performance_ok = bool(
        None not in (speed_mape, two_mape, speed_rmse, two_rmse, speed_bias, two_bias)
        and two_mape <= speed_mape
        and two_rmse <= speed_rmse
        and abs(two_bias) <= 8
    )

    if engineering_ok and performance_ok:
        selected_key = "stw_displacement"
        selected_model = two_model
        selected_validation = two_validation
        alternative_model = speed_model
        reason = (
            "STW-and-displacement Huber was selected using pre-DD data only: its loading effect "
            "was non-negative in the full fit and every valid chronological fold, and it was no "
            "worse than speed-only Huber on MAPE and RMSE, while absolute bias remained inside "
            "the declared project gate."
        )
    else:
        selected_key = "stw_only"
        selected_model = speed_model
        selected_validation = speed_validation
        alternative_model = two_model
        if not engineering_ok:
            reason = (
                "Speed-only Huber was selected because the STW-and-displacement candidate did "
                "not produce a consistently non-negative displacement effect in the pre-DD fit "
                "and chronological validation folds. Displacement remains part of the operating-"
                "support check."
            )
        else:
            reason = (
                "Speed-only Huber was selected because adding displacement did not improve both "
                "MAPE and RMSE while keeping absolute bias inside the declared project gate during "
                "pre-DD chronological validation. "
                "Displacement remains part of the operating-support check."
            )

    rows = []
    for key, label, validation, model in [
        ("stw_only", "Huber ML - STW", speed_validation, speed_model),
        ("stw_displacement", "Huber ML - STW and displacement", two_validation, two_model),
    ]:
        coefficients = (
            model.coefficient_table()
            .set_index("Feature")["Coefficient in transformed units"]
            .astype(float)
        )
        displacement = coefficients.get("LogDisplacementRatio", np.nan)
        review = (
            "Not included"
            if not np.isfinite(displacement)
            else "Acceptable" if float(displacement) >= 0 else "Negative displacement effect"
        )
        rows.append(
            {
                "ML specification": label,
                "MAPE (%)": metric(validation, "mape_pct"),
                "Bias (%)": metric(validation, "bias_pct"),
                "RMSE (FOC MT/day)": metric(validation, "rmse"),
                "Displacement review": review,
                "Decision": "Selected" if key == selected_key else "Sensitivity only",
            }
        )
    return {
        "selected_key": selected_key,
        "selected_label": (
            "Huber ML - STW" if selected_key == "stw_only"
            else "Huber ML - STW and displacement"
        ),
        "selected_model": selected_model,
        "selected_validation": selected_validation,
        "alternative_model": alternative_model,
        "alternative_label": (
            "Huber ML - STW and displacement" if selected_key == "stw_only"
            else "Huber ML - STW"
        ),
        "selection_reason": reason,
        "candidate_table": pd.DataFrame(rows),
        "candidate_validations": {
            "stw_only": speed_validation,
            "stw_displacement": two_validation,
        },
    }


def assess_operating_support(
    train: pd.DataFrame,
    target: pd.DataFrame,
    model: FittedFOCModel,
    percentile: float = 95.0,
    k_neighbors: int = 3,
    comparison_mode: str = "strict",
) -> pd.DataFrame:
    """Mark rows inside vessel-specific pre-DD STW/displacement support.

    ``strict`` requires the same confirmed route and operating leg. ``expanded``
    allows a different route but retains the same confirmed operating leg. Both
    modes use the same pre-DD-only local-density test in log(STW) and loading.
    """
    if comparison_mode not in {"strict", "expanded"}:
        raise ValueError("comparison_mode must be 'strict' or 'expanded'.")
    result = target.copy()
    result["InsideSupport"] = False
    result["SupportDistance"] = np.nan
    result["SupportThreshold"] = np.nan
    result["SupportGroupRows"] = 0
    result["SupportMode"] = comparison_mode
    if target.empty:
        return result
    train_features = _feature_frame(train, model.displacement_reference)
    target_features = _feature_frame(target, model.displacement_reference)
    # Operating support always uses both speed and displacement, even when the
    # selected prediction model is speed-only. This prevents a simpler model
    # from treating an unrepresented loading condition as comparable.
    support_columns = list(MODEL_NUMERIC)
    support_numeric = train_features[support_columns]
    medians = support_numeric.median(numeric_only=True)
    filled_support = support_numeric.fillna(medians)
    means = filled_support.mean()
    scales = filled_support.std(ddof=0).replace(0, 1.0).fillna(1.0)
    train_numeric = (train_features[support_columns].fillna(medians) - means) / scales
    target_numeric = (target_features[support_columns].fillna(medians) - means) / scales

    group_columns = SUPPORT_GROUPS if comparison_mode == "strict" else "ServiceLeg"
    groups = target_features.groupby(group_columns, dropna=False).groups
    for key, target_indexes in groups.items():
        if comparison_mode == "strict":
            analysis_route, service_leg = key if isinstance(key, tuple) else (key, "Unknown")
            train_mask = (
                (train_features["AnalysisRoute"] == analysis_route)
                & (train_features["ServiceLeg"] == service_leg)
            )
        else:
            service_leg = key
            train_mask = train_features["ServiceLeg"] == service_leg
        train_indexes = train_features.index[train_mask]
        result.loc[target_indexes, "SupportGroupRows"] = int(len(train_indexes))
        if len(train_indexes) < max(6, k_neighbors + 2):
            continue
        train_matrix = train_numeric.loc[train_indexes].to_numpy(dtype=float)
        target_matrix = target_numeric.loc[target_indexes].to_numpy(dtype=float)
        k = min(k_neighbors, len(train_matrix) - 1)
        within = NearestNeighbors(n_neighbors=k + 1).fit(train_matrix)
        within_distances, _ = within.kneighbors(train_matrix)
        within_scores = within_distances[:, 1:].mean(axis=1)
        threshold = float(np.quantile(within_scores, percentile / 100.0))
        neighbours = NearestNeighbors(n_neighbors=k).fit(train_matrix)
        distances, _ = neighbours.kneighbors(target_matrix)
        scores = distances.mean(axis=1)
        result.loc[target_indexes, "SupportDistance"] = scores
        result.loc[target_indexes, "SupportThreshold"] = threshold
        result.loc[target_indexes, "InsideSupport"] = scores <= threshold
    return result


def aggregate_improvement(
    actual_foc: Iterable[float],
    expected_foc: Iterable[float],
    hours: Iterable[float],
) -> float:
    actual = np.asarray(list(actual_foc), dtype=float)
    expected = np.asarray(list(expected_foc), dtype=float)
    weights = np.asarray(list(hours), dtype=float) / 24.0
    mask = (
        np.isfinite(actual)
        & np.isfinite(expected)
        & np.isfinite(weights)
        & (actual > 0)
        & (expected > 0)
        & (weights > 0)
    )
    if not mask.any():
        return np.nan
    expected_total = float(np.sum(expected[mask] * weights[mask]))
    actual_total = float(np.sum(actual[mask] * weights[mask]))
    return float((1.0 - actual_total / expected_total) * 100.0) if expected_total > 0 else np.nan


def _fuel_coverage_pct(assessed: pd.DataFrame) -> float:
    if assessed.empty:
        return 0.0
    weights = assessed["FOC_MT_Day"] * assessed["PropellingHours"] / 24.0
    total = float(weights.sum())
    supported = float(weights[assessed["InsideSupport"]].sum())
    return supported / total * 100.0 if total > 0 else 0.0


def _service_leg_scope(supported: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Describe which route/leg combinations actually contribute to the estimate."""
    if supported.empty:
        return pd.DataFrame(), "No supported post-DD operating scope was available."
    work = supported.copy()
    if "ServiceLeg" not in work:
        work["ServiceLeg"] = work.get("Rotation", "Unknown")
    work["SupportedFuelMT"] = work["FOC_MT_Day"] * work["PropellingHours"] / 24.0
    grouped = (
        work.groupby(["AnalysisRoute", "ServiceLeg"], dropna=False)
        .agg(Rows=("Date", "size"), SupportedFuelMT=("SupportedFuelMT", "sum"))
        .reset_index()
        .sort_values("SupportedFuelMT", ascending=False)
        .reset_index(drop=True)
    )
    total = float(grouped["SupportedFuelMT"].sum())
    grouped["FuelSharePct"] = np.where(
        total > 0, grouped["SupportedFuelMT"] / total * 100.0, np.nan
    )
    dominant = grouped.iloc[0]
    if len(grouped) == 1:
        scope = (
            f"The estimate applies to supported {dominant['AnalysisRoute']} / "
            f"{dominant['ServiceLeg']} observations only."
        )
    elif float(dominant["FuelSharePct"]) >= 80.0:
        scope = (
            f"The estimate is dominated by {dominant['AnalysisRoute']} / "
            f"{dominant['ServiceLeg']} ({float(dominant['FuelSharePct']):.1f}% of supported fuel)."
        )
    else:
        scope = "The estimate covers the route/service-leg combinations listed in the scope table."
    return grouped, scope


def run_placebo_analysis(
    before: pd.DataFrame,
    actual_effect_pct: float | None,
    desired_placebos: int = 5,
    minimum_train_rows: int = 20,
    minimum_test_rows: int = 5,
    evaluation_rows: int = 12,
    numeric_features: list[str] | None = None,
) -> dict:
    """Run repeated in-time pseudo-interventions entirely within pre-DD data."""
    ordered = before.sort_values("Date").reset_index(drop=True)
    max_evaluation = max(minimum_test_rows, min(evaluation_rows, max(0, len(ordered) // 3)))
    latest_split = len(ordered) - max_evaluation
    if latest_split < minimum_train_rows:
        return {
            "valid_count": 0,
            "results": pd.DataFrame(),
            "actual_percentile": None,
            "empirical_p_one_sided": None,
            "separated_from_placebos": None,
            "reason": "Insufficient pre-DD rows for repeated placebo splits.",
        }
    candidate_positions = np.linspace(
        minimum_train_rows,
        latest_split,
        num=min(desired_placebos, max(1, latest_split - minimum_train_rows + 1)),
        dtype=int,
    )
    candidate_positions = sorted(set(int(position) for position in candidate_positions))
    rows: list[dict] = []
    for position in candidate_positions:
        train = ordered.iloc[:position].copy()
        pseudo = ordered.iloc[position : position + max_evaluation].copy()
        if len(train) < minimum_train_rows or len(pseudo) < minimum_test_rows:
            continue
        record = {
            "PlaceboDate": pd.to_datetime(pseudo["Date"]).min(),
            "TrainingRows": int(len(train)),
            "PseudoPostRows": int(len(pseudo)),
            "SupportedRows": 0,
            "CoveragePct": 0.0,
            "PlaceboSavingPct": None,
            "Status": "Invalid",
        }
        try:
            model = fit_foc_model(
                train,
                minimum_rows=minimum_train_rows,
                numeric_features=numeric_features,
            )
            assessed = assess_operating_support(
                train, pseudo, model, comparison_mode="expanded"
            )
            assessed["ExpectedNoInterventionFOC"] = model.predict(assessed)
            supported = assessed[assessed["InsideSupport"]].copy()
            record["SupportedRows"] = int(len(supported))
            record["CoveragePct"] = _fuel_coverage_pct(assessed)
            if len(supported) >= minimum_test_rows:
                effect = aggregate_improvement(
                    supported["FOC_MT_Day"],
                    supported["ExpectedNoInterventionFOC"],
                    supported["PropellingHours"],
                )
                record["PlaceboSavingPct"] = float(effect)
                record["Status"] = "Valid"
            else:
                record["Status"] = "Too few supported pseudo-post rows"
        except Exception as exc:
            record["Status"] = f"Failed: {exc}"
        rows.append(record)
    results = pd.DataFrame(rows)
    valid = results.loc[results.get("Status", pd.Series(dtype=str)).eq("Valid")].copy()
    effects = pd.to_numeric(valid.get("PlaceboSavingPct"), errors="coerce").dropna().to_numpy()
    if len(effects) == 0 or actual_effect_pct is None or not np.isfinite(actual_effect_pct):
        return {
            "valid_count": int(len(effects)),
            "results": results,
            "actual_percentile": None,
            "empirical_p_one_sided": None,
            "separated_from_placebos": None,
            "reason": "No valid placebo comparison was available.",
        }
    percentile = float(100.0 * np.mean(effects < float(actual_effect_pct)))
    empirical_p = float((1 + np.sum(effects >= float(actual_effect_pct))) / (len(effects) + 1))
    return {
        "valid_count": int(len(effects)),
        "results": results,
        "actual_percentile": percentile,
        "empirical_p_one_sided": empirical_p,
        "separated_from_placebos": bool(float(actual_effect_pct) > float(np.max(effects))),
        "reason": None,
    }


def _monthly_persistence(supported: pd.DataFrame) -> pd.DataFrame:
    if supported.empty:
        return pd.DataFrame()
    work = supported.copy()
    work["Month"] = pd.to_datetime(work["Date"]).dt.to_period("M").dt.to_timestamp()
    work["ActualIntervalFuelMT"] = work["FOC_MT_Day"] * work["PropellingHours"] / 24.0
    work["ExpectedIntervalFuelMT"] = (
        work["ExpectedPreDDCondition_FOC_MT_Day"] * work["PropellingHours"] / 24.0
    )
    grouped = work.groupby("Month", as_index=False).agg(
        ActualFuelMT=("ActualIntervalFuelMT", "sum"),
        ExpectedPreDDConditionFuelMT=("ExpectedIntervalFuelMT", "sum"),
        SupportedRows=("InsideSupport", "size"),
        PropellingHours=("PropellingHours", "sum"),
    )
    grouped["SavingEquivalentMT"] = (
        grouped["ExpectedPreDDConditionFuelMT"] - grouped["ActualFuelMT"]
    )
    grouped["SavingPct"] = np.where(
        grouped["ExpectedPreDDConditionFuelMT"] > 0,
        grouped["SavingEquivalentMT"] / grouped["ExpectedPreDDConditionFuelMT"] * 100.0,
        np.nan,
    )
    grouped["PerformanceIndex"] = np.where(
        grouped["ExpectedPreDDConditionFuelMT"] > 0,
        grouped["ActualFuelMT"] / grouped["ExpectedPreDDConditionFuelMT"],
        np.nan,
    )
    return grouped


def detect_foc_anomalies(
    reference: pd.DataFrame,
    target: pd.DataFrame,
    model: FittedFOCModel,
    robust_z_threshold: float = 4.5,
    minimum_ratio: float = 0.25,
    maximum_ratio: float = 4.0,
) -> pd.DataFrame:
    """Flag gross FOC anomalies without deleting or changing source reports.

    The expected FOC relationship and residual scale are learned from pre-DD
    data only. A report is flagged only when it is both statistically extreme
    on the log scale and outside a deliberately broad expected/actual ratio.
    This is a data-review screen, not an automatic exclusion rule.
    """
    columns = [
        "ReportID", "Date", "Period", "Voyage", "STW", "DisplacementMT",
        "FOC_MT_Day", "ExpectedFOCForAnomalyScreen", "ActualExpectedRatio",
        "RobustResidualScore", "FlagReason",
    ]
    if reference.empty or target.empty:
        return pd.DataFrame(columns=columns)
    reference_actual = pd.to_numeric(reference["FOC_MT_Day"], errors="coerce").to_numpy(dtype=float)
    reference_expected = model.predict(reference)
    reference_mask = (
        np.isfinite(reference_actual)
        & np.isfinite(reference_expected)
        & (reference_actual > 0)
        & (reference_expected > 0)
    )
    if reference_mask.sum() < 10:
        return pd.DataFrame(columns=columns)
    reference_residual = np.log(reference_actual[reference_mask] / reference_expected[reference_mask])
    centre = float(np.median(reference_residual))
    mad = float(np.median(np.abs(reference_residual - centre)))
    # The 0.05 floor avoids unstable scores when a synthetic or exceptionally
    # uniform training set has an almost-zero median absolute deviation.
    scale = max(1.4826 * mad, 0.05)

    work = target.copy()
    actual = pd.to_numeric(work["FOC_MT_Day"], errors="coerce").to_numpy(dtype=float)
    expected = model.predict(work)
    valid = np.isfinite(actual) & np.isfinite(expected) & (actual > 0) & (expected > 0)
    ratio = np.full(len(work), np.nan, dtype=float)
    score = np.full(len(work), np.nan, dtype=float)
    ratio[valid] = actual[valid] / expected[valid]
    score[valid] = (np.log(ratio[valid]) - centre) / scale
    flagged = (
        valid
        & (np.abs(score) > robust_z_threshold)
        & ((ratio < minimum_ratio) | (ratio > maximum_ratio))
    )
    if not flagged.any():
        return pd.DataFrame(columns=columns)
    work = work.loc[flagged].copy()
    work["ExpectedFOCForAnomalyScreen"] = expected[flagged]
    work["ActualExpectedRatio"] = ratio[flagged]
    work["RobustResidualScore"] = score[flagged]
    work["FlagReason"] = np.where(
        ratio[flagged] < minimum_ratio,
        "Reported FOC is less than one quarter of the pre-DD model expectation",
        "Reported FOC is more than four times the pre-DD model expectation",
    )
    for column in columns:
        if column not in work:
            work[column] = np.nan
    return work[columns].reset_index(drop=True)


def _leave_one_out_effect_range(supported: pd.DataFrame) -> tuple[float | None, float | None]:
    """Return the report-influence range while keeping the fitted model fixed."""
    if len(supported) < 3:
        return None, None
    effects: list[float] = []
    for index in supported.index:
        subset = supported.drop(index=index)
        effect = aggregate_improvement(
            subset["FOC_MT_Day"],
            subset["ExpectedPreDDCondition_FOC_MT_Day"],
            subset["PropellingHours"],
        )
        if np.isfinite(effect):
            effects.append(float(effect))
    if not effects:
        return None, None
    return float(min(effects)), float(max(effects))


def _effect_with_retrained_clean_data(
    before: pd.DataFrame,
    after: pd.DataFrame,
    anomalies: pd.DataFrame,
    comparison_mode: str,
    minimum_train_rows: int,
    minimum_supported_after: int,
) -> float | None:
    """Sensitivity estimate after excluding flagged rows and retraining."""
    if anomalies.empty or "ReportID" not in before or "ReportID" not in after:
        return None
    flagged_pre = set(
        anomalies.loc[anomalies["Period"].eq("Before DD"), "ReportID"].dropna().tolist()
    )
    flagged_post = set(
        anomalies.loc[anomalies["Period"].eq("After DD"), "ReportID"].dropna().tolist()
    )
    clean_before = before.loc[~before["ReportID"].isin(flagged_pre)].copy()
    clean_after = after.loc[~after["ReportID"].isin(flagged_post)].copy()
    if len(clean_before) < minimum_train_rows or clean_after.empty:
        return None
    try:
        clean_selection = select_huber_specification(clean_before, minimum_train_rows)
        clean_model = clean_selection["selected_model"]
        assessed = assess_operating_support(
            clean_before, clean_after, clean_model, comparison_mode=comparison_mode
        )
        assessed["ExpectedFOC"] = clean_model.predict(assessed)
        supported = assessed.loc[assessed["InsideSupport"]].copy()
        if len(supported) < minimum_supported_after:
            return None
        effect = aggregate_improvement(
            supported["FOC_MT_Day"], supported["ExpectedFOC"], supported["PropellingHours"]
        )
        return float(effect) if np.isfinite(effect) else None
    except Exception:
        return None


def _stability_summary(
    primary: float | None,
    alternative: float | None,
    anomaly_sensitivity: float | None,
    route_sensitivity: float | None,
    leave_one_out_min: float | None,
    leave_one_out_max: float | None,
    anomaly_count: int,
) -> tuple[str, str, float | None, float | None]:
    """Classify direction stability without inventing a materiality threshold."""
    values = [
        primary, alternative, anomaly_sensitivity, route_sensitivity,
        leave_one_out_min, leave_one_out_max,
    ]
    finite = [float(value) for value in values if value is not None and np.isfinite(value)]
    if primary is None or not np.isfinite(primary) or not finite:
        return "Not assessable", "No valid sensitivity range was available.", None, None
    primary_sign = np.sign(float(primary))
    sign_change = any(np.sign(value) != primary_sign for value in finite if value != 0)
    if primary_sign == 0:
        sign_change = any(value != 0 for value in finite)
    low, high = min(finite), max(finite)
    if sign_change or low <= 0 <= high:
        return (
            "Unstable",
            "The estimated effect changed direction during reasonable model, anomaly or report-influence checks.",
            low,
            high,
        )
    if anomaly_count:
        return (
            "Sensitive but direction stable",
            "The estimated direction remained the same, but flagged FOC reports require engineering review.",
            low,
            high,
        )
    return (
        "Stable",
        "The estimated direction remained the same across the available model and report-influence checks.",
        low,
        high,
    )


def classify_evidence(
    result: dict,
    period_coverage_ok: bool,
    route_confirmed: bool,
    service_cycle_confirmed: bool,
) -> tuple[str, list[str]]:
    """Apply declared prototype gates; these are not regulatory standards."""
    limitations: list[str] = []
    if not result.get("valid"):
        return "Inconclusive", [result.get("reason") or "No package estimate was available."]
    validation = result["validation"]
    huber = validation["huber"]
    mape = huber.get("mape_pct")
    bias = huber.get("bias_pct")
    coverage = float(result.get("coverage_pct") or 0.0)
    supported_rows = int(result.get("supported_after_rows") or 0)
    comparison_basis = str(result.get("comparison_basis") or "Unavailable")
    stability_status = str(result.get("stability_status") or "Not assessable")
    anomaly_count = int(result.get("flagged_foc_rows") or 0)
    exponent = result.get("learned_speed_exponent")
    coefficient_table = result.get("coefficients", pd.DataFrame())
    coefficient_sign_review: list[str] = []
    if isinstance(coefficient_table, pd.DataFrame) and not coefficient_table.empty:
        transformed = coefficient_table.set_index("Feature")["Coefficient in transformed units"]
        for feature in ["LogDisplacementRatio"]:
            value = transformed.get(feature)
            if value is not None and np.isfinite(value) and float(value) < 0:
                coefficient_sign_review.append(feature)

    if not period_coverage_ok:
        limitations.append("The workbook does not cover the declared pre-DD and post-DD monitoring windows.")
    if not route_confirmed:
        limitations.append("The route and operating-leg mappings have not been confirmed from operational records.")
    if comparison_basis == "Expanded cross-route":
        limitations.append(
            "The strict same-route comparison was insufficient, so the estimate uses different routes "
            "with the same operating leg and comparable pre-DD speed/loading support under Beaufort <=4."
        )
    if not service_cycle_confirmed:
        limitations.append(
            "The selected post-DD monitoring period has not been confirmed to contain one complete normal service cycle."
        )
    if anomaly_count:
        limitations.append(
            f"{anomaly_count} gross FOC report(s) were flagged for engineering review; source rows were retained."
        )
    if stability_status == "Unstable":
        limitations.append(
            "The estimated effect changed direction during the automatic stability checks."
        )
    elif stability_status == "Sensitive but direction stable":
        limitations.append(
            "The estimated direction remained stable, but the result is sensitive to flagged FOC data."
        )
    if mape is None:
        limitations.append("Chronological Huber validation could not be completed.")
    elif mape > 12:
        limitations.append(
            f"Chronological Huber MAPE is {mape:.1f}% (project gate: <=12%)."
        )
    if bias is None:
        limitations.append("Chronological Huber bias is unavailable.")
    elif abs(bias) > 8:
        limitations.append(f"Absolute chronological bias is {abs(bias):.1f}% (project gate: <=8%).")
    if coverage < 70:
        limitations.append(f"Comparable post-DD fuel coverage is {coverage:.1f}% (project gate: >=70%).")
    if supported_rows < 20:
        limitations.append(
            f"Only {supported_rows} comparable post-DD reports are available "
            "(project target for an indicative result: >=20)."
        )
    elif supported_rows < 30:
        limitations.append(
            f"{supported_rows} comparable post-DD reports are available "
            "(preferred target for supported prototype screening: >=30)."
        )
    if exponent is None or not 1.5 <= float(exponent) <= 4.5:
        limitations.append("The learned speed exponent is outside the 1.5-4.5 engineering review band.")
    if coefficient_sign_review:
        limitations.append(
            "A negative displacement coefficient requires engineering review: "
            + ", ".join(coefficient_sign_review)
            + ". This can indicate insufficient loading variation or correlation with operating pattern."
        )
    if validation.get("huber_outperforms_cubic") is False:
        huber_mape_text = _display_metric(validation.get("huber", {}).get("mape_pct"))
        cubic_mape_text = _display_metric(validation.get("cubic", {}).get("mape_pct"))
        limitations.append(
            "ML superiority was not demonstrated on unseen pre-DD reports: selected Huber MAPE "
            f"{huber_mape_text} versus public cubic-speed benchmark MAPE {cubic_mape_text}. "
            "The benchmark remains a "
            "comparison, not the primary ML method."
        )
    placebo = result.get("placebo", {})
    if placebo.get("valid_count", 0) >= 3 and placebo.get("separated_from_placebos") is False:
        limitations.append("The actual saving did not exceed all valid pre-DD placebo effects.")

    model_validation_passed = (
        mape is not None
        and mape <= 12
        and bias is not None
        and abs(bias) <= 8
        and exponent is not None
        and 1.5 <= float(exponent) <= 4.5
        and not coefficient_sign_review
    )

    # A calculated percentage is not useful evidence when the pre-DD model
    # fails its declared validation gates.  Earlier versions allowed these
    # cases to fall through to Preliminary whenever the basic row and coverage
    # floor was met.  That could make a grossly inaccurate model look merely
    # data-limited rather than unusable.
    if not model_validation_passed:
        limitations.append(
            "The pre-DD prediction model did not pass the minimum validation gates, "
            "so the calculated percentage is not suitable for interpretation."
        )
        return "Inconclusive", limitations

    if stability_status == "Unstable":
        return "Unstable", limitations

    if supported_rows < 10 or coverage < 40:
        return "Inconclusive", limitations

    supported = (
        period_coverage_ok
        and route_confirmed
        and service_cycle_confirmed
        and comparison_basis == "Strict same-route"
        and model_validation_passed
        and coverage >= 70
        and supported_rows >= 30
        and stability_status == "Stable"
        and anomaly_count == 0
        and validation.get("huber_outperforms_cubic") is not False
        and not (
            placebo.get("valid_count", 0) >= 3
            and placebo.get("separated_from_placebos") is False
        )
    )
    if supported:
        return "Supported (prototype screening)", limitations
    if comparison_basis == "Expanded cross-route":
        return "Preliminary", limitations
    indicative = (
        period_coverage_ok
        and route_confirmed
        and service_cycle_confirmed
        and comparison_basis == "Strict same-route"
        and model_validation_passed
        and supported_rows >= 20
        and coverage >= 70
        and stability_status == "Stable"
        and anomaly_count == 0
    )
    if indicative:
        return "Indicative", limitations
    return "Preliminary", limitations


def run_package_assessment(
    eligible: pd.DataFrame,
    period_coverage_ok: bool,
    route_confirmed: bool,
    service_cycle_confirmed: bool = True,
    minimum_train_rows: int = 20,
    minimum_supported_after: int = 10,
    placebo_count: int = 5,
    progress_callback: ProgressCallback | None = None,
) -> dict:
    """Run the complete package-level counterfactual workflow."""
    before = eligible[eligible["Period"] == "Before DD"].copy()
    after = eligible[eligible["Period"] == "After DD"].copy()
    result = {
        "valid": False,
        "reason": None,
        "before_rows": int(len(before)),
        "after_rows": int(len(after)),
        "supported_after_rows": 0,
        "coverage_pct": 0.0,
        "comparison_basis": "Unavailable",
        "strict_supported_after_rows": 0,
        "strict_coverage_pct": 0.0,
        "expanded_supported_after_rows": 0,
        "expanded_coverage_pct": 0.0,
        "pre_dd_speed_profile": {},
        "improvement_pct": None,
        "observed_equivalent_fuel_saved_mt": None,
        "observed_propelling_days": None,
        "saving_rate_mt_per_propelling_day": None,
        "learned_speed_exponent": None,
        "numeric_features": [],
        "selected_ml_model": None,
        "model_selection_reason": None,
        "ml_candidate_comparison": pd.DataFrame(),
        "alternative_ml_saving_pct": None,
        "alternative_ml_label": None,
        "model_sensitivity": pd.DataFrame(),
        "foc_anomalies": pd.DataFrame(),
        "flagged_foc_rows": 0,
        "anomaly_sensitivity_pct": None,
        "strict_saving_pct": None,
        "expanded_saving_pct": None,
        "leave_one_out_min_pct": None,
        "leave_one_out_max_pct": None,
        "stability_status": "Not assessable",
        "stability_summary": "No valid sensitivity range was available.",
        "stability_min_pct": None,
        "stability_max_pct": None,
        "model_details": {},
        "coefficients": pd.DataFrame(),
        "assessed_rows": pd.DataFrame(),
        "service_leg_summary": pd.DataFrame(),
        "scope_statement": "No supported post-DD operating scope was available.",
        "validation": {
            "folds": 0,
            "huber": {"mape_pct": None, "bias_pct": None, "rmse": None},
            "cubic": {"mape_pct": None, "bias_pct": None, "rmse": None},
            "predictions": pd.DataFrame(),
            "huber_outperforms_cubic": None,
            "candidate_test_rows": 0,
            "supported_test_rows": 0,
        },
        "placebo": {},
        "persistence": pd.DataFrame(),
        "cubic_benchmark_saving_pct": None,
        "evidence_tier": "Inconclusive",
        "limitations": [],
    }
    pre_speed = pd.to_numeric(before.get("STW", pd.Series(dtype=float)), errors="coerce").dropna()
    if not pre_speed.empty:
        result["pre_dd_speed_profile"] = {
            "minimum": float(pre_speed.min()),
            "p05": float(pre_speed.quantile(0.05)),
            "median": float(pre_speed.median()),
            "p95": float(pre_speed.quantile(0.95)),
            "maximum": float(pre_speed.max()),
        }
    if len(before) < minimum_train_rows:
        result["reason"] = f"Only {len(before)} eligible pre-DD rows; {minimum_train_rows} required."
        result["evidence_tier"], result["limitations"] = classify_evidence(
            result, period_coverage_ok, route_confirmed, service_cycle_confirmed
        )
        return result
    if after.empty:
        result["reason"] = "No eligible post-DD reports are available."
        result["evidence_tier"], result["limitations"] = classify_evidence(
            result, period_coverage_ok, route_confirmed, service_cycle_confirmed
        )
        return result
    try:
        _emit(
            progress_callback,
            20,
            f"Stage 3/6 - Training input confirmed: {len(before)} eligible pre-DD rows; "
            "post-DD rows remain unseen by training.",
        )
        _emit(
            progress_callback,
            32,
            "Stage 3/6 - Comparing the STW-only and STW-and-displacement Huber specifications "
            "using pre-DD data only.",
        )
        selection = select_huber_specification(before, minimum_train_rows)
        model = selection["selected_model"]
        alternative_model = selection["alternative_model"]
        result["selected_ml_model"] = selection["selected_label"]
        result["model_selection_reason"] = selection["selection_reason"]
        result["ml_candidate_comparison"] = selection["candidate_table"]
        result["alternative_ml_label"] = selection["alternative_label"]
        result["validation"] = selection["selected_validation"]
        pre_anomalies = detect_foc_anomalies(before, before, model)
        post_anomalies = detect_foc_anomalies(before, after, model)
        anomaly_frames = [frame for frame in [pre_anomalies, post_anomalies] if not frame.empty]
        result["foc_anomalies"] = (
            pd.concat(anomaly_frames, ignore_index=True)
            if anomaly_frames
            else pre_anomalies.copy()
        )
        result["flagged_foc_rows"] = int(len(result["foc_anomalies"]))
        coefficient_map = (
            model.coefficient_table()
            .set_index("Feature")["Coefficient in transformed units"]
            .astype(float)
            .to_dict()
        )
        displacement_coefficient = coefficient_map.get("LogDisplacementRatio")
        _emit(
            progress_callback,
            45,
            f"Stage 3/6 - Selected {selection['selected_label']}: learned speed exponent "
            f"{model.learned_speed_exponent:.3f}"
            + (
                f"; displacement coefficient {displacement_coefficient:.3f}."
                if displacement_coefficient is not None
                else ". Displacement remains in the operating-support check."
            ),
        )

        _emit(
            progress_callback,
            55,
            "Stage 4/6 - Reviewing chronological validation for the selected Huber specification "
            "and the public cubic-speed benchmark.",
        )
        huber_mape = result["validation"]["huber"].get("mape_pct")
        cubic_mape = result["validation"]["cubic"].get("mape_pct")
        _emit(
            progress_callback,
            64,
            "Stage 4/6 - Validation completed: "
            f"Huber MAPE {_display_metric(huber_mape)}; cubic-speed benchmark MAPE "
            f"{_display_metric(cubic_mape)} "
            f"across {result['validation'].get('folds', 0)} valid fold(s).",
        )

        _emit(
            progress_callback,
            72,
            "Stage 5/6 - Learning vessel-specific STW/loading support from pre-DD data, "
            "starting with the same route and operating leg.",
        )
        strict_assessed = assess_operating_support(
            before, after, model, comparison_mode="strict"
        )
        expanded_assessed = assess_operating_support(
            before, after, model, comparison_mode="expanded"
        )
        strict_rows = int(strict_assessed["InsideSupport"].sum())
        strict_coverage = _fuel_coverage_pct(strict_assessed)
        expanded_rows = int(expanded_assessed["InsideSupport"].sum())
        expanded_coverage = _fuel_coverage_pct(expanded_assessed)
        result["strict_supported_after_rows"] = strict_rows
        result["strict_coverage_pct"] = strict_coverage
        result["expanded_supported_after_rows"] = expanded_rows
        result["expanded_coverage_pct"] = expanded_coverage

        for label, frame in [("strict_saving_pct", strict_assessed), ("expanded_saving_pct", expanded_assessed)]:
            branch = frame.loc[frame["InsideSupport"]].copy()
            if len(branch) >= minimum_supported_after:
                branch_expected = model.predict(branch)
                branch_effect = aggregate_improvement(
                    branch["FOC_MT_Day"], branch_expected, branch["PropellingHours"]
                )
                if np.isfinite(branch_effect):
                    result[label] = float(branch_effect)

        # Use the strict same-route result whenever it reaches the declared
        # minimum evidence floor.  Only then fall back to a cross-route result
        # that keeps the confirmed operating leg and the same pre-DD-only
        # STW/displacement density test.  Exactly one basis becomes the headline.
        strict_usable = strict_rows >= minimum_supported_after and strict_coverage >= 40.0
        if strict_usable:
            assessed = strict_assessed.copy()
            result["comparison_basis"] = "Strict same-route"
        else:
            assessed = expanded_assessed.copy()
            result["comparison_basis"] = "Expanded cross-route"
        assessed["StrictInsideSupport"] = strict_assessed["InsideSupport"].reindex(assessed.index)
        assessed["ExpandedInsideSupport"] = expanded_assessed["InsideSupport"].reindex(assessed.index)
        assessed["ExpectedPreDDCondition_FOC_MT_Day"] = model.predict(assessed)
        assessed["ActualIntervalFuelMT"] = assessed["FOC_MT_Day"] * assessed["PropellingHours"] / 24.0
        assessed["ExpectedIntervalFuelMT"] = (
            assessed["ExpectedPreDDCondition_FOC_MT_Day"] * assessed["PropellingHours"] / 24.0
        )
        assessed["SavingEquivalentMT"] = (
            assessed["ExpectedIntervalFuelMT"] - assessed["ActualIntervalFuelMT"]
        )
        supported = assessed[assessed["InsideSupport"]].copy()
        result["service_leg_summary"], result["scope_statement"] = _service_leg_scope(supported)
        if result["comparison_basis"] == "Expanded cross-route":
            result["scope_statement"] = (
                "Expanded cross-route comparison: " + result["scope_statement"]
            )
        result["assessed_rows"] = assessed
        result["supported_after_rows"] = int(len(supported))
        result["coverage_pct"] = _fuel_coverage_pct(assessed)
        result["learned_speed_exponent"] = model.learned_speed_exponent
        result["numeric_features"] = model.numeric_features
        result["coefficients"] = model.coefficient_table()
        transformed = result["coefficients"].set_index("Feature")["Coefficient in transformed units"]
        result["model_details"] = {
            "transformed_intercept": model.transformed_intercept,
            "speed_coefficient": float(transformed["LogSTW"]),
            "displacement_coefficient": (
                float(transformed["LogDisplacementRatio"])
                if "LogDisplacementRatio" in transformed.index
                else None
            ),
            "displacement_reference_mt": model.displacement_reference,
            "smearing_factor": model.smearing_factor,
            "huber_epsilon": 1.35,
            "huber_alpha": 0.0001,
        }
        _emit(
            progress_callback,
            80,
            f"Stage 5/6 - {result['comparison_basis']} support selected: {len(supported)} of "
            f"{len(after)} post-DD rows; fuel-weighted coverage {result['coverage_pct']:.1f}%.",
        )
        if len(supported) < minimum_supported_after:
            result["reason"] = (
                f"Only {len(supported)} post-DD reports are inside pre-DD operating support; "
                f"{minimum_supported_after} required to calculate an exploratory result."
            )
        else:
            effect = aggregate_improvement(
                supported["FOC_MT_Day"],
                supported["ExpectedPreDDCondition_FOC_MT_Day"],
                supported["PropellingHours"],
            )
            saved = float(supported["SavingEquivalentMT"].sum())
            propelling_days = float(supported["PropellingHours"].sum() / 24.0)
            result["valid"] = bool(np.isfinite(effect))
            result["improvement_pct"] = float(effect)
            result["observed_equivalent_fuel_saved_mt"] = saved
            result["observed_propelling_days"] = propelling_days
            result["saving_rate_mt_per_propelling_day"] = (
                saved / propelling_days if propelling_days > 0 else None
            )
            result["persistence"] = _monthly_persistence(supported)
            alternative_expected = alternative_model.predict(supported)
            result["alternative_ml_saving_pct"] = aggregate_improvement(
                supported["FOC_MT_Day"],
                alternative_expected,
                supported["PropellingHours"],
            )
            leave_min, leave_max = _leave_one_out_effect_range(supported)
            result["leave_one_out_min_pct"] = leave_min
            result["leave_one_out_max_pct"] = leave_max
            comparison_mode = (
                "strict" if result["comparison_basis"] == "Strict same-route" else "expanded"
            )
            result["anomaly_sensitivity_pct"] = _effect_with_retrained_clean_data(
                before,
                after,
                result["foc_anomalies"],
                comparison_mode,
                minimum_train_rows,
                minimum_supported_after,
            )
            route_sensitivity = (
                result["expanded_saving_pct"]
                if result["comparison_basis"] == "Strict same-route"
                else None
            )
            (
                result["stability_status"],
                result["stability_summary"],
                result["stability_min_pct"],
                result["stability_max_pct"],
            ) = _stability_summary(
                result["improvement_pct"],
                result["alternative_ml_saving_pct"],
                result["anomaly_sensitivity_pct"],
                route_sensitivity,
                leave_min,
                leave_max,
                result["flagged_foc_rows"],
            )
            _emit(
                progress_callback,
                88,
                f"Stage 6/6 - Counterfactual aggregation completed: primary Huber package estimate {effect:.2f}%.",
            )
            try:
                cubic = fit_cubic_speed_benchmark(before)
                cubic_expected = cubic.predict(supported)
                result["cubic_benchmark_saving_pct"] = aggregate_improvement(
                    supported["FOC_MT_Day"], cubic_expected, supported["PropellingHours"]
                )
            except Exception:
                pass
            if result["cubic_benchmark_saving_pct"] is not None:
                _emit(
                    progress_callback,
                    92,
                    f"Public cubic-speed benchmark completed: V^3 estimate "
                    f"{result['cubic_benchmark_saving_pct']:.2f}% on the same supported rows.",
                )
            result["model_sensitivity"] = pd.DataFrame(
                [
                    {
                        "Method": result["selected_ml_model"] + " - primary ML",
                        "Estimated package FOC saving (%)": result["improvement_pct"],
                        "Assessment role": "Primary ML estimate",
                    },
                    {
                        "Method": result["alternative_ml_label"],
                        "Estimated package FOC saving (%)": result["alternative_ml_saving_pct"],
                        "Assessment role": "ML specification sensitivity",
                    },
                    {
                        "Method": "Public cubic-speed benchmark (V^3)",
                        "Estimated package FOC saving (%)": result["cubic_benchmark_saving_pct"],
                        "Assessment role": "Public physics-based benchmark",
                    },
                    {
                        "Method": "Retrained without flagged gross FOC reports",
                        "Estimated package FOC saving (%)": result["anomaly_sensitivity_pct"],
                        "Assessment role": "Data-quality sensitivity",
                    },
                ]
            )
        _emit(progress_callback, 95, "Running repeated fake pre-DD intervention dates for the placebo check.")
        result["placebo"] = run_placebo_analysis(
            before,
            result["improvement_pct"],
            desired_placebos=placebo_count,
            minimum_train_rows=minimum_train_rows,
            evaluation_rows=max(5, min(15, len(after))),
            numeric_features=model.numeric_features,
        )
        _emit(
            progress_callback,
            99,
            f"Placebo check completed with {result['placebo'].get('valid_count', 0)} valid fake intervention(s).",
        )
    except Exception as exc:
        result["reason"] = f"The FOC prediction model failed: {exc}"
    result["evidence_tier"], result["limitations"] = classify_evidence(
        result, period_coverage_ok, route_confirmed, service_cycle_confirmed
    )
    _emit(progress_callback, 100, f"Assessment classified as: {result['evidence_tier']}.")
    return result
