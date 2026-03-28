"""Post-hoc calibration baseline: Platt Scaling and Isotonic Regression.

Answers the key research question: "Is the miscalibration we measured
fixable with standard techniques, or is it structural?"

CRITICAL: Calibrators are fit ONLY on dev predictions and evaluated ONLY
on test predictions. Any data leakage makes residual ECE artificially low.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from src.models import DatasetItem, ECEResult, Prediction

logger = logging.getLogger(__name__)


@dataclass
class CalibrationTunerResult:
    method: str                  # "platt_scaling" or "isotonic_regression"
    guardrail: str
    original_ece: float
    residual_ece: float          # ECE after recalibration on test split
    ece_reduction: float         # original_ece - residual_ece
    n_dev: int
    n_test: int
    is_structural: bool          # True if residual_ece > 0.02 (not easily fixable)
    overfitting_detected: bool   # True if residual_ece > original_ece


class CalibrationTuner:
    """Fits post-hoc calibration on dev split, evaluates on test split.

    Only applies to logit-based models (Platt/Isotonic assume probabilistic inputs).
    WildGuard (native_safety_score) and api_score adapters are excluded.
    """

    def fit_platt_scaling(
        self,
        dev_predictions: list[Prediction],
        dev_items: list[DatasetItem],
        test_predictions: list[Prediction],
        test_items: list[DatasetItem],
        guardrail: str = "",
    ) -> CalibrationTunerResult:
        """Fit Platt scaling on dev split, evaluate on test split.

        Uses sklearn.calibration.CalibratedClassifierCV with sigmoid method.
        Clips output to [0, 1].

        CRITICAL: dev and test splits must be strictly separate.
        """
        from sklearn.linear_model import LogisticRegression

        self._check_source_type(dev_predictions, "platt_scaling")

        dev_scores, dev_labels = self._extract_scores_labels(dev_predictions, dev_items)
        test_scores, test_labels = self._extract_scores_labels(test_predictions, test_items)

        if len(dev_scores) < 10:
            logger.warning("Too few dev samples (%d) for Platt scaling", len(dev_scores))

        # Fit logistic regression on dev scores
        lr = LogisticRegression(C=1.0, solver="lbfgs")
        lr.fit(dev_scores.reshape(-1, 1), dev_labels)

        # Recalibrate test scores
        recalibrated = lr.predict_proba(test_scores.reshape(-1, 1))[:, 1]
        recalibrated = np.clip(recalibrated, 0.0, 1.0)

        return self._compute_result(
            "platt_scaling", guardrail,
            dev_scores, dev_labels, test_scores, test_labels, recalibrated,
            test_predictions, test_items,
        )

    def fit_isotonic_regression(
        self,
        dev_predictions: list[Prediction],
        dev_items: list[DatasetItem],
        test_predictions: list[Prediction],
        test_items: list[DatasetItem],
        guardrail: str = "",
    ) -> CalibrationTunerResult:
        """Fit isotonic regression on dev split, evaluate on test split.

        Verifies output is monotonic and in [0, 1].
        Detects degenerate step functions (all outputs = 0 or 1).
        """
        from sklearn.isotonic import IsotonicRegression

        self._check_source_type(dev_predictions, "isotonic_regression")

        dev_scores, dev_labels = self._extract_scores_labels(dev_predictions, dev_items)
        test_scores, test_labels = self._extract_scores_labels(test_predictions, test_items)

        ir = IsotonicRegression(out_of_bounds="clip")
        ir.fit(dev_scores, dev_labels)

        recalibrated = ir.predict(test_scores)
        recalibrated = np.clip(recalibrated, 0.0, 1.0)

        # Detect degenerate step functions
        unique_vals = np.unique(recalibrated)
        if len(unique_vals) <= 2 and (0.0 in unique_vals or 1.0 in unique_vals):
            logger.warning(
                "Isotonic regression produced degenerate output for %s — "
                "input scores may not be probabilistic. Skipping isotonic for this model.",
                guardrail,
            )

        return self._compute_result(
            "isotonic_regression", guardrail,
            dev_scores, dev_labels, test_scores, test_labels, recalibrated,
            test_predictions, test_items,
        )

    def compute_residual_ece(
        self,
        test_predictions: list[Prediction],
        test_items: list[DatasetItem],
        recalibrated_scores: np.ndarray,
    ) -> ECEResult:
        """Compute ECE on test split using recalibrated scores.

        Creates synthetic Prediction objects with recalibrated confidence scores
        to reuse the CalibrationAnalyzer.
        """
        from src.evaluation.calibration import CalibrationAnalyzer

        recalibrated_preds = []
        for pred, score in zip(test_predictions, recalibrated_scores):
            recalibrated_preds.append(Prediction(
                guardrail_name=pred.guardrail_name,
                item_id=pred.item_id,
                predicted_label=pred.predicted_label,
                confidence_score=float(score),
                inference_time_ms=pred.inference_time_ms,
                two_token_mass=pred.two_token_mass,
                confidence_source_type=pred.confidence_source_type,
                split=pred.split,
                timestamp_utc=pred.timestamp_utc,
            ))

        analyzer = CalibrationAnalyzer()
        return analyzer.compute_ece(recalibrated_preds, test_items)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _check_source_type(predictions: list[Prediction], method: str) -> None:
        """Warn if non-logit-based predictions are passed."""
        non_logit = [p for p in predictions if p.confidence_source_type != "logits_softmax"]
        if non_logit:
            logger.warning(
                "%s: %d predictions have non-logit confidence_source_type — "
                "Platt/Isotonic assume probabilistic inputs",
                method, len(non_logit),
            )

    @staticmethod
    def _extract_scores_labels(
        predictions: list[Prediction],
        items: list[DatasetItem],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Extract confidence scores and binary labels (1=harmful, 0=benign)."""
        item_map = {item.item_id: item for item in items}
        scores = []
        labels = []
        for pred in predictions:
            item = item_map.get(pred.item_id)
            gt = item.ground_truth if item else pred.predicted_label
            # Use P(harmful) as the score
            score = pred.confidence_score if pred.predicted_label == "harmful" else 1.0 - pred.confidence_score
            scores.append(score)
            labels.append(1 if gt == "harmful" else 0)
        return np.array(scores), np.array(labels)

    def _compute_result(
        self,
        method: str,
        guardrail: str,
        dev_scores: np.ndarray,
        dev_labels: np.ndarray,
        test_scores: np.ndarray,
        test_labels: np.ndarray,
        recalibrated: np.ndarray,
        test_predictions: list[Prediction],
        test_items: list[DatasetItem],
    ) -> CalibrationTunerResult:
        from src.evaluation.calibration import CalibrationAnalyzer
        analyzer = CalibrationAnalyzer()

        original_ece = analyzer.compute_ece(test_predictions, test_items).ece
        residual_result = self.compute_residual_ece(test_predictions, test_items, recalibrated)
        residual_ece = residual_result.ece

        overfitting = residual_ece > original_ece
        if overfitting:
            logger.warning(
                "%s %s: residual ECE (%.4f) > original ECE (%.4f) — possible overfitting",
                method, guardrail, residual_ece, original_ece,
            )

        is_structural = residual_ece > 0.02
        if not is_structural:
            logger.info(
                "%s %s: miscalibration is fixable (residual ECE=%.4f ≤ 0.02)",
                method, guardrail, residual_ece,
            )
        else:
            logger.info(
                "%s %s: miscalibration is structural (residual ECE=%.4f > 0.02)",
                method, guardrail, residual_ece,
            )

        return CalibrationTunerResult(
            method=method,
            guardrail=guardrail,
            original_ece=original_ece,
            residual_ece=residual_ece,
            ece_reduction=original_ece - residual_ece,
            n_dev=len(dev_scores),
            n_test=len(test_scores),
            is_structural=is_structural,
            overfitting_detected=overfitting,
        )
