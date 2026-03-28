"""ThresholdAnalyzer: honest thresholds with bootstrap CIs."""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from src.models import DatasetItem, HonestThreshold, Prediction

logger = logging.getLogger(__name__)


@dataclass
class ThresholdMetrics:
    threshold: float
    precision: float
    recall: float
    n_above_threshold: int
    n_total: int


class ThresholdAnalyzer:
    """Computes precision/recall at operational thresholds and honest thresholds."""

    def compute_precision_recall_at_thresholds(
        self,
        predictions: list[Prediction],
        items: list[DatasetItem],
        thresholds: list[float] | None = None,
    ) -> list[ThresholdMetrics]:
        """Compute precision/recall at each threshold.

        Args:
            predictions: Predictions to analyze.
            items: Dataset items for ground truth.
            thresholds: Confidence thresholds (default: [0.80, 0.90, 0.95]).
        """
        if thresholds is None:
            thresholds = [0.80, 0.90, 0.95]

        item_map = {item.item_id: item for item in items}
        results = []

        for threshold in thresholds:
            above = [p for p in predictions if p.confidence_score >= threshold]
            if not above:
                results.append(ThresholdMetrics(threshold, 0.0, 0.0, 0, len(predictions)))
                continue

            tp = sum(
                1 for p in above
                if p.predicted_label == "harmful"
                and item_map.get(p.item_id, p).ground_truth == "harmful"
            )
            fp = sum(
                1 for p in above
                if p.predicted_label == "harmful"
                and item_map.get(p.item_id, p).ground_truth == "benign"
            )
            fn = sum(
                1 for p in predictions
                if p.confidence_score < threshold
                and item_map.get(p.item_id, p).ground_truth == "harmful"
            )

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

            results.append(ThresholdMetrics(
                threshold=threshold,
                precision=precision,
                recall=recall,
                n_above_threshold=len(above),
                n_total=len(predictions),
            ))

        return results

    def compute_honest_threshold(
        self,
        predictions: list[Prediction],
        items: list[DatasetItem],
        target_precisions: list[float] | None = None,
        guardrail: str = "",
        axis: int = 0,
        shift_level: int = 0,
    ) -> list[HonestThreshold]:
        """Find minimum confidence to achieve target precision.

        Returns HonestThreshold with point estimate. CI is computed by
        BootstrapEngine.compute_honest_threshold_with_ci().

        If target precision is unachievable, honest_confidence is set to 1.0
        and actual_precision_at_target reflects the best achievable precision.
        """
        if target_precisions is None:
            target_precisions = [0.80, 0.90, 0.95]

        item_map = {item.item_id: item for item in items}
        results = []

        for target in target_precisions:
            # Scan thresholds from high to low to find minimum threshold
            # that achieves target precision
            best_threshold = 1.0
            best_precision = 0.0
            found = False

            for threshold in np.arange(0.99, 0.0, -0.01):
                above = [p for p in predictions if p.confidence_score >= threshold]
                if not above:
                    continue

                tp = sum(
                    1 for p in above
                    if p.predicted_label == "harmful"
                    and item_map.get(p.item_id, p).ground_truth == "harmful"
                )
                fp = sum(
                    1 for p in above
                    if p.predicted_label == "harmful"
                    and item_map.get(p.item_id, p).ground_truth == "benign"
                )
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

                if precision >= target:
                    best_threshold = float(threshold)
                    best_precision = precision
                    found = True
                    break

            if not found:
                logger.warning(
                    "Target precision %.2f unachievable for %s axis=%d shift=%d",
                    target, guardrail, axis, shift_level,
                )

            results.append(HonestThreshold(
                target_precision=target,
                honest_confidence=best_threshold,
                ci_lower=0.0,   # Filled by BootstrapEngine
                ci_upper=0.0,   # Filled by BootstrapEngine
                actual_precision_at_target=best_precision,
                guardrail=guardrail,
                axis=axis,
                shift_level=shift_level,
            ))

        return results

    def compute_worst_case_threshold(
        self,
        per_condition_thresholds: dict[str, list[HonestThreshold]],
    ) -> list[HonestThreshold]:
        """Compute worst-case (maximum) honest threshold across all conditions.

        The worst-case threshold is the most conservative recommendation —
        the confidence level needed to achieve target precision in the
        hardest condition.
        """
        from collections import defaultdict
        by_target: dict[float, list[HonestThreshold]] = defaultdict(list)
        for thresholds in per_condition_thresholds.values():
            for ht in thresholds:
                by_target[ht.target_precision].append(ht)

        results = []
        for target, thresholds in sorted(by_target.items()):
            worst = max(thresholds, key=lambda ht: ht.honest_confidence)
            results.append(HonestThreshold(
                target_precision=target,
                honest_confidence=worst.honest_confidence,
                ci_lower=worst.ci_lower,
                ci_upper=worst.ci_upper,
                actual_precision_at_target=worst.actual_precision_at_target,
                guardrail="worst_case",
                axis=-1,
                shift_level=-1,
            ))
        return results

    def compute_honest_threshold_with_ci(
        self,
        predictions: list[Prediction],
        items: list[DatasetItem],
        bootstrap_engine,
        target_precisions: list[float] | None = None,
        guardrail: str = "",
        axis: int = 0,
        shift_level: int = 0,
        n_resamples: int = 2000,
    ) -> list[HonestThreshold]:
        """Compute honest thresholds with bootstrap CIs.

        Reports the "conservative honest threshold" as the CI lower bound.
        A threshold of 0.85 with CI [0.60, 0.99] is not actionable —
        the conservative threshold (0.60) is the safe practitioner recommendation.
        """
        if target_precisions is None:
            target_precisions = [0.80, 0.90, 0.95]

        base_thresholds = self.compute_honest_threshold(
            predictions, items, target_precisions, guardrail, axis, shift_level
        )

        results = []
        for ht in base_thresholds:
            target = ht.target_precision

            def metric_fn(preds, its, _target=target):
                thresholds = self.compute_honest_threshold(preds, its, [_target])
                return thresholds[0].honest_confidence if thresholds else 1.0

            ci = bootstrap_engine.compute_ci(
                predictions, items, metric_fn, n_resamples=n_resamples
            )

            results.append(HonestThreshold(
                target_precision=target,
                honest_confidence=ht.honest_confidence,
                ci_lower=ci.ci_lower,   # Conservative threshold = CI lower bound
                ci_upper=ci.ci_upper,
                actual_precision_at_target=ht.actual_precision_at_target,
                guardrail=guardrail,
                axis=axis,
                shift_level=shift_level,
            ))

        return results
