"""BootstrapEngine: cluster bootstrap CIs and statistical tests.

Critical implementation notes:
- Bootstrap resamples by seed_id (cluster bootstrap), NOT individual predictions
- Permutation test permutes within-pair assignments, NOT ECE values
- Holm-Bonferroni correction scoped per-axis, NOT globally
"""
from __future__ import annotations

import logging
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable

import numpy as np

from src.models import DatasetItem, Prediction

logger = logging.getLogger(__name__)


@dataclass
class ConfidenceInterval:
    point_estimate: float
    ci_lower: float
    ci_upper: float
    n_resamples: int
    alpha: float
    is_degenerate: bool  # True if lower == upper


@dataclass
class McNemarResult:
    statistic: float
    p_value: float
    n_concordant: int
    n_discordant_ab: int  # A correct, B wrong
    n_discordant_ba: int  # B correct, A wrong
    n_dropped: int        # Unmatched items


@dataclass
class CorrectedPValue:
    test_name: str
    uncorrected_p: float
    corrected_p: float
    significant: bool  # After correction


@dataclass
class PermutationTestResult:
    observed_delta_ece: float
    p_value: float
    n_permutations: int
    null_distribution_mean: float
    null_distribution_std: float


class BootstrapEngine:
    """Cluster bootstrap CIs and statistical tests for calibration metrics."""

    def compute_ci(
        self,
        predictions: list[Prediction],
        items: list[DatasetItem],
        metric_fn: Callable[[list[Prediction], list[DatasetItem]], float],
        n_resamples: int = 2000,
        alpha: float = 0.05,
        random_seed: int = 42,
    ) -> ConfidenceInterval:
        """Compute bootstrap CI using cluster resampling by seed_id.

        CRITICAL: Resamples by seed_id (cluster bootstrap), NOT individual
        predictions. Multiple variants of the same seed are correlated —
        resampling items would produce artificially narrow CIs.

        Verifies that resampled dataset preserves within-seed structure
        (all variants of a resampled seed are included).

        Args:
            predictions: Predictions to bootstrap.
            items: Dataset items for ground truth and seed_id lookup.
            metric_fn: callable(predictions, items) -> float
            n_resamples: Number of bootstrap resamples (default: 2000).
            alpha: Significance level (default: 0.05).

        Returns ConfidenceInterval with point estimate and CI bounds.
        """
        rng = random.Random(random_seed)

        # Build seed_id -> (predictions, items) mapping
        item_map = {item.item_id: item for item in items}
        pred_map = {pred.item_id: pred for pred in predictions}

        seed_to_item_ids: dict[str, list[str]] = defaultdict(list)
        for item in items:
            if item.item_id in pred_map:
                seed_to_item_ids[item.seed_id].append(item.item_id)

        seed_ids = list(seed_to_item_ids.keys())
        if not seed_ids:
            logger.warning("No seed_ids found — falling back to item-level bootstrap")
            seed_ids = [pred.item_id for pred in predictions]
            seed_to_item_ids = {id_: [id_] for id_ in seed_ids}

        # Point estimate
        point_estimate = metric_fn(predictions, items)

        # Bootstrap resamples
        bootstrap_estimates = []
        for _ in range(n_resamples):
            # Resample seeds with replacement
            resampled_seeds = rng.choices(seed_ids, k=len(seed_ids))
            # Expand to all item_ids for each resampled seed
            resampled_item_ids = []
            for seed_id in resampled_seeds:
                resampled_item_ids.extend(seed_to_item_ids[seed_id])

            resampled_preds = [pred_map[id_] for id_ in resampled_item_ids if id_ in pred_map]
            resampled_items = [item_map[id_] for id_ in resampled_item_ids if id_ in item_map]

            if not resampled_preds:
                continue
            try:
                estimate = metric_fn(resampled_preds, resampled_items)
                bootstrap_estimates.append(estimate)
            except Exception as e:
                logger.debug("Bootstrap resample failed: %s", e)

        if not bootstrap_estimates:
            return ConfidenceInterval(point_estimate, point_estimate, point_estimate,
                                      n_resamples, alpha, True)

        arr = np.array(bootstrap_estimates)
        ci_lower = float(np.percentile(arr, 100 * alpha / 2))
        ci_upper = float(np.percentile(arr, 100 * (1 - alpha / 2)))

        is_degenerate = abs(ci_upper - ci_lower) < 1e-9
        if is_degenerate:
            logger.warning("Degenerate CI: lower=%.4f == upper=%.4f", ci_lower, ci_upper)

        return ConfidenceInterval(
            point_estimate=point_estimate,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            n_resamples=len(bootstrap_estimates),
            alpha=alpha,
            is_degenerate=is_degenerate,
        )

    def pairwise_mcnemar(
        self,
        predictions_a: list[Prediction],
        predictions_b: list[Prediction],
        items: list[DatasetItem],
    ) -> McNemarResult:
        """McNemar's test for paired ACCURACY comparison.

        Aligns prediction sets by item_id before testing.
        Drops and logs unmatched items.

        Note: McNemar tests accuracy, NOT calibration. Use
        permutation_test_delta_ece() for calibration significance.
        """
        from scipy.stats import chi2

        item_map = {item.item_id: item for item in items}
        pred_a_map = {p.item_id: p for p in predictions_a}
        pred_b_map = {p.item_id: p for p in predictions_b}

        common_ids = set(pred_a_map) & set(pred_b_map)
        n_dropped = len(pred_a_map) + len(pred_b_map) - 2 * len(common_ids)
        if n_dropped > 0:
            logger.warning("McNemar: dropped %d unmatched items", n_dropped)

        n_ab = 0  # A correct, B wrong
        n_ba = 0  # B correct, A wrong
        n_concordant = 0

        for item_id in common_ids:
            item = item_map.get(item_id)
            if item is None:
                continue
            gt = item.ground_truth
            a_correct = pred_a_map[item_id].predicted_label == gt
            b_correct = pred_b_map[item_id].predicted_label == gt

            if a_correct and b_correct:
                n_concordant += 1
            elif a_correct and not b_correct:
                n_ab += 1
            elif not a_correct and b_correct:
                n_ba += 1

        # McNemar statistic with continuity correction
        if n_ab + n_ba == 0:
            return McNemarResult(0.0, 1.0, n_concordant, n_ab, n_ba, n_dropped)

        statistic = (abs(n_ab - n_ba) - 1) ** 2 / (n_ab + n_ba)
        p_value = float(1 - chi2.cdf(statistic, df=1))

        return McNemarResult(
            statistic=statistic,
            p_value=p_value,
            n_concordant=n_concordant,
            n_discordant_ab=n_ab,
            n_discordant_ba=n_ba,
            n_dropped=n_dropped,
        )

    def permutation_test_delta_ece(
        self,
        predictions_a: list[Prediction],
        predictions_b: list[Prediction],
        items: list[DatasetItem],
        n_permutations: int = 5000,
        random_seed: int = 42,
    ) -> PermutationTestResult:
        """Permutation test for significant difference in ECE between two guardrails.

        CRITICAL: Permutes guardrail assignments within each paired prediction
        (swaps which model produced which prediction for each item), NOT the
        ECE values themselves. Permuting ECE values would be invalid.

        Test statistic: absolute difference in ECE.
        p-value: fraction of permutations where |ΔECE_perm| >= |ΔECE_obs|.
        """
        from src.evaluation.calibration import CalibrationAnalyzer

        analyzer = CalibrationAnalyzer()
        pred_a_map = {p.item_id: p for p in predictions_a}
        pred_b_map = {p.item_id: p for p in predictions_b}
        common_ids = sorted(set(pred_a_map) & set(pred_b_map))

        if not common_ids:
            logger.warning("No common items for permutation test")
            return PermutationTestResult(0.0, 1.0, 0, 0.0, 0.0)

        paired_a = [pred_a_map[id_] for id_ in common_ids]
        paired_b = [pred_b_map[id_] for id_ in common_ids]

        ece_a = analyzer.compute_ece(paired_a, items).ece
        ece_b = analyzer.compute_ece(paired_b, items).ece
        observed_delta = abs(ece_a - ece_b)

        rng = random.Random(random_seed)
        null_deltas = []

        for _ in range(n_permutations):
            # Permute within-pair assignments: for each item, randomly swap A and B
            perm_a = []
            perm_b = []
            for a, b in zip(paired_a, paired_b):
                if rng.random() < 0.5:
                    perm_a.append(a)
                    perm_b.append(b)
                else:
                    perm_a.append(b)
                    perm_b.append(a)

            try:
                perm_ece_a = analyzer.compute_ece(perm_a, items).ece
                perm_ece_b = analyzer.compute_ece(perm_b, items).ece
                null_deltas.append(abs(perm_ece_a - perm_ece_b))
            except Exception:
                pass

        if not null_deltas:
            return PermutationTestResult(observed_delta, 1.0, 0, 0.0, 0.0)

        null_arr = np.array(null_deltas)
        p_value = float(np.mean(null_arr >= observed_delta))

        return PermutationTestResult(
            observed_delta_ece=observed_delta,
            p_value=p_value,
            n_permutations=len(null_deltas),
            null_distribution_mean=float(np.mean(null_arr)),
            null_distribution_std=float(np.std(null_arr)),
        )

    def apply_holm_bonferroni(
        self,
        p_values: list[tuple[str, float]],
    ) -> list[CorrectedPValue]:
        """Holm-Bonferroni step-down correction.

        Args:
            p_values: list of (test_name, p_value) tuples.
                      Should be scoped per-axis (all pairwise comparisons
                      within one axis), NOT globally across all axes.

        Returns list of CorrectedPValue with both uncorrected and corrected p-values.
        """
        if not p_values:
            return []

        n = len(p_values)
        # Sort by p-value ascending
        sorted_tests = sorted(enumerate(p_values), key=lambda x: x[1][1])
        results = [None] * n

        for rank, (orig_idx, (name, p)) in enumerate(sorted_tests):
            # Holm correction: multiply by (n - rank)
            corrected = min(1.0, p * (n - rank))
            # Ensure monotonicity: corrected p >= previous corrected p
            if rank > 0:
                prev_corrected = results[sorted_tests[rank - 1][0]].corrected_p
                corrected = max(corrected, prev_corrected)
            results[orig_idx] = CorrectedPValue(
                test_name=name,
                uncorrected_p=p,
                corrected_p=corrected,
                significant=corrected < 0.05,
            )

        return results
