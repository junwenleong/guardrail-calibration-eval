"""ValidationPipeline: LLM-as-judge validation and human review scaffolding.

Disputed item resolution policy:
- When LLM-Judge disagrees with assigned ground_truth, item is flagged "disputed"
- When human review overrides the label, item gets "disputed_human_override"
- Items are NEVER discarded — discarding biases toward easy examples
- Disagreement rates are reported per axis for calibration context

LLM judge uses Ollama (qwen2.5:14b) locally — no API keys required.
The judge model is the SAME model as the generator (qwen2.5:14b) but this is
acceptable because the judge uses a different prompt format and task framing.
For stricter independence, configure a different Ollama model tag.
"""
from __future__ import annotations

import logging
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable

from src.models import DatasetItem
from src.utils.ollama_client import OllamaClient

logger = logging.getLogger(__name__)


@dataclass
class DisagreementReport:
    axis: int
    n_items: int
    n_disputed: int
    disagreement_rate: float


@dataclass
class InterRaterReport:
    axis: int
    metric: str          # "cohens_kappa" or "spearman"
    value: float
    n_items: int
    below_threshold: bool  # True if kappa < 0.6


class ValidationPipeline:
    """Validates dataset items using LLM-as-judge and human review.

    The judge uses Ollama (qwen2.5:14b) locally. For stricter model
    independence, configure a different model tag (e.g. mistral:7b).
    """

    def __init__(self, ollama_client: OllamaClient | None = None):
        self._ollama = ollama_client

    def make_judge_fn(self) -> Callable[[str], str] | None:
        """Return a judge_fn backed by Ollama, or None if no client configured."""
        if self._ollama is None:
            return None
        return self._ollama.judge

    def validate_with_llm_judge(
        self,
        items: list[DatasetItem],
        judge_fn: Callable[[str], str] | None = None,
    ) -> list[DatasetItem]:
        """Use an LLM judge to validate ground truth labels.

        Args:
            items: Dataset items to validate.
            judge_fn: callable(variant_text) -> "harmful" | "benign".
                      If None, uses the Ollama client configured at init.

        Returns items with validation_status updated:
        - "validated" if judge agrees with ground_truth
        - "disputed" if judge disagrees (flagged for human review)
        """
        if judge_fn is None:
            judge_fn = self.make_judge_fn()
        if judge_fn is None:
            logger.warning("No judge_fn provided and no Ollama client — skipping validation")
            return items
        for item in items:
            try:
                judge_label = judge_fn(item.variant_text)
                if judge_label == item.ground_truth:
                    if item.validation_status == "pending":
                        item.validation_status = "validated"
                else:
                    item.validation_status = "disputed"
                    logger.debug(
                        "Disputed: item=%s judge=%s assigned=%s",
                        item.item_id, judge_label, item.ground_truth,
                    )
            except Exception as e:
                logger.warning("Judge failed for item %s: %s", item.item_id, e)
        return items

    def select_human_review_sample(
        self,
        items: list[DatasetItem],
        fraction: float = 0.20,
        random_seed: int = 42,
        oversample_disputed: bool = True,
    ) -> list[DatasetItem]:
        """Select a stratified 20% sample for human review.

        Stratified by axis. If oversample_disputed=True, ALL disputed items
        are included in the sample (in addition to the stratified 20%).
        This ensures disputed items are corrected, not left with wrong labels.
        """
        rng = random.Random(random_seed)

        # Always include all disputed items
        disputed = [i for i in items if i.validation_status == "disputed"]
        disputed_ids = {i.item_id for i in disputed}

        # Stratified sample from non-disputed items
        by_axis: dict[int, list[DatasetItem]] = defaultdict(list)
        for item in items:
            if item.item_id not in disputed_ids:
                by_axis[item.axis].append(item)

        sampled: list[DatasetItem] = list(disputed) if oversample_disputed else []
        for axis, axis_items in by_axis.items():
            rng.shuffle(axis_items)
            n = max(1, round(len(axis_items) * fraction))
            sampled.extend(axis_items[:n])

        logger.info(
            "Human review sample: %d items (%d disputed + %d stratified)",
            len(sampled), len(disputed), len(sampled) - len(disputed),
        )
        return sampled

    def apply_human_overrides(
        self,
        items: list[DatasetItem],
        human_labels: dict[str, str],
    ) -> list[DatasetItem]:
        """Apply human label overrides to disputed items.

        Args:
            items: All dataset items.
            human_labels: dict mapping item_id -> human-assigned label.

        When human label differs from current ground_truth:
        - Updates ground_truth to human label
        - Sets validation_status = "disputed_human_override"
        - Item STAYS in dataset (not discarded)

        When human label matches ground_truth:
        - Sets validation_status = "validated"
        """
        for item in items:
            if item.item_id in human_labels:
                human_label = human_labels[item.item_id]
                if human_label != item.ground_truth:
                    item.ground_truth = human_label
                    item.validation_status = "disputed_human_override"
                    logger.debug(
                        "Human override: item=%s new_label=%s", item.item_id, human_label
                    )
                else:
                    item.validation_status = "validated"
        return items

    def compute_judge_error_rate(
        self,
        judge_labels: dict[str, str],
        human_labels: dict[str, str],
    ) -> float:
        """Compute fraction of items where judge differs from human label.

        Args:
            judge_labels: dict item_id -> judge-assigned label
            human_labels: dict item_id -> human-assigned label (ground truth)

        Returns error rate in [0, 1].
        """
        common_ids = set(judge_labels) & set(human_labels)
        if not common_ids:
            logger.warning("No common items between judge and human labels")
            return 0.0
        errors = sum(
            1 for item_id in common_ids
            if judge_labels[item_id] != human_labels[item_id]
        )
        rate = errors / len(common_ids)
        logger.info(
            "Judge error rate: %.3f (%d/%d items disagree)",
            rate, errors, len(common_ids),
        )
        return rate

    def compute_disagreement_rates(
        self, items: list[DatasetItem]
    ) -> list[DisagreementReport]:
        """Report LLM-Judge/human disagreement rate per axis.

        High disagreement on Axis 2 (Cultural) vs Axis 1 (Register) is
        critical context for interpreting calibration results under ambiguity.
        """
        by_axis: dict[int, list[DatasetItem]] = defaultdict(list)
        for item in items:
            by_axis[item.axis].append(item)

        reports = []
        for axis, axis_items in sorted(by_axis.items()):
            n_disputed = sum(
                1 for i in axis_items
                if i.validation_status in ("disputed", "disputed_human_override")
            )
            rate = n_disputed / len(axis_items) if axis_items else 0.0
            if rate > 0.15:
                logger.warning(
                    "High disagreement rate on Axis %d: %.1f%% (%d/%d items)",
                    axis, 100 * rate, n_disputed, len(axis_items),
                )
            reports.append(DisagreementReport(
                axis=axis,
                n_items=len(axis_items),
                n_disputed=n_disputed,
                disagreement_rate=rate,
            ))
        return reports

    def compute_inter_rater_reliability(
        self,
        ratings_a: list[float | str],
        ratings_b: list[float | str],
        metric: str = "cohens_kappa",
        axis: int = 2,
    ) -> InterRaterReport:
        """Compute inter-rater reliability between two annotators.

        Args:
            ratings_a, ratings_b: Paired ratings from two annotators.
            metric: "cohens_kappa" for Axis 2 (binary), "spearman" for Axis 3 (continuous).
            axis: Which axis these ratings are for.

        Returns InterRaterReport. Flags kappa < 0.6 for Axis 2.
        """
        if len(ratings_a) != len(ratings_b):
            raise ValueError(
                f"Rating lists must have equal length: {len(ratings_a)} vs {len(ratings_b)}"
            )
        if not ratings_a:
            raise ValueError("Rating lists must not be empty")

        if metric == "cohens_kappa":
            value = self._cohens_kappa(ratings_a, ratings_b)
            below_threshold = value < 0.6
            if below_threshold:
                logger.warning(
                    "Axis %d Cohen's kappa=%.3f < 0.6 — review protocol revision needed",
                    axis, value,
                )
        elif metric == "spearman":
            value = self._spearman_correlation(
                [float(r) for r in ratings_a],
                [float(r) for r in ratings_b],
            )
            below_threshold = False  # No threshold for Spearman
        else:
            raise ValueError(f"Unknown metric: {metric}")

        return InterRaterReport(
            axis=axis,
            metric=metric,
            value=value,
            n_items=len(ratings_a),
            below_threshold=below_threshold,
        )

    # ------------------------------------------------------------------
    # Statistical helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _cohens_kappa(a: list, b: list) -> float:
        """Compute Cohen's kappa for two raters."""
        from collections import Counter
        labels = sorted(set(a) | set(b))
        n = len(a)
        if n == 0:
            return 0.0

        # Observed agreement
        p_o = sum(1 for x, y in zip(a, b) if x == y) / n

        # Expected agreement
        count_a = Counter(a)
        count_b = Counter(b)
        p_e = sum(count_a[label] * count_b[label] for label in labels) / (n ** 2)

        if p_e == 1.0:
            return 1.0
        return (p_o - p_e) / (1.0 - p_e)

    @staticmethod
    def _spearman_correlation(a: list[float], b: list[float]) -> float:
        """Compute Spearman rank correlation."""
        from scipy.stats import spearmanr
        corr, _ = spearmanr(a, b)
        return float(corr)
