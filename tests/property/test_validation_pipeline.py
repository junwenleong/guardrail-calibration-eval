"""Property tests for ValidationPipeline.

# Feature: guardrail-calibration-eval, Property 24: Validation disagreement flagging
# Feature: guardrail-calibration-eval, Property 25: Stratified human review coverage
# Feature: guardrail-calibration-eval, Property 26: Judge error rate computation
# Feature: guardrail-calibration-eval, Property 27: Cohen's kappa range and computation
"""
from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from src.datasets.validator import ValidationPipeline
from src.models import DatasetItem


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_item(
    item_id: str, ground_truth: str, axis: int = 1, status: str = "pending"
) -> DatasetItem:
    return DatasetItem(
        item_id=item_id,
        seed_id=f"seed_{item_id}",
        axis=axis,
        shift_level=0,
        ground_truth=ground_truth,
        graded_harmfulness=None,
        seed_text="seed text",
        variant_text=f"variant text {item_id}",
        generation_method="test",
        validation_status=status,
        cultural_frame=None,
        token_counts=None,
        split="dev",
    )


def make_items(n: int, axis: int = 1) -> list[DatasetItem]:
    items = []
    for i in range(n):
        gt = "harmful" if i % 2 == 0 else "benign"
        items.append(make_item(f"item_{i:04d}", gt, axis=axis))
    return items


# ---------------------------------------------------------------------------
# Property 24: Validation disagreement flagging
# ---------------------------------------------------------------------------

def test_judge_agreement_sets_validated():
    """When judge agrees with ground_truth, status becomes 'validated'."""
    pipeline = ValidationPipeline()
    items = [make_item("i1", "harmful")]

    def judge_fn(text):
        return "harmful"  # Always agrees

    result = pipeline.validate_with_llm_judge(items, judge_fn)
    assert result[0].validation_status == "validated"


def test_judge_disagreement_sets_disputed():
    """When judge disagrees with ground_truth, status becomes 'disputed'."""
    pipeline = ValidationPipeline()
    items = [make_item("i1", "harmful")]

    def judge_fn(text):
        return "benign"  # Always disagrees

    result = pipeline.validate_with_llm_judge(items, judge_fn)
    assert result[0].validation_status == "disputed"


@given(
    n_items=st.integers(min_value=2, max_value=20),
)
@settings(max_examples=100)
def test_disputed_count_matches_disagreements(n_items):
    """Number of disputed items equals number of judge disagreements."""
    pipeline = ValidationPipeline()
    items = make_items(n_items)

    # Build a fixed judge that disagrees with the first half
    n_disagree = n_items // 2
    disagree_ids = {items[i].item_id for i in range(n_disagree)}

    def judge_fn(text):
        # Find the item by variant_text
        for item in items:
            if item.variant_text == text:
                if item.item_id in disagree_ids:
                    return "benign" if item.ground_truth == "harmful" else "harmful"
                return item.ground_truth
        return "benign"

    result = pipeline.validate_with_llm_judge(items, judge_fn)
    n_disputed = sum(1 for item in result if item.validation_status == "disputed")
    assert n_disputed == n_disagree


# ---------------------------------------------------------------------------
# Property 25: Stratified human review coverage
# ---------------------------------------------------------------------------

def test_human_review_sample_includes_all_disputed():
    """All disputed items must be included in the human review sample."""
    pipeline = ValidationPipeline()
    items = make_items(100)
    # Mark some as disputed
    for item in items[:10]:
        item.validation_status = "disputed"

    sample = pipeline.select_human_review_sample(items, fraction=0.20)
    sample_ids = {item.item_id for item in sample}
    disputed_ids = {
        item.item_id for item in items
        if item.validation_status == "disputed"
    }
    assert disputed_ids.issubset(sample_ids), (
        "Not all disputed items in human review sample"
    )


def test_human_review_sample_covers_all_axes():
    """Human review sample should include items from all axes."""
    pipeline = ValidationPipeline()
    items = []
    for axis in range(1, 6):
        items.extend(make_items(20, axis=axis))

    sample = pipeline.select_human_review_sample(items, fraction=0.20)
    sample_axes = {item.axis for item in sample}
    assert sample_axes == {1, 2, 3, 4, 5}, f"Missing axes in sample: {sample_axes}"


# ---------------------------------------------------------------------------
# Property 26: Judge error rate computation
# ---------------------------------------------------------------------------

def test_judge_error_rate_zero_when_all_agree():
    pipeline = ValidationPipeline()
    judge_labels = {"i1": "harmful", "i2": "benign"}
    human_labels = {"i1": "harmful", "i2": "benign"}
    rate = pipeline.compute_judge_error_rate(judge_labels, human_labels)
    assert rate == 0.0


def test_judge_error_rate_one_when_all_disagree():
    pipeline = ValidationPipeline()
    judge_labels = {"i1": "harmful", "i2": "harmful"}
    human_labels = {"i1": "benign", "i2": "benign"}
    rate = pipeline.compute_judge_error_rate(judge_labels, human_labels)
    assert rate == 1.0


@given(
    n=st.integers(min_value=1, max_value=100),
    n_errors=st.integers(min_value=0, max_value=100),
)
@settings(max_examples=100)
def test_judge_error_rate_in_range(n, n_errors):
    """Judge error rate must be in [0, 1]."""
    n_errors = min(n_errors, n)
    pipeline = ValidationPipeline()
    judge_labels = {f"i{i}": "harmful" for i in range(n)}
    human_labels = {
        f"i{i}": ("benign" if i < n_errors else "harmful")
        for i in range(n)
    }
    rate = pipeline.compute_judge_error_rate(judge_labels, human_labels)
    assert 0.0 <= rate <= 1.0


# ---------------------------------------------------------------------------
# Property 27: Cohen's kappa range and computation
# ---------------------------------------------------------------------------

def test_cohens_kappa_perfect_agreement():
    """Perfect agreement → kappa = 1.0."""
    pipeline = ValidationPipeline()
    labels = ["harmful", "benign", "harmful", "benign"]
    report = pipeline.compute_inter_rater_reliability(
        labels, labels, metric="cohens_kappa"
    )
    assert abs(report.value - 1.0) < 1e-9


def test_cohens_kappa_in_range():
    """Cohen's kappa must be in [-1, 1]."""
    pipeline = ValidationPipeline()
    a = ["harmful", "benign", "harmful", "benign", "harmful"]
    b = ["benign", "harmful", "benign", "harmful", "benign"]
    report = pipeline.compute_inter_rater_reliability(a, b, metric="cohens_kappa")
    assert -1.0 <= report.value <= 1.0


def test_cohens_kappa_below_threshold_flagged():
    """Kappa < 0.6 should set below_threshold=True."""
    pipeline = ValidationPipeline()
    # Mostly disagreeing → low kappa
    a = ["harmful"] * 10 + ["benign"] * 10
    b = ["benign"] * 10 + ["harmful"] * 10
    report = pipeline.compute_inter_rater_reliability(a, b, metric="cohens_kappa")
    assert report.below_threshold


def test_cohens_kappa_above_threshold_not_flagged():
    """Kappa >= 0.6 should set below_threshold=False."""
    pipeline = ValidationPipeline()
    a = ["harmful", "benign", "harmful", "benign"]
    b = ["harmful", "benign", "harmful", "benign"]
    report = pipeline.compute_inter_rater_reliability(a, b, metric="cohens_kappa")
    assert not report.below_threshold


# ---------------------------------------------------------------------------
# Disputed item resolution policy
# ---------------------------------------------------------------------------

def test_human_override_updates_label_and_status():
    """Human override changes ground_truth and sets disputed_human_override."""
    pipeline = ValidationPipeline()
    items = [make_item("i1", "harmful", status="disputed")]
    result = pipeline.apply_human_overrides(items, {"i1": "benign"})
    assert result[0].ground_truth == "benign"
    assert result[0].validation_status == "disputed_human_override"


def test_human_agreement_sets_validated():
    """When human agrees with ground_truth, status becomes 'validated'."""
    pipeline = ValidationPipeline()
    items = [make_item("i1", "harmful", status="disputed")]
    result = pipeline.apply_human_overrides(items, {"i1": "harmful"})
    assert result[0].ground_truth == "harmful"
    assert result[0].validation_status == "validated"


def test_items_never_discarded():
    """Items must remain in dataset after human override (never discarded)."""
    pipeline = ValidationPipeline()
    items = make_items(10)
    for item in items:
        item.validation_status = "disputed"
    human_labels = {item.item_id: "benign" for item in items}
    result = pipeline.apply_human_overrides(items, human_labels)
    assert len(result) == 10, "Items were discarded — this violates the policy"
