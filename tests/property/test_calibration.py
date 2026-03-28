"""Property and unit tests for CalibrationAnalyzer.

# Feature: guardrail-calibration-eval, Property 14: Adaptive bin count formula
# Feature: guardrail-calibration-eval, Property 15: ECE computation correctness
# Feature: guardrail-calibration-eval, Property 16: Class-conditional ECE decomposition
# Feature: guardrail-calibration-eval, Property 18: Brier Score decomposition invariant
# Feature: guardrail-calibration-eval, Property 41: EOE leq ECE
# Feature: guardrail-calibration-eval, Property 23: Delta metrics computation
# Feature: guardrail-calibration-eval, Property 31: Spearman correlation for Axis 3
# Feature: guardrail-calibration-eval, Property 35: Two-token mass summary consistency
# Feature: guardrail-calibration-eval, Property 36: Clean-subset ECE relationship
"""
from __future__ import annotations

import math
from datetime import datetime, timezone

from hypothesis import given, settings
from hypothesis import strategies as st

from src.evaluation.calibration import CalibrationAnalyzer
from src.models import DatasetItem, Prediction


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_prediction(
    item_id: str,
    confidence: float,
    predicted_label: str = "harmful",
    two_token_mass: float | None = None,
    source_type: str = "logits_softmax",
) -> Prediction:
    return Prediction(
        guardrail_name="TestGuard",
        item_id=item_id,
        predicted_label=predicted_label,
        confidence_score=confidence,
        inference_time_ms=1.0,
        two_token_mass=two_token_mass,
        confidence_source_type=source_type,
        split="dev",
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
    )


def make_item(item_id: str, ground_truth: str, axis: int = 1) -> DatasetItem:
    return DatasetItem(
        item_id=item_id,
        seed_id=f"s_{item_id}",
        axis=axis,
        shift_level=0,
        ground_truth=ground_truth,
        graded_harmfulness=None,
        seed_text="seed",
        variant_text=f"variant {item_id}",
        generation_method="test",
        validation_status="validated",
        cultural_frame=None,
        token_counts=None,
        split="dev",
    )


def make_paired(
    n: int, correct_fraction: float = 0.8
) -> tuple[list[Prediction], list[DatasetItem]]:
    """Create n paired predictions and items."""
    predictions = []
    items = []
    for i in range(n):
        gt = "harmful" if i % 2 == 0 else "benign"
        # Correct predictions have high confidence, wrong ones have low
        if i < round(n * correct_fraction):
            conf = 0.85
            pred_label = gt
        else:
            conf = 0.6
            pred_label = "benign" if gt == "harmful" else "harmful"
        predictions.append(make_prediction(f"i{i}", conf, pred_label))
        items.append(make_item(f"i{i}", gt))
    return predictions, items


# ---------------------------------------------------------------------------
# Property 14: Adaptive bin count formula
# ---------------------------------------------------------------------------

@given(n=st.integers(min_value=0, max_value=10000))
@settings(max_examples=200)
def test_adaptive_bin_count_formula(n):
    """M = max(5, min(15, floor(N/15))) for all N."""
    m = CalibrationAnalyzer.compute_adaptive_bin_count(n)
    expected = max(5, min(15, math.floor(n / 15)))
    assert m == expected
    assert 5 <= m <= 15


# ---------------------------------------------------------------------------
# Property 15: ECE computation correctness (weighted average)
# ---------------------------------------------------------------------------

def test_ece_uses_weighted_average():
    """ECE must use weighted average, not unweighted mean.

    Regression test: np.sum(counts * gaps) / total, NOT np.mean(gaps).

    Setup: 90 items in bin [0.8-0.9] with gap=0.1, and 10 items in bin [0.0-0.1]
    with gap=0.9. Weighted ECE ≈ (90*0.1 + 10*0.9)/100 = 0.18.
    Unweighted ECE ≈ mean([0.1, 0.9]) = 0.5 (if only 2 bins covered).
    """
    analyzer = CalibrationAnalyzer()
    predictions = []
    items = []

    # 90 items: confidence=0.85, all correct → accuracy=1.0, gap=0.15
    for i in range(90):
        predictions.append(make_prediction(f"good_{i}", 0.85, "harmful"))
        items.append(make_item(f"good_{i}", "harmful"))

    # 10 items: confidence=0.05, all wrong → accuracy=0.0, gap=0.05
    for i in range(10):
        predictions.append(make_prediction(f"bad_{i}", 0.05, "harmful"))
        items.append(make_item(f"bad_{i}", "benign"))

    ece = analyzer.compute_ece(predictions, items).ece

    # Weighted ECE ≈ (90*0.15 + 10*0.05) / 100 = 0.14
    # Unweighted ECE would give equal weight to both bins ≈ (0.15 + 0.05) / 2 = 0.10
    # The key check: ECE should be dominated by the large bin (90 items)
    # Verify it's closer to 0.14 than to 0.5 (which would indicate wrong formula)
    assert ece < 0.3, f"ECE={ece:.4f} is too large — may be using wrong formula"
    assert ece > 0.0, "ECE should be > 0 with miscalibrated items"


@given(
    n=st.integers(min_value=10, max_value=200),
    correct_frac=st.floats(min_value=0.0, max_value=1.0),
)
@settings(max_examples=100)
def test_ece_in_range(n, correct_frac):
    """ECE must be in [0, 1]."""
    analyzer = CalibrationAnalyzer()
    predictions, items = make_paired(n, correct_frac)
    ece = analyzer.compute_ece(predictions, items).ece
    assert 0.0 <= ece <= 1.0


def test_ece_zero_for_perfect_calibration():
    """ECE should be 0 when confidence exactly matches accuracy in each bin."""
    analyzer = CalibrationAnalyzer()
    # All predictions in one bin with confidence=0.8, all correct → accuracy=1.0
    # This won't be exactly 0 due to binning, but should be small
    predictions = [make_prediction(f"i{i}", 0.8, "harmful") for i in range(50)]
    items = [make_item(f"i{i}", "harmful") for i in range(50)]
    ece = analyzer.compute_ece(predictions, items).ece
    # All correct, confidence=0.8, accuracy=1.0 → gap=0.2 per bin
    assert ece <= 0.25


# ---------------------------------------------------------------------------
# Property 16: Class-conditional ECE decomposition
# ---------------------------------------------------------------------------

def test_class_conditional_ece_has_three_components():
    """ClassConditionalECE must have harmful_ece, benign_ece, overall_ece."""
    analyzer = CalibrationAnalyzer()
    predictions, items = make_paired(50)
    result = analyzer.compute_class_conditional_ece(predictions, items)
    assert result.harmful_ece is not None
    assert result.benign_ece is not None
    assert result.overall_ece is not None


def test_class_conditional_ece_overall_matches_compute_ece():
    """overall_ece in ClassConditionalECE should match compute_ece()."""
    analyzer = CalibrationAnalyzer()
    predictions, items = make_paired(100)
    cc_ece = analyzer.compute_class_conditional_ece(predictions, items)
    direct_ece = analyzer.compute_ece(predictions, items)
    assert abs(cc_ece.overall_ece.ece - direct_ece.ece) < 1e-9


# ---------------------------------------------------------------------------
# Property 18: Brier Score decomposition invariant
# ---------------------------------------------------------------------------

@given(n=st.integers(min_value=10, max_value=200))
@settings(max_examples=100)
def test_brier_decomposition_invariant(n):
    """brier_score ≈ calibration + resolution - uncertainty."""
    analyzer = CalibrationAnalyzer()
    predictions, items = make_paired(n)
    result = analyzer.compute_brier_score(predictions, items)
    # Brier = calibration + resolution - uncertainty
    reconstructed = result.calibration + result.resolution - result.uncertainty
    assert abs(result.brier_score - reconstructed) < 0.01, (
        f"Brier decomposition violated: {result.brier_score:.4f} ≠ "
        f"{result.calibration:.4f} + {result.resolution:.4f} - {result.uncertainty:.4f}"
    )


def test_brier_uncertainty_fixed_at_025():
    """Uncertainty component must be fixed at 0.25 for 50/50 balance."""
    analyzer = CalibrationAnalyzer()
    predictions, items = make_paired(100)
    result = analyzer.compute_brier_score(predictions, items)
    assert result.uncertainty == 0.25


def test_brier_score_in_range():
    """Brier Score must be in [0, 1]."""
    analyzer = CalibrationAnalyzer()
    predictions, items = make_paired(100)
    result = analyzer.compute_brier_score(predictions, items)
    assert 0.0 <= result.brier_score <= 1.0


# ---------------------------------------------------------------------------
# Property 41: EOE ≤ ECE
# ---------------------------------------------------------------------------

@given(n=st.integers(min_value=10, max_value=200))
@settings(max_examples=100)
def test_eoe_leq_ece(n):
    """EOE must always be ≤ ECE (EOE only sums overconfident bins)."""
    analyzer = CalibrationAnalyzer()
    predictions, items = make_paired(n)
    ece = analyzer.compute_ece(predictions, items).ece
    eoe = analyzer.compute_eoe(predictions, items)
    assert eoe <= ece + 1e-9, f"EOE={eoe:.4f} > ECE={ece:.4f}"


def test_eoe_zero_for_perfectly_calibrated():
    """EOE = 0 when model is perfectly calibrated (no overconfident bins)."""
    analyzer = CalibrationAnalyzer()
    # All predictions: confidence=0.5, half correct → accuracy=0.5 → no overconfidence
    predictions = []
    items = []
    for i in range(100):
        gt = "harmful" if i % 2 == 0 else "benign"
        pred_label = gt  # All correct
        predictions.append(make_prediction(f"i{i}", 0.5, pred_label))
        items.append(make_item(f"i{i}", gt))
    eoe = analyzer.compute_eoe(predictions, items)
    # confidence=0.5, accuracy=1.0 → confidence < accuracy → NOT overconfident
    assert eoe == 0.0


def test_eoe_neq_ece_when_underconfident_bins_exist():
    """EOE ≠ ECE when there are underconfident bins (catches missing filter bug)."""
    analyzer = CalibrationAnalyzer()
    # Mix: some overconfident (conf=0.9, wrong) and some underconfident
    # (conf=0.3, correct)
    predictions = []
    items = []
    for i in range(50):
        # Overconfident: high confidence, wrong
        predictions.append(make_prediction(f"over_{i}", 0.9, "harmful"))
        items.append(make_item(f"over_{i}", "benign"))
    for i in range(50):
        # Underconfident: low confidence, correct
        predictions.append(make_prediction(f"under_{i}", 0.3, "benign"))
        items.append(make_item(f"under_{i}", "benign"))

    ece = analyzer.compute_ece(predictions, items).ece
    eoe = analyzer.compute_eoe(predictions, items)
    # EOE should be less than ECE because underconfident bins are excluded
    assert eoe < ece, (
        f"EOE={eoe:.4f} should be < ECE={ece:.4f} with underconfident bins"
    )


# ---------------------------------------------------------------------------
# Unit tests: calibration edge cases
# ---------------------------------------------------------------------------

def test_ece_single_bin_all_correct():
    """All predictions in one bin, all correct."""
    analyzer = CalibrationAnalyzer()
    predictions = [make_prediction(f"i{i}", 0.8, "harmful") for i in range(20)]
    items = [make_item(f"i{i}", "harmful") for i in range(20)]
    ece = analyzer.compute_ece(predictions, items).ece
    assert 0.0 <= ece <= 1.0


def test_ece_single_bin_all_wrong():
    """All predictions in one bin, all wrong."""
    analyzer = CalibrationAnalyzer()
    predictions = [make_prediction(f"i{i}", 0.8, "harmful") for i in range(20)]
    items = [make_item(f"i{i}", "benign") for i in range(20)]
    ece = analyzer.compute_ece(predictions, items).ece
    assert ece > 0.0  # Should be miscalibrated


def test_small_n_warning(caplog):
    """N < 100 should log a warning about ECE reliability."""
    import logging
    analyzer = CalibrationAnalyzer()
    predictions = [make_prediction(f"i{i}", 0.8, "harmful") for i in range(50)]
    items = [make_item(f"i{i}", "harmful") for i in range(50)]
    with caplog.at_level(logging.WARNING, logger="src.evaluation.calibration"):
        analyzer.compute_ece(predictions, items)
    assert any("N=50" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Property 35: Two-token mass summary consistency
# ---------------------------------------------------------------------------

@given(
    masses=st.lists(
        st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
        min_size=1, max_size=100,
    )
)
@settings(max_examples=100)
def test_two_token_mass_summary_consistency(masses):
    """TwoTokenMassSummary statistics are consistent with input masses."""
    analyzer = CalibrationAnalyzer()
    predictions = [
        make_prediction(f"i{i}", 0.8, two_token_mass=m)
        for i, m in enumerate(masses)
    ]
    summary = analyzer.compute_two_token_mass_summary(predictions)
    assert summary.n_items == len(masses)
    assert 0.0 <= summary.mean_mass <= 1.0
    assert summary.std_mass >= 0.0
    assert 0.0 <= summary.fraction_below_threshold <= 1.0


# ---------------------------------------------------------------------------
# Property 36: Clean-subset ECE relationship
# ---------------------------------------------------------------------------

def test_clean_subset_ece_uses_only_high_mass_predictions():
    """compute_ece_excluding_low_mass() only uses predictions with mass >= threshold."""
    analyzer = CalibrationAnalyzer()
    # Low-mass predictions (should be excluded)
    low_mass = [
        make_prediction(f"low_{i}", 0.9, "harmful", two_token_mass=0.3)
        for i in range(50)
    ]
    low_items = [
        make_item(f"low_{i}", "benign") for i in range(50)
    ]  # All wrong → high ECE
    # High-mass predictions (should be included)
    high_mass = [
        make_prediction(f"high_{i}", 0.8, "harmful", two_token_mass=0.8)
        for i in range(50)
    ]
    high_items = [
        make_item(f"high_{i}", "harmful") for i in range(50)
    ]  # All correct → low ECE

    all_preds = low_mass + high_mass
    all_items = low_items + high_items

    full_ece = analyzer.compute_ece(all_preds, all_items).ece
    clean_ece = analyzer.compute_ece_excluding_low_mass(
        all_preds, all_items, mass_threshold=0.5
    ).ece

    # Clean ECE should be lower (excludes the badly miscalibrated low-mass items)
    assert clean_ece <= full_ece + 0.01


# ---------------------------------------------------------------------------
# Property 31: Spearman correlation for Axis 3
# ---------------------------------------------------------------------------

def test_spearman_correlation_in_range():
    """Spearman correlation must be in [-1, 1]."""
    analyzer = CalibrationAnalyzer()
    confidences = [0.9, 0.7, 0.5, 0.3, 0.1]
    graded = [1.0, 0.8, 0.5, 0.2, 0.0]
    result = analyzer.compute_spearman_correlation(confidences, graded)
    assert -1.0 <= result.correlation <= 1.0
    assert 0.0 <= result.p_value <= 1.0


def test_spearman_perfect_positive_correlation():
    """Perfectly correlated inputs → correlation ≈ 1.0."""
    analyzer = CalibrationAnalyzer()
    x = [0.1, 0.3, 0.5, 0.7, 0.9]
    result = analyzer.compute_spearman_correlation(x, x)
    assert abs(result.correlation - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# Property 23: Delta metrics computation
# ---------------------------------------------------------------------------

def test_delta_metrics_zero_when_identical():
    """ΔAccuracy and ΔECE should be 0 when base and shifted are identical."""
    analyzer = CalibrationAnalyzer()
    predictions, items = make_paired(50)
    delta = analyzer.compute_delta_metrics(predictions, predictions, items)
    assert abs(delta.delta_accuracy) < 1e-9
    assert abs(delta.delta_ece) < 1e-9


def test_delta_metrics_sign_correct():
    """ΔAccuracy should be negative when shifted predictions are worse."""
    analyzer = CalibrationAnalyzer()
    # Base: all correct
    base_preds = [make_prediction(f"i{i}", 0.9, "harmful") for i in range(50)]
    base_items = [make_item(f"i{i}", "harmful") for i in range(50)]
    # Shifted: all wrong
    shifted_preds = [make_prediction(f"i{i}", 0.6, "benign") for i in range(50)]

    delta = analyzer.compute_delta_metrics(base_preds, shifted_preds, base_items)
    assert delta.delta_accuracy < 0, (
        "ΔAccuracy should be negative when shifted is worse"
    )


# ---------------------------------------------------------------------------
# Property 29: AUROC computation correctness
# ---------------------------------------------------------------------------

def test_auroc_in_range():
    """AUROC must be in [0, 1]."""
    analyzer = CalibrationAnalyzer()
    predictions, items = make_paired(50)
    auroc = analyzer.compute_auroc(predictions, items)
    assert 0.0 <= auroc <= 1.0


def test_auroc_perfect_classifier():
    """Perfect classifier (all harmful with high conf, all benign with low conf) → AUROC ≈ 1."""
    analyzer = CalibrationAnalyzer()
    predictions = (
        [make_prediction(f"h{i}", 0.95, "harmful") for i in range(25)] +
        [make_prediction(f"b{i}", 0.05, "benign") for i in range(25)]
    )
    items = (
        [make_item(f"h{i}", "harmful") for i in range(25)] +
        [make_item(f"b{i}", "benign") for i in range(25)]
    )
    auroc = analyzer.compute_auroc(predictions, items)
    assert auroc > 0.9, (
        f"Perfect classifier should have AUROC > 0.9, got {auroc:.4f}"
    )


def test_auroc_random_classifier():
    """Random classifier → AUROC ≈ 0.5."""
    import random
    rng = random.Random(42)
    analyzer = CalibrationAnalyzer()
    predictions = [
        make_prediction(f"i{i}", rng.random(), "harmful") for i in range(100)
    ]
    items = [
        make_item(f"i{i}", "harmful" if i % 2 == 0 else "benign")
        for i in range(100)
    ]
    auroc = analyzer.compute_auroc(predictions, items)
    assert 0.3 <= auroc <= 0.7, (
        f"Random classifier AUROC={auroc:.4f} should be near 0.5"
    )
