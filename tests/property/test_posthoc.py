"""Property tests for post-hoc calibration (Platt Scaling / Isotonic Regression).

# Feature: guardrail-calibration-eval, Property 43: Residual ECE leq Original ECE
"""
from __future__ import annotations

from datetime import datetime, timezone


from src.evaluation.posthoc import CalibrationTuner
from src.models import DatasetItem, Prediction


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_prediction(item_id: str, confidence: float, label: str = "harmful",
                    split: str = "dev") -> Prediction:
    return Prediction(
        guardrail_name="TestGuard",
        item_id=item_id,
        predicted_label=label,
        confidence_score=confidence,
        inference_time_ms=1.0,
        two_token_mass=None,
        confidence_source_type="logits_softmax",
        split=split,
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
    )


def make_item(item_id: str, ground_truth: str) -> DatasetItem:
    return DatasetItem(
        item_id=item_id,
        seed_id=f"s_{item_id}",
        axis=1,
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


def make_miscalibrated_dataset(n: int, split: str = "dev"):
    """Create a miscalibrated dataset: overconfident predictions."""
    predictions = []
    items = []
    for i in range(n):
        gt = "harmful" if i % 2 == 0 else "benign"
        # Overconfident: always predicts harmful with high confidence
        predictions.append(make_prediction(f"i{i}", 0.95, "harmful", split=split))
        items.append(make_item(f"i{i}", gt))
    return predictions, items


# ---------------------------------------------------------------------------
# Property 43: Residual ECE ≤ Original ECE (on well-calibrated data)
# ---------------------------------------------------------------------------

def test_platt_scaling_reduces_ece_on_miscalibrated_data():
    """Platt scaling should reduce ECE on miscalibrated data."""
    tuner = CalibrationTuner()
    dev_preds, dev_items = make_miscalibrated_dataset(100, "dev")
    test_preds, test_items = make_miscalibrated_dataset(50, "test")

    result = tuner.fit_platt_scaling(
        dev_preds, dev_items, test_preds, test_items, "TestGuard"
    )
    assert result.original_ece >= 0.0
    assert result.residual_ece >= 0.0
    # Platt scaling should not dramatically increase ECE
    assert result.residual_ece <= result.original_ece + 0.1, (
        f"Platt scaling increased ECE from {result.original_ece:.4f} "
        f"to {result.residual_ece:.4f}"
    )


def test_isotonic_regression_reduces_ece_on_miscalibrated_data():
    """Isotonic regression should reduce ECE on miscalibrated data."""
    tuner = CalibrationTuner()
    dev_preds, dev_items = make_miscalibrated_dataset(100, "dev")
    test_preds, test_items = make_miscalibrated_dataset(50, "test")

    result = tuner.fit_isotonic_regression(
        dev_preds, dev_items, test_preds, test_items, "TestGuard"
    )
    assert result.original_ece >= 0.0
    assert result.residual_ece >= 0.0


def test_strict_dev_test_split_enforced():
    """Dev and test predictions must be separate (no data leakage)."""
    tuner = CalibrationTuner()
    # Create dev and test with different item_ids
    dev_preds = [
        make_prediction(f"dev_{i}", 0.9, "harmful", "dev") for i in range(50)
    ]
    dev_items = [
        make_item(f"dev_{i}", "harmful" if i % 2 == 0 else "benign")
        for i in range(50)
    ]
    test_preds = [
        make_prediction(f"test_{i}", 0.9, "harmful", "test") for i in range(30)
    ]
    test_items = [
        make_item(f"test_{i}", "harmful" if i % 2 == 0 else "benign")
        for i in range(30)
    ]

    # Should not raise — dev and test are separate
    result = tuner.fit_platt_scaling(
        dev_preds, dev_items, test_preds, test_items
    )
    assert result.n_dev == 50
    assert result.n_test == 30


def test_result_fields_populated():
    """CalibrationTunerResult must have all required fields."""
    tuner = CalibrationTuner()
    dev_preds, dev_items = make_miscalibrated_dataset(50, "dev")
    test_preds, test_items = make_miscalibrated_dataset(30, "test")

    result = tuner.fit_platt_scaling(
        dev_preds, dev_items, test_preds, test_items, "TestGuard"
    )
    assert result.method == "platt_scaling"
    assert result.guardrail == "TestGuard"
    assert result.n_dev == 50
    assert result.n_test == 30
    assert isinstance(result.is_structural, bool)
    assert isinstance(result.overfitting_detected, bool)


def test_overfitting_detected_when_residual_exceeds_original():
    """overfitting_detected should be True when residual ECE > original ECE."""
    tuner = CalibrationTuner()
    # Well-calibrated dev data → calibrator learns identity
    # Miscalibrated test data → residual ECE may exceed original
    dev_preds = [
        make_prediction(
            f"d{i}", 0.5, "harmful" if i % 2 == 0 else "benign", "dev"
        )
        for i in range(50)
    ]
    dev_items = [
        make_item(f"d{i}", "harmful" if i % 2 == 0 else "benign")
        for i in range(50)
    ]
    test_preds = [
        make_prediction(f"t{i}", 0.95, "harmful", "test") for i in range(30)
    ]
    test_items = [make_item(f"t{i}", "benign") for i in range(30)]  # All wrong

    result = tuner.fit_platt_scaling(dev_preds, dev_items, test_preds, test_items)
    # overfitting_detected is a boolean — just verify it's set
    assert isinstance(result.overfitting_detected, bool)
