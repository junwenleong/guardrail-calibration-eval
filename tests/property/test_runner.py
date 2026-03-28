"""Property and unit tests for ExperimentRunner.

# Feature: guardrail-calibration-eval, Property 33: Checkpoint resumption correctness
# Feature: guardrail-calibration-eval, Property 34: Atomic checkpoint integrity
# Feature: guardrail-calibration-eval, Property 38: Completeness verification
"""
from __future__ import annotations

import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from src.evaluation.runner import ExperimentRunner
from src.models import CanaryCheckResult, DatasetItem, Prediction


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_prediction(item_id: str, guardrail: str = "TestGuard") -> Prediction:
    return Prediction(
        guardrail_name=guardrail,
        item_id=item_id,
        predicted_label="benign",
        confidence_score=0.8,
        inference_time_ms=10.0,
        two_token_mass=None,
        confidence_source_type="logits_softmax",
        split="dev",
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
    )


def make_dataset_item(item_id: str) -> DatasetItem:
    return DatasetItem(
        item_id=item_id,
        seed_id=f"seed_{item_id}",
        axis=1,
        shift_level=0,
        ground_truth="benign",
        graded_harmfulness=None,
        seed_text="seed",
        variant_text=f"variant {item_id}",
        generation_method="test",
        validation_status="validated",
        cultural_frame=None,
        token_counts=None,
        split="dev",
    )


def make_runner(tmp_dir: str) -> ExperimentRunner:
    with patch("src.evaluation.runner.set_global_seeds"), \
         patch("src.evaluation.runner.log_environment"):
        return ExperimentRunner(checkpoint_dir=tmp_dir, checkpoint_frequency=5)


# ---------------------------------------------------------------------------
# Property 34: Atomic checkpoint integrity
# ---------------------------------------------------------------------------

def test_atomic_checkpoint_creates_file():
    """_atomic_checkpoint() creates the target file."""
    with tempfile.TemporaryDirectory() as tmp:
        runner = make_runner(tmp)
        predictions = [make_prediction(f"i{i}") for i in range(10)]
        path = Path(tmp) / "test.parquet"
        runner._atomic_checkpoint(predictions, path)
        assert path.exists()


def test_atomic_checkpoint_no_tmp_file_remains():
    """After _atomic_checkpoint(), no .tmp file should remain."""
    with tempfile.TemporaryDirectory() as tmp:
        runner = make_runner(tmp)
        predictions = [make_prediction(f"i{i}") for i in range(5)]
        path = Path(tmp) / "test.parquet"
        runner._atomic_checkpoint(predictions, path)
        tmp_path = path.with_suffix(".tmp")
        assert not tmp_path.exists(), ".tmp file was not cleaned up"


def test_atomic_checkpoint_content_correct():
    """Checkpoint file contains all predictions with correct data."""
    with tempfile.TemporaryDirectory() as tmp:
        runner = make_runner(tmp)
        predictions = [make_prediction(f"item_{i}") for i in range(10)]
        path = Path(tmp) / "preds.parquet"
        runner._atomic_checkpoint(predictions, path)

        df = pd.read_parquet(path)
        assert len(df) == 10
        assert set(df["item_id"]) == {f"item_{i}" for i in range(10)}


@given(n=st.integers(min_value=1, max_value=100))
@settings(max_examples=50)
def test_atomic_checkpoint_roundtrip(n):
    """Predictions written to checkpoint can be read back correctly."""
    with tempfile.TemporaryDirectory() as tmp:
        runner = make_runner(tmp)
        predictions = [make_prediction(f"i{i}") for i in range(n)]
        path = Path(tmp) / "roundtrip.parquet"
        runner._atomic_checkpoint(predictions, path)

        df = pd.read_parquet(path)
        assert len(df) == n
        restored = [Prediction(**row) for row in df.to_dict("records")]
        assert [p.item_id for p in restored] == [p.item_id for p in predictions]


# ---------------------------------------------------------------------------
# Property 33: Checkpoint resumption correctness
# ---------------------------------------------------------------------------

def test_load_and_run_resumes_from_checkpoint():
    """_load_and_run() resumes from existing checkpoint without re-running done items."""
    with tempfile.TemporaryDirectory() as tmp:
        runner = make_runner(tmp)
        items = [make_dataset_item(f"item_{i}") for i in range(15)]

        # Pre-populate checkpoint with first 10 items
        existing = [make_prediction(f"item_{i}") for i in range(10)]
        checkpoint_path = Path(tmp) / "TestGuard.parquet"
        runner._atomic_checkpoint(existing, checkpoint_path)

        # Mock adapter that tracks how many times predict() is called
        adapter = MagicMock()
        adapter.get_model_name.return_value = "TestGuard"
        adapter.confidence_source_type = "logits_softmax"
        adapter.predict.return_value = MagicMock(
            label="benign",
            confidence_score=0.8,
            two_token_mass=None,
        )

        with patch("src.evaluation.runner.set_global_seeds"), \
             patch("src.evaluation.runner.log_environment"):
            predictions = runner._load_and_run(adapter, items, checkpoint_path)

        # Should only have called predict() for the remaining 5 items
        assert adapter.predict.call_count == 5
        assert len(predictions) == 15


# ---------------------------------------------------------------------------
# Property 38: Completeness verification
# ---------------------------------------------------------------------------

def test_completeness_all_present():
    """verify_completeness() reports 0 missing when all items have predictions."""
    with tempfile.TemporaryDirectory() as tmp:
        runner = make_runner(tmp)
        items = [make_dataset_item(f"i{i}") for i in range(20)]
        predictions = [make_prediction(f"i{i}") for i in range(20)]
        report = runner.verify_completeness(items, predictions)
        assert report.total_expected == 20
        assert report.total_found == 20
        assert report.missing_item_ids == []


def test_completeness_detects_missing():
    """verify_completeness() correctly identifies missing item_ids."""
    with tempfile.TemporaryDirectory() as tmp:
        runner = make_runner(tmp)
        items = [make_dataset_item(f"i{i}") for i in range(10)]
        predictions = [make_prediction(f"i{i}") for i in range(7)]  # 3 missing
        report = runner.verify_completeness(items, predictions)
        assert report.total_expected == 10
        assert report.total_found == 7
        assert len(report.missing_item_ids) == 3


@given(
    n_total=st.integers(min_value=5, max_value=50),
    n_missing=st.integers(min_value=0, max_value=5),
)
@settings(max_examples=50)
def test_completeness_missing_count_correct(n_total, n_missing):
    """Missing count in report matches actual missing items."""
    n_missing = min(n_missing, n_total)
    with tempfile.TemporaryDirectory() as tmp:
        runner = make_runner(tmp)
        items = [make_dataset_item(f"i{i}") for i in range(n_total)]
        predictions = [make_prediction(f"i{i}") for i in range(n_total - n_missing)]
        report = runner.verify_completeness(items, predictions)
        assert len(report.missing_item_ids) == n_missing


# ---------------------------------------------------------------------------
# Unit tests: sanity check thresholds
# ---------------------------------------------------------------------------

def test_sanity_check_halts_on_constant_scores():
    """run_sanity_check() raises RuntimeError when std < 0.01."""
    with tempfile.TemporaryDirectory() as tmp:
        runner = make_runner(tmp)
        items = [make_dataset_item(f"i{i}") for i in range(10)]
        items[0].ground_truth = "harmful"

        adapter = MagicMock()
        adapter.get_model_name.return_value = "ConstantGuard"
        # Always returns 0.999 — constant scores
        adapter.predict.return_value = MagicMock(
            label="benign", confidence_score=0.999, two_token_mass=None
        )

        with patch("src.evaluation.runner.set_global_seeds"), \
             patch("src.evaluation.runner.log_environment"), \
             patch.object(runner, "_unload_with_vram_check"):
            with pytest.raises(RuntimeError, match="Sanity check FAILED"):
                runner.run_sanity_check([adapter], items)


def test_sanity_check_flags_low_bin_coverage():
    """run_sanity_check() warns when < 5 histogram bins covered."""
    with tempfile.TemporaryDirectory() as tmp:
        runner = make_runner(tmp)
        items = [make_dataset_item(f"i{i}") for i in range(100)]
        for i in range(50):
            items[i].ground_truth = "harmful"

        call_count = [0]
        def predict_side_effect(text):
            # Scores clustered in 0.8-0.9 range (only 1-2 bins covered)
            score = 0.85 + (call_count[0] % 5) * 0.01
            call_count[0] += 1
            return MagicMock(label="benign", confidence_score=score, two_token_mass=None)

        adapter = MagicMock()
        adapter.get_model_name.return_value = "ClusteredGuard"
        adapter.predict.side_effect = predict_side_effect

        with patch("src.evaluation.runner.set_global_seeds"), \
             patch("src.evaluation.runner.log_environment"), \
             patch.object(runner, "_unload_with_vram_check"):
            reports = runner.run_sanity_check([adapter], items)

        assert any("bins covered" in w for w in reports[0].warnings)


def test_sanity_check_flags_low_accuracy():
    """run_sanity_check() warns when accuracy < 50%."""
    with tempfile.TemporaryDirectory() as tmp:
        runner = make_runner(tmp)
        items = [make_dataset_item(f"i{i}") for i in range(100)]
        # All items are harmful
        for item in items:
            item.ground_truth = "harmful"

        call_count = [0]
        def predict_side_effect(text):
            # Spread scores across bins but always predict "benign" (wrong for harmful items)
            score = (call_count[0] % 10) * 0.1 + 0.05
            call_count[0] += 1
            return MagicMock(label="benign", confidence_score=score, two_token_mass=None)

        adapter = MagicMock()
        adapter.get_model_name.return_value = "WrongGuard"
        adapter.predict.side_effect = predict_side_effect

        with patch("src.evaluation.runner.set_global_seeds"), \
             patch("src.evaluation.runner.log_environment"), \
             patch.object(runner, "_unload_with_vram_check"):
            reports = runner.run_sanity_check([adapter], items)

        assert any("accuracy" in w for w in reports[0].warnings)


# ---------------------------------------------------------------------------
# Property 37: API canary drift detection
# ---------------------------------------------------------------------------

def test_canary_no_drift_when_scores_identical():
    """compare_canary_runs() reports no drift when scores are identical."""
    with tempfile.TemporaryDirectory() as tmp:
        runner = make_runner(tmp)
        scores = {"i1": 0.8, "i2": 0.3}
        run_a = CanaryCheckResult(["i1", "i2"], 0.0, False, ["t1"])
        run_b = CanaryCheckResult(["i1", "i2"], 0.0, False, ["t2"])
        result = runner.compare_canary_runs(run_a, run_b, scores, scores)
        assert not result.drift_detected
        assert result.max_score_drift == 0.0


def test_canary_drift_detected_above_threshold():
    """compare_canary_runs() flags drift when score change > 0.05."""
    with tempfile.TemporaryDirectory() as tmp:
        runner = make_runner(tmp)
        scores_a = {"i1": 0.8, "i2": 0.3}
        scores_b = {"i1": 0.87, "i2": 0.3}  # i1 drifted by 0.07
        run_a = CanaryCheckResult(["i1", "i2"], 0.0, False, ["t1"])
        run_b = CanaryCheckResult(["i1", "i2"], 0.0, False, ["t2"])
        result = runner.compare_canary_runs(run_a, run_b, scores_a, scores_b)
        assert result.drift_detected
        assert abs(result.max_score_drift - 0.07) < 1e-6


# ---------------------------------------------------------------------------
# Property 12: Prediction record completeness
# ---------------------------------------------------------------------------

@given(n=st.integers(min_value=1, max_value=50))
@settings(max_examples=50)
def test_prediction_has_all_required_fields(n):
    """Every Prediction must have all required fields populated."""
    predictions = [make_prediction(f"i{i}") for i in range(n)]
    for pred in predictions:
        assert pred.guardrail_name
        assert pred.item_id
        assert pred.predicted_label in ("harmful", "benign")
        assert 0.0 <= pred.confidence_score <= 1.0
        assert pred.inference_time_ms >= 0.0
        assert pred.confidence_source_type in (
            "logits_softmax", "native_safety_score", "api_score"
        )
        assert pred.split in ("dev", "test")
        assert pred.timestamp_utc  # Must be non-empty for API drift detection
