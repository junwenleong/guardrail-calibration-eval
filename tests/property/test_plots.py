"""Property tests for PlotGenerator result persistence.

# Feature: guardrail-calibration-eval, Property 5: Prediction persistence round-trip
"""
from __future__ import annotations

import tempfile
from datetime import datetime, timezone

import pandas as pd
from hypothesis import given, settings
from hypothesis import strategies as st

from src.analysis.plots import PlotGenerator
from src.models import Prediction


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_prediction(item_id: str, confidence: float = 0.8) -> Prediction:
    return Prediction(
        guardrail_name="TestGuard",
        item_id=item_id,
        predicted_label="harmful",
        confidence_score=confidence,
        inference_time_ms=1.0,
        two_token_mass=None,
        confidence_source_type="logits_softmax",
        split="dev",
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
    )


# ---------------------------------------------------------------------------
# Property 5: Prediction persistence round-trip
# ---------------------------------------------------------------------------

def test_persist_predictions_creates_parquet_and_csv():
    """persist_predictions() creates both Parquet and CSV files."""
    with tempfile.TemporaryDirectory() as tmp:
        plotter = PlotGenerator(output_dir=tmp)
        predictions = [make_prediction(f"i{i}", 0.8) for i in range(10)]
        plotter.persist_predictions(predictions, "TestGuard")

        parquet_files = list((plotter.output_dir / "predictions").glob("*.parquet"))
        csv_files = list((plotter.output_dir / "predictions").glob("*.csv"))
        assert len(parquet_files) == 1
        assert len(csv_files) == 1


@given(n=st.integers(min_value=1, max_value=100))
@settings(max_examples=50)
def test_prediction_persistence_roundtrip(n):
    """Predictions written to Parquet can be read back with correct data."""
    with tempfile.TemporaryDirectory() as tmp:
        plotter = PlotGenerator(output_dir=tmp)
        predictions = [make_prediction(f"i{i}", i / 100) for i in range(n)]
        plotter.persist_predictions(predictions, "TestGuard")

        parquet_path = list((plotter.output_dir / "predictions").glob("*.parquet"))[0]
        df = pd.read_parquet(parquet_path)

        assert len(df) == n
        restored = [Prediction(**row) for row in df.to_dict("records")]
        assert [p.item_id for p in restored] == [p.item_id for p in predictions]
        for orig, rest in zip(predictions, restored):
            assert abs(orig.confidence_score - rest.confidence_score) < 1e-6


def test_persist_metrics_creates_json():
    """persist_metrics() creates a JSON file with correct content."""
    with tempfile.TemporaryDirectory() as tmp:
        plotter = PlotGenerator(output_dir=tmp)
        metrics = {"ece": 0.123, "brier": 0.456, "guardrail": "TestGuard"}
        plotter.persist_metrics(metrics, "test_metrics")

        import json
        path = plotter.output_dir / "metrics" / "test_metrics.json"
        assert path.exists()
        with open(path) as f:
            loaded = json.load(f)
        assert loaded["ece"] == 0.123
        assert loaded["guardrail"] == "TestGuard"


def test_honest_threshold_table_has_conservative_threshold():
    """generate_honest_threshold_table() includes conservative_threshold column."""
    from src.models import HonestThreshold
    with tempfile.TemporaryDirectory() as tmp:
        plotter = PlotGenerator(output_dir=tmp)
        thresholds = [
            HonestThreshold(
                target_precision=0.90,
                honest_confidence=0.85,
                ci_lower=0.75,
                ci_upper=0.95,
                actual_precision_at_target=0.91,
                guardrail="TestGuard",
                axis=1,
                shift_level=0,
            )
        ]
        df = plotter.generate_honest_threshold_table(thresholds, save=False)
        assert "conservative_threshold" in df.columns
        assert df["conservative_threshold"].iloc[0] == 0.75  # CI lower bound
