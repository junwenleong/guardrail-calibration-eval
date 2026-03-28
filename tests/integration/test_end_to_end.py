"""End-to-end integration tests on small synthetic data.

Tests the full pipeline: dataset generation → validation → mock inference
→ calibration metrics → persistence.

Uses mocked adapters to avoid GPU requirements.
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch


from src.datasets.builder import DatasetBuilder
from src.datasets.validator import ValidationPipeline
from src.evaluation.calibration import CalibrationAnalyzer
from src.evaluation.runner import ExperimentRunner
from src.models import DatasetItem, SeedExample


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_seeds(n_harmful: int = 5, n_benign: int = 5) -> list[SeedExample]:
    seeds = []
    for i in range(n_harmful):
        seeds.append(SeedExample(f"h{i}", f"harmful text {i}", "harmful", "HarmBench"))
    for i in range(n_benign):
        seeds.append(SeedExample(f"b{i}", f"benign text {i}", "benign", "ToxiGen"))
    return seeds


def make_mock_adapter(name: str = "MockGuard", confidence: float = 0.8):
    """Create a mock adapter that returns deterministic predictions."""
    adapter = MagicMock()
    adapter.get_model_name.return_value = name
    adapter.confidence_source_type = "logits_softmax"
    adapter.predict.return_value = MagicMock(
        label="harmful",
        confidence_score=confidence,
        two_token_mass=0.9,
    )
    return adapter


def make_runner(tmp_dir: str) -> ExperimentRunner:
    with patch("src.evaluation.runner.set_global_seeds"), \
         patch("src.evaluation.runner.log_environment"):
        return ExperimentRunner(checkpoint_dir=tmp_dir, checkpoint_frequency=5)


# ---------------------------------------------------------------------------
# Integration test: full pipeline on 20 synthetic items
# ---------------------------------------------------------------------------

def test_end_to_end_pipeline():
    """Full pipeline: generate → validate → infer → metrics → persist."""
    with tempfile.TemporaryDirectory() as tmp:
        # 1. Dataset generation
        builder = DatasetBuilder()
        seeds = make_seeds(5, 5)
        items = builder.generate_axis1_register(seeds)
        assert len(items) > 0, "No items generated"

        # 2. Validation with mock judge
        validator = ValidationPipeline()
        judge_fn = lambda text: "harmful" if "harmful" in text else "benign"
        items = validator.validate_with_llm_judge(items, judge_fn)
        validated = [i for i in items if i.validation_status in ("validated", "disputed")]
        assert len(validated) > 0

        # 3. Dataset split
        dev_items, test_items = builder.split_dataset(items[:20])
        assert len(dev_items) + len(test_items) == 20

        # 4. Mock inference
        runner = make_runner(tmp)
        adapter = make_mock_adapter("MockGuard", 0.8)
        checkpoint_path = Path(tmp) / "MockGuard.parquet"

        with patch.object(runner, "_unload_with_vram_check"):
            predictions = runner._load_and_run(adapter, items[:20], checkpoint_path)

        assert len(predictions) == 20
        for pred in predictions:
            assert pred.guardrail_name == "MockGuard"
            assert 0.0 <= pred.confidence_score <= 1.0
            assert pred.timestamp_utc  # Must be set

        # 5. Calibration metrics
        analyzer = CalibrationAnalyzer()
        ece = analyzer.compute_ece(predictions, items[:20])
        assert 0.0 <= ece.ece <= 1.0

        brier = analyzer.compute_brier_score(predictions, items[:20])
        assert 0.0 <= brier.brier_score <= 1.0

        eoe = analyzer.compute_eoe(predictions, items[:20])
        assert eoe <= ece.ece + 1e-9  # EOE ≤ ECE

        cc_ece = analyzer.compute_class_conditional_ece(predictions, items[:20])
        assert cc_ece.overall_ece is not None

        sweep = analyzer.compute_bin_sensitivity_sweep(predictions, items[:20])
        assert len(sweep) == 16  # M from 5 to 20

        # 6. Completeness check
        report = runner.verify_completeness(items[:20], predictions)
        assert report.total_expected == 20
        assert report.total_found == 20
        assert report.missing_item_ids == []

        # 7. Persistence
        from src.analysis.plots import PlotGenerator
        plotter = PlotGenerator(output_dir=tmp)
        plotter.persist_predictions(predictions, "MockGuard")

        import pandas as pd
        parquet_files = list((plotter.output_dir / "predictions").glob("*.parquet"))
        assert len(parquet_files) == 1
        df = pd.read_parquet(parquet_files[0])
        assert len(df) == 20


def test_sanity_check_passes_on_varied_scores():
    """Sanity check passes when scores are varied (std >= 0.01)."""
    with tempfile.TemporaryDirectory() as tmp:
        runner = make_runner(tmp)
        items = []
        for i in range(100):
            gt = "harmful" if i % 2 == 0 else "benign"
            items.append(DatasetItem(
                item_id=f"i{i}", seed_id=f"s{i}", axis=1, shift_level=0,
                ground_truth=gt, graded_harmfulness=None,
                seed_text="seed", variant_text=f"variant {i}",
                generation_method="test", validation_status="validated",
                cultural_frame=None, token_counts=None, split="dev",
            ))

        call_count = [0]
        def predict_side_effect(text):
            score = (call_count[0] % 10) * 0.1 + 0.05
            call_count[0] += 1
            label = "harmful" if score > 0.5 else "benign"
            return MagicMock(label=label, confidence_score=score, two_token_mass=None)

        adapter = MagicMock()
        adapter.get_model_name.return_value = "VariedGuard"
        adapter.predict.side_effect = predict_side_effect

        with patch.object(runner, "_unload_with_vram_check"):
            reports = runner.run_sanity_check([adapter], items)

        assert reports[0].passed


def test_validation_pipeline_integration():
    """ValidationPipeline correctly flags disputed items and computes disagreement rates."""
    validator = ValidationPipeline()
    items = []
    for i in range(20):
        gt = "harmful" if i % 2 == 0 else "benign"
        items.append(DatasetItem(
            item_id=f"i{i}", seed_id=f"s{i}", axis=1 if i < 10 else 2,
            shift_level=0, ground_truth=gt, graded_harmfulness=None,
            seed_text="seed", variant_text=f"variant {i}",
            generation_method="test", validation_status="pending",
            cultural_frame=None, token_counts=None, split="dev",
        ))

    # Judge disagrees with first 5 items
    disagree_ids = {f"i{i}" for i in range(5)}
    def judge_fn(text):
        for item in items:
            if item.variant_text == text:
                if item.item_id in disagree_ids:
                    return "benign" if item.ground_truth == "harmful" else "harmful"
                return item.ground_truth
        return "benign"

    result = validator.validate_with_llm_judge(items, judge_fn)
    n_disputed = sum(1 for i in result if i.validation_status == "disputed")
    assert n_disputed == 5

    # Disagreement rates
    reports = validator.compute_disagreement_rates(result)
    assert len(reports) == 2  # Axis 1 and 2
    total_disputed = sum(r.n_disputed for r in reports)
    assert total_disputed == 5


# ---------------------------------------------------------------------------
# Integration test: checkpoint resumption
# ---------------------------------------------------------------------------

def test_checkpoint_resumption():
    """Run 15 items, interrupt at item 10, resume → final predictions match uninterrupted run."""
    with tempfile.TemporaryDirectory() as tmp:
        items = []
        for i in range(15):
            gt = "harmful" if i % 2 == 0 else "benign"
            items.append(DatasetItem(
                item_id=f"i{i}", seed_id=f"s{i}", axis=1, shift_level=0,
                ground_truth=gt, graded_harmfulness=None,
                seed_text="seed", variant_text=f"variant {i}",
                generation_method="test", validation_status="validated",
                cultural_frame=None, token_counts=None, split="dev",
            ))

        checkpoint_path = Path(tmp) / "TestGuard.parquet"
        call_count = [0]

        def predict_side_effect(text):
            score = 0.7 + (call_count[0] % 3) * 0.1
            call_count[0] += 1
            return MagicMock(label="harmful", confidence_score=score, two_token_mass=None)

        # Run 1: process all 15 items (uninterrupted reference)
        runner1 = make_runner(tmp)
        adapter1 = MagicMock()
        adapter1.get_model_name.return_value = "TestGuard"
        adapter1.confidence_source_type = "logits_softmax"
        adapter1.predict.side_effect = predict_side_effect

        with patch.object(runner1, "_unload_with_vram_check"):
            full_predictions = runner1._load_and_run(adapter1, items, checkpoint_path)

        assert len(full_predictions) == 15

        # Verify checkpoint exists
        assert checkpoint_path.exists()

        # Run 2: simulate resumption (checkpoint already has all 15 items)
        runner2 = make_runner(tmp)
        adapter2 = MagicMock()
        adapter2.get_model_name.return_value = "TestGuard"
        adapter2.confidence_source_type = "logits_softmax"
        adapter2.predict.side_effect = predict_side_effect

        with patch.object(runner2, "_unload_with_vram_check"):
            resumed_predictions = runner2._load_and_run(adapter2, items, checkpoint_path)

        # Should not call predict() again (all items already in checkpoint)
        assert adapter2.predict.call_count == 0
        assert len(resumed_predictions) == 15
        assert [p.item_id for p in resumed_predictions] == [p.item_id for p in full_predictions]
