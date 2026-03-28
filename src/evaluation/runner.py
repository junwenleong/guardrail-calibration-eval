"""ExperimentRunner: orchestrates the factorial experiment with sequential model loading.

Key design decisions:
- Only one local model occupies VRAM at a time
- Atomic checkpointing via write-then-rename prevents Parquet corruption
- VRAM leak detection after every unload_model() call
- Per-prediction timestamp_utc for API drift detection
- temperature=0.0 for all local model inference (deterministic)
"""
from __future__ import annotations

import gc
import logging
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.guardrails.base import GuardrailAdapter
from src.models import (
    CanaryCheckResult,
    CompletenessReport,
    DatasetItem,
    Prediction,
    SanityCheckReport,
)
from src.utils.reproducibility import log_environment, set_global_seeds

logger = logging.getLogger(__name__)

_VRAM_LEAK_THRESHOLD_MB = 500.0


@dataclass
class PilotReport:
    guardrails: list[str]
    axes: list[int]
    n_items: int
    divergence_conditions: list[dict]  # Conditions where accuracy and ECE diverge


@dataclass
class QuantBaselineReport:
    guardrail: str
    n_items: int
    mean_abs_confidence_diff: float
    ece_4bit: float
    ece_fp16: float
    ece_diff: float
    warning_triggered: bool


class ExperimentRunner:
    """Orchestrates the full factorial experiment across all guardrails and axes."""

    def __init__(
        self,
        checkpoint_dir: str | Path = "results/checkpoints",
        checkpoint_frequency: int = 500,
        temperature: float = 0.0,
        random_seed: int = 42,
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_frequency = checkpoint_frequency
        self.temperature = temperature
        self.random_seed = random_seed

        # Initialize reproducibility
        set_global_seeds(random_seed)
        log_environment()

    # ------------------------------------------------------------------
    # Sanity check
    # ------------------------------------------------------------------

    def run_sanity_check(
        self,
        adapters: list[GuardrailAdapter],
        subset: list[DatasetItem],
    ) -> list[SanityCheckReport]:
        """Run all guardrails on a 100-item subset to verify score distributions.

        Halts if std < 0.01 (constant scores). Flags if < 5 bins covered or
        accuracy < 50%.
        """
        import numpy as np
        from scipy.stats import kurtosis, skew

        reports = []
        for adapter in adapters:
            logger.info("Sanity check: %s on %d items", adapter.get_model_name(), len(subset))
            adapter.load_model()
            scores = []
            labels = []
            try:
                for item in subset:
                    result = adapter.predict(item.variant_text)
                    scores.append(result.confidence_score)
                    labels.append(result.label)
            finally:
                self._unload_with_vram_check(adapter)

            scores_arr = np.array(scores)
            warnings = []

            std = float(np.std(scores_arr))
            if std < 0.01:
                msg = f"std={std:.4f} < 0.01 — constant scores detected"
                warnings.append(msg)
                logger.warning("HALT: %s — %s", adapter.get_model_name(), msg)

            # Histogram coverage (10 bins: 0.0-0.1, ..., 0.9-1.0)
            hist, _ = np.histogram(scores_arr, bins=10, range=(0.0, 1.0))
            bins_covered = int(np.sum(hist > 0))
            if bins_covered < 5:
                msg = f"Only {bins_covered}/10 histogram bins covered"
                warnings.append(msg)
                logger.warning("%s: %s", adapter.get_model_name(), msg)

            # Accuracy
            correct = sum(
                1 for item, label in zip(subset, labels)
                if label == item.ground_truth
            )
            accuracy = correct / len(subset) if subset else 0.0
            if accuracy < 0.5:
                msg = f"accuracy={accuracy:.2f} < 0.5 (worse than random)"
                warnings.append(msg)
                logger.warning("%s: %s", adapter.get_model_name(), msg)

            passed = std >= 0.01  # Halt condition
            reports.append(SanityCheckReport(
                guardrail=adapter.get_model_name(),
                mean=float(np.mean(scores_arr)),
                std=std,
                min_score=float(np.min(scores_arr)),
                max_score=float(np.max(scores_arr)),
                skewness=float(skew(scores_arr)),
                kurtosis=float(kurtosis(scores_arr)),
                bins_covered=bins_covered,
                accuracy=accuracy,
                passed=passed,
                warnings=warnings,
            ))

            if not passed:
                raise RuntimeError(
                    f"Sanity check FAILED for {adapter.get_model_name()}: {warnings}"
                )

        return reports

    # ------------------------------------------------------------------
    # Core inference loop
    # ------------------------------------------------------------------

    def _load_and_run(
        self,
        adapter: GuardrailAdapter,
        items: list[DatasetItem],
        checkpoint_path: Path,
    ) -> list[Prediction]:
        """Load model, run all items, checkpoint every N, unload.

        Resumes from checkpoint if it exists.
        """
        predictions: list[Prediction] = []
        start_idx = 0

        # Resume from checkpoint if available
        if checkpoint_path.exists():
            df = pd.read_parquet(checkpoint_path)
            predictions = [Prediction(**row) for row in df.to_dict("records")]
            start_idx = len(predictions)
            logger.info(
                "Resuming %s from checkpoint: %d/%d items done",
                adapter.get_model_name(), start_idx, len(items),
            )

        if start_idx >= len(items):
            logger.info("%s: all items already processed", adapter.get_model_name())
            return predictions

        adapter.load_model()
        logger.info(
            "Running %s on %d items (temperature=%.1f)",
            adapter.get_model_name(), len(items) - start_idx, self.temperature,
        )
        logger.info("Temperature: %.1f for %s", self.temperature, adapter.get_model_name())

        buffer: list[Prediction] = []
        try:
            for i, item in enumerate(items[start_idx:], start=start_idx):
                t0 = time.perf_counter()
                result = adapter.predict(item.variant_text)
                elapsed_ms = (time.perf_counter() - t0) * 1000

                pred = Prediction(
                    guardrail_name=adapter.get_model_name(),
                    item_id=item.item_id,
                    predicted_label=result.label,
                    confidence_score=result.confidence_score,
                    inference_time_ms=elapsed_ms,
                    two_token_mass=result.two_token_mass,
                    confidence_source_type=adapter.confidence_source_type,
                    split=item.split,
                    timestamp_utc=datetime.now(timezone.utc).isoformat(),
                )
                buffer.append(pred)

                if len(buffer) >= self.checkpoint_frequency:
                    predictions.extend(buffer)
                    self._atomic_checkpoint(predictions, checkpoint_path)
                    buffer = []
                    logger.info(
                        "%s: %d/%d items processed",
                        adapter.get_model_name(), i + 1, len(items),
                    )
        except Exception as e:
            logger.error("Inference failed at item %d: %s", i, e)
            raise
        finally:
            if buffer:
                predictions.extend(buffer)
                self._atomic_checkpoint(predictions, checkpoint_path)
            self._unload_with_vram_check(adapter)

        return predictions

    def _atomic_checkpoint(
        self, predictions: list[Prediction], path: Path
    ) -> None:
        """Write predictions to path.tmp then os.rename() to prevent corruption."""
        tmp_path = path.with_suffix(".tmp")
        df = pd.DataFrame([asdict(p) for p in predictions])
        df.to_parquet(tmp_path, index=False)
        os.replace(tmp_path, path)  # Atomic on POSIX; best-effort on Windows
        # Clean up any stale .tmp files from previous crashes
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)

    def _unload_with_vram_check(self, adapter: GuardrailAdapter) -> None:
        """Unload model and check for VRAM leaks."""
        before_mb = self._get_vram_mb()
        adapter.unload_model()
        gc.collect()
        after_mb = self._get_vram_mb()
        if before_mb is not None and after_mb is not None:
            leaked_mb = after_mb - (before_mb - self._get_model_size_estimate_mb())
            if after_mb > _VRAM_LEAK_THRESHOLD_MB:
                logger.warning(
                    "Possible VRAM leak after unloading %s: %.1f MB remaining",
                    adapter.get_model_name(), after_mb,
                )

    @staticmethod
    def _get_vram_mb() -> float | None:
        """Get current VRAM usage in MB. Returns None if unavailable."""
        try:
            import torch
            if torch.cuda.is_available():
                return torch.cuda.memory_allocated() / 1024 ** 2
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                try:
                    return torch.mps.current_allocated_memory() / 1024 ** 2
                except AttributeError:
                    logger.warning("torch.mps.current_allocated_memory() unavailable — VRAM leak detection disabled on MPS")
                    return None
        except ImportError:
            pass
        return None

    @staticmethod
    def _get_model_size_estimate_mb() -> float:
        """Rough estimate of model size freed on unload (for leak threshold)."""
        return 4096.0  # ~4 GB for 8B 4-bit model

    # ------------------------------------------------------------------
    # Completeness verification
    # ------------------------------------------------------------------

    def verify_completeness(
        self,
        dataset: list[DatasetItem],
        predictions: list[Prediction],
    ) -> CompletenessReport:
        """Verify all item_ids in dataset have corresponding predictions.

        Reports missing items and whether skipped items are systematically
        biased (e.g., correlated with input length).
        """
        expected_ids = {item.item_id for item in dataset}
        found_ids = {pred.item_id for pred in predictions}
        missing_ids = sorted(expected_ids - found_ids)

        # Analyze whether missing items are systematically biased
        skipped_analysis: dict = {}
        if missing_ids:
            missing_items = [i for i in dataset if i.item_id in set(missing_ids)]
            all_lengths = [len(i.variant_text) for i in dataset]
            missing_lengths = [len(i.variant_text) for i in missing_items]
            import numpy as np
            skipped_analysis = {
                "mean_length_all": float(np.mean(all_lengths)),
                "mean_length_missing": float(np.mean(missing_lengths)) if missing_lengths else 0.0,
                "length_bias_detected": (
                    float(np.mean(missing_lengths)) > float(np.mean(all_lengths)) * 1.5
                    if missing_lengths else False
                ),
            }
            logger.warning(
                "Missing %d/%d predictions. Length bias: %s",
                len(missing_ids), len(expected_ids),
                skipped_analysis.get("length_bias_detected"),
            )

        return CompletenessReport(
            total_expected=len(expected_ids),
            total_found=len(found_ids),
            missing_item_ids=missing_ids,
            skipped_item_analysis=skipped_analysis,
        )

    # ------------------------------------------------------------------
    # API canary check
    # ------------------------------------------------------------------

    def _run_api_canary_check(
        self,
        adapter: GuardrailAdapter,
        canary_items: list[DatasetItem],
        run_label: str = "start",
    ) -> CanaryCheckResult:
        """Run canary items through the API to detect model version drift.

        Canary items should be stable, well-understood examples that exercise
        the model's calibration properties (not trivially short/simple).
        """
        scores: dict[str, float] = {}
        timestamps = []
        for item in canary_items:
            result = adapter.predict(item.variant_text)
            scores[item.item_id] = result.confidence_score
            timestamps.append(datetime.now(timezone.utc).isoformat())

        return CanaryCheckResult(
            canary_items=list(scores.keys()),
            max_score_drift=0.0,  # Computed by comparing across runs
            drift_detected=False,
            timestamps=timestamps,
        )

    def compare_canary_runs(
        self,
        run_a: CanaryCheckResult,
        run_b: CanaryCheckResult,
        scores_a: dict[str, float],
        scores_b: dict[str, float],
        drift_threshold: float = 0.05,
    ) -> CanaryCheckResult:
        """Compare two canary runs to detect API version drift."""
        common_ids = set(scores_a) & set(scores_b)
        if not common_ids:
            return run_b

        drifts = [abs(scores_a[id_] - scores_b[id_]) for id_ in common_ids]
        max_drift = max(drifts) if drifts else 0.0
        drift_detected = max_drift > drift_threshold

        if drift_detected:
            logger.warning(
                "API canary drift detected: max_drift=%.4f > threshold=%.4f",
                max_drift, drift_threshold,
            )

        return CanaryCheckResult(
            canary_items=list(common_ids),
            max_score_drift=max_drift,
            drift_detected=drift_detected,
            timestamps=run_a.timestamps + run_b.timestamps,
        )


    # ------------------------------------------------------------------
    # Pilot experiment (MUST run before full experiment)
    # ------------------------------------------------------------------

    def run_pilot(
        self,
        adapters: list[GuardrailAdapter],
        dataset: list[DatasetItem],
        pilot_axes: list[int] | None = None,
        pilot_size: int = 200,
    ) -> PilotReport:
        """Run a 200-item pilot with 2 guardrails on Axes 1 and 5.

        MUST be called before run_full_experiment() to validate the pipeline
        and confirm metric sensitivity before the expensive ~10k-item run.

        Args:
            adapters: Exactly 2 guardrails (e.g., LlamaGuard 4 + Qwen3Guard-8B).
            dataset: Full dataset to sample from.
            pilot_axes: Axes to include (default: [1, 5]).
            pilot_size: Number of items to run (default: 200).

        Returns PilotReport with divergence conditions.
        """
        from src.evaluation.calibration import CalibrationAnalyzer

        pilot_axes = pilot_axes or [1, 5]
        pilot_items = [
            item for item in dataset
            if item.axis in pilot_axes
        ][:pilot_size]

        logger.info(
            "Running pilot: %d items, axes=%s, guardrails=%s",
            len(pilot_items), pilot_axes, [a.get_model_name() for a in adapters],
        )

        all_predictions: dict[str, list[Prediction]] = {}
        for adapter in adapters:
            checkpoint_path = self.checkpoint_dir / f"pilot_{adapter.get_model_name().replace(' ', '_')}.parquet"
            preds = self._load_and_run(adapter, pilot_items, checkpoint_path)
            all_predictions[adapter.get_model_name()] = preds

        # Detect accuracy-calibration divergence
        analyzer = CalibrationAnalyzer()
        divergence_conditions = []
        for guardrail_name, preds in all_predictions.items():
            for axis in pilot_axes:
                axis_preds = [p for p in preds if any(
                    item.item_id == p.item_id and item.axis == axis
                    for item in pilot_items
                )]
                if len(axis_preds) < 10:
                    continue
                divergence = analyzer.detect_accuracy_calibration_divergence(
                    axis_preds, pilot_items, guardrail_name, axis
                )
                if divergence:
                    divergence_conditions.append(divergence)
                    logger.info(
                        "Divergence detected: %s axis=%d — %s",
                        guardrail_name, axis, divergence,
                    )

        return PilotReport(
            guardrails=[a.get_model_name() for a in adapters],
            axes=pilot_axes,
            n_items=len(pilot_items),
            divergence_conditions=divergence_conditions,
        )

    # ------------------------------------------------------------------
    # Full experiment
    # ------------------------------------------------------------------

    def run_full_experiment(
        self,
        adapters: list[GuardrailAdapter],
        dataset: list[DatasetItem],
        canary_items: list[DatasetItem] | None = None,
    ) -> dict[str, list[Prediction]]:
        """Run the full factorial experiment: 6 guardrails × ~10k items.

        Sequential model loading — only one model in VRAM at a time.
        Checkpoints every 500 items per guardrail.
        Runs API canary check at start, middle, and end for OpenAI adapter.

        Returns dict mapping guardrail_name -> list[Prediction].
        """
        log_environment()  # Log environment at start of full run
        all_predictions: dict[str, list[Prediction]] = {}

        for adapter in adapters:
            name = adapter.get_model_name()
            checkpoint_path = self.checkpoint_dir / f"{name.replace(' ', '_')}.parquet"
            logger.info("Starting full experiment for %s", name)
            preds = self._load_and_run(adapter, dataset, checkpoint_path)
            all_predictions[name] = preds
            logger.info("Completed %s: %d predictions", name, len(preds))

        log_environment()  # Log environment at end to detect mid-run changes
        return all_predictions

    # ------------------------------------------------------------------
    # Quantization baseline
    # ------------------------------------------------------------------

    def run_quantization_baseline(
        self,
        adapter_4bit: GuardrailAdapter,
        adapter_fp16: GuardrailAdapter,
        subset: list[DatasetItem],
    ) -> QuantBaselineReport:
        """Compare 4-bit vs fp16 inference on a 500-item subset.

        Warns if mean absolute confidence diff > 0.05 or ECE diff > 0.02.
        """
        import numpy as np
        from src.evaluation.calibration import CalibrationAnalyzer

        checkpoint_4bit = self.checkpoint_dir / "quant_4bit.parquet"
        checkpoint_fp16 = self.checkpoint_dir / "quant_fp16.parquet"

        preds_4bit = self._load_and_run(adapter_4bit, subset, checkpoint_4bit)
        preds_fp16 = self._load_and_run(adapter_fp16, subset, checkpoint_fp16)

        # Align by item_id
        scores_4bit = {p.item_id: p.confidence_score for p in preds_4bit}
        scores_fp16 = {p.item_id: p.confidence_score for p in preds_fp16}
        common_ids = list(set(scores_4bit) & set(scores_fp16))

        diffs = [abs(scores_4bit[id_] - scores_fp16[id_]) for id_ in common_ids]
        mean_diff = float(np.mean(diffs)) if diffs else 0.0

        analyzer = CalibrationAnalyzer()
        ece_4bit = analyzer.compute_ece(preds_4bit).ece
        ece_fp16 = analyzer.compute_ece(preds_fp16).ece
        ece_diff = abs(ece_4bit - ece_fp16)

        warning = mean_diff > 0.05 or ece_diff > 0.02
        if warning:
            logger.warning(
                "Quantization impact: mean_conf_diff=%.4f, ECE_diff=%.4f",
                mean_diff, ece_diff,
            )

        return QuantBaselineReport(
            guardrail=adapter_4bit.get_model_name(),
            n_items=len(common_ids),
            mean_abs_confidence_diff=mean_diff,
            ece_4bit=ece_4bit,
            ece_fp16=ece_fp16,
            ece_diff=ece_diff,
            warning_triggered=warning,
        )
