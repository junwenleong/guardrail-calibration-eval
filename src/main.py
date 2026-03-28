"""Guardrail Calibration Evaluation Framework — main pipeline entry point.

Orchestrates the full pipeline:
  dataset generation → validation → sanity check → pilot → full experiment
  → analysis → post-hoc calibration → visualization

Usage:
    python -m src.main --config config.yaml

Hardware profiles (set in config.yaml):
  Mac Studio M3 Ultra (96GB): parallel_loading=true  → ~50-60 min total
  MacBook Pro M4 Pro (32GB):  parallel_loading=false → ~3 hours total

LLM generation & validation: Ollama (qwen2.5:14b) — no API keys required.
  Start Ollama: ollama serve
  Pull models:  ollama pull qwen2.5:14b && ollama pull shieldgemma:9b

Key sequencing:
- Pilot (Task 11) runs BEFORE full experiment (Task 13)
- Dev split used for all exploratory analysis and post-hoc calibration fitting
- Test split used ONLY for final reported metrics
"""
from __future__ import annotations

import argparse
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def run_pipeline(config_path: str) -> None:
    """Run the full calibration evaluation pipeline."""
    from src.models import ExperimentConfig
    from src.utils.reproducibility import log_environment, set_global_seeds

    # 1. Load and validate config
    logger.info("Loading config from %s", config_path)
    try:
        config = ExperimentConfig.from_yaml(config_path)
    except (FileNotFoundError, ValueError) as e:
        logger.error("Config error: %s", e)
        sys.exit(1)

    # 2. Initialize reproducibility BEFORE any computation
    set_global_seeds(config.random_seed)
    log_environment()

    # 3. Initialize Ollama client
    from src.utils.ollama_client import OllamaClient
    ollama_cfg = getattr(config, "ollama", {}) or {}
    if isinstance(ollama_cfg, dict):
        ollama_url = ollama_cfg.get("base_url", "http://localhost:11434")
        gen_model = ollama_cfg.get("generation_model", "qwen2.5:14b")
    else:
        ollama_url = "http://localhost:11434"
        gen_model = "qwen2.5:14b"

    ollama_client = OllamaClient(base_url=ollama_url, model=gen_model)
    if not ollama_client.health_check():
        logger.warning(
            "Ollama not available at %s — dataset generation will use placeholders. "
            "Start Ollama with: ollama serve && ollama pull %s",
            ollama_url, gen_model,
        )
        ollama_client = None

    # 4. Initialize components
    from src.analysis.plots import PlotGenerator
    from src.datasets.builder import DatasetBuilder
    from src.datasets.validator import ValidationPipeline
    from src.evaluation.calibration import CalibrationAnalyzer
    from src.evaluation.runner import ExperimentRunner

    builder = DatasetBuilder(ollama_client=ollama_client)
    validator = ValidationPipeline(ollama_client=ollama_client)

    # Hardware profile from config
    hw_cfg = getattr(config, "hardware", {}) or {}
    if isinstance(hw_cfg, dict):
        parallel_loading = hw_cfg.get("parallel_loading", False)
        max_workers = hw_cfg.get("max_parallel_workers", 1)
    else:
        parallel_loading = False
        max_workers = 1

    runner = ExperimentRunner(
        checkpoint_frequency=config.checkpoint_frequency,
        temperature=config.temperature,
        random_seed=config.random_seed,
        parallel_loading=parallel_loading,
        max_parallel_workers=max_workers,
    )
    analyzer = CalibrationAnalyzer()
    plotter = PlotGenerator()

    # 5. Dataset generation
    logger.info("=== Phase 1: Dataset Generation ===")
    all_items = []
    for axis in config.axes:
        seeds = builder.load_seeds(axis)
        if not seeds:
            logger.warning("No seeds for axis %d — skipping", axis)
            continue
        if axis == 1:
            items = builder.generate_axis1_register(seeds)
        elif axis == 2:
            items = builder.generate_axis2_cultural(seeds)
        elif axis == 3:
            items = builder.generate_axis3_indirection(seeds)
        elif axis == 4:
            items = builder.generate_axis4_domain(seeds)
        elif axis == 5:
            items = builder.generate_axis5_language(seeds)
        else:
            continue
        all_items.extend(items)

    if not all_items:
        logger.error("No dataset items generated — check seed files in data/seeds/")
        sys.exit(1)

    # 6. Validation
    logger.info("=== Phase 2: Validation ===")
    all_items = validator.validate_with_llm_judge(all_items)
    logger.info("Validation complete — %d items", len(all_items))

    # 7. Dataset split
    dev_items, test_items = builder.split_dataset(
        all_items, random_seed=config.random_seed
    )
    logger.info(
        "Dataset split: %d dev / %d test", len(dev_items), len(test_items)
    )

    # 8. Load guardrail adapters
    adapters = _load_adapters(config.guardrails)
    if not adapters:
        logger.error("No adapters loaded — check guardrails config")
        sys.exit(1)

    # 9. Sanity check
    logger.info("=== Phase 3: Sanity Check ===")
    import random
    sanity_subset = random.sample(
        all_items, min(config.sanity_check_size, len(all_items))
    )
    try:
        sanity_reports = runner.run_sanity_check(adapters, sanity_subset)
        for report in sanity_reports:
            logger.info("Sanity check %s: passed=%s", report.guardrail, report.passed)
    except RuntimeError as e:
        logger.error("Sanity check FAILED: %s", e)
        sys.exit(1)

    # 10. Pilot experiment (MUST run before full experiment)
    logger.info("=== Phase 4: Pilot Experiment ===")
    pilot_adapters = [a for a in adapters if any(
        name in a.get_model_name() for name in config.pilot_guardrails
    )]
    if pilot_adapters:
        pilot_report = runner.run_pilot(
            pilot_adapters, all_items,
            pilot_axes=config.pilot_axes,
            pilot_size=config.pilot_size,
        )
        logger.info(
            "Pilot complete: %d items, %d divergence conditions found",
            pilot_report.n_items, len(pilot_report.divergence_conditions),
        )
    else:
        logger.warning("No pilot adapters found — skipping pilot")

    # 11. Full experiment
    logger.info("=== Phase 5: Full Experiment ===")
    all_predictions = runner.run_full_experiment(adapters, all_items)

    # 12. Completeness check
    for guardrail_name, preds in all_predictions.items():
        report = runner.verify_completeness(all_items, preds)
        if report.missing_item_ids:
            logger.warning(
                "%s: %d missing predictions",
                guardrail_name, len(report.missing_item_ids),
            )

    # 13. Analysis (dev split only for exploratory)
    logger.info("=== Phase 6: Analysis ===")
    for guardrail_name, preds in all_predictions.items():
        dev_preds = [p for p in preds if p.split == "dev"]
        test_preds = [p for p in preds if p.split == "test"]

        analyzer.compute_ece(dev_preds, dev_items)
        dev_eoe = analyzer.compute_eoe(dev_preds, dev_items)
        analyzer.compute_bin_sensitivity_sweep(dev_preds, dev_items)

        test_ece = analyzer.compute_ece(test_preds, test_items)
        test_brier = analyzer.compute_brier_score(test_preds, test_items)
        analyzer.compute_class_conditional_ece(test_preds, test_items)

        logger.info(
            "%s | test ECE=%.4f | test Brier=%.4f | EOE=%.4f",
            guardrail_name, test_ece.ece, test_brier.brier_score, dev_eoe,
        )

        plotter.persist_predictions(preds, guardrail_name)
        plotter.persist_metrics({
            "guardrail": guardrail_name,
            "test_ece": test_ece.ece,
            "test_brier": test_brier.brier_score,
            "dev_eoe": dev_eoe,
        }, f"metrics_{guardrail_name.replace(' ', '_')}")

    # 14. Post-hoc calibration
    logger.info("=== Phase 7: Post-hoc Calibration ===")
    from src.evaluation.posthoc import CalibrationTuner
    tuner = CalibrationTuner()
    for guardrail_name, preds in all_predictions.items():
        dev_preds = [
            p for p in preds
            if p.split == "dev" and p.confidence_source_type == "logits_softmax"
        ]
        test_preds = [
            p for p in preds
            if p.split == "test" and p.confidence_source_type == "logits_softmax"
        ]
        if not dev_preds or not test_preds:
            continue
        result = tuner.fit_platt_scaling(
            dev_preds, dev_items, test_preds, test_items, guardrail_name
        )
        logger.info(
            "%s | Platt residual ECE=%.4f (original=%.4f, structural=%s)",
            guardrail_name,
            result.residual_ece, result.original_ece, result.is_structural,
        )

    log_environment()
    logger.info("=== Pipeline Complete ===")


def _load_adapters(guardrail_names: list[str]) -> list:
    """Load guardrail adapters by name."""
    adapters = []
    for name in guardrail_names:
        try:
            adapter = _get_adapter(name)
            if adapter:
                adapters.append(adapter)
        except Exception as e:
            logger.warning("Failed to load adapter %s: %s", name, e)
    return adapters


def _get_adapter(name: str):
    """Map guardrail name to adapter class."""
    name_lower = name.lower()
    if "llamaguard" in name_lower or "llama_guard" in name_lower:
        from src.guardrails.llamaguard import LlamaGuardAdapter
        return LlamaGuardAdapter()
    elif "wildguard" in name_lower:
        from src.guardrails.wildguard import WildGuardAdapter
        return WildGuardAdapter()
    elif "granite" in name_lower:
        from src.guardrails.granite import GraniteGuardianAdapter
        return GraniteGuardianAdapter()
    elif "qwen" in name_lower:
        from src.guardrails.qwen3guard import Qwen3GuardAdapter
        return Qwen3GuardAdapter()
    elif "nemo" in name_lower:
        from src.guardrails.nemoguard import NemoGuardAdapter
        return NemoGuardAdapter()
    elif "shieldgemma" in name_lower:
        from src.guardrails.shieldgemma import ShieldGemmaAdapter
        return ShieldGemmaAdapter()
    else:
        logger.warning("Unknown guardrail: %s", name)
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Guardrail Calibration Evaluation Framework"
    )
    parser.add_argument(
        "--config", default="config.yaml",
        help="Path to config.yaml (default: config.yaml)"
    )
    args = parser.parse_args()
    run_pipeline(args.config)


if __name__ == "__main__":
    main()
