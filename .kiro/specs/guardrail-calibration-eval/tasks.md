# Implementation Plan: Guardrail Calibration Evaluation Framework

## Overview

Incremental implementation of the calibration evaluation pipeline in five phases: (1) core interfaces, data models, and reproducibility infrastructure, (2) guardrail adapters, (3) dataset generation and validation, (4) experiment runner with pilot-first sequencing, (5) analysis, post-hoc calibration, and visualization. Each phase produces runnable, tested code before the next begins.

Key sequencing principle: the pilot experiment (Task 11) runs BEFORE the full experiment (Task 13) to validate the pipeline and catch signal/methodology issues before committing to the expensive ~10k-item run.

## Tasks

- [x] 1. Set up project structure, dependencies, core data models, and reproducibility infrastructure
  - Create directory layout: `src/guardrails/`, `src/datasets/`, `src/evaluation/`, `src/analysis/`, `src/utils/`, `tests/unit/`, `tests/property/`, `tests/integration/`
  - Create `pyproject.toml` (or `requirements.txt`) with dependencies: `torch`, `transformers`, `bitsandbytes`, `hypothesis`, `pytest`, `scipy`, `numpy`, `pandas`, `pyarrow`, `pyyaml`, `openai`, `matplotlib`, `seaborn`, `scikit-learn`
  - Implement all dataclasses from the Data Models section in `src/models.py`: `SeedExample`, `DatasetItem`, `PredictionResult`, `Prediction`, `CalibrationCurve`, `ECEResult`, `ClassConditionalECE`, `BrierDecomposition`, `SpearmanResult`, `HonestThreshold`, `DeltaMetrics`, `SanityCheckReport`, `TwoTokenMassSummary`, `TokenLengthAnalysis`, `CanaryCheckResult`, `CompletenessReport`, `ExperimentConfig`
  - Add `timestamp_utc: str` field to `Prediction` dataclass for per-prediction timestamping (API drift detection)
  - Implement `DatasetItem.to_json()` and `DatasetItem.from_json()` with explicit `None` preservation (not `NaN`) for `graded_harmfulness` and `token_counts`
  - Implement `ExperimentConfig.from_yaml()` and `ExperimentConfig.validate()`
  - Implement `src/utils/reproducibility.py`: `set_global_seeds(seed)` to lock `random`, `numpy`, `torch` seeds; also set `torch.backends.cudnn.deterministic = True`, `torch.backends.cudnn.benchmark = False`, and `CUBLAS_WORKSPACE_CONFIG=:4096:8` env var to eliminate non-deterministic CUDA ops; `log_environment()` to record GPU model, CUDA/MPS version, torch version, bitsandbytes version, Python version, and OS — call at both start AND end of experiment to detect mid-run environment changes
  - _Requirements: 14.4, 14.5, 30.1, 30.2, 30.3_

  - [x] 1.1 Write property test for DatasetItem JSON round-trip
    - **Property 4: Metadata JSON round-trip**
    - Verify `None` fields survive round-trip (not converted to `NaN`); verify `graded_harmfulness is None` check works after deserialization
    - **Validates: Requirements 14.4, 14.5**
    - `# Feature: guardrail-calibration-eval, Property 4: Metadata JSON round-trip`

  - [x] 1.2 Write property test for ExperimentConfig round-trip and validation
    - **Property 30: Config validation round-trip**
    - **Property 39: Config semantic validation**
    - **Validates: Requirements 30.1, 30.2**
    - `# Feature: guardrail-calibration-eval, Property 30: Config validation round-trip`
    - `# Feature: guardrail-calibration-eval, Property 39: Config semantic validation`

- [x] 2. Implement GuardrailAdapter base class and LogitBasedAdapterMixin
  - Implement `GuardrailAdapter` ABC in `src/guardrails/base.py` with abstract methods: `predict`, `format_prompt`, `get_model_name`, `load_model`, `unload_model`, and properties `confidence_source`, `confidence_source_type`
  - Implement `LogitBasedAdapterMixin` in `src/guardrails/base.py` with `verify_token_mapping()` and `compute_confidence()` (full-vocabulary softmax, 2-token mass extraction, warning on mass < 0.5)
  - Raise `ValueError` on empty/null input in `predict()`
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7_

  - [x]* 2.1 Write property tests for adapter output contract and confidence source type
    - **Property 1: Adapter output contract**
    - **Property 2: Confidence source type derivation**
    - **Property 40: Prompt template application**
    - **Validates: Requirements 1.1, 1.5**
    - `# Feature: guardrail-calibration-eval, Property 1: Adapter output contract`
    - `# Feature: guardrail-calibration-eval, Property 2: Confidence source type derivation`
    - `# Feature: guardrail-calibration-eval, Property 40: Prompt template application`

  - [x]* 2.2 Write property test for two-token probability mass invariant
    - **Property 3: Two-token probability mass invariant**
    - **Validates: Requirements 1.7**
    - `# Feature: guardrail-calibration-eval, Property 3: Two-token probability mass invariant`

  - [x]* 2.3 Write unit tests for adapter base class
    - Test `ValueError` on empty string, `None`, and whitespace-only input
    - Test `confidence_source_type` mapping for all three `confidence_source` values
    - _Requirements: 1.3, 1.5_

- [x] 3. Checkpoint — Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 4. Implement the five local guardrail adapters (4-bit quantized)
  - [x] 4.1 Implement `LlamaGuardAdapter` in `src/guardrails/llamaguard.py`
    - Load `meta-llama/Llama-Guard-4-8B` with `load_in_4bit=True` via bitsandbytes
    - Apply LlamaGuard chat template in `format_prompt()`
    - Derive confidence from logits via `LogitBasedAdapterMixin.compute_confidence()`
    - Call `verify_token_mapping()` at init; log quantization config and VRAM footprint
    - _Requirements: 2.1, 2.2, 2.3, 8a.1, 8a.2_

  - [x] 4.2 Implement `WildGuardAdapter` in `src/guardrails/wildguard.py`
    - Load `allenai/wildguard` with `load_in_4bit=True`
    - Use model's native safety score output (NOT logit softmax); document score semantics from model card
    - Set `confidence_source = "native_safety_score"`; note in docstring that this is not a proper probability
    - Log quantization config and VRAM footprint at init
    - _Requirements: 3.1, 3.2, 3.3_

  - [x] 4.3 Implement `GraniteGuardianAdapter` in `src/guardrails/granite.py`
    - Load `ibm-granite/granite-guardian-3.3-8b` with `load_in_4bit=True`
    - Map yes/no tokens to benign/harmful; call `verify_token_mapping()` at init to catch case-sensitivity issues (e.g., "Yes" vs "yes")
    - Apply Granite chat template in `format_prompt()`
    - Log quantization config and VRAM footprint at init
    - _Requirements: 4.1, 4.2, 4.3_

  - [x] 4.4 Implement `Qwen3GuardAdapter` in `src/guardrails/qwen3guard.py`
    - Load `Qwen/Qwen3Guard-8B` with `load_in_4bit=True`
    - Apply Qwen chat template in `format_prompt()`
    - Derive confidence from logits via `LogitBasedAdapterMixin.compute_confidence()`
    - Log quantization config and VRAM footprint at init
    - _Requirements: 5.1, 5.2, 5.3_

  - [x] 4.5 Implement `NemoGuardAdapter` in `src/guardrails/nemoguard.py`
    - Load `nvidia/Llama-3.1-8B-Instruct-NemoGuard` with `load_in_4bit=True`
    - Apply Llama-3.1 chat template in `format_prompt()`
    - Derive confidence from logits via `LogitBasedAdapterMixin.compute_confidence()`
    - Log quantization config and VRAM footprint at init
    - _Requirements: 6.1, 6.2, 6.3_

  - [x]* 4.6 Write unit tests for all five local adapters
    - Test that `get_model_name()` returns the `(4-bit)` suffix
    - Test `confidence_source` and `confidence_source_type` values
    - Test `format_prompt()` produces non-identity output (template is applied)
    - Mock model loading to avoid VRAM requirements in unit tests
    - _Requirements: 2.1–6.3_

- [x] 5. Implement `OpenAIModerationAdapter`
  - Implement `OpenAIModerationAdapter` in `src/guardrails/openai_moderation.py`
  - Call OpenAI Moderation API; return `category_scores` as confidence
  - Set `confidence_source = "category_scores"`; document in docstring that scores are NOT probabilities
  - Implement retry logic: 3 retries with exponential backoff (1s, 2s, 4s); respect `Retry-After` on 429
  - Record per-prediction `timestamp_utc` for API drift analysis
  - _Requirements: 7.1, 7.2, 7.3, 7.4_

  - [x]* 5.1 Write unit tests for OpenAI adapter retry logic
    - Test 3 failures → skip item (mock API errors)
    - Test exponential backoff timing
    - Test rate limit (429) handling with `Retry-After`
    - _Requirements: 7.3_

- [x] 6. Implement `DatasetBuilder` — axes 1, 2, 4 (binary ground truth axes)
  - Implement `DatasetBuilder` in `src/datasets/builder.py`
  - Implement seed ingestion from standardized benchmarks (HarmBench, AdvBench, ToxiGen, or similar public datasets); document seed sources in code and config
  - Implement `generate_axis1_register()`: 50-100 harmful + 50-100 benign seeds → 4-5 register variants, shift_level 0-4, ~200-500 items, 50/50 balance per shift_level
  - Implement `generate_axis2_cultural()`: 30-40 harmful + 30-40 benign seeds → cultural frames (filial piety, traditional medicine, cultural idioms, Singlish), tag `cultural_frame`, 50/50 balance
  - Implement `generate_axis4_domain()`: 100 harmful + 100 benign seeds → 4 domain framings including APAC public sector language, ~800 items, 50/50 balance
  - Implement `validate_class_balance()` returning `BalanceReport`
  - Implement `generate_unique_item_id()` ensuring uniqueness across the full dataset
  - Attach all required metadata fields to every item
  - Implement variant sanitization: strip meta-instructions and label-leaking cues from LLM-generated variants (e.g., "Here is a harmful request…") before storing `variant_text`
  - _Requirements: 9.1–9.5, 10.1–10.5, 12.1–12.5, 14.1–14.3_

  - [x]* 6.1 Write property tests for dataset structural invariants
    - **Property 6: Ground truth preservation across variants**
    - **Property 7: Shift level range**
    - **Property 8: Class balance per shift level**
    - **Property 9: Axis-specific required fields**
    - **Property 10: Unique item IDs**
    - **Property 11: Dataset metadata completeness**
    - **Validates: Requirements 9.3, 9.5, 10.3, 10.5, 12.3, 12.5, 14.1, 14.2**
    - `# Feature: guardrail-calibration-eval, Property 6: Ground truth preservation across variants`
    - `# Feature: guardrail-calibration-eval, Property 7: Shift level range`
    - `# Feature: guardrail-calibration-eval, Property 8: Class balance per shift level`
    - `# Feature: guardrail-calibration-eval, Property 10: Unique item IDs`
    - `# Feature: guardrail-calibration-eval, Property 11: Dataset metadata completeness`

- [x] 7. Implement `DatasetBuilder` — axes 3 and 5 (special axes)
  - Implement `generate_axis3_indirection()`: 100 harmful seeds → 5 indirection levels, `graded_harmfulness` field (0.0-1.0), ~500 items + 50-item benign refusals subset; shift_level 0 = direct, 4 = most indirect
  - Add `adversarial_pressure_type` metadata tag to each Axis 3 item: one of `"soft_indirection"` (roleplay, hypothetical, academic framing) or `"hard_adversarial"` (Base64 encoding, leetspeak, character substitution); this enables downstream analysis of whether calibration collapse is specific to adversarial encoding or general indirection
  - Implement `generate_axis5_language()`: 200-item English subset → Malay, Mandarin (Simplified), Indonesian; preserve ground truth; flag `western_norm_flag` where translation introduces ambiguity; record `token_counts` per model
  - Implement `split_dataset()`: 80/20 stratified split by `(axis, shift_level, ground_truth)`
  - _Requirements: 11.1–11.5, 13.1–13.7, 18a.1_

  - [x]* 7.1 Write property tests for split and axis-3/5 invariants
    - **Property 9: Axis-specific required fields** (graded_harmfulness for Axis 3, token_counts for Axis 5)
    - **Property 13: Stratified split preserves proportions**
    - **Validates: Requirements 11.5, 13.7, 18a.1**
    - `# Feature: guardrail-calibration-eval, Property 9: Axis-specific required fields`
    - `# Feature: guardrail-calibration-eval, Property 13: Stratified split preserves proportions`

- [x] 8. Implement `ValidationPipeline`
  - Implement `ValidationPipeline` in `src/datasets/validator.py`
  - Implement `validate_with_llm_judge()`: use a different LLM from the generator; flag disagreements as `"disputed"`
  - Implement `select_human_review_sample()`: stratified 20% sample across ALL axes
  - Implement `compute_judge_error_rate()`: fraction of items where judge differs from human label
  - Implement `compute_inter_rater_reliability()`: Cohen's kappa for Axis 2, Spearman for Axis 3
  - Set `validation_status = "ambiguous"` when two Axis 2 reviewers disagree; flag sub-categories with kappa < 0.6
  - Implement disputed item resolution policy: when LLM-Judge says "harmful" but human says "benign" (or vice versa), the human label overrides and `validation_status` is set to `"disputed_human_override"`; the item remains in the dataset with the corrected label, NOT discarded (discarding biases toward easy items)
  - Implement `compute_disagreement_rates()`: report LLM-Judge/human disagreement rate per axis; a high rate on Axis 2 (Cultural) vs Axis 1 (Register) is critical context for interpreting calibration results under ambiguity
  - _Requirements: 15.1–15.4, 16.1–16.5_

  - [x]* 8.1 Write property tests for validation pipeline invariants
    - **Property 24: Validation disagreement flagging**
    - **Property 25: Stratified human review coverage**
    - **Property 26: Judge error rate computation**
    - **Property 27: Cohen's kappa range and computation**
    - **Validates: Requirements 15.2, 15.3, 15.4, 16.2, 16.4**
    - `# Feature: guardrail-calibration-eval, Property 24: Validation disagreement flagging`
    - `# Feature: guardrail-calibration-eval, Property 25: Stratified human review coverage`
    - `# Feature: guardrail-calibration-eval, Property 26: Judge error rate computation`
    - `# Feature: guardrail-calibration-eval, Property 27: Cohen's kappa range and computation`

- [x] 9. Checkpoint — Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 10. Implement `ExperimentRunner` — sequential loading, checkpointing, sanity check, VRAM monitoring
  - Implement `ExperimentRunner` in `src/evaluation/runner.py`
  - Implement `_load_and_run()`: load model → run all items → checkpoint every 500 → unload; verify VRAM at baseline before loading next model
  - Implement `_atomic_checkpoint()`: write to `path.tmp` then `os.rename()` to prevent Parquet corruption
  - Implement `run_sanity_check()`: 100-item check (50 harmful, 50 benign); compute mean, std, min, max, skewness, kurtosis, histogram; halt if std < 0.01; flag if < 5 bins covered; flag if accuracy < 50%
  - Set `temperature=0.0` for all local model inference; log temperature per guardrail
  - Record `inference_time_ms` and `timestamp_utc` for every prediction
  - Implement VRAM leak detection: log `torch.cuda.memory_allocated()` before and after every `unload_model()` call; on MPS use `torch.mps.current_allocated_memory()` with a fallback log warning if the API is unavailable; warn if post-unload memory exceeds baseline by > 500MB
  - Call `set_global_seeds()` and `log_environment()` from `src/utils/reproducibility.py` at runner initialization
  - _Requirements: 8.1, 8.2, 8.3, 8a.1, 8a.2, 8a.3, 17.1–17.5, 18.1–18.4_

  - [x]* 10.1 Write property tests for checkpoint and completeness
    - **Property 33: Checkpoint resumption correctness**
    - **Property 34: Atomic checkpoint integrity**
    - **Property 38: Completeness verification**
    - **Validates: Requirements 18.3, 18.4**
    - `# Feature: guardrail-calibration-eval, Property 33: Checkpoint resumption correctness`
    - `# Feature: guardrail-calibration-eval, Property 34: Atomic checkpoint integrity`
    - `# Feature: guardrail-calibration-eval, Property 38: Completeness verification`

  - [x]* 10.2 Write unit tests for sanity check thresholds
    - Test constant scores (std < 0.01) → halt with warning
    - Test < 5 bins covered → flag guardrail
    - Test accuracy < 50% → flag guardrail
    - _Requirements: 17.3, 17.4, 17.5_

- [x] 11. Implement pilot experiment and accuracy-calibration divergence detection (BEFORE full experiment)
  - Implement `run_pilot()` in `ExperimentRunner`: 2 guardrails (LlamaGuard 4 + Qwen3Guard-8B), 200-item subset, Axes 1 and 5
  - Implement divergence detection in `CalibrationAnalyzer`: identify conditions where accuracy and ECE diverge (accuracy drops but ECE stable, or vice versa)
  - Report divergence conditions as concrete motivating examples
  - The pilot MUST run before the full experiment (Task 13) to validate the pipeline, catch methodology issues, and confirm signal exists in the metrics before committing to the expensive ~10k-item run
  - _Requirements: 32.1–32.4_

  - [x]* 11.1 Write property test for accuracy-calibration divergence detection
    - **Property 32: Accuracy-calibration divergence detection**
    - **Validates: Requirements 32.3**
    - `# Feature: guardrail-calibration-eval, Property 32: Accuracy-calibration divergence detection`

- [x] 12. Implement ecological validation subset support
  - Add `ecological_validation` flag and source metadata to `DatasetItem`
  - Implement collection scaffolding in `DatasetBuilder`: 100-150 items from public APAC sources, covering ≥ 3 axes, in Singlish/Malay/Mandarin/Indonesian
  - Implement two-annotator labeling workflow (binary harmfulness + axis + approximate shift_level)
  - Implement ecological comparison in `CalibrationAnalyzer`: compute ECE and accuracy on ecological subset per guardrail; compute Spearman correlation between per-guardrail ECE on synthetic data and per-guardrail ECE on ecological subset — if model ranking is preserved, the synthetic methodology is ecologically plausible; report as exploratory plausibility check, not formal validation
  - _Requirements: 31.1–31.5_

- [x] 13. Implement `ExperimentRunner` — full experiment, API snapshot, quantization baseline
  - Implement `run_full_experiment()`: 6 guardrails × ~10k items, sequential loading, checkpointing
  - Implement `_run_api_canary_check()`: run 5-10 stable canary items at start/middle/end of 48-hour window; flag drift > 0.05; compare per-prediction timestamps against canary timestamps to detect mid-run API version changes
  - Implement `verify_completeness()`: after experiment, verify all item_ids have predictions; report missing items and whether skipped items are systematically biased (e.g., correlated with input length)
  - Implement `run_quantization_baseline()`: LlamaGuard 4 in 4-bit vs fp16 on 500-item subset; compute mean absolute confidence diff and ECE diff; warn if diff > 0.05 or ECE diff > 0.02
  - Record API model version, UTC timestamp for all OpenAI evaluations; complete within 48-hour window
  - _Requirements: 8b.1–8b.4, 18.1–18.4, 18a.2–18a.4, 18b.1–18b.3, 29.1, 29.2_

  - [x]* 13.1 Write property test for API canary drift detection
    - **Property 37: API canary drift detection**
    - **Validates: Requirements 18b**
    - `# Feature: guardrail-calibration-eval, Property 37: API canary drift detection`

  - [x]* 13.2 Write property test for prediction record completeness
    - **Property 12: Prediction record completeness**
    - **Validates: Requirements 18.2**
    - `# Feature: guardrail-calibration-eval, Property 12: Prediction record completeness`

- [x] 14. Implement `CalibrationAnalyzer` — core metrics
  - Implement `CalibrationAnalyzer` in `src/evaluation/calibration.py`
  - Implement `compute_adaptive_bin_count()`: `M = max(5, min(15, floor(N/15)))`
  - Implement `compute_calibration_curve()`: equal-width (primary) and adaptive binning; output bin edges, counts, mean confidence, actual accuracy
  - Implement `compute_ece()`: WEIGHTED average `np.sum(bin_counts * np.abs(mean_confidence - actual_accuracy)) / total_count` across bins (NOT `np.mean`); switch to Brier Score as primary when N < 100; exclude empty bins from computation and log them
  - Implement `compute_class_conditional_ece()`: separate ECE for harmful-class and benign-class predictions; required in all primary results
  - Implement `compute_brier_score()`: full decomposition into calibration + resolution + uncertainty (uncertainty fixed at 0.25)
  - Implement `compute_eoe()` (Expected Overconfidence Error): sum calibration gaps only where confidence > accuracy per bin; for safety guardrails, overconfidence (claiming benign with 0.99 when wrong) is the dangerous failure mode
  - Implement `compute_bin_sensitivity_sweep()`: for a given set of predictions, compute ECE for M in range [5, 20] and return the sweep as a list of (M, ECE) pairs; re-bin from scratch for each M value (do NOT reuse bin edges); run on a full guardrail's predictions across all axes (not a small subset) to get a stable curve that empirically justifies the adaptive formula
  - Stratify all cross-model ECE comparisons by `confidence_source_type`
  - _Requirements: 19.1–19.5, 20.1, 20.3–20.5, 21.1–21.5_

  - [x] 14.1 Write property tests for calibration metric correctness
    - **Property 14: Adaptive bin count formula**
    - **Property 15: ECE computation correctness** — verify weighted average, not unweighted mean
    - **Property 16: Class-conditional ECE decomposition**
    - **Property 18: Brier Score decomposition invariant**
    - **Property 41: EOE ≤ ECE** — EOE only sums overconfident bins (`confidence > accuracy`), so EOE ≤ ECE must hold; also verify EOE = 0 for a perfectly calibrated set; verify EOE ≠ ECE when underconfident bins exist (catches the bug of omitting the `confidence > accuracy` filter)
    - **Validates: Requirements 19.1, 20.1, 20.5, 21.1, 21.2, 21.5**
    - `# Feature: guardrail-calibration-eval, Property 14: Adaptive bin count formula`
    - `# Feature: guardrail-calibration-eval, Property 15: ECE computation correctness`
    - `# Feature: guardrail-calibration-eval, Property 16: Class-conditional ECE decomposition`
    - `# Feature: guardrail-calibration-eval, Property 18: Brier Score decomposition invariant`
    - `# Feature: guardrail-calibration-eval, Property 41: EOE leq ECE`

  - [x]* 14.2 Write unit tests for calibration edge cases
    - All predictions in one bin, all correct, all wrong
    - N < 100 → Brier Score as primary metric
    - Empty bin → excluded from ECE, logged
    - Verify ECE uses weighted formula (regression test: `np.sum(counts * gaps) / total` not `np.mean(gaps)`)
    - _Requirements: 19.5_

- [x] 15. Implement `CalibrationAnalyzer` — Axis 3, Axis 5, two-token mass, delta metrics
  - Implement `compute_spearman_correlation()`: Axis 3 only — confidence vs graded harmfulness; do NOT compute binary ECE for Axis 3
  - Implement `compute_delta_metrics()`: ΔAccuracy and ΔECE between shift_level=0 and max; produce side-by-side table
  - Implement `compute_two_token_mass_summary()`: mean, std, fraction below 0.5; required in all results for logit-based models
  - Implement `compute_ece_excluding_low_mass()`: ECE on clean subset (two_token_mass >= 0.5); report alongside full-dataset ECE
  - Implement `compute_token_length_correlation()`: Axis 5 — Spearman between token count and confidence, and between token count and correctness; STRATIFY by model (do not pool across models, as tokenizer differences confound the correlation); flag if p < 0.05
  - Implement `run_sensitivity_analysis()`: re-compute metrics with ambiguous items excluded; also re-compute with `western_norm_flag=True` items excluded for Axis 5
  - _Requirements: 11.6, 11.7, 13.7, 16.5, 20.4, 24.1, 24.3, 24.4_

  - [x] 15.1 Write property tests for Spearman, delta metrics, and two-token mass
    - **Property 31: Spearman correlation for Axis 3**
    - **Property 23: Delta metrics computation**
    - **Property 35: Two-token mass summary consistency**
    - **Property 36: Clean-subset ECE relationship**
    - **Validates: Requirements 11.8, 24.1, 1.7**
    - `# Feature: guardrail-calibration-eval, Property 31: Spearman correlation for Axis 3`
    - `# Feature: guardrail-calibration-eval, Property 23: Delta metrics computation`
    - `# Feature: guardrail-calibration-eval, Property 35: Two-token mass summary consistency`
    - `# Feature: guardrail-calibration-eval, Property 36: Clean-subset ECE relationship`

- [x] 16. Implement `BootstrapEngine` and `ThresholdAnalyzer`
  - Implement `BootstrapEngine` in `src/evaluation/bootstrap.py`
  - Implement `compute_ci()`: bootstrap CI for any metric function, 2000 resamples, alpha=0.05; resample by `seed_id` (cluster bootstrap), NOT by individual prediction, to respect within-seed correlation; verify resampled dataset preserves within-seed structure (all variants of a resampled seed are included); log warning if CI is degenerate (lower == upper)
  - Implement `pairwise_mcnemar()`: McNemar's test for paired ACCURACY comparison; align prediction sets by `item_id` before testing; drop and log any unmatched items; report count of dropped items
  - Implement `permutation_test_delta_ece()`: permutation test (N=5000 permutations) for statistically significant difference in ECE between two guardrails; permute guardrail assignments within each paired prediction (NOT the ECE values themselves); test statistic is absolute difference in ECE; p-value from null distribution of permuted differences
  - Implement `apply_holm_bonferroni()`: step-down correction; report both uncorrected and corrected p-values; scope the family of comparisons per-axis (all pairwise guardrail comparisons within one axis), NOT globally across all axes
  - Implement `compute_honest_threshold_with_ci()`: apply bootstrap (cluster by seed_id) to honest threshold computation; report the point estimate AND the 95% CI; report the "conservative honest threshold" as the CI lower bound — this is the only valid practitioner advice (a threshold of 0.85 with CI [0.60, 0.99] is not actionable)
  - Implement `ThresholdAnalyzer` in `src/evaluation/thresholds.py`
  - Implement `compute_precision_recall_at_thresholds()`: precision/recall at 0.80, 0.90, 0.95
  - Implement `compute_honest_threshold()`: minimum confidence to achieve target precision, per guardrail × axis × shift_level
  - Implement `compute_worst_case_threshold()`: max honest threshold across all conditions
  - Stratify honest threshold reporting by `confidence_source_type`
  - _Requirements: 20.2, 22.1–22.5, 23.1–23.3, 24.2_

  - [x] 16.1 Write property tests for bootstrap, thresholds, and statistical tests
    - **Property 17: Bootstrap CI coverage** — verify cluster bootstrap resamples by seed_id; verify resampled dataset contains all variants of each resampled seed (within-seed structure preserved); verify CI is wider than naive item-level bootstrap on the same data
    - **Property 19: Precision at threshold monotonicity**
    - **Property 20: Honest threshold correctness** — verify conservative honest threshold equals CI lower bound
    - **Property 21: Holm-Bonferroni correction ordering**
    - **Property 22: McNemar's test symmetry**
    - **Property 42: Permutation test ΔECE symmetry** — swapping model A and B should negate the delta but preserve the p-value; verify test permutes within-pair assignments, not ECE values
    - **Validates: Requirements 20.2, 22.1, 22.2, 22.4, 23.1, 23.2, 24.2**
    - `# Feature: guardrail-calibration-eval, Property 17: Bootstrap CI coverage`
    - `# Feature: guardrail-calibration-eval, Property 19: Precision at threshold monotonicity`
    - `# Feature: guardrail-calibration-eval, Property 20: Honest threshold correctness`
    - `# Feature: guardrail-calibration-eval, Property 21: Holm-Bonferroni correction ordering`
    - `# Feature: guardrail-calibration-eval, Property 22: McNemar's test symmetry`
    - `# Feature: guardrail-calibration-eval, Property 42: Permutation test delta ECE symmetry`

  - [x]* 16.2 Write unit tests for threshold edge cases
    - No predictions above threshold → handle gracefully
    - All predictions above threshold → precision = base rate
    - Target precision unachievable → report None or max confidence
    - _Requirements: 22.2_

- [x] 17. Checkpoint — Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 18. Implement `CalibrationAnalyzer` — OpenAI Moderation as separate practitioner analysis
  - Implement AUROC computation in `src/evaluation/calibration.py` (primary metric for OpenAI)
  - Present all OpenAI results in a dedicated section; frame as "can category_scores be used as calibrated confidence proxies?"
  - Report ECE and Brier Score for OpenAI with explicit caveats that they assume probabilistic inputs
  - Compute same threshold metrics (precision/recall at 0.80, 0.90, 0.95) for OpenAI
  - Do NOT include OpenAI in ranking tables alongside logit-based models without prominent caveat
  - _Requirements: 33.1–33.5_

  - [x]* 18.1 Write property test for AUROC computation
    - **Property 29: AUROC computation correctness**
    - **Validates: Requirements 33.3**
    - `# Feature: guardrail-calibration-eval, Property 29: AUROC computation correctness`

- [x] 19. Implement post-hoc calibration baseline (Platt Scaling / Isotonic Regression)
  - Implement `CalibrationTuner` in `src/evaluation/calibration.py`
  - Implement `fit_platt_scaling()`: fit logistic regression on dev split confidence scores → recalibrated probabilities; use `sklearn.calibration.CalibratedClassifierCV`; enforce strict dev/test split — the calibrator is fit ONLY on dev predictions and evaluated ONLY on test predictions (no data leakage); clip output to [0, 1]
  - Implement `fit_isotonic_regression()`: fit isotonic regression on dev split confidence scores → recalibrated probabilities; verify output is monotonic and in [0, 1]; degenerate step functions (e.g., all outputs = 0 or 1) indicate the input scores are not probabilistic — log a warning and skip isotonic for that model
  - Implement `compute_residual_ece()`: compute ECE on test split using recalibrated scores; report as "Residual ECE" — the miscalibration that remains after the simplest possible post-hoc fix
  - This answers the key research question: "Is the miscalibration we measured fixable with standard techniques, or is it structural?"
  - Stratify by `confidence_source_type`; only apply to logit-based models (Platt/Isotonic assume probabilistic inputs)

  - [x]* 19.1 Write property test for post-hoc calibration
    - **Property 43: Residual ECE ≤ Original ECE** — post-hoc calibration on dev split should not increase ECE on test split (or if it does, flag overfitting)
    - `# Feature: guardrail-calibration-eval, Property 43: Residual ECE leq Original ECE`

- [x] 20. Implement `PlotGenerator` and result persistence
  - Implement `PlotGenerator` in `src/analysis/plots.py`
  - Implement `plot_ece_vs_shift()`: ECE vs shift_level, one line per guardrail, 95% CI bands, per axis; publication-quality vector format, colorblind-safe palette
  - Implement `plot_reliability_heatmap()`: guardrails × axes, ECE as color
  - Implement `plot_safety_risk_heatmap()`: overlay EOE onto the honest threshold table — rows are guardrails, columns are axes, cells show EOE value with honest threshold as annotation; this shows practitioners not just whether a model is miscalibrated, but whether it is miscalibrated in the dangerous direction (false negatives / overconfident benign predictions)
  - Implement `generate_honest_threshold_table()`: per-guardrail per-axis honest thresholds with bootstrap CI and conservative honest threshold (CI lower bound); logit-based and api_score in separate sections; include baseline and worst-case per axis
  - Implement `plot_apac_language_comparison()`: ECE, Brier, precision-at-threshold across EN/MS/ZH/ID; highlight statistically significant differences
  - Implement `plot_bin_sensitivity_sweep()`: line plot of ECE vs M for the sweep from Task 14, showing the adaptive formula's M as a vertical marker
  - Persist all raw predictions to Parquet (primary) + CSV (backup) in `results/predictions/`; persist metrics to JSON in `results/metrics/`; persist figures to PDF in `results/figures/`
  - Log exact model versions, quantization settings, and random seeds for each run
  - _Requirements: 25.1, 25.2, 26.1, 26.2, 27.1–27.4, 28.1, 28.2, 29.1–29.3_

  - [x]* 20.1 Write property test for prediction persistence round-trip
    - **Property 5: Prediction persistence round-trip**
    - **Validates: Requirements 29.3**
    - `# Feature: guardrail-calibration-eval, Property 5: Prediction persistence round-trip`

- [x] 21. Wire everything together — end-to-end pipeline entry point
  - Implement `src/main.py` (or `src/pipeline.py`) as the single entry point
  - Load and validate `config.yaml` at startup; exit with descriptive error on failure
  - Call `set_global_seeds()` and `log_environment()` before any computation
  - Orchestrate: dataset generation → validation → sanity check → pilot (Task 11) → full experiment (Task 13) → analysis → post-hoc calibration → visualization
  - Ensure dev split used for all exploratory analysis and post-hoc calibration fitting; test split used only for final reported metrics
  - _Requirements: 18a.2, 18a.3, 18a.4, 30.1–30.3_

- [x] 22. Integration tests and final checkpoint
  - [x] 22.1 Write end-to-end integration test on small synthetic data
    - Generate 20 synthetic items → validate with mock judge → run mock adapters → compute metrics (including EOE and bin sensitivity sweep) → verify output structure
    - _Requirements: 17.1–17.5, 18.1–18.4, 19.1–19.5_

  - [x]* 22.2 Write checkpoint resumption integration test
    - Run 15 items, interrupt at item 10, resume → verify final predictions match uninterrupted run
    - _Requirements: 18.3_

- [x] 23. Final checkpoint — Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP; all others are required
- Each task references specific requirements for traceability
- Property tests use `hypothesis` with `@settings(max_examples=100)` minimum
- All local model adapters require GPU/MPS hardware; unit tests must mock model loading
- The design uses Python throughout — all code examples and implementations are Python
- Checkpoints ensure incremental validation at tasks 3, 9, 17, and 23
- **Pilot-first sequencing**: Task 11 (pilot) runs before Task 13 (full experiment) — this is intentional to validate the pipeline before the expensive run
- **Bootstrap resamples by seed_id**: All bootstrap CIs use cluster resampling grouped by `seed_id` to respect within-seed correlation; resampling individual predictions would produce artificially narrow CIs
- **ECE must use weighted average**: `np.sum(bin_counts * gaps) / total`, never `np.mean(gaps)` — this is the single most likely implementation bug that would invalidate results
- **EOE filter is mandatory**: `compute_eoe()` must filter `confidence > accuracy` per bin; omitting this filter makes EOE = ECE, losing the safety-relevant signal
- **Post-hoc calibration requires strict split**: Platt/Isotonic fit on dev, evaluate on test — any leakage makes residual ECE artificially low
- **Permutation test permutes within-pair assignments**: `permutation_test_delta_ece()` swaps which model produced which prediction for each item, NOT the ECE values themselves
- **Disputed items stay in dataset**: When LLM-Judge and human disagree, the human label overrides and the item remains with `validation_status="disputed_human_override"`; discarding disputed items biases toward easy examples
- **Conservative honest threshold = CI lower bound**: A threshold of 0.85 with CI [0.60, 0.99] is not actionable; always report the lower bound as the practitioner-safe recommendation
