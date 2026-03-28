"""Core data models for the guardrail calibration evaluation framework."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Literal


# ---------------------------------------------------------------------------
# Dataset layer
# ---------------------------------------------------------------------------

@dataclass
class SeedExample:
    seed_id: str
    text: str
    ground_truth: Literal["harmful", "benign"]
    source: str  # e.g. "HarmBench", "AdvBench", "ToxiGen"


@dataclass
class DatasetItem:
    item_id: str
    seed_id: str
    axis: int                                    # 1-5
    shift_level: int                             # 0-4
    ground_truth: Literal["harmful", "benign"]   # Axes 1,2,4,5
    graded_harmfulness: float | None             # Axis 3 only (0.0-1.0)
    seed_text: str
    variant_text: str
    generation_method: str                       # e.g. "gpt-4o-mini", "human_translation"
    validation_status: Literal["validated", "disputed", "disputed_human_override",
                               "ambiguous", "pending"]
    cultural_frame: str | None                   # Axis 2 only
    token_counts: dict[str, int] | None          # Axis 5: {model_name: count}
    split: Literal["dev", "test"]
    western_norm_flag: bool = False              # True if translation introduced ambiguity
    ecological_validation: bool = False          # True if from real-world APAC source
    ecological_source: str | None = None         # e.g. "reddit_sg", "twitter_my"
    adversarial_pressure_type: str | None = None # Axis 3: "soft_indirection"|"hard_adversarial"

    def to_json(self) -> str:
        """Serialize to JSON string preserving None (not NaN)."""
        return json.dumps(self.__dict__, ensure_ascii=False, sort_keys=True)

    @classmethod
    def from_json(cls, json_str: str) -> "DatasetItem":
        """Deserialize from JSON string. None fields remain None (not NaN)."""
        data = json.loads(json_str)
        # Ensure optional fields default correctly when absent from older serializations
        data.setdefault("western_norm_flag", False)
        data.setdefault("ecological_validation", False)
        data.setdefault("ecological_source", None)
        data.setdefault("adversarial_pressure_type", None)
        return cls(**data)


# ---------------------------------------------------------------------------
# Prediction layer
# ---------------------------------------------------------------------------

@dataclass
class PredictionResult:
    label: Literal["harmful", "benign"]
    confidence_score: float                      # 0.0-1.0, normalized
    two_token_mass: float | None                 # Logit-based only
    raw_logits: dict[str, float] | None          # {"safe_logit": p, "unsafe_logit": p}


@dataclass
class Prediction:
    guardrail_name: str
    item_id: str
    predicted_label: Literal["harmful", "benign"]
    confidence_score: float
    inference_time_ms: float
    two_token_mass: float | None
    confidence_source_type: Literal["logits_softmax", "native_safety_score", "api_score"]
    split: Literal["dev", "test"]
    timestamp_utc: str                           # ISO-8601 UTC for API drift detection


# ---------------------------------------------------------------------------
# Metrics layer
# ---------------------------------------------------------------------------

@dataclass
class CalibrationCurve:
    bin_edges: list[float]
    bin_counts: list[int]
    mean_confidence: list[float]
    actual_accuracy: list[float]
    binning_method: Literal["equal_width", "adaptive"]


@dataclass
class ECEResult:
    ece: float
    ci_lower: float
    ci_upper: float
    n_bins: int
    n_items: int


@dataclass
class ClassConditionalECE:
    harmful_ece: ECEResult
    benign_ece: ECEResult
    overall_ece: ECEResult


@dataclass
class BrierDecomposition:
    brier_score: float
    calibration: float
    resolution: float
    uncertainty: float  # Fixed at 0.25 for 50/50 balance


@dataclass
class SpearmanResult:
    correlation: float
    p_value: float
    n_items: int


@dataclass
class HonestThreshold:
    target_precision: float
    honest_confidence: float          # Point estimate
    ci_lower: float                   # 95% CI lower bound (conservative threshold)
    ci_upper: float
    actual_precision_at_target: float
    guardrail: str
    axis: int
    shift_level: int


@dataclass
class DeltaMetrics:
    delta_accuracy: float
    delta_accuracy_ci: tuple[float, float]
    delta_ece: float
    delta_ece_ci: tuple[float, float]
    guardrail: str
    axis: int


@dataclass
class SanityCheckReport:
    guardrail: str
    mean: float
    std: float
    min_score: float
    max_score: float
    skewness: float
    kurtosis: float
    bins_covered: int       # Out of 10
    accuracy: float
    passed: bool
    warnings: list[str]


@dataclass
class TwoTokenMassSummary:
    mean_mass: float
    std_mass: float
    fraction_below_threshold: float  # Fraction of items with mass < threshold
    n_items: int
    threshold: float = 0.5


@dataclass
class TokenLengthAnalysis:
    guardrail_name: str
    correlation_with_confidence: float   # Spearman: token_count vs confidence
    correlation_with_correctness: float  # Spearman: token_count vs correct/incorrect
    p_value_confidence: float
    p_value_correctness: float
    mean_token_ratio: float              # Mean(non-EN tokens / EN tokens) per seed


@dataclass
class CanaryCheckResult:
    canary_items: list[str]   # item_ids used as canaries
    max_score_drift: float    # Max absolute score change across runs
    drift_detected: bool      # True if any drift > 0.05
    timestamps: list[str]     # UTC timestamps of each canary run


@dataclass
class CompletenessReport:
    total_expected: int
    total_found: int
    missing_item_ids: list[str]
    skipped_item_analysis: dict   # Distribution stats of skipped items


# ---------------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------------

@dataclass
class ExperimentConfig:
    guardrails: list[str]
    axes: list[int]
    bootstrap_resamples: int = 2000
    operational_thresholds: list[float] = field(
        default_factory=lambda: [0.80, 0.90, 0.95]
    )
    checkpoint_frequency: int = 500
    sanity_check_size: int = 100
    pilot_guardrails: list[str] = field(
        default_factory=lambda: ["llamaguard4", "qwen3guard"]
    )
    pilot_axes: list[int] = field(default_factory=lambda: [1, 5])
    pilot_size: int = 200
    quantization_baseline_size: int = 500
    temperature: float = 0.0
    api_snapshot_window_hours: int = 48
    random_seed: int = 42

    @classmethod
    def from_yaml(cls, path: str) -> "ExperimentConfig":
        """Load and validate config from YAML file."""
        import yaml
        with open(path) as f:
            data = yaml.safe_load(f)
        cfg = cls(**data)
        errors = cfg.validate()
        if errors:
            raise ValueError("Config validation failed:\n" + "\n".join(errors))
        return cfg

    def validate(self) -> list[str]:
        """Return list of validation errors (empty = valid)."""
        errors: list[str] = []
        if self.bootstrap_resamples < 500:
            errors.append(f"bootstrap_resamples={self.bootstrap_resamples} < 500")
        if self.checkpoint_frequency < 100:
            errors.append(f"checkpoint_frequency={self.checkpoint_frequency} < 100")
        if not (0.0 <= self.temperature <= 1.0):
            errors.append(f"temperature={self.temperature} not in [0.0, 1.0]")
        if self.api_snapshot_window_hours < 12:
            errors.append(f"api_snapshot_window_hours={self.api_snapshot_window_hours} < 12")
        for t in self.operational_thresholds:
            if not (0.0 < t < 1.0):
                errors.append(f"threshold={t} not in (0.0, 1.0)")
        if not self.guardrails:
            errors.append("guardrails list is empty")
        if not self.axes:
            errors.append("axes list is empty")
        for a in self.axes:
            if a not in {1, 2, 3, 4, 5}:
                errors.append(f"axis={a} not in {{1,2,3,4,5}}")
        return errors
