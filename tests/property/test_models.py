"""Property tests for core data models.

# Feature: guardrail-calibration-eval, Property 4: Metadata JSON round-trip
# Feature: guardrail-calibration-eval, Property 30: Config validation round-trip
# Feature: guardrail-calibration-eval, Property 39: Config semantic validation
"""
from __future__ import annotations

import math

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from src.models import DatasetItem, ExperimentConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

VALID_GROUND_TRUTH = st.sampled_from(["harmful", "benign"])
VALID_VALIDATION_STATUS = st.sampled_from([
    "validated", "disputed", "disputed_human_override", "ambiguous", "pending"
])
VALID_SPLIT = st.sampled_from(["dev", "test"])


def dataset_item_strategy(axis: int | None = None):
    """Hypothesis strategy that generates valid DatasetItem instances."""
    axis_st = (
        st.just(axis)
        if axis is not None
        else st.integers(min_value=1, max_value=5)
    )
    char_set = st.characters(
        whitelist_categories=("Lu", "Ll", "Nd"), whitelist_characters="-_"
    )
    return axis_st.flatmap(lambda ax: st.builds(
        DatasetItem,
        item_id=st.text(min_size=1, max_size=32, alphabet=char_set),
        seed_id=st.text(min_size=1, max_size=32, alphabet=char_set),
        axis=st.just(ax),
        shift_level=st.integers(min_value=0, max_value=4),
        ground_truth=VALID_GROUND_TRUTH,
        graded_harmfulness=(
            st.none()
            if ax != 3
            else st.floats(min_value=0.0, max_value=1.0, allow_nan=False)
        ),
        seed_text=st.text(min_size=1, max_size=200),
        variant_text=st.text(min_size=1, max_size=200),
        generation_method=st.sampled_from(
            ["gpt-4o-mini", "human_translation", "manual"]
        ),
        validation_status=VALID_VALIDATION_STATUS,
        cultural_frame=(
            st.none()
            if ax != 2
            else st.sampled_from(
                ["filial_piety", "traditional_medicine", "singlish"]
            )
        ),
        token_counts=(
            st.none()
            if ax != 5
            else st.dictionaries(
                keys=st.sampled_from(["llamaguard4", "wildguard"]),
                values=st.integers(min_value=1, max_value=2048),
                min_size=1,
            )
        ),
        split=VALID_SPLIT,
        western_norm_flag=st.booleans(),
        ecological_validation=st.booleans(),
        ecological_source=st.none(),
        adversarial_pressure_type=(
            st.none()
            if ax != 3
            else st.sampled_from(
                ["soft_indirection", "hard_adversarial", None]
            )
        ),
    ))


# ---------------------------------------------------------------------------
# Property 4: Metadata JSON round-trip
# ---------------------------------------------------------------------------

@given(dataset_item_strategy())
@settings(max_examples=200)
def test_dataset_item_json_roundtrip(item: DatasetItem):
    """Serializing then deserializing a DatasetItem produces an equivalent object."""
    json_str = item.to_json()
    restored = DatasetItem.from_json(json_str)
    assert restored == item, f"Round-trip failed:\noriginal={item}\nrestored={restored}"


@given(dataset_item_strategy(axis=3))
@settings(max_examples=100)
def test_none_graded_harmfulness_preserved(item: DatasetItem):
    """None graded_harmfulness survives JSON round-trip as None, not NaN."""
    # Force None for this test
    item.graded_harmfulness = None
    restored = DatasetItem.from_json(item.to_json())
    assert restored.graded_harmfulness is None
    # Ensure the None check works (would fail if NaN was returned)
    assert not (restored.graded_harmfulness is not None)


@given(dataset_item_strategy())
@settings(max_examples=100)
def test_none_token_counts_preserved(item: DatasetItem):
    """None token_counts survives JSON round-trip as None."""
    item.token_counts = None
    restored = DatasetItem.from_json(item.to_json())
    assert restored.token_counts is None


@given(dataset_item_strategy(axis=3))
@settings(max_examples=100)
def test_float_graded_harmfulness_preserved(item: DatasetItem):
    """Float graded_harmfulness survives round-trip without NaN corruption."""
    if item.graded_harmfulness is not None:
        restored = DatasetItem.from_json(item.to_json())
        assert restored.graded_harmfulness is not None
        assert not math.isnan(restored.graded_harmfulness)
        assert abs(restored.graded_harmfulness - item.graded_harmfulness) < 1e-9


# ---------------------------------------------------------------------------
# Property 30 & 39: ExperimentConfig validation
# ---------------------------------------------------------------------------

def make_valid_config(**overrides) -> ExperimentConfig:
    defaults = dict(
        guardrails=["llamaguard4", "wildguard"],
        axes=[1, 2, 3, 4, 5],
        bootstrap_resamples=500,
        operational_thresholds=[0.80, 0.90, 0.95],
        checkpoint_frequency=100,
        sanity_check_size=100,
        pilot_guardrails=["llamaguard4"],
        pilot_axes=[1, 5],
        pilot_size=200,
        quantization_baseline_size=500,
        temperature=0.0,
        api_snapshot_window_hours=48,
        random_seed=42,
    )
    defaults.update(overrides)
    return ExperimentConfig(**defaults)


def test_valid_config_has_no_errors():
    cfg = make_valid_config()
    assert cfg.validate() == []


@pytest.mark.parametrize("field,bad_value,expected_fragment", [
    ("bootstrap_resamples", 499, "bootstrap_resamples"),
    ("checkpoint_frequency", 99, "checkpoint_frequency"),
    ("temperature", 1.5, "temperature"),
    ("api_snapshot_window_hours", 11, "api_snapshot_window_hours"),
    ("guardrails", [], "guardrails"),
    ("axes", [], "axes"),
])
def test_invalid_config_reports_error(field, bad_value, expected_fragment):
    cfg = make_valid_config(**{field: bad_value})
    errors = cfg.validate()
    assert any(expected_fragment in e for e in errors), (
        f"Expected error containing '{expected_fragment}', got: {errors}"
    )


def test_invalid_axis_value_reported():
    cfg = make_valid_config(axes=[1, 6])
    errors = cfg.validate()
    assert any("axis=6" in e for e in errors)


def test_threshold_out_of_range_reported():
    cfg = make_valid_config(operational_thresholds=[0.0, 0.9])
    errors = cfg.validate()
    assert any("threshold=0.0" in e for e in errors)


@given(st.integers(min_value=500, max_value=10000))
@settings(max_examples=50)
def test_valid_bootstrap_resamples_no_error(n):
    cfg = make_valid_config(bootstrap_resamples=n)
    errors = cfg.validate()
    assert not any("bootstrap_resamples" in e for e in errors)
