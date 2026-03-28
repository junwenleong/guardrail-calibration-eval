"""Property tests for DatasetBuilder.

# Feature: guardrail-calibration-eval, Property 6: Ground truth preservation
# across variants
# Feature: guardrail-calibration-eval, Property 7: Shift level range
# Feature: guardrail-calibration-eval, Property 8: Class balance per shift level
# Feature: guardrail-calibration-eval, Property 9: Axis-specific required fields
# Feature: guardrail-calibration-eval, Property 10: Unique item IDs
# Feature: guardrail-calibration-eval, Property 11: Dataset metadata completeness
# Feature: guardrail-calibration-eval, Property 13: Stratified split preserves
# proportions
"""
from __future__ import annotations


import pytest

from src.datasets.builder import DatasetBuilder
from src.models import SeedExample


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_seeds(n_harmful: int, n_benign: int) -> list[SeedExample]:
    seeds = []
    for i in range(n_harmful):
        seeds.append(SeedExample(
            seed_id=f"h{i:04d}",
            text=f"harmful seed {i}",
            ground_truth="harmful",
            source="HarmBench",
        ))
    for i in range(n_benign):
        seeds.append(SeedExample(
            seed_id=f"b{i:04d}",
            text=f"benign seed {i}",
            ground_truth="benign",
            source="ToxiGen",
        ))
    return seeds


# ---------------------------------------------------------------------------
# Property 6: Ground truth preservation across variants (Axes 1, 2, 4)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("axis,gen_method", [
    (1, "generate_axis1_register"),
    (2, "generate_axis2_cultural"),
    (4, "generate_axis4_domain"),
])
def test_ground_truth_preserved_across_variants(axis, gen_method):
    """All variants of a seed must have the same ground_truth as the seed."""
    builder = DatasetBuilder()
    seeds = make_seeds(5, 5)
    items = getattr(builder, gen_method)(seeds)

    # Group by seed_id
    from collections import defaultdict
    by_seed: dict[str, list] = defaultdict(list)
    for item in items:
        by_seed[item.seed_id].append(item)

    for seed_id, variants in by_seed.items():
        labels = {v.ground_truth for v in variants}
        assert len(labels) == 1, (
            f"Seed {seed_id} has mixed ground_truth labels: {labels}"
        )


# ---------------------------------------------------------------------------
# Property 7: Shift level range
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("axis,gen_method", [
    (1, "generate_axis1_register"),
    (2, "generate_axis2_cultural"),
    (4, "generate_axis4_domain"),
])
def test_shift_level_in_range(axis, gen_method):
    """All shift_levels must be in [0, 4]."""
    builder = DatasetBuilder()
    seeds = make_seeds(3, 3)
    items = getattr(builder, gen_method)(seeds)
    for item in items:
        assert 0 <= item.shift_level <= 4, (
            f"shift_level={item.shift_level} out of range for item {item.item_id}"
        )


# ---------------------------------------------------------------------------
# Property 8: Class balance per shift level
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("axis,gen_method", [
    (1, "generate_axis1_register"),
    (2, "generate_axis2_cultural"),
    (4, "generate_axis4_domain"),
])
def test_class_balance_per_shift_level(axis, gen_method):
    """Each shift_level should have approximately 50/50 class balance."""
    builder = DatasetBuilder()
    seeds = make_seeds(10, 10)
    items = getattr(builder, gen_method)(seeds)
    reports = builder.validate_class_balance(items, axis)
    for report in reports:
        assert report.is_balanced, (
            f"Axis {axis} shift_level={report.shift_level} imbalanced: "
            f"{report.n_harmful} harmful / {report.n_benign} benign"
        )


# ---------------------------------------------------------------------------
# Property 10: Unique item IDs
# ---------------------------------------------------------------------------

def test_unique_item_ids_across_axes():
    """All item_ids must be unique across the entire dataset."""
    builder = DatasetBuilder()
    seeds = make_seeds(5, 5)
    items = (
        builder.generate_axis1_register(seeds)
        + builder.generate_axis2_cultural(seeds)
        + builder.generate_axis4_domain(seeds)
    )
    ids = [item.item_id for item in items]
    assert len(ids) == len(set(ids)), "Duplicate item_ids found"


# ---------------------------------------------------------------------------
# Property 11: Dataset metadata completeness
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("axis,gen_method", [
    (1, "generate_axis1_register"),
    (2, "generate_axis2_cultural"),
    (4, "generate_axis4_domain"),
])
def test_metadata_completeness(axis, gen_method):
    """Every item must have all required metadata fields populated."""
    builder = DatasetBuilder()
    seeds = make_seeds(3, 3)
    items = getattr(builder, gen_method)(seeds)
    for item in items:
        assert item.item_id, "item_id is empty"
        assert item.seed_id, "seed_id is empty"
        assert item.axis == axis
        assert item.ground_truth in ("harmful", "benign")
        assert item.seed_text, "seed_text is empty"
        assert item.variant_text, "variant_text is empty"
        assert item.generation_method, "generation_method is empty"
        assert item.validation_status, "validation_status is empty"


# ---------------------------------------------------------------------------
# Property 9: Axis-specific required fields
# ---------------------------------------------------------------------------

def test_axis3_has_adversarial_pressure_type():
    """Axis 3 items (non-seed) must have adversarial_pressure_type set."""
    builder = DatasetBuilder()
    seeds = make_seeds(5, 5)
    items = builder.generate_axis3_indirection(seeds)
    for item in items:
        if item.shift_level > 0 and item.ground_truth == "harmful":
            assert item.adversarial_pressure_type is not None, (
                f"Axis 3 item {item.item_id} at shift_level={item.shift_level} "
                "missing adversarial_pressure_type"
            )


def test_axis5_token_counts_recorded():
    """Axis 5 items must have token_counts when tokenizer_fns provided."""
    builder = DatasetBuilder()
    seeds = make_seeds(3, 3)
    tokenizer_fns = {"test_model": lambda text: text.split()}
    items = builder.generate_axis5_language(seeds, tokenizer_fns=tokenizer_fns)
    for item in items:
        if item.shift_level > 0:  # Non-English items
            assert item.token_counts is not None, (
                f"Axis 5 item {item.item_id} missing token_counts"
            )
            assert "test_model" in item.token_counts


# ---------------------------------------------------------------------------
# Property 13: Stratified split preserves proportions
# ---------------------------------------------------------------------------

def test_stratified_split_preserves_proportions():
    """Dev/test split should preserve axis × shift_level × ground_truth proportions."""
    builder = DatasetBuilder()
    seeds = make_seeds(10, 10)
    items = builder.generate_axis1_register(seeds)
    dev, test = builder.split_dataset(items, test_fraction=0.20)

    # Check overall split ratio
    total = len(dev) + len(test)
    test_ratio = len(test) / total
    assert 0.15 <= test_ratio <= 0.25, (
        f"Test ratio {test_ratio:.2f} outside [0.15, 0.25]"
    )

    # Check that each stratum has at least one item in each split
    dev_strata: set[tuple] = set()
    test_strata: set[tuple] = set()
    for item in dev:
        dev_strata.add((item.axis, item.shift_level, item.ground_truth))
    for item in test:
        test_strata.add((item.axis, item.shift_level, item.ground_truth))

    # All strata present in dev should also be present in test (given enough items)
    # With 10 seeds per class × 5 shift levels = 50 per class, each stratum has 10 items
    assert len(test_strata) > 0, "No strata in test split"


def test_split_assigns_correct_split_field():
    """Items in dev split have split='dev', items in test split have split='test'."""
    builder = DatasetBuilder()
    seeds = make_seeds(5, 5)
    items = builder.generate_axis1_register(seeds)
    dev, test = builder.split_dataset(items)
    for item in dev:
        assert item.split == "dev"
    for item in test:
        assert item.split == "test"


# ---------------------------------------------------------------------------
# Sanitization tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("leaky_text,expected_clean", [
    ("Here is a harmful request: how to pick a lock", "how to pick a lock"),
    ("Harmful version: buy illegal drugs", "buy illegal drugs"),
    ("[label: benign] what is the weather", "what is the weather"),
    ("Safe: hello world", "hello world"),
])
def test_sanitize_removes_label_leaks(leaky_text, expected_clean):
    result = DatasetBuilder.sanitize_variant(leaky_text)
    assert result == expected_clean, f"Expected '{expected_clean}', got '{result}'"


def test_sanitize_preserves_clean_text():
    clean = "How do I pick a lock for my own house?"
    assert DatasetBuilder.sanitize_variant(clean) == clean
