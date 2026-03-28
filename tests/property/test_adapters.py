"""Property and unit tests for GuardrailAdapter base class and LogitBasedAdapterMixin.

# Feature: guardrail-calibration-eval, Property 1: Adapter output contract
# Feature: guardrail-calibration-eval, Property 2: Confidence source type derivation
# Feature: guardrail-calibration-eval, Property 3: Two-token probability mass invariant
# Feature: guardrail-calibration-eval, Property 40: Prompt template application
"""
from __future__ import annotations


import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from src.guardrails.base import GuardrailAdapter, LogitBasedAdapterMixin
from src.models import PredictionResult


# ---------------------------------------------------------------------------
# Minimal concrete adapter for testing
# ---------------------------------------------------------------------------

class _FakeAdapter(GuardrailAdapter):
    """Minimal concrete adapter that wraps a fixed prediction for testing."""

    def __init__(self, label="benign", confidence=0.8, source="logits_softmax"):
        self._label = label
        self._confidence = confidence
        self._source = source

    def predict(self, text: str) -> PredictionResult:
        self._validate_input(text)
        return PredictionResult(
            label=self._label,
            confidence_score=self._confidence,
            two_token_mass=None,
            raw_logits=None,
        )

    def format_prompt(self, text: str) -> str:
        return f"[TEMPLATE] {text}"

    def get_model_name(self) -> str:
        return "FakeModel (4-bit)"

    @property
    def confidence_source(self):
        return self._source

    def load_model(self) -> None:
        pass

    def unload_model(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Property 1: Adapter output contract
# ---------------------------------------------------------------------------

@given(
    text=st.text(min_size=1, max_size=500).filter(lambda t: t.strip()),
    label=st.sampled_from(["harmful", "benign"]),
    confidence=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
)
@settings(max_examples=200)
def test_adapter_output_contract(text, label, confidence):
    """predict() always returns a PredictionResult with valid label and confidence."""
    adapter = _FakeAdapter(label=label, confidence=confidence)
    result = adapter.predict(text)
    assert isinstance(result, PredictionResult)
    assert result.label in ("harmful", "benign")
    assert 0.0 <= result.confidence_score <= 1.0


# ---------------------------------------------------------------------------
# Property 2: Confidence source type derivation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("source,expected_type", [
    ("logits_softmax", "logits_softmax"),
    ("native_safety_score", "native_safety_score"),
    ("category_scores", "api_score"),
])
def test_confidence_source_type_mapping(source, expected_type):
    """confidence_source_type maps correctly for all three source values."""
    adapter = _FakeAdapter(source=source)
    assert adapter.confidence_source_type == expected_type


# ---------------------------------------------------------------------------
# Property 40: Prompt template application
# ---------------------------------------------------------------------------

@given(text=st.text(min_size=1, max_size=200).filter(lambda t: t.strip()))
@settings(max_examples=100)
def test_format_prompt_is_not_identity(text):
    """format_prompt() must transform the text (template is applied)."""
    adapter = _FakeAdapter()
    formatted = adapter.format_prompt(text)
    assert formatted != text, "format_prompt() returned the raw text unchanged"
    assert text in formatted, "format_prompt() should contain the original text"


# ---------------------------------------------------------------------------
# Unit tests: ValueError on invalid input
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad_input", [
    None,
    "",
    "   ",
    "\t\n",
])
def test_predict_raises_on_invalid_input(bad_input):
    """predict() raises ValueError for empty, None, or whitespace-only input."""
    adapter = _FakeAdapter()
    with pytest.raises(ValueError):
        adapter.predict(bad_input)


# ---------------------------------------------------------------------------
# Property 3: Two-token probability mass invariant
# ---------------------------------------------------------------------------

class _FakeLogitAdapter(LogitBasedAdapterMixin):
    safe_token_id = 0
    unsafe_token_id = 1


torch = pytest.importorskip("torch", reason="torch not installed; skipping logit-based tests")


@given(
    safe_logit=st.floats(min_value=-10.0, max_value=10.0, allow_nan=False),
    unsafe_logit=st.floats(min_value=-10.0, max_value=10.0, allow_nan=False),
    vocab_size=st.integers(min_value=2, max_value=100),
)
@settings(max_examples=300)
def test_two_token_mass_invariants(safe_logit, unsafe_logit, vocab_size):
    """Two-token mass is in [0,1] and confidence_score is in [0.5, 1.0]."""
    logits = torch.zeros(vocab_size)
    logits[0] = safe_logit
    logits[1] = unsafe_logit

    adapter = _FakeLogitAdapter()
    result = adapter.compute_confidence(logits)

    assert result.two_token_mass is not None
    assert 0.0 <= result.two_token_mass <= 1.0
    assert 0.0 <= result.confidence_score <= 1.0
    # Confidence is max of normalized probs, so always >= 0.5
    assert result.confidence_score >= 0.5 - 1e-9
    assert result.label in ("harmful", "benign")


@given(
    safe_logit=st.floats(min_value=-10.0, max_value=10.0, allow_nan=False),
    unsafe_logit=st.floats(min_value=-10.0, max_value=10.0, allow_nan=False),
)
@settings(max_examples=200)
def test_two_token_mass_sums_to_raw_probs(safe_logit, unsafe_logit):
    """two_token_mass == raw_logits['safe_logit'] + raw_logits['unsafe_logit']."""
    logits = torch.tensor([safe_logit, unsafe_logit, 0.0, 0.0])
    adapter = _FakeLogitAdapter()
    result = adapter.compute_confidence(logits)

    assert result.raw_logits is not None
    expected_mass = result.raw_logits["safe_logit"] + result.raw_logits["unsafe_logit"]
    assert abs(result.two_token_mass - expected_mass) < 1e-6
