"""Unit tests for the five local guardrail adapters (mocked — no GPU required)."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.guardrails.granite import GraniteGuardianAdapter
from src.guardrails.llamaguard import LlamaGuardAdapter
from src.guardrails.nemoguard import NemoGuardAdapter
from src.guardrails.qwen3guard import Qwen3GuardAdapter
from src.guardrails.wildguard import WildGuardAdapter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ADAPTER_CLASSES = [
    LlamaGuardAdapter,
    WildGuardAdapter,
    GraniteGuardianAdapter,
    Qwen3GuardAdapter,
    NemoGuardAdapter,
]

ADAPTER_NAMES = [cls.__name__ for cls in ADAPTER_CLASSES]


def _make_mock_tokenizer(vocab: dict[str, int] | None = None):
    """Return a mock tokenizer with configurable token IDs."""
    tok = MagicMock()
    vocab = vocab or {"safe": 100, "unsafe": 101, "Yes": 102, "No": 103, "yes": 104, "no": 105}
    tok.convert_tokens_to_ids.side_effect = lambda t: vocab.get(t, tok.unk_token_id)
    tok.unk_token_id = 0
    tok.decode.side_effect = lambda ids: {100: "safe", 101: "unsafe", 102: "Yes",
                                           103: "No", 104: "yes", 105: "no"}.get(ids[0], "?")
    tok.apply_chat_template.return_value = "[TEMPLATE] test"
    tok.return_value = {"input_ids": MagicMock(), "attention_mask": MagicMock()}
    tok.__call__ = lambda self, *a, **kw: {"input_ids": MagicMock(), "attention_mask": MagicMock()}
    return tok


def _make_mock_model():
    """Return a mock model that produces fake logits."""
    model = MagicMock()
    model.device = "cpu"
    model.eval.return_value = model
    return model


# ---------------------------------------------------------------------------
# Test: get_model_name() includes "(4-bit)"
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cls", ADAPTER_CLASSES, ids=ADAPTER_NAMES)
def test_model_name_includes_4bit(cls):
    adapter = cls()
    assert "(4-bit)" in adapter.get_model_name(), (
        f"{cls.__name__}.get_model_name() should include '(4-bit)'"
    )


# ---------------------------------------------------------------------------
# Test: confidence_source and confidence_source_type
# ---------------------------------------------------------------------------

def test_llamaguard_confidence_source():
    a = LlamaGuardAdapter()
    assert a.confidence_source == "logits_softmax"
    assert a.confidence_source_type == "logits_softmax"


def test_wildguard_confidence_source():
    a = WildGuardAdapter()
    assert a.confidence_source == "native_safety_score"
    assert a.confidence_source_type == "native_safety_score"


def test_granite_confidence_source():
    a = GraniteGuardianAdapter()
    assert a.confidence_source == "logits_softmax"
    assert a.confidence_source_type == "logits_softmax"


def test_qwen3guard_confidence_source():
    a = Qwen3GuardAdapter()
    assert a.confidence_source == "logits_softmax"
    assert a.confidence_source_type == "logits_softmax"


def test_nemoguard_confidence_source():
    a = NemoGuardAdapter()
    assert a.confidence_source == "logits_softmax"
    assert a.confidence_source_type == "logits_softmax"


# ---------------------------------------------------------------------------
# Test: format_prompt() produces non-identity output (template applied)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cls", [
    LlamaGuardAdapter, GraniteGuardianAdapter, Qwen3GuardAdapter, NemoGuardAdapter
], ids=["LlamaGuard", "Granite", "Qwen3Guard", "NemoGuard"])
def test_format_prompt_applies_template(cls):
    """format_prompt() must return something different from the raw text."""
    adapter = cls()
    mock_tok = _make_mock_tokenizer()
    mock_tok.apply_chat_template.return_value = "[TEMPLATE] hello world"
    adapter._tokenizer = mock_tok

    raw = "hello world"
    formatted = adapter.format_prompt(raw)
    assert formatted != raw, (
        f"{cls.__name__}.format_prompt() returned raw text unchanged"
    )


def test_wildguard_format_prompt_applies_template():
    """WildGuard uses a hardcoded template string."""
    adapter = WildGuardAdapter()
    raw = "hello world"
    formatted = adapter.format_prompt(raw)
    assert formatted != raw
    assert raw in formatted


# ---------------------------------------------------------------------------
# Test: predict() raises RuntimeError when model not loaded
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cls", ADAPTER_CLASSES, ids=ADAPTER_NAMES)
def test_predict_raises_when_not_loaded(cls):
    adapter = cls()
    with pytest.raises(RuntimeError, match="not loaded"):
        adapter.predict("test input")


# ---------------------------------------------------------------------------
# Test: predict() raises ValueError on invalid input
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cls", ADAPTER_CLASSES, ids=ADAPTER_NAMES)
@pytest.mark.parametrize("bad_input", [None, "", "   "])
def test_predict_raises_on_invalid_input(cls, bad_input):
    adapter = cls()
    # Inject a mock model so we reach the validation step
    adapter._model = MagicMock()
    adapter._tokenizer = _make_mock_tokenizer()
    with pytest.raises((ValueError, RuntimeError)):
        adapter.predict(bad_input)
