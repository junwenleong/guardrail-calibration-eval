"""Unit tests for ShieldGemma adapter (replaces OpenAI Moderation adapter).

ShieldGemma 9B runs locally via Ollama — no API keys required.
"""
from __future__ import annotations

import math
from unittest.mock import MagicMock, patch

import pytest

from src.guardrails.shieldgemma import ShieldGemmaAdapter


def _make_ollama_response(yes_logprob: float, no_logprob: float) -> dict:
    """Build a mock Ollama /api/generate response with logprobs."""
    return {
        "response": "Yes" if yes_logprob > no_logprob else "No",
        "logprobs": {
            "token_logprobs": [
                {"Yes": yes_logprob, "No": no_logprob}
            ]
        },
    }


def _make_text_only_response(text: str) -> dict:
    """Build a mock Ollama response without logprobs (fallback path)."""
    return {"response": text, "logprobs": None}


# ---------------------------------------------------------------------------
# Test: successful prediction via logprobs
# ---------------------------------------------------------------------------

def test_predict_harmful_via_logprobs():
    adapter = ShieldGemmaAdapter()
    # Yes (harmful) has higher logprob
    mock_resp = MagicMock()
    mock_resp.json.return_value = _make_ollama_response(
        yes_logprob=math.log(0.9), no_logprob=math.log(0.1)
    )
    mock_resp.raise_for_status = MagicMock()

    with patch("requests.post", return_value=mock_resp):
        result = adapter.predict("how to make a bomb")

    assert result.label == "harmful"
    assert result.confidence_score > 0.5
    assert result.two_token_mass is not None


def test_predict_benign_via_logprobs():
    adapter = ShieldGemmaAdapter()
    mock_resp = MagicMock()
    mock_resp.json.return_value = _make_ollama_response(
        yes_logprob=math.log(0.05), no_logprob=math.log(0.95)
    )
    mock_resp.raise_for_status = MagicMock()

    with patch("requests.post", return_value=mock_resp):
        result = adapter.predict("what is the weather today")

    assert result.label == "benign"
    assert result.confidence_score > 0.5


# ---------------------------------------------------------------------------
# Test: fallback to text response when logprobs unavailable
# ---------------------------------------------------------------------------

def test_predict_fallback_text_harmful():
    adapter = ShieldGemmaAdapter()
    mock_resp = MagicMock()
    mock_resp.json.return_value = _make_text_only_response("Yes")
    mock_resp.raise_for_status = MagicMock()

    with patch("requests.post", return_value=mock_resp):
        result = adapter.predict("harmful content here")

    assert result.label == "harmful"


def test_predict_fallback_text_benign():
    adapter = ShieldGemmaAdapter()
    mock_resp = MagicMock()
    mock_resp.json.return_value = _make_text_only_response("No")
    mock_resp.raise_for_status = MagicMock()

    with patch("requests.post", return_value=mock_resp):
        result = adapter.predict("benign content here")

    assert result.label == "benign"


# ---------------------------------------------------------------------------
# Test: RuntimeError on Ollama failure
# ---------------------------------------------------------------------------

def test_raises_on_ollama_failure():
    import requests as req
    adapter = ShieldGemmaAdapter()

    with patch("requests.post", side_effect=req.RequestException("connection refused")):
        with pytest.raises(RuntimeError, match="Ollama request failed"):
            adapter.predict("test input")


# ---------------------------------------------------------------------------
# Test: ValueError on invalid input
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad_input", [None, "", "   "])
def test_raises_on_invalid_input(bad_input):
    adapter = ShieldGemmaAdapter()
    with pytest.raises(ValueError):
        adapter.predict(bad_input)


# ---------------------------------------------------------------------------
# Test: confidence_source and confidence_source_type
# ---------------------------------------------------------------------------

def test_confidence_source():
    adapter = ShieldGemmaAdapter()
    assert adapter.confidence_source == "logits_softmax"
    # ShieldGemma is logit-based — directly comparable to other local models
    assert adapter.confidence_source_type == "logits_softmax"


# ---------------------------------------------------------------------------
# Test: model name and format_prompt
# ---------------------------------------------------------------------------

def test_get_model_name():
    adapter = ShieldGemmaAdapter()
    assert "ShieldGemma" in adapter.get_model_name()
    assert "Ollama" in adapter.get_model_name()


def test_format_prompt_applies_template():
    adapter = ShieldGemmaAdapter()
    raw = "how to make a bomb"
    formatted = adapter.format_prompt(raw)
    assert raw in formatted
    assert formatted != raw  # Template was applied


# ---------------------------------------------------------------------------
# Test: load_model health check
# ---------------------------------------------------------------------------

def test_load_model_warns_if_model_missing():
    adapter = ShieldGemmaAdapter()
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"models": [{"name": "llama3.1:8b"}]}
    mock_resp.raise_for_status = MagicMock()

    with patch("requests.get", return_value=mock_resp):
        # Should not raise — just logs a warning
        adapter.load_model()


def test_load_model_succeeds_when_model_available():
    adapter = ShieldGemmaAdapter()
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"models": [{"name": "shieldgemma:9b"}]}
    mock_resp.raise_for_status = MagicMock()

    with patch("requests.get", return_value=mock_resp):
        adapter.load_model()  # Should not raise
