"""Unit tests for OpenAI Moderation adapter retry logic."""
from __future__ import annotations

from unittest.mock import MagicMock, patch
import pytest

from src.guardrails.openai_moderation import OpenAIModerationAdapter


def _make_moderation_response(flagged: bool, max_score: float = 0.9):
    """Build a mock OpenAI moderation response."""
    result = MagicMock()
    result.flagged = flagged
    scores = MagicMock()
    scores.model_dump.return_value = {"hate": max_score, "violence": 0.1}
    result.category_scores = scores
    response = MagicMock()
    response.results = [result]
    return response


# ---------------------------------------------------------------------------
# Test: successful prediction
# ---------------------------------------------------------------------------

def test_predict_harmful():
    adapter = OpenAIModerationAdapter()
    adapter._client = MagicMock()
    adapter._client.moderations.create.return_value = _make_moderation_response(
        flagged=True, max_score=0.95
    )
    result = adapter.predict("how to make a bomb")
    assert result.label == "harmful"
    assert 0.0 <= result.confidence_score <= 1.0


def test_predict_benign():
    adapter = OpenAIModerationAdapter()
    adapter._client = MagicMock()
    adapter._client.moderations.create.return_value = _make_moderation_response(
        flagged=False, max_score=0.05
    )
    result = adapter.predict("what is the weather today")
    assert result.label == "benign"
    assert result.confidence_score >= 0.5


# ---------------------------------------------------------------------------
# Test: retry on API errors
# ---------------------------------------------------------------------------

def test_retries_on_api_error_then_succeeds():
    """Should retry on APIError and succeed on the third attempt."""
    from openai import APIError

    adapter = OpenAIModerationAdapter()
    adapter._client = MagicMock()

    success_response = _make_moderation_response(flagged=False, max_score=0.1)
    adapter._client.moderations.create.side_effect = [
        APIError("error", request=MagicMock(), body=None),
        APIError("error", request=MagicMock(), body=None),
        success_response,
    ]

    with patch("time.sleep"):
        result = adapter.predict("test input")

    assert result.label == "benign"
    assert adapter._client.moderations.create.call_count == 3


def test_raises_after_all_retries_exhausted():
    """Should raise RuntimeError after 3 failed attempts."""
    from openai import APIError

    adapter = OpenAIModerationAdapter()
    adapter._client = MagicMock()
    adapter._client.moderations.create.side_effect = APIError(
        "persistent error", request=MagicMock(), body=None
    )

    with patch("time.sleep"):
        with pytest.raises(RuntimeError, match="3 retries"):
            adapter.predict("test input")

    assert adapter._client.moderations.create.call_count == 3


# ---------------------------------------------------------------------------
# Test: rate limit (429) handling with Retry-After
# ---------------------------------------------------------------------------

def test_respects_retry_after_header():
    """Should wait for Retry-After seconds on 429 RateLimitError."""
    from openai import RateLimitError

    adapter = OpenAIModerationAdapter()
    adapter._client = MagicMock()

    # Build a RateLimitError with a Retry-After header
    mock_response = MagicMock()
    mock_response.headers = {"retry-after": "5"}
    rate_limit_err = RateLimitError(
        "rate limited", response=mock_response, body=None
    )
    success_response = _make_moderation_response(flagged=False, max_score=0.1)
    adapter._client.moderations.create.side_effect = [rate_limit_err, success_response]

    sleep_calls = []
    with patch("time.sleep", side_effect=lambda s: sleep_calls.append(s)):
        result = adapter.predict("test input")

    assert result.label == "benign"
    # Should have slept for the Retry-After value (5s), not the default backoff (1s)
    assert any(s == 5.0 for s in sleep_calls), (
        f"Expected 5.0s sleep, got: {sleep_calls}"
    )


# ---------------------------------------------------------------------------
# Test: ValueError on invalid input
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad_input", [None, "", "   "])
def test_raises_on_invalid_input(bad_input):
    adapter = OpenAIModerationAdapter()
    adapter._client = MagicMock()
    with pytest.raises(ValueError):
        adapter.predict(bad_input)


# ---------------------------------------------------------------------------
# Test: RuntimeError when client not initialized
# ---------------------------------------------------------------------------

def test_raises_when_not_loaded():
    adapter = OpenAIModerationAdapter()
    with pytest.raises(RuntimeError, match="not initialized"):
        adapter.predict("test")


# ---------------------------------------------------------------------------
# Test: confidence_source and confidence_source_type
# ---------------------------------------------------------------------------

def test_confidence_source():
    adapter = OpenAIModerationAdapter()
    assert adapter.confidence_source == "category_scores"
    assert adapter.confidence_source_type == "api_score"
