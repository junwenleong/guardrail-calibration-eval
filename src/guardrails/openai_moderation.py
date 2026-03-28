"""OpenAI Moderation API adapter.

IMPORTANT: category_scores returned by the OpenAI Moderation API are explicitly
NOT probabilities per OpenAI documentation. ECE and Brier Score computed on
these scores assume probabilistic semantics that the API does not guarantee.
All OpenAI results must be presented in a dedicated section with this caveat,
and must NOT be included in ranking tables alongside logit-based models without
a prominent disclaimer.
"""
from __future__ import annotations

import logging
import time
from typing import Literal

from src.guardrails.base import GuardrailAdapter
from src.models import PredictionResult

logger = logging.getLogger(__name__)

_MAX_RETRIES = 3
_BACKOFF_SECONDS = [1.0, 2.0, 4.0]


class OpenAIModerationAdapter(GuardrailAdapter):
    """OpenAI Moderation API adapter.

    Returns the maximum category_score across all categories as the confidence
    score. category_scores are NOT probabilities — they are model-specific
    scalars that correlate with harmfulness but lack calibration guarantees.

    Per-prediction timestamp_utc is recorded for API drift detection.
    """

    def __init__(self, api_key: str | None = None):
        """
        Args:
            api_key: OpenAI API key. If None, uses OPENAI_API_KEY env var.
        """
        self._api_key = api_key
        self._client = None

    def get_model_name(self) -> str:
        return "OpenAI Moderation API"

    @property
    def confidence_source(self) -> Literal["category_scores"]:
        """category_scores are explicitly NOT probabilities per OpenAI docs."""
        return "category_scores"

    def load_model(self) -> None:
        """Initialize the OpenAI client."""
        from openai import OpenAI
        self._client = OpenAI(api_key=self._api_key)
        logger.info("OpenAI Moderation adapter initialized (no VRAM required)")

    def unload_model(self) -> None:
        self._client = None
        logger.info("OpenAI Moderation adapter unloaded")

    def format_prompt(self, text: str) -> str:
        """No template needed — the API handles formatting internally."""
        return text

    def predict(self, text: str) -> PredictionResult:
        """Call the OpenAI Moderation API with retry logic.

        Retries up to 3 times with exponential backoff (1s, 2s, 4s).
        Respects Retry-After header on 429 responses.

        Raises:
            ValueError: if text is empty, None, or whitespace-only.
            RuntimeError: if all retries are exhausted.
        """
        self._validate_input(text)
        if self._client is None:
            raise RuntimeError("Client not initialized — call load_model() first")

        from openai import APIError, RateLimitError

        last_exc: Exception | None = None
        for attempt, backoff in enumerate(_BACKOFF_SECONDS, start=1):
            try:
                response = self._client.moderations.create(input=text)
                result = response.results[0]

                # Max category score across all categories
                scores = result.category_scores.model_dump()
                max_score = max(scores.values())
                label: Literal["harmful", "benign"] = (
                    "harmful" if result.flagged else "benign"
                )
                # Confidence: max_score if harmful, else 1 - max_score
                confidence = max_score if label == "harmful" else 1.0 - max_score

                return PredictionResult(
                    label=label,
                    confidence_score=float(confidence),
                    two_token_mass=None,   # Not applicable for API
                    raw_logits=None,
                )

            except RateLimitError as e:
                retry_after = _parse_retry_after(e)
                wait = retry_after if retry_after else backoff
                logger.warning(
                    "Rate limit on attempt %d/%d — waiting %.1fs",
                    attempt, _MAX_RETRIES, wait,
                )
                time.sleep(wait)
                last_exc = e

            except APIError as e:
                logger.warning(
                    "API error on attempt %d/%d: %s — retrying in %.1fs",
                    attempt, _MAX_RETRIES, e, backoff,
                )
                time.sleep(backoff)
                last_exc = e

        raise RuntimeError(
            f"OpenAI Moderation API failed after {_MAX_RETRIES} retries"
        ) from last_exc


def _parse_retry_after(exc) -> float | None:
    """Extract Retry-After seconds from a RateLimitError if available."""
    try:
        headers = exc.response.headers
        if "retry-after" in headers:
            return float(headers["retry-after"])
    except Exception:
        pass
    return None
