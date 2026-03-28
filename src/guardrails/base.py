"""Abstract base class and shared mixin for all guardrail adapters."""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Literal

from src.models import PredictionResult

logger = logging.getLogger(__name__)


class GuardrailAdapter(ABC):
    """Uniform interface for all guardrail models.

    Subclasses must implement predict(), format_prompt(), get_model_name(),
    load_model(), unload_model(), and the confidence_source property.
    """

    @abstractmethod
    def predict(self, text: str) -> PredictionResult:
        """Classify text and return prediction with confidence.

        Applies model-specific chat template before inference.

        Raises:
            ValueError: if text is empty, None, or whitespace-only.
        """
        ...

    @abstractmethod
    def format_prompt(self, text: str) -> str:
        """Apply model-specific chat template to raw text.

        Raw text must NEVER be passed directly to instruction-tuned models.
        """
        ...

    @abstractmethod
    def get_model_name(self) -> str:
        """Return display name, e.g. 'LlamaGuard 4 (4-bit)'."""
        ...

    @property
    @abstractmethod
    def confidence_source(self) -> Literal[
        "logits_softmax", "native_safety_score", "category_scores"
    ]:
        """How the confidence score is derived."""
        ...

    @property
    def confidence_source_type(self) -> Literal[
        "logits_softmax", "native_safety_score", "api_score"
    ]:
        """Three-way stratification key for cross-model analysis.

        logits_softmax and native_safety_score are NOT directly comparable
        because they derive confidence from fundamentally different mechanisms.
        category_scores (OpenAI) maps to api_score.
        """
        if self.confidence_source == "category_scores":
            return "api_score"
        return self.confidence_source  # type: ignore[return-value]

    @abstractmethod
    def load_model(self) -> None:
        """Load model into VRAM/memory."""
        ...

    @abstractmethod
    def unload_model(self) -> None:
        """Fully unload model from VRAM/memory."""
        ...

    # ------------------------------------------------------------------
    # Shared input validation
    # ------------------------------------------------------------------

    def _validate_input(self, text: str | None) -> None:
        """Raise ValueError for empty, None, or whitespace-only input."""
        if text is None:
            raise ValueError("Input text must not be None")
        if not isinstance(text, str):
            raise ValueError(f"Input text must be a str, got {type(text)}")
        if not text.strip():
            raise ValueError("Input text must not be empty or whitespace-only")


class LogitBasedAdapterMixin:
    """Mixin for adapters that derive confidence from token logit softmax.

    Subclasses must set safe_token_id and unsafe_token_id at initialization,
    then call verify_token_mapping(tokenizer) to confirm the IDs are correct.
    """

    safe_token_id: int
    unsafe_token_id: int
    # Subclasses may set these to enable strict token verification
    expected_safe_token: str | None = None
    expected_unsafe_token: str | None = None

    def verify_token_mapping(self, tokenizer) -> None:
        """Verify that mapped token IDs decode to expected tokens.

        Must be called at initialization. Raises ValueError if mapping is wrong.
        Catches case-sensitivity issues (e.g., Granite's "Yes" vs "yes").
        """
        safe_decoded = tokenizer.decode([self.safe_token_id]).strip().lower()
        unsafe_decoded = tokenizer.decode([self.unsafe_token_id]).strip().lower()
        logger.info(
            "Token mapping: safe=%d -> '%s', unsafe=%d -> '%s'",
            self.safe_token_id, safe_decoded,
            self.unsafe_token_id, unsafe_decoded,
        )
        if self.expected_safe_token and safe_decoded != self.expected_safe_token.lower():
            raise ValueError(
                f"Safe token ID {self.safe_token_id} decodes to '{safe_decoded}', "
                f"expected '{self.expected_safe_token}'"
            )
        if self.expected_unsafe_token and unsafe_decoded != self.expected_unsafe_token.lower():
            raise ValueError(
                f"Unsafe token ID {self.unsafe_token_id} decodes to '{unsafe_decoded}', "
                f"expected '{self.expected_unsafe_token}'"
            )

    def compute_confidence(self, logits) -> PredictionResult:
        """Apply full-vocabulary softmax, extract 2-token mass, normalize.

        Logs a warning if two_token_mass < 0.5 (normalized confidence is
        artificially inflated when the model prefers non-classification tokens).

        Args:
            logits: 1-D torch.Tensor of raw logits over the full vocabulary.

        Returns:
            PredictionResult with label, confidence_score, two_token_mass,
            and raw_logits.
        """
        import torch

        full_softmax = torch.softmax(logits.float(), dim=-1)
        safe_prob = full_softmax[self.safe_token_id].item()
        unsafe_prob = full_softmax[self.unsafe_token_id].item()
        two_token_mass = safe_prob + unsafe_prob

        if two_token_mass < 0.5:
            logger.warning(
                "2-token mass=%.4f < 0.5 — normalized confidence is artificially inflated "
                "(model prefers non-classification tokens)",
                two_token_mass,
            )

        # Normalize to 2-token distribution; guard against zero mass
        if two_token_mass > 0:
            normalized_unsafe = unsafe_prob / two_token_mass
            normalized_safe = safe_prob / two_token_mass
        else:
            logger.error("Two-token mass is zero — defaulting to 0.5 confidence")
            normalized_unsafe = 0.5
            normalized_safe = 0.5

        label: Literal["harmful", "benign"] = (
            "harmful" if normalized_unsafe > normalized_safe else "benign"
        )
        confidence = max(normalized_unsafe, normalized_safe)

        return PredictionResult(
            label=label,
            confidence_score=confidence,
            two_token_mass=two_token_mass,
            raw_logits={
                "safe_logit": safe_prob,
                "unsafe_logit": unsafe_prob,
            },
        )
