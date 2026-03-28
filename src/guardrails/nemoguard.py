"""NemoGuard-8B adapter — 4-bit quantized via bitsandbytes.

Note: This is NVIDIA's Llama-3.1-8B-Instruct-NemoGuard classifier model,
distinct from the NeMo Guardrails framework.
"""
from __future__ import annotations

import gc
import logging
from typing import Literal

from src.guardrails.base import GuardrailAdapter, LogitBasedAdapterMixin
from src.models import PredictionResult

logger = logging.getLogger(__name__)

MODEL_ID = "nvidia/Llama-3.1-8B-Instruct-NemoGuard"


class NemoGuardAdapter(LogitBasedAdapterMixin, GuardrailAdapter):
    """NemoGuard-8B (4-bit quantized) guardrail adapter.

    Uses Llama-3.1 chat template. Confidence derived from logit softmax.
    """

    expected_safe_token = "safe"
    expected_unsafe_token = "unsafe"

    def __init__(self):
        self._model = None
        self._tokenizer = None
        self.safe_token_id: int = -1
        self.unsafe_token_id: int = -1

    def get_model_name(self) -> str:
        return "NemoGuard-8B (4-bit)"

    @property
    def confidence_source(self) -> Literal["logits_softmax"]:
        return "logits_softmax"

    def load_model(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        logger.info("Loading %s in 4-bit quantization…", MODEL_ID)
        bnb_config = BitsAndBytesConfig(load_in_4bit=True)
        self._tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        self._model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            quantization_config=bnb_config,
            device_map="auto",
        )
        self._model.eval()

        self.safe_token_id = self._tokenizer.convert_tokens_to_ids("safe")
        self.unsafe_token_id = self._tokenizer.convert_tokens_to_ids("unsafe")
        self.verify_token_mapping(self._tokenizer)

        if torch.cuda.is_available():
            vram_mb = torch.cuda.memory_allocated() / 1024 ** 2
            logger.info("VRAM after loading %s: %.1f MB", self.get_model_name(), vram_mb)
        logger.info("Loaded %s | quant=4bit", self.get_model_name())

    def unload_model(self) -> None:
        import torch
        self._model = None
        self._tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Unloaded %s", self.get_model_name())

    def format_prompt(self, text: str) -> str:
        """Apply Llama-3.1 chat template."""
        if self._tokenizer is None:
            raise RuntimeError("Model not loaded — call load_model() first")
        messages = [{"role": "user", "content": text}]
        return self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def predict(self, text: str) -> PredictionResult:
        self._validate_input(text)
        if self._model is None or self._tokenizer is None:
            raise RuntimeError("Model not loaded — call load_model() first")

        import torch

        prompt = self.format_prompt(text)
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
        with torch.no_grad():
            outputs = self._model(**inputs)
        logits = outputs.logits[0, -1, :]
        return self.compute_confidence(logits)
