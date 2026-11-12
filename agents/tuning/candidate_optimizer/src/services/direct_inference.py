"""Direct structured inference for candidate repair and local reflection."""

from __future__ import annotations

import os
from typing import Any, Mapping, Protocol

from agents.llm_model_policy import effective_deepseek_api_model
from agents.tuning.experience_summarizer.src.services.direct_inference import (
    OpenAICompatibleStructuredClient,
)


DIRECT_DECISION_PROVIDER = "direct_openai_compatible_json_v1"
CANDIDATE_DECISION_TIMEOUT_SECONDS = 120.0
CANDIDATE_DECISION_REQUEST_RETRIES = 1


class StructuredInferenceClient(Protocol):
    def complete_structured(
        self,
        *,
        system: str,
        user: str,
        response_schema: Mapping[str, Any],
    ) -> dict[str, Any]: ...


def default_candidate_structured_inference_client(
    *,
    environ: Mapping[str, str] | None = None,
) -> OpenAICompatibleStructuredClient:
    """Return the bounded direct DeepSeek client without a Reflect tool loop.

    Candidate decisions are latency-sensitive and fail open at their call sites,
    so they must not inherit the Observation summarizer's long, batch-oriented
    transport timeout.
    """

    values = os.environ if environ is None else environ
    api_key = values.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError(
            "direct Candidate decisions require DEEPSEEK_API_KEY"
        )
    base_url = values.get(
        "DEEPSEEK_BASE_URL", "https://api.deepseek.com"
    )
    model = effective_deepseek_api_model(
        base_url,
        values.get("DEEPSEEK_MODEL", "deepseek-flash"),
    )
    return OpenAICompatibleStructuredClient(
        base_url=base_url,
        model=model,
        api_key=api_key,
        timeout_seconds=CANDIDATE_DECISION_TIMEOUT_SECONDS,
        temperature=0.0,
        max_tokens=8192,
        request_retries=CANDIDATE_DECISION_REQUEST_RETRIES,
        thinking_mode="disabled",
    )


__all__ = [
    "DIRECT_DECISION_PROVIDER",
    "CANDIDATE_DECISION_REQUEST_RETRIES",
    "CANDIDATE_DECISION_TIMEOUT_SECONDS",
    "StructuredInferenceClient",
    "default_candidate_structured_inference_client",
]
