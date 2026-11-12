"""Structured inference transport for local experience summarization."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any, Mapping, Protocol

import httpx

from agents.llm_model_policy import (
    effective_deepseek_api_model,
    effective_deepseek_thinking_mode,
)


class StructuredInferenceClient(Protocol):
    """The only model capability required by Observation discovery/refinement."""

    def complete_structured(
        self,
        *,
        system: str,
        user: str,
        response_schema: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Return a receipt containing a JSON-object ``structured_output``."""


@dataclass(frozen=True)
class OpenAICompatibleStructuredClient:
    """Call an OpenAI-compatible JSON endpoint directly, without memory tools."""

    base_url: str
    model: str
    api_key: str | None = None
    timeout_seconds: float = 300.0
    temperature: float = 0.0
    max_tokens: int = 8192
    request_retries: int = 2
    thinking_mode: str | None = None

    @property
    def transport_name(self) -> str:
        return "direct_openai_compatible_json_v1"

    def checkpoint_identity(self) -> dict[str, Any]:
        """Non-secret identity used to prevent cross-transport checkpoint reuse."""

        return {
            "transport": self.transport_name,
            "base_url": self.base_url,
            "model": effective_deepseek_api_model(self.base_url, self.model),
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "thinking_mode": effective_deepseek_thinking_mode(
                self.base_url, self.thinking_mode
            ),
        }

    def complete_structured(
        self,
        *,
        system: str,
        user: str,
        response_schema: Mapping[str, Any],
    ) -> dict[str, Any]:
        if self.temperature != 0.0:
            raise ValueError(
                "Observation structured inference requires temperature=0.0"
            )
        effective_model = effective_deepseek_api_model(self.base_url, self.model)
        effective_thinking = effective_deepseek_thinking_mode(
            self.base_url, self.thinking_mode
        )
        schema_text = json.dumps(
            dict(response_schema),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        payload: dict[str, Any] = {
            "model": effective_model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        system
                        + "\nReturn exactly one JSON object matching JSON_SCHEMA. "
                        "Include every required field, obey enums and additionalProperties=false, "
                        "and emit no prose or Markdown fences.\nJSON_SCHEMA="
                        + schema_text
                    ),
                },
                {"role": "user", "content": user},
            ],
            "temperature": 0.0,
            "max_tokens": self.max_tokens,
            "response_format": {"type": "json_object"},
        }
        if effective_thinking is not None:
            if effective_thinking not in {"enabled", "disabled"}:
                raise ValueError("thinking_mode must be enabled, disabled, or None")
            payload["thinking"] = {"type": effective_thinking}
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": "SemBench-experience-summarizer/1.0",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        last_error: Exception | None = None
        for attempt in range(self.request_retries + 1):
            try:
                with httpx.Client(
                    timeout=self.timeout_seconds,
                    follow_redirects=False,
                    trust_env=False,
                ) as client:
                    response = client.post(
                        f"{self.base_url.rstrip('/')}/chat/completions",
                        headers=headers,
                        json=payload,
                    )
                if 300 <= response.status_code < 400:
                    raise RuntimeError(
                        f"HTTP {response.status_code} redirect to "
                        f"{response.headers.get('location')!r}"
                    )
                response.raise_for_status()
                body = response.json()
                choice = body["choices"][0]
                content = choice["message"].get("content")
                if isinstance(content, list):
                    content = "".join(
                        str(part.get("text", ""))
                        for part in content
                        if isinstance(part, Mapping)
                    )
                structured = json.loads(str(content or ""))
                if not isinstance(structured, Mapping):
                    raise TypeError("direct structured response is not a JSON object")
                return {
                    "structured_output": dict(structured),
                    "usage": body.get("usage"),
                    "model": body.get("model") or effective_model,
                    "finish_reason": choice.get("finish_reason"),
                    "request_id": body.get("id"),
                    "transport": self.transport_name,
                }
            except (
                httpx.HTTPError,
                RuntimeError,
                KeyError,
                IndexError,
                TypeError,
                ValueError,
                json.JSONDecodeError,
            ) as exc:
                last_error = exc
                if attempt < self.request_retries:
                    time.sleep(min(2**attempt, 4))
        assert last_error is not None
        raise RuntimeError(
            "direct structured request failed after "
            f"{self.request_retries + 1} attempts: {last_error}"
        ) from last_error


def default_observation_structured_inference_client(
    *,
    environ: Mapping[str, str] | None = None,
) -> OpenAICompatibleStructuredClient:
    """Recreate the direct structured transport used by the 2026-08-26 v10 run."""

    values = os.environ if environ is None else environ
    api_key = values.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError(
            "direct local summarization requires DEEPSEEK_API_KEY"
        )
    return OpenAICompatibleStructuredClient(
        base_url=values.get(
            "DEEPSEEK_BASE_URL", "https://api.deepseek.com"
        ),
        model=values.get("DEEPSEEK_MODEL", "deepseek-flash"),
        api_key=api_key,
        timeout_seconds=600.0,
        temperature=0.0,
        max_tokens=8192,
        request_retries=2,
        thinking_mode="disabled",
    )
