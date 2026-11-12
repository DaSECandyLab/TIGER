"""最小化的 OpenAI-compatible JSON 对话客户端。"""

from __future__ import annotations

import http.client
import json
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Protocol

from agents.llm_model_policy import (
    effective_deepseek_api_model,
    effective_deepseek_thinking_mode,
)


class JSONChatClient(Protocol):
    """参数筛选器所需的最小客户端接口。"""

    def complete_json(self, *, system: str, user: str) -> dict[str, Any]:
        """返回模型生成的 JSON object。"""


_FENCED_JSON_PATTERN = re.compile(
    r"```(?:json)?\s*(\{.*\})\s*```", re.IGNORECASE | re.DOTALL
)


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    """让临时 3xx 进入重试，避免 POST 被 urllib 自动改写成 GET。"""

    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> None:
        """禁止 urllib 把 3xx POST 自动改写成 GET。"""

        return None


_NO_REDIRECT_OPENER = urllib.request.build_opener(
    urllib.request.ProxyHandler({}),
    _NoRedirectHandler(),
)


def parse_json_object(content: str) -> dict[str, Any]:
    """兼容纯 JSON 和 Markdown fenced JSON。"""

    stripped = content.strip()
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        match = _FENCED_JSON_PATTERN.search(stripped)
        if match is None:
            first = stripped.find("{")
            last = stripped.rfind("}")
            if first < 0 or last <= first:
                raise
            parsed = json.loads(stripped[first : last + 1])
        else:
            parsed = json.loads(match.group(1))
    if not isinstance(parsed, dict):
        raise ValueError("LLM response must be a JSON object")
    return parsed


@dataclass
class OpenAICompatibleClient:
    """通过 `/chat/completions` 调用远程或本地兼容服务。"""

    base_url: str
    model: str
    api_key: str | None = None
    timeout: float = 180.0
    temperature: float = 0.0
    max_tokens: int = 8192
    use_json_response_format: bool = True
    request_retries: int = 2
    thinking_mode: str | None = None
    usage_callback: Callable[[dict[str, Any]], None] | None = None

    def complete_json(self, *, system: str, user: str) -> dict[str, Any]:
        """调用兼容接口，并在网络或 JSON 错误时按策略重试。"""

        url = f"{self.base_url.rstrip('/')}/chat/completions"
        effective_thinking_mode = effective_deepseek_thinking_mode(
            self.base_url, self.thinking_mode
        )
        payload: dict[str, Any] = {
            "model": effective_deepseek_api_model(self.base_url, self.model),
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if self.use_json_response_format:
            payload["response_format"] = {"type": "json_object"}
        if effective_thinking_mode is not None:
            if effective_thinking_mode not in {"enabled", "disabled"}:
                raise ValueError("thinking_mode must be 'enabled', 'disabled', or None")
            payload["thinking"] = {"type": effective_thinking_mode}

        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "SCOOT-vLLM-parameter-screening/1.0",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        last_error: Exception | None = None
        for attempt in range(self.request_retries + 1):
            started = time.monotonic()
            event = {"started_at": datetime.now(timezone.utc).isoformat(),
                     "attempt": attempt + 1, "requested_model": payload["model"],
                     "status": "error", "usage": None}
            try:
                with _NO_REDIRECT_OPENER.open(
                    request, timeout=self.timeout
                ) as response:
                    body = json.loads(response.read().decode("utf-8"))
                event.update(response_id=body.get("id"), response_model=body.get("model"),
                             usage=body.get("usage"))
                choice = body["choices"][0]
                event["finish_reason"] = choice.get("finish_reason")
                message = choice["message"]
                content = message["content"]
                if isinstance(content, list):
                    content = "".join(
                        str(part.get("text", ""))
                        for part in content
                        if isinstance(part, dict)
                    )
                if not str(content).strip():
                    usage = body.get("usage", {})
                    completion_details = usage.get("completion_tokens_details", {})
                    reasoning_content = message.get("reasoning_content") or ""
                    raise ValueError(
                        "empty assistant content "
                        f"(finish_reason={choice.get('finish_reason')!r}, "
                        f"reasoning_chars={len(str(reasoning_content))}, "
                        "reasoning_tokens="
                        f"{completion_details.get('reasoning_tokens')!r})"
                    )
                parsed = parse_json_object(str(content))
                event["status"] = "success"
                return parsed
            except (
                http.client.HTTPException,
                urllib.error.HTTPError,
                urllib.error.URLError,
                ConnectionError,
                TimeoutError,
                KeyError,
                IndexError,
                TypeError,
                ValueError,
                json.JSONDecodeError,
            ) as exc:
                event["error_type"] = type(exc).__name__
                if isinstance(exc, urllib.error.HTTPError) and 300 <= exc.code < 400:
                    location = exc.headers.get("Location")
                    last_error = RuntimeError(
                        f"HTTP {exc.code} redirect to {location!r}"
                    )
                else:
                    last_error = exc
            finally:
                event["duration_s"] = time.monotonic() - started
                event["completed_at"] = datetime.now(timezone.utc).isoformat()
                if self.usage_callback is not None:
                    self.usage_callback(event)
            if attempt >= self.request_retries:
                break
            time.sleep(min(2**attempt, 4))
        assert last_error is not None
        raise RuntimeError(
            f"OpenAI-compatible request failed after "
            f"{self.request_retries + 1} attempts: {last_error}"
        ) from last_error
