"""Helpers for controlling FlockMTL SQL-side CacheBlend markers."""

from __future__ import annotations

import os
import re


CACHEBLEND_MODE_ENV = "SEMBENCH_CACHEBLEND_MODE"
DEFAULT_BLEND_SPECIAL_STR = " # # "


def normalize_cacheblend_mode(value: str | None) -> str:
    if value is None:
        return "auto"
    normalized = value.strip().lower()
    aliases = {
        "": "auto",
        "default": "auto",
        "true": "on",
        "1": "on",
        "yes": "on",
        "false": "off",
        "0": "off",
        "no": "off",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in {"auto", "on", "off"}:
        raise ValueError(
            f"Invalid {CACHEBLEND_MODE_ENV}={value!r}; expected auto, on, or off"
        )
    return normalized


def current_cacheblend_mode() -> str:
    return normalize_cacheblend_mode(os.environ.get(CACHEBLEND_MODE_ENV))


def remove_cacheblend_fields(sql_text: str) -> str:
    updated = re.sub(
        r"^[ \t]*'cacheblend'\s*:\s*(?:true|false)\s*,\s*\n?",
        "",
        sql_text,
        flags=re.IGNORECASE | re.MULTILINE,
    )
    updated = re.sub(
        r"^[ \t]*'blend_special_str'\s*:\s*'[^']*'\s*,\s*\n?",
        "",
        updated,
        flags=re.IGNORECASE | re.MULTILINE,
    )
    return updated


def _ensure_prompt_has_blend_markers(prompt_text: str, blend_special_str: str) -> str:
    if blend_special_str in prompt_text:
        return prompt_text
    return re.sub(
        r"(\{\{[^}]+\}\})",
        lambda match: f"{blend_special_str}{match.group(1)}",
        prompt_text,
    )


def enable_cacheblend_fields(
    sql_text: str,
    blend_special_str: str = DEFAULT_BLEND_SPECIAL_STR,
) -> str:
    updated = remove_cacheblend_fields(sql_text)
    if "context_columns" not in updated:
        raise ValueError("cacheblend rewrite requires SQL to define context_columns")

    prompt_match = re.search(
        r"('prompt'\s*:\s*')(?P<prompt>(?:''|[^'])*)(')",
        updated,
        flags=re.DOTALL,
    )
    if prompt_match is not None:
        prompt_text = prompt_match.group("prompt")
        updated = (
            updated[: prompt_match.start("prompt")]
            + _ensure_prompt_has_blend_markers(prompt_text, blend_special_str)
            + updated[prompt_match.end("prompt") :]
        )

    insertion = (
        f"        'cacheblend': true,\n"
        f"        'blend_special_str': '{blend_special_str}',\n"
    )
    return re.sub(
        r"('prompt'\s*:\s*'(?:''|[^'])*'\s*,)",
        lambda match: match.group(1) + "\n" + insertion.rstrip("\n"),
        updated,
        count=1,
        flags=re.MULTILINE | re.DOTALL,
    )


def apply_cacheblend_mode_to_sql(sql_text: str, mode: str | None = None) -> str:
    resolved_mode = (
        normalize_cacheblend_mode(mode)
        if mode is not None
        else current_cacheblend_mode()
    )
    if resolved_mode == "auto":
        return sql_text
    if resolved_mode == "off":
        return remove_cacheblend_fields(sql_text)
    return enable_cacheblend_fields(sql_text)
