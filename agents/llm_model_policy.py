"""Repository-wide model policy for direct DeepSeek API calls."""

from urllib.parse import urlparse

FIXED_DEEPSEEK_API_MODEL = "deepseek-flash"
FIXED_DEEPSEEK_API_MODEL_CHOICES = (FIXED_DEEPSEEK_API_MODEL,)

# Backward-compatible names used by the tuning CLIs.
FIXED_SCREENING_MODEL = FIXED_DEEPSEEK_API_MODEL
FIXED_SCREENING_MODEL_CHOICES = FIXED_DEEPSEEK_API_MODEL_CHOICES


def is_direct_deepseek_api(base_url: str) -> bool:
    """Return whether an OpenAI-compatible URL targets DeepSeek directly."""

    parsed = urlparse(base_url if "://" in base_url else f"https://{base_url}")
    return parsed.hostname == "api.deepseek.com"


def effective_deepseek_api_model(base_url: str, requested_model: str) -> str:
    """Force Flash only for direct DeepSeek, preserving other compatible APIs."""

    if is_direct_deepseek_api(base_url):
        return FIXED_DEEPSEEK_API_MODEL
    return requested_model


def effective_deepseek_thinking_mode(
    base_url: str, requested_mode: str | None
) -> str | None:
    """Reserve direct-Flash completion tokens for JSON unless explicitly set."""

    if is_direct_deepseek_api(base_url) and requested_mode is None:
        return "disabled"
    return requested_mode
