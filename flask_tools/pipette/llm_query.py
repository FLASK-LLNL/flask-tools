from __future__ import annotations

from typing import Any

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - exercised through availability checks.
    OpenAI = None


_CHAT_COMPLETIONS_SUFFIX = "/chat/completions"


def _get_field(value: object, field_name: str) -> object:
    if isinstance(value, dict):
        return value.get(field_name)
    return getattr(value, field_name, None)


def _resolve_base_url(url: str | None) -> str | None:
    if not url:
        return None

    trimmed = url.rstrip("/")
    if trimmed.endswith(_CHAT_COMPLETIONS_SUFFIX):
        return trimmed[: -len(_CHAT_COMPLETIONS_SUFFIX)]
    return trimmed


def create_openai_client(
    *,
    api_key: str,
    url: str | None = None,
    client: OpenAI | None = None,
) -> OpenAI:
    if client is not None:
        return client
    if OpenAI is None:
        raise RuntimeError("The openai package is required for LLM queries.")

    base_url = _resolve_base_url(url)
    if base_url:
        return OpenAI(api_key=api_key, base_url=base_url)
    return OpenAI(api_key=api_key)


def extract_message_text(response: object) -> str:
    choices = _get_field(response, "choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("LLM response did not contain any choices.")

    message = _get_field(choices[0], "message")
    if message is None:
        raise ValueError("LLM response choice did not contain a message.")

    content = _get_field(message, "content")
    if isinstance(content, str):
        stripped = content.strip()
        if stripped:
            return stripped

    if isinstance(content, list):
        text_parts: list[str] = []
        for block in content:
            text = _get_field(block, "text")
            if isinstance(text, str) and text.strip():
                text_parts.append(text.strip())
        if text_parts:
            return "\n".join(text_parts)

    raise ValueError("LLM response message content was not text.")


def query_messages(
    *,
    client: OpenAI,
    model: str,
    messages: list[dict[str, str]],
) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=messages,
    )
    return extract_message_text(response)
