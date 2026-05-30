from __future__ import annotations

import asyncio
import hashlib
import threading
from typing import TYPE_CHECKING, Any, Awaitable

from charge.clients.autogen import AutoGenBackend
from charge.tasks.task import Task
from pydantic import BaseModel

if TYPE_CHECKING:
    from charge.clients.autogen import AutoGenAgent


_CHAT_COMPLETIONS_SUFFIX = "/chat/completions"
_BACKEND_REGISTRY_LOCK = threading.Lock()
_BACKEND_CACHE: dict[str, AutoGenBackend] = {}


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


def _backend_alias(*, model: str, api_key: str, url: str | None) -> str:
    digest = hashlib.sha256(
        f"{model}\0{api_key}\0{_resolve_base_url(url) or ''}".encode("utf-8")
    ).hexdigest()
    return f"pipette_llm_{digest[:16]}"


def get_autogen_backend(
    *,
    model: str,
    api_key: str,
    url: str | None = None,
) -> AutoGenBackend:
    alias = _backend_alias(model=model, api_key=api_key, url=url)
    with _BACKEND_REGISTRY_LOCK:
        if alias not in _BACKEND_CACHE:
            _BACKEND_CACHE[alias] = AutoGenBackend(
                model=model,
                backend="openai",
                api_key=api_key,
                base_url=_resolve_base_url(url),
            )
    return _BACKEND_CACHE[alias]


async def query_task_async(
    *,
    system_prompt: str,
    user_prompt: str,
    model: str,
    api_key: str,
    url: str | None = None,
    structured_output_schema: type[BaseModel] | None = None,
    agent_name: str = "Pipette",
    max_retries: int = 1,
    max_tool_calls: int = 1,
) -> str:
    task = Task(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        structured_output_schema=structured_output_schema,
    )
    backend = get_autogen_backend(
        model=model,
        api_key=api_key,
        url=url,
    )
    agent: AutoGenAgent = backend.create_agent(
        task=task,
        agent_name=agent_name,
        max_retries=max_retries,
        max_tool_calls=max_tool_calls,
    )
    result = await agent.run()
    return str(result)


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
    client: Any,
    model: str,
    messages: list[dict[str, str]],
) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=messages,
    )
    return extract_message_text(response)


def _run_coroutine_sync(coro: Awaitable[str]) -> str:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    result: dict[str, str] = {}
    error: dict[str, BaseException] = {}

    def runner() -> None:
        try:
            result["value"] = asyncio.run(coro)
        except BaseException as exc:  # pragma: no cover - only hit inside active loop.
            error["value"] = exc

    thread = threading.Thread(target=runner, daemon=True)
    thread.start()
    thread.join()

    if "value" in error:
        raise error["value"]
    return result["value"]


def query_task(
    *,
    system_prompt: str,
    user_prompt: str,
    model: str,
    api_key: str,
    url: str | None = None,
    structured_output_schema: type[BaseModel] | None = None,
    agent_name: str = "Pipette",
    max_retries: int = 1,
    max_tool_calls: int = 1,
) -> str:
    return _run_coroutine_sync(
        query_task_async(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=model,
            api_key=api_key,
            url=url,
            structured_output_schema=structured_output_schema,
            agent_name=agent_name,
            max_retries=max_retries,
            max_tool_calls=max_tool_calls,
        )
    )
