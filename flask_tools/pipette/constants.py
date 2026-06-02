from __future__ import annotations

from dataclasses import dataclass, field, is_dataclass, asdict
from enum import Enum
import json
import os
from typing import Any

DEFAULT_LLM_BASE_URL = "https://livai-api.llnl.gov/v1/"

LLM_API_KEY_ENV_VARS = (
    "API_KEY",
    "JUDGE_API_KEY",
    "PIPETTE_API_KEY",
    "PIPETTE_LLM_API_KEY",
    "OPENAI_API_KEY",
)

LLM_BASE_URL_ENV_VARS = (
    "LLM_BASE_URL",
    "PIPETTE_LLM_BASE_URL",
)


def resolve_llm_api_key(explicit_api_key: str | None = None) -> str | None:
    if explicit_api_key:
        return explicit_api_key
    for env_var in LLM_API_KEY_ENV_VARS:
        api_key = os.environ.get(env_var)
        if api_key:
            return api_key
    return None


def resolve_llm_base_url(explicit_base_url: str | None = None) -> str:
    if explicit_base_url:
        return explicit_base_url
    for env_var in LLM_BASE_URL_ENV_VARS:
        base_url = os.environ.get(env_var)
        if base_url:
            return base_url
    return DEFAULT_LLM_BASE_URL


class ToolStatus(str, Enum):
    FAIL = "fail"
    POTENTIAL = "potential"
    PASS = "pass"
    UNKNOWN = "unknown"
    NOT_RUN = "not_run"
    ERROR = "error"


class FinalGrade(str, Enum):
    LIKELY = "likely"
    POSSIBLE = "possible"
    UNCERTAIN = "uncertain"
    UNLIKELY = "unlikely"
    IMPOSSIBLE = "impossible"


@dataclass
class ToolResult:
    name: str
    status: ToolStatus
    grade_hint: FinalGrade | None = None  # Kind of a confidence level
    data: dict[str, Any] = field(default_factory=dict)
    comment: str = ""
    skipped_reason: str | None = (
        None  # If a priority checker skipped this tool, like in an exact rule pipeline
    )

    def to_json_dict(self) -> dict:
        return {
            "name": self.name,
            "status": self.status.value,
            "data": self.data,
            "comment": self.comment,
            "skipped_reason": self.skipped_reason,
        }


@dataclass
class ReactionGrade:
    final_grade: FinalGrade
    short_reason: str
    results: list[ToolResult]
    comment: str = ""

    def __str__(self) -> str:
        lines = [
            f"ReactionGrade(final_grade={self.final_grade.value}, short_reason={self.short_reason})"
        ]
        if self.comment:
            lines.append(f"comment: {self.comment}")

        if not self.results:
            lines.append("tool_results: none")
            return "\n".join(lines)

        lines.append("tool_results:")
        for result in self.results:
            tool_line = f"  - {result.name}: {result.status.value}"
            if result.grade_hint is not None:
                tool_line += f" ({result.grade_hint.value})"
            if result.comment:
                tool_line += f" - {result.comment}"
            if result.skipped_reason:
                tool_line += f" [skipped: {result.skipped_reason}]"
            lines.append(tool_line)
            if result.data:
                for data_line in json.dumps(
                    result.data,
                    indent=2,
                    default=lambda o: (
                        asdict(o)
                        if is_dataclass(o)
                        else TypeError(f"not serializable: {o}")
                    ),
                ).splitlines():
                    lines.append(f"    {data_line}")
        return "\n".join(lines)


@dataclass(frozen=True)
class MissingProductRule:
    name: str
    smiles: str
    formula: dict[str, int]
    confidence: str
