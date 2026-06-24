from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field, is_dataclass, asdict
from enum import Enum, StrEnum
import json
import os
from typing import Any

LLM_API_KEY_ENV_VARS = (
    "FLASK_ORCHESTRATOR_API_KEY",
    "PIPETTE_API_KEY",
    "OPENAI_API_KEY",
)

LLM_BASE_URL_ENV_VARS = (
    "FLASK_ORCHESTRATOR_URL",
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
    else:
        raise ValueError(f"One of {LLM_BASE_URL_ENV_VARS} is required.")


DEFAULT_LLM_BASE_URL = resolve_llm_base_url()


class ToolStatus(str, Enum):
    FAIL = "fail"  # Result indicating a rxn is invalid
    PASS = "pass"  # Result indicating a rxn is definitely valid
    UNKNOWN = "unknown"  # Result was ambiguous
    NOT_RUN = "not_run"  # Tool was not called (skipped because of prior tool)
    ERROR = "error"  # Tool call errored out


class FinalGrade(str, Enum):
    LIKELY = "likely"  # The reaction is valid and the product(s) are a likely outcome
    POSSIBLE = "possible but unlikely"  # The reaction is plausible but other products exist that would have significantly higher likelihood
    IMPOSSIBLE = "impossible"  # The reaction is physically or chemically impossible
    UNCERTAIN = "uncertain"  #  With the given tools and knowledge, it is not possible to ascertain whether this reaction is possible or not. (Use this, e.g., when certain tools cannot run, or the results are ambiguous)


@dataclass
class ToolResult:
    name: str
    status: ToolStatus
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
    short_comment: str
    results: list[ToolResult]
    comment: str = ""

    def __str__(self) -> str:
        lines = [
            f"ReactionGrade(final_grade={self.final_grade.value}, short_comment={self.short_comment})"
        ]
        if self.comment:
            lines.append(f"comment: {self.comment}")

        if not self.results:
            lines.append("tool_results: none")
            return "\n".join(lines)

        lines.append("tool_results:")
        for result in self.results:
            tool_line = f"  - {result.name}: {result.status.value}"
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


Smi = str
CheckerName = str
ToolResultsDict = dict[tuple[Smi, CheckerName], ToolResult]


class Confidence(StrEnum):
    HIGH = "high"
    MED = "medium"
    LOW = "low"


class RxnSide(StrEnum):
    REACTANTS = "reactants"
    PRODUCTS = "products"


@dataclass(frozen=True)
class AllowedUnbalancedMolecule:
    # A molecule that is commonly missing from one side of a rxn. For example, a solvent may be on the left hand side.
    name: str
    smiles: str
    formula: dict[str, int]
    confidence: Confidence


@dataclass
class ReactionMassImbalanceExplanation:
    name: str
    smiles: str
    missing_side: RxnSide
    mass_amu: float
    confidence: Confidence
