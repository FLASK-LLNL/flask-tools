from __future__ import annotations

import json
from pathlib import Path
from typing import Literal, TypeVar

from pydantic import BaseModel

from .config import PipetteConfig
from .constants import (
    FinalGrade,
    LLM_API_KEY_ENV_VARS,
    ReactionGrade,
    ToolResult,
    resolve_llm_api_key,
)
from .llm_query import query_task, query_task_async


class ReactionGradeResponse(BaseModel):
    final_grade: Literal["likely", "possible", "uncertain", "unlikely", "impossible"]
    short_reason: str = "ai.llm_judge"
    comment: str = ""


_JudgeT = TypeVar("_JudgeT", bound="BaseLLMJudge")


class BaseLLMJudge:
    def __init__(
        self,
        *,
        url: str,
        model: str,
        api_key: str,
        prompt_path: Path,
        prompt: str | None = None,
    ) -> None:
        self.url = url
        self.model = model
        self.api_key = api_key
        self.prompt_path = prompt_path
        self._prompt = prompt

    @classmethod
    def from_config(cls: type[_JudgeT], config: PipetteConfig) -> _JudgeT:
        api_key = resolve_llm_api_key(config.llm_judge.api_key)
        if not api_key:
            raise ValueError(
                "LLM Judge requires an API key via llm_judge.api_key, "
                + ", ".join(LLM_API_KEY_ENV_VARS)
                + "."
            )
        return cls(
            url=config.llm_judge.url,
            model=config.llm_judge.model,
            api_key=api_key,
            prompt_path=config.llm_judge.prompt_path,
            prompt=config.llm_judge.prompt,
        )

    @property
    def prompt(self) -> str:
        if self._prompt is not None:
            return self._prompt
        return self.prompt_path.read_text(encoding="utf-8")

    def _serialize_results(self, results: list[ToolResult]) -> list[dict[str, object]]:
        serialized: list[dict[str, object]] = [r.to_json_dict() for r in results]
        for s in serialized:
            del s["skipped_reason"]
        return serialized

    def _build_messages(
        self, rxn_smiles: str, results: list[ToolResult]
    ) -> list[dict[str, str]]:
        user_payload = {
            "reaction_smiles": rxn_smiles,
            "tool_results": self._serialize_results(results),
        }
        return [
            {"role": "system", "content": self.prompt},
            {
                "role": "user",
                "content": json.dumps(
                    user_payload, indent=2, sort_keys=True, default=str
                ),
            },
        ]

    def _parse_reaction_grade(
        self,
        response_text: str,
        results: list[ToolResult],
    ) -> ReactionGrade:
        try:
            parsed = ReactionGradeResponse.model_validate_json(response_text)
        except Exception as exc:
            raise ValueError(
                f"LLM judge did not return valid JSON: {response_text}"
            ) from exc

        try:
            final_grade = FinalGrade(parsed.final_grade)
        except ValueError as exc:
            raise ValueError(
                f"LLM judge returned unsupported final_grade: {parsed.final_grade!r}"
            ) from exc

        return ReactionGrade(
            final_grade=final_grade,
            short_reason=parsed.short_reason,
            results=list(results),
            comment=parsed.comment,
        )


class AsyncLLMJudge(BaseLLMJudge):
    async def judge(
        self,
        rxn_smiles: str,
        results: list[ToolResult],
    ) -> ReactionGrade:
        messages = self._build_messages(rxn_smiles, results)
        response_text = await query_task_async(
            system_prompt=messages[0]["content"],
            user_prompt=messages[1]["content"],
            model=self.model,
            api_key=self.api_key,
            url=self.url,
            structured_output_schema=ReactionGradeResponse,
            agent_name="PipetteJudge",
        )
        return self._parse_reaction_grade(response_text, results)


class LLMJudge(BaseLLMJudge):
    def judge(
        self,
        rxn_smiles: str,
        results: list[ToolResult],
    ) -> ReactionGrade:
        messages = self._build_messages(rxn_smiles, results)
        response_text = query_task(
            system_prompt=messages[0]["content"],
            user_prompt=messages[1]["content"],
            model=self.model,
            api_key=self.api_key,
            url=self.url,
            structured_output_schema=ReactionGradeResponse,
            agent_name="PipetteJudge",
        )
        return self._parse_reaction_grade(response_text, results)
