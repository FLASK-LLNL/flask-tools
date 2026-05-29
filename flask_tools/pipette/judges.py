from __future__ import annotations

import json
from pathlib import Path

from pipette.config import PipetteConfig
from pipette.constants import (
    FinalGrade,
    LLM_API_KEY_ENV_VARS,
    ReactionGrade,
    ToolResult,
    resolve_llm_api_key,
)
from pipette.llm_query import OpenAI, create_openai_client, query_messages


class LLMJudge:
    def __init__(
        self,
        *,
        url: str,
        model: str,
        api_key: str,
        prompt_path: Path,
        client: OpenAI | None = None,
    ) -> None:
        self.url = url
        self.model = model
        self.api_key = api_key
        self.prompt_path = prompt_path
        self.client = create_openai_client(url=url, api_key=api_key, client=client)

    @classmethod
    def from_config(cls, config: PipetteConfig) -> LLMJudge:
        api_key = resolve_llm_api_key(config.llm_judge.api_key)
        if not api_key:
            raise ValueError(
                "AI mode requires an API key via llm_judge.api_key, "
                + ", ".join(LLM_API_KEY_ENV_VARS)
                + "."
            )
        if OpenAI is None:
            raise RuntimeError("The openai package is required for AI judge mode.")
        return cls(
            url=config.llm_judge.url,
            model=config.llm_judge.model,
            api_key=api_key,
            prompt_path=config.llm_judge.prompt_path,
        )

    def _load_prompt(self) -> str:
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
            {"role": "system", "content": self._load_prompt()},
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
            parsed = json.loads(response_text)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"LLM judge did not return valid JSON: {response_text}"
            ) from exc

        if not isinstance(parsed, dict):
            raise ValueError("LLM judge response JSON must be an object.")

        final_grade_value = parsed.get("final_grade")
        if not isinstance(final_grade_value, str):
            raise ValueError("LLM judge response must include a string final_grade.")

        try:
            final_grade = FinalGrade(final_grade_value)
        except ValueError as exc:
            raise ValueError(
                f"LLM judge returned unsupported final_grade: {final_grade_value!r}"
            ) from exc

        short_reason = parsed.get("short_reason", "ai.llm_judge")
        if not isinstance(short_reason, str):
            raise ValueError("LLM judge response short_reason must be a string.")

        comment = parsed.get("comment", "")
        if not isinstance(comment, str):
            raise ValueError("LLM judge response comment must be a string.")

        return ReactionGrade(
            final_grade=final_grade,
            short_reason=short_reason,
            results=list(results),
            comment=comment,
        )

    def judge(
        self,
        rxn_smiles: str,
        results: list[ToolResult],
    ) -> ReactionGrade:
        response_text = query_messages(
            client=self.client,
            model=self.model,
            messages=self._build_messages(rxn_smiles, results),
        )
        return self._parse_reaction_grade(response_text, results)
