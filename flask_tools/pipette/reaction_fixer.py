from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
from typing import Any

from pydantic import BaseModel

from .config import PipetteConfig
from .constants import ToolResult, resolve_llm_api_key
from .llm_query import query_task, query_task_async
from .smiles import (
    canonicalize_reaction_smiles,
    canonicalize_smiles,
    split_reaction_smiles,
)


# The [H+] part is needed for caffeine, Cn1cnc2c1c(=O)[nH]c(=O)n2C.CI>>CN1C=NC2=C1C(=O)N(C(=O)N2C)C
SYSTEM_PROMPT = """You repair reaction SMILES for chemistry validation pipelines.

Your job:
- Remove non-participating agents from the middle section.
- Balance the reaction by adding missing byproducts or counter-species to the reactant or product side when needed. Do not ignore [H+]
- Preserve the main transformation instead of inventing a different reaction.
- Return valid SMILES
- Return exactly one JSON object with this schema:
{
  "fixed_reaction_smiles": "...",
  "comment": "brief explanation"
}
- Do not return markdown or any text outside the JSON object.
"""


@dataclass(frozen=True)
class ReactionFix:
    fixed_reaction_smiles: str
    removed_agents: list[str]
    added_reactants: list[str]
    added_products: list[str]
    reasoning_summary: str


class ReactionFixResponse(BaseModel):
    fixed_reaction_smiles: str
    comment: str = ""


class LLMReactionFixer:
    def __init__(
        self,
        *,
        model: str,
        url: str,
        api_key: str,
        client: object | None = None,
    ) -> None:
        self.model = model
        self.api_key = api_key
        self.url = url
        self.client = client

    @classmethod
    def from_config(cls, config: PipetteConfig) -> LLMReactionFixer | None:
        fixer_config = config.llm_reaction_fixer
        if not fixer_config.enabled:
            return None

        api_key = resolve_llm_api_key(fixer_config.api_key)
        if not api_key:
            return None

        return cls(
            model=fixer_config.model,
            url=config.llm_judge.url,
            api_key=api_key,
        )

    @staticmethod
    def _canonical_component_list(side: str) -> list[str]:
        if not side:
            return []
        return [
            canonicalize_smiles(component) for component in side.split(".") if component
        ]

    @staticmethod
    def _multiset_difference(updated: list[str], original: list[str]) -> list[str]:
        diff = Counter(updated) - Counter(original)
        added: list[str] = []
        for component in sorted(diff):
            added.extend([component] * diff[component])
        return added

    def _parse_reaction_fix(
        self, original_reaction_smiles: str, response_text: str
    ) -> ReactionFix:
        try:
            parsed = ReactionFixResponse.model_validate_json(response_text)
        except Exception as exc:
            raise ValueError(
                f"Reaction fixer did not return valid JSON: {response_text}"
            ) from exc

        canonical_fixed = canonicalize_reaction_smiles(
            parsed.fixed_reaction_smiles.strip(), include_agents=True
        )
        original_reactants, original_agents, original_products = split_reaction_smiles(
            canonicalize_reaction_smiles(original_reaction_smiles, include_agents=True)
        )
        fixed_reactants, fixed_agents, fixed_products = split_reaction_smiles(
            canonical_fixed
        )

        return ReactionFix(
            fixed_reaction_smiles=canonical_fixed,
            removed_agents=self._multiset_difference(
                self._canonical_component_list(original_agents),
                self._canonical_component_list(fixed_agents),
            ),
            added_reactants=self._multiset_difference(
                self._canonical_component_list(fixed_reactants),
                self._canonical_component_list(original_reactants),
            ),
            added_products=self._multiset_difference(
                self._canonical_component_list(fixed_products),
                self._canonical_component_list(original_products),
            ),
            reasoning_summary=parsed.comment,
        )

    @staticmethod
    def _build_user_payload(
        rxn_smiles: str,
        results: list[ToolResult],
    ) -> dict[str, Any]:
        serialized_results = [r.to_json_dict() for r in results]
        for s in serialized_results:
            del s["skipped_reason"]
        return {
            "reaction_smiles": rxn_smiles,
            "tool_results": serialized_results,
            "instructions": (
                "Return a repaired reaction SMILES that removes agents and adds "
                "missing byproducts or counter-species on either side when needed. "
                "If no credible repair is possible, return the original reaction_smiles."
            ),
        }

    def fix(
        self,
        rxn_smiles: str,
        results: list[ToolResult],
        *,
        experiment: object | None = None,
    ) -> ReactionFix:
        user_prompt = json.dumps(
            self._build_user_payload(rxn_smiles, results),
            indent=2,
            sort_keys=True,
        )
        if self.client is not None:
            from .llm_query import query_messages

            response_text = query_messages(
                client=self.client,
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
            )
        else:
            response_text = query_task(
                system_prompt=SYSTEM_PROMPT,
                user_prompt=user_prompt,
                model=self.model,
                api_key=self.api_key,
                url=self.url,
                structured_output_schema=ReactionFixResponse,
                agent_name="PipetteFixer",
            )
        try:
            return self._parse_reaction_fix(rxn_smiles, response_text)
        except Exception as exc:
            raise ValueError(
                f"Reaction fixer failed to parse response: {exc} for {rxn_smiles} with response {response_text}"
            ) from exc

    async def fix_async(
        self,
        rxn_smiles: str,
        results: list[ToolResult],
        *,
        experiment: object | None = None,
    ) -> ReactionFix:
        user_prompt = json.dumps(
            self._build_user_payload(rxn_smiles, results),
            indent=2,
            sort_keys=True,
        )
        if self.client is not None:
            from .llm_query import query_messages

            response_text = query_messages(
                client=self.client,
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
            )
        else:
            response_text = await query_task_async(
                system_prompt=SYSTEM_PROMPT,
                user_prompt=user_prompt,
                model=self.model,
                api_key=self.api_key,
                url=self.url,
                structured_output_schema=ReactionFixResponse,
                agent_name="PipetteFixer",
            )
        try:
            return self._parse_reaction_fix(rxn_smiles, response_text)
        except Exception as exc:
            raise ValueError(
                f"Reaction fixer failed to parse response: {exc} for {rxn_smiles} with response {response_text}"
            ) from exc
