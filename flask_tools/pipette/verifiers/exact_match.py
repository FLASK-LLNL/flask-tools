from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .base import ReactionChecker
from ..constants import FinalGrade, ToolResult, ToolStatus
from ..smiles import canonicalize_reaction_smiles


@dataclass
class DatabaseMatch:
    source: str
    record_id: str
    canonical_reaction_smiles: str
    matched_without_agents: bool = False


class ReactionDatabase(Protocol):
    def find_exact_match(
        self, canonical_reaction_smiles: str
    ) -> DatabaseMatch | None: ...


class ExactMatchChecker(ReactionChecker):
    name = "exact_match"

    def __init__(self, database: ReactionDatabase | None = None) -> None:
        self.database = database

    def run(self, rxn_smiles: str, context: dict[str, ToolResult]) -> ToolResult:
        if self.database is None:
            return ToolResult(
                name=self.name,
                status=ToolStatus.UNKNOWN,
                comment="No reaction database backend is configured.",
            )

        with_agents = canonicalize_reaction_smiles(rxn_smiles, include_agents=True)
        match = self.database.find_exact_match(with_agents)
        if match is not None:
            return ToolResult(
                name=self.name,
                status=ToolStatus.PASS,
                data={
                    "found": {
                        "source": match.source,
                        "record_id": match.record_id,
                        "matched_without_agents": False,
                    }
                },
                comment="Found an exact reaction match in the configured database.",
            )

        without_agents = canonicalize_reaction_smiles(rxn_smiles, include_agents=False)
        match = self.database.find_exact_match(without_agents)
        if match is not None:
            return ToolResult(
                name=self.name,
                status=ToolStatus.POTENTIAL,
                data={
                    "found": {
                        "source": match.source,
                        "record_id": match.record_id,
                        "matched_without_agents": True,
                    }
                },
                comment="Found an exact reaction match after dropping agents.",
            )

        return ToolResult(
            name=self.name,
            status=ToolStatus.UNKNOWN,
            data={"found": None},
            comment="No exact reaction match was found.",
        )
