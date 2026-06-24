###############################################################################
## Copyright 2025-2026 Lawrence Livermore National Security, LLC.
## See the top-level LICENSE file for details.
##
## SPDX-License-Identifier: Apache-2.0
###############################################################################

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .base import ReactionChecker
from ..constants import (
    FinalGrade,
    ToolResult,
    ToolStatus,
    ToolResultsDict,
    ToolResultDetails,
)
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


class ExactResultDetails(ToolResultDetails):
    source: str  # Name of the database
    record_id: str | int | None  # Id within the database
    matched_without_agents: bool  # True if only matched after removing agents. False if there were no agents to begin with


class ExactMatchChecker(ReactionChecker):
    """Check if a rxn exists in the database. For rxns with reagents separated (A>C>B), will check without
    the reagent if it cannot find a match with them.
    ToolResult example:
        ToolResult(
            name="exact_match",
            status=ToolStatus.PASS,
            data=ExactResultDetails(
                source = match.source,
                record_id = match.record_id,
                matched_without_agents = False,
            )
            comment="Found an exact reaction match in the configured database.",
        )
    """

    name = "exact_match"

    def __init__(self, database: ReactionDatabase | None = None) -> None:
        self.database = database

    def run(self, rxn_smiles: str, context: ToolResultsDict) -> ToolResult:
        if self.database is None:
            return ToolResult(
                name=self.name,
                status=ToolStatus.NOT_RUN,
                data=None,
                comment="No reaction database backend is configured.",
            )

        with_agents = canonicalize_reaction_smiles(rxn_smiles, include_agents=True)
        match = self.database.find_exact_match(with_agents)
        if match is not None:
            return ToolResult(
                name=self.name,
                status=ToolStatus.PASS,
                data=ExactResultDetails(
                    source=match.source,
                    record_id=match.record_id,
                    matched_without_agents=False,
                ),
                comment="Found an exact reaction match in the configured database.",
            )

        without_agents = canonicalize_reaction_smiles(rxn_smiles, include_agents=False)
        match = self.database.find_exact_match(without_agents)
        if match is not None:
            return ToolResult(
                name=self.name,
                status=ToolStatus.PASS,
                data=ExactResultDetails(
                    source=match.source,
                    record_id=match.record_id,
                    matched_without_agents=True,
                ),
                comment="Found an exact reaction match after dropping agents.",
            )

        return ToolResult(
            name=self.name,
            status=ToolStatus.UNKNOWN,
            data=None,
            comment="No exact reaction match was found.",
        )
