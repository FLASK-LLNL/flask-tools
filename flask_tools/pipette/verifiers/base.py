from __future__ import annotations

from abc import ABC, abstractmethod
from enum import StrEnum

from ..constants import FinalGrade, ToolResult, ToolStatus, ToolResultsDict


class ReactionChecker(ABC):
    name: str
    stops_on_fail: bool = False

    @abstractmethod
    def run(self, rxn_smiles: str, context: ToolResultsDict) -> ToolResult:
        raise NotImplementedError

    def skipped(self, reason: str) -> ToolResult:
        return ToolResult(
            name=self.name,
            status=ToolStatus.NOT_RUN,
            comment="Tool was not run.",
            skipped_reason=reason,
        )

    def errored(self, reason: str, traceback_text: str | None = None) -> ToolResult:
        return ToolResult(
            name=self.name,
            status=ToolStatus.ERROR,
            data={"traceback": traceback_text} if traceback_text else {},
            comment=reason,
        )


class CacheableReactionChecker(ReactionChecker):
    def __init__(self, database: str | None = None) -> None:
        self.database = database

    @abstractmethod
    def check_cache(self, rxn_smiles: str) -> None | ToolResult:
        # Remember to canonicalize first
        raise NotImplementedError
