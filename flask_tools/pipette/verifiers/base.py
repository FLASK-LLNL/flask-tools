from __future__ import annotations

from abc import ABC, abstractmethod
from enum import StrEnum

from ..constants import FinalGrade, ToolResult, ToolStatus


class Speed(StrEnum):
    FAST = "fast"
    CACHEABLE_SLOW = "cacheable"  # Can be fast or slow. Ideally, SLOW operations should instead be CACHEABLE_SLOW.
    SLOW = "slow"


class ReactionChecker(ABC):
    name: str
    stops_on_fail: bool = False
    speed = Speed.FAST

    @abstractmethod
    def run(self, rxn_smiles: str, context: dict[str, ToolResult]) -> ToolResult:
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
            grade_hint=FinalGrade.IMPOSSIBLE,
            data={"traceback": traceback_text} if traceback_text else {},
            comment=reason,
        )


class CacheableReactionChecker(ReactionChecker):
    speed = Speed.CACHEABLE_SLOW

    def __init__(self, database: str | None = None) -> None:
        self.database = database

    @abstractmethod
    def check_cache(self, rxn_smiles: str) -> None | ToolResult:
        raise NotImplementedError
