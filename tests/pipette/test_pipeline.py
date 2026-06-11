from __future__ import annotations

import pytest

from flask_tools.pipette.verifiers import ReactionEnergyChecker
from flask_tools.pipette.verifiers.base import ReactionChecker, CacheableReactionChecker
from flask_tools.pipette.config import PipetteConfig, TopLevelConfig
from flask_tools.pipette.constants import ToolResult, ToolStatus, ToolResultsDict
from flask_tools.pipette.pipeline import (
    GradingPipeline,
    build_default_pipeline,
    resolve_tool_list,
)
from conftest import RecordingJudge
import helpers


class StubChecker(ReactionChecker):
    """A ReactionChecker that returns result it was initialized with."""

    def __init__(
        self,
        name: str,
        result: ToolResult | Exception,
        *,
        stops_on_fail: bool = False,
    ) -> None:
        self.name = name
        self._result = result
        self.stops_on_fail = stops_on_fail

    def run(self, rxn_smiles: str, context: ToolResultsDict) -> ToolResult:
        if isinstance(self._result, Exception):
            raise self._result
        return self._result


class DFTMockedReactionEnergyChecker(ReactionEnergyChecker):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs, dft_executor=helpers.FakeDFTExecutor())


def _pass_result(name: str) -> ToolResult:
    return ToolResult(name=name, status=ToolStatus.PASS, comment=f"{name} passed")


def test_resolve_tool_list_supports_all_and_explicit_lists() -> None:
    available = ["a", "b", "c"]

    assert resolve_tool_list("all", available) == available
    assert resolve_tool_list(["c", "a"], available) == ["c", "a"]


def test_resolve_tool_list_rejects_unknown_and_duplicate_tools() -> None:
    with pytest.raises(ValueError, match="Unknown tool names"):
        resolve_tool_list(["missing"], ["a", "b"])

    with pytest.raises(ValueError, match="must not contain duplicate"):
        resolve_tool_list(["a", "a"], ["a", "b"])


def test_build_default_pipeline_uses_tool_list_order() -> None:
    # Check that tool call order respects tool_list
    config = PipetteConfig(
        tool_list=["third", "first"], rules=TopLevelConfig(use_dft=False)
    )
    pipeline = build_default_pipeline(
        config=config,
        checker_factories={
            "first": lambda _: StubChecker("first", _pass_result("first")),
            "second": lambda _: StubChecker("second", _pass_result("second")),
            "third": lambda _: StubChecker("third", _pass_result("third")),
        },
    )

    assert [checker.name for checker in pipeline.checkers] == ["third", "first"]
