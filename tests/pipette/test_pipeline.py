###############################################################################
## Copyright 2025-2026 Lawrence Livermore National Security, LLC.
## See the top-level LICENSE file for details.
##
## SPDX-License-Identifier: Apache-2.0
###############################################################################

from __future__ import annotations

import pytest

from flask_tools.pipette.verifiers import ReactionEnergyChecker
from flask_tools.pipette.verifiers.base import ReactionChecker, CacheableReactionChecker
from flask_tools.pipette.config import PipetteConfig, PipelineConfig
from flask_tools.pipette.constants import ToolResult, ToolStatus, ToolResultsDict
from flask_tools.pipette.pipeline import (
    GradingPipeline,
    build_default_pipeline,
    resolve_tool_list,
)
from flask_tools.pipette.reaction_fixer import ReactionFixResultDetails
from conftest import RecordingJudge, RecordingReactionFixer
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
    return ToolResult(
        name=name, data=None, status=ToolStatus.PASS, comment=f"{name} passed"
    )


def test_resolve_tool_list_supports_all_and_explicit_lists() -> None:
    available = ["a", "b", "c"]

    assert resolve_tool_list("all", available) == available
    assert resolve_tool_list(None, available) == []
    assert resolve_tool_list(["c", "a"], available) == ["c", "a"]


def test_resolve_tool_list_rejects_unknown_and_duplicate_tools() -> None:
    with pytest.raises(ValueError, match="Unknown tool names"):
        resolve_tool_list(["missing"], ["a", "b"])

    with pytest.raises(ValueError, match="must not contain duplicate"):
        resolve_tool_list(["a", "a"], ["a", "b"])


def test_build_default_pipeline_uses_tool_list_order() -> None:
    # Check that tool call order respects tool_list
    config = PipetteConfig(
        tool_list=["third", "first"], settings=PipelineConfig(use_dft=False)
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


@pytest.mark.parametrize(
    "use_tools, use_fixing",
    [[True, False], [True, True], [False, False], [False, True]],
)
def test_pipeline_with_or_without_tools_and_fixing(
    use_tools: bool, use_fixing: bool
) -> None:
    original = "CCO>>C=C"
    fixed = "CCO.O>>CC=O"

    smiles_validation = StubChecker(
        "basic_smiles_validation", _pass_result("basic_smiles_validation")
    )
    exact = StubChecker(
        "exact_match",
        ToolResult(
            name="exact_match",
            status=ToolStatus.UNKNOWN,
            data=None,
            comment="No exact match.",
        ),
    )
    charge = StubChecker("charge_conservation", _pass_result("charge_conservation"))
    mass = StubChecker("mass_conservation", _pass_result("mass_conservation"))

    class FailingFixer:
        def fix(self, rxn_smiles: str, results: list[ToolResult]) -> None:
            raise AssertionError("Fixer should not be called when fixing is disabled.")

    good_fixer = RecordingReactionFixer(
        ReactionFixResultDetails(
            original_reaction_smiles=original,
            fixed_reaction_smiles=fixed,
            removed_agents=[],
            added_reactants=["O"],
            removed_products=[],
            added_products=[],
            reasoning_summary="Added missing water reactant.",
        )
    )

    pipeline = GradingPipeline(
        checkers=[smiles_validation, exact, charge, mass] if use_tools else [],
        config=PipetteConfig(
            mode="exact",
            # the tool_list is not really necessary since we are constructing GradingPipeline explicitly instead of from the config
            tool_list="all" if use_tools else None,
            settings=PipelineConfig(use_fixing=use_fixing),
        ),
        reaction_fixer=good_fixer if use_fixing else FailingFixer(),  # noqa
    )

    result = next(pipeline.grade([original]))

    basic_before_fixer_tool_names = [
        "basic_smiles_validation",
        "exact_match",
    ]
    basic_tool_names = [
        "basic_smiles_validation",
        "exact_match",
        "charge_conservation",
        "mass_conservation",
    ]
    if use_tools:
        if use_fixing:
            assert [
                tool.name for tool in result.results
            ] == basic_before_fixer_tool_names + ["llm_reaction_fix"] + basic_tool_names
        else:
            assert [tool.name for tool in result.results] == basic_tool_names
    else:
        if use_fixing:
            assert [tool.name for tool in result.results] == ["llm_reaction_fix"]
        else:
            assert result.results == []

    if use_fixing:
        tool_res = [t for t in result.results if t.name == "llm_reaction_fix"][0]
        assert tool_res.data.original_reaction_smiles == original
        assert tool_res.data.fixed_reaction_smiles == fixed
