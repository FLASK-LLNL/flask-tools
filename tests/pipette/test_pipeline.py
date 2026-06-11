from __future__ import annotations

import json
from copy import copy
from dataclasses import asdict

import pytest
import pandas as pd

import flask_tools.pipette.verifiers.reaction_energy
from flask_tools.pipette.verifiers import ReactionEnergyChecker
from flask_tools.pipette.verifiers.base import ReactionChecker, CacheableReactionChecker
from flask_tools.pipette.config import LLMJudgeConfig, PipetteConfig, TopLevelConfig
from flask_tools.pipette.constants import FinalGrade, ToolResult, ToolStatus
from flask_tools.pipette.reaction_fixer import ReactionFix
from flask_tools.pipette.pipeline import (
    GradingPipeline,
    PendingReaction,
    build_default_pipeline,
    resolve_tool_list,
)
from conftest import RecordingJudge
import helpers


# todo check this file


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

    def run(self, rxn_smiles: str, context: dict[str, ToolResult]) -> ToolResult:
        if isinstance(self._result, Exception):
            raise self._result
        return self._result


class RoutingChecker(ReactionChecker):
    """A reaction checker that returns result of handler function it was initialized with."""

    def __init__(self, name: str, handler, *, stops_on_fail: bool = False) -> None:
        self.name = name
        self._handler = handler
        self.stops_on_fail = stops_on_fail
        self.calls: list[str] = []

    def run(self, rxn_smiles: str, context: dict[str, ToolResult]) -> ToolResult:
        self.calls.append(rxn_smiles)
        return self._handler(rxn_smiles, context)


class CacheableRoutingChecker(RoutingChecker, CacheableReactionChecker):
    def check_cache(self, rxn_smiles: str) -> None | ToolResult:
        self.calls.append(rxn_smiles)
        return self._handler(rxn_smiles, {})


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


# peggy: possibly remove after discuss w/ Tal
# def test_build_default_pipeline_passes_reaction_energy_database_string(
#     tmp_path,
# ) -> None:
#     database_path = tmp_path / "fake_molecule_energies.csv"
#     database_path.write_text("inchi_key,energy_ev_mol\nTEST,-1.0\n", encoding="utf-8")
#     config = PipetteConfig(tool_list=["reaction_energy"])
#     config.tools_settings.reaction_energy.database = database_path
#
#     pipeline = build_default_pipeline(config=config)
#
#     checker = pipeline.checkers[0]
#     assert checker.name == "reaction_energy"
#     assert checker.database == str(database_path.resolve())


def test_ai_mode_continues_after_allowed_tool_error() -> None:
    judge = RecordingJudge()
    pipeline = GradingPipeline(
        checkers=[
            StubChecker(
                "exact_match", RuntimeError("database offline"), stops_on_fail=True
            ),
            StubChecker("reaction_energy", _pass_result("reaction_energy")),
        ],
        config=PipetteConfig(
            mode="ai",
            llm_judge=LLMJudgeConfig(allow_fail=["exact_match"]),
        ),
        judge=judge,
    )

    result = next(pipeline.grade(["CCO>>CC=O"]))

    assert result.short_reason == "ai.mock_judge"
    assert [tool.status for tool in result.results] == [
        ToolStatus.ERROR,
        ToolStatus.PASS,
    ]
    assert "traceback" in result.results[0].data
    assert "RuntimeError: database offline" in result.results[0].data["traceback"]
    assert len(judge.calls) == 1


def test_ai_mode_stops_after_unallowed_tool_error() -> None:
    judge = RecordingJudge()
    pipeline = GradingPipeline(
        checkers=[
            StubChecker(
                "exact_match", RuntimeError("database offline"), stops_on_fail=True
            ),
            StubChecker("reaction_energy", _pass_result("reaction_energy")),
        ],
        config=PipetteConfig(mode="ai"),
        judge=judge,
    )

    result = next(pipeline.grade(["CCO>>CC=O"]))

    assert [tool.status for tool in result.results] == [
        ToolStatus.ERROR,
        ToolStatus.NOT_RUN,
    ]
    assert (
        result.results[1].skipped_reason == "Skipped after hard failure in exact_match."
    )


def test_ai_mode_requires_api_key_for_default_judge(monkeypatch) -> None:
    for env_var in (
        "FLASK_ORCHESTRATOR_API_KEY",
        "PIPETTE_API_KEY",
        "OPENAI_API_KEY",
    ):
        monkeypatch.delenv(env_var, raising=False)
    with pytest.raises(ValueError, match="API key"):
        GradingPipeline(
            checkers=[
                StubChecker(
                    "basic_smiles_validation", _pass_result("basic_smiles_validation")
                )
            ],
            config=PipetteConfig(mode="ai"),
        )


def test_ai_mode_uses_default_judge_when_none_is_provided(monkeypatch) -> None:
    judge = RecordingJudge()
    monkeypatch.setattr(
        "flask_tools.pipette.pipeline.LLMJudge.from_config", lambda config: judge
    )

    pipeline = GradingPipeline(
        checkers=[
            StubChecker(
                "basic_smiles_validation", _pass_result("basic_smiles_validation")
            )
        ],
        config=PipetteConfig(mode="ai"),
    )

    result = next(pipeline.grade(["CCO>>CC=O"]))

    assert result.short_reason == "ai.mock_judge"
    assert len(judge.calls) == 1


def test_ai_mode_rejects_unknown_allow_fail_tool() -> None:
    with pytest.raises(ValueError, match="llm_judge.allow_fail"):
        GradingPipeline(
            checkers=[
                StubChecker(
                    "basic_smiles_validation", _pass_result("basic_smiles_validation")
                )
            ],
            config=PipetteConfig(
                mode="ai", llm_judge=LLMJudgeConfig(allow_fail=["missing_tool"])
            ),
            judge=RecordingJudge(),
        )


def test_grade_one_tiered_run_returns_pending_reaction_with_tuple_prefix_results() -> (
    None
):
    rxn_smiles = "CCO>>CC=O"
    # peggy: these can be StubCheckers instead of RoutingChecker. And it doesn't need to be _pass_result necessarily, although it more closely replicaetes the usual return type
    pipeline = GradingPipeline(
        checkers=[
            RoutingChecker(
                "basic_smiles_validation",
                lambda _rxn_smiles, _: _pass_result("basic_smiles_validation"),
            ),
            RoutingChecker(
                "exact_match", lambda _rxn_smiles, _: _pass_result("exact_match")
            ),
            RoutingChecker(
                "charge_conservation",
                lambda _rxn_smiles, _: _pass_result("charge_conservation"),
            ),
            CacheableRoutingChecker(
                "reaction_energy",
                lambda _rxn_smiles, _: None,  # return None = Not found in cache
            ),
        ],
        config=PipetteConfig(mode="exact"),
    )

    pending = pipeline.grade_one(
        rxn_smiles, tiered_run=True
    )  # tiered_run: Will stop if cached result not found

    assert isinstance(pending, PendingReaction)
    assert pending.rxn_smiles == rxn_smiles
    assert pending.prefix_results == [
        (
            rxn_smiles,
            "basic_smiles_validation",
            _pass_result("basic_smiles_validation"),
        ),
        (rxn_smiles, "exact_match", _pass_result("exact_match")),
        (rxn_smiles, "charge_conservation", _pass_result("charge_conservation")),
    ]


@pytest.mark.llm_query
def test_pipeline_reruns_with_llm_fixed_reaction_once_for_tiered_flow(
    tests_relative_path,
) -> None:
    # peggy: this could go into the e2e tests... what to name the e2e? pipeline? rxn?
    original = "CCO>>CC"
    fixed = "CCO>>CC.O"

    smiles_validation = RoutingChecker(
        "basic_smiles_validation",
        lambda rxn_smiles, _: ToolResult(
            name="basic_smiles_validation",
            status=ToolStatus.PASS,
            comment=f"parsed {rxn_smiles}",
        ),
    )
    exact = RoutingChecker(
        "exact_match",
        lambda rxn_smiles, _: ToolResult(
            name="exact_match",
            status=ToolStatus.UNKNOWN,
            data={"found": None},
            comment="No exact match.",
        ),
    )
    charge = RoutingChecker(
        "charge_conservation",
        lambda rxn_smiles, _: ToolResult(
            name="charge_conservation",
            status=ToolStatus.PASS,
            data={"charge_difference": 0},
            comment="Charge is conserved.",
        ),
        stops_on_fail=True,
    )
    mass = RoutingChecker(
        "mass_conservation",
        lambda rxn_smiles, _: ToolResult(
            name="mass_conservation",
            status=ToolStatus.FAIL if rxn_smiles == original else ToolStatus.PASS,
            comment="Mass failed." if rxn_smiles == original else "Mass passed.",
        ),
    )

    def make_re_res(_rxn_smiles, _):
        res = ToolResult(
            name="reaction_energy",
            status=ToolStatus.PASS,
            comment="Energy passed.",
        )
        return res

    reaction_energy = RoutingChecker(
        "reaction_energy",
        # lambda rxn_smiles, _: ToolResult(
        #     name="reaction_energy",
        #     status=ToolStatus.PASS,
        #     comment="Energy passed.",
        # ),
        make_re_res,
    )

    class StubReactionFixer:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def fix(self, rxn_smiles: str, results: list[ToolResult]) -> ReactionFix:
            self.calls.append(rxn_smiles)
            assert [result.name for result in results] == [
                "basic_smiles_validation",
                "exact_match",
                "charge_conservation",
                "mass_conservation",
            ]
            return ReactionFix(
                fixed_reaction_smiles=fixed,
                removed_agents=[],
                added_reactants=[],
                added_products=["O"],
                reasoning_summary="Removed the agent and balanced both sides with water.",
            )

    fixer = StubReactionFixer()
    pipeline = GradingPipeline(
        checkers=[smiles_validation, exact, charge, mass, reaction_energy],
        config=PipetteConfig(mode="exact"),
        reaction_fixer=fixer,  # noqa
    )

    result = next(pipeline.grade([original], tiered_run=True))

    # debugging: i think two runs have all but the llm judge result be different... which sucks
    dbg_file = tests_relative_path / "tool_res.json"
    result2 = copy(result)
    result2.final_grade = None
    result2.short_reason = None
    result2.short_comment = None
    result2.comment = None
    res_dict = asdict(result2)
    # Remove the llm judge answer
    if not dbg_file.exists():
        prev_res_dict = None
        with open(dbg_file, "w") as f:
            json.dump(res_dict, f, indent=2)
    else:
        with open(dbg_file, "r") as f:
            prev_res_dict = json.load(f)

    if prev_res_dict is not None:
        assert res_dict == prev_res_dict

    assert fixer.calls == [original]
    assert smiles_validation.calls == [original, fixed]
    assert exact.calls == [original, fixed]
    assert charge.calls == [original, fixed]
    assert mass.calls == [original, fixed]
    assert reaction_energy.calls == [fixed]
    assert result.short_reason.startswith("ai.")
    assert [tool.name for tool in result.results] == [
        "basic_smiles_validation",
        "exact_match",
        "charge_conservation",
        "mass_conservation",
        "llm_reaction_fix",
        "basic_smiles_validation",
        "exact_match",
        "charge_conservation",
        "mass_conservation",
        "reaction_energy",
    ]
    llm_fix_i = 4
    assert result.results[llm_fix_i].data["original_reaction_smiles"] == original
    assert result.results[llm_fix_i].data["fixed_reaction_smiles"] == fixed
    assert result.final_grade == FinalGrade.POSSIBLE, result
