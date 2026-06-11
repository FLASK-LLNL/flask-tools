from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from flask_tools.pipette.config import PipetteConfig
from flask_tools.pipette.constants import FinalGrade, ReactionGrade, ToolResult
from flask_tools.pipette.reaction_fixer import ReactionFix
from flask_tools.pipette.verifiers import ReactionChecker


@pytest.fixture
def test_data_path() -> Path:
    data_dir = Path(__file__).parent / "data"
    return data_dir


@pytest.fixture
def tests_relative_path() -> Path:
    return Path(__file__).parent


@pytest.fixture(autouse=True)
def disable_dft_for_default_test_configs(monkeypatch) -> None:
    original_init = PipetteConfig.__init__

    def patched_init(self, *args, **kwargs) -> None:
        original_init(self, *args, **kwargs)
        if "rules" not in kwargs and len(args) < 4:
            self.rules.use_dft = False

    monkeypatch.setattr(PipetteConfig, "__init__", patched_init)


class RecordingReactionFixer:
    def __init__(self, fix: ReactionFix) -> None:
        self.fix_result = fix
        self.calls: list[tuple[str, list[ToolResult]]] = []

    def fix(self, rxn_smiles: str, results: list[ToolResult]) -> ReactionFix:
        self.calls.append((rxn_smiles, list(results)))
        return self.fix_result


class RecordingJudge:
    def __init__(
        self,
        *,
        final_grade: FinalGrade = FinalGrade.POSSIBLE,
        short_reason: str = "ai.mock_judge",
        comment: str = "Stub AI judge result.",
    ) -> None:
        self.final_grade = final_grade
        self.short_reason = short_reason
        self.comment = comment
        self.calls: list[tuple[str, list[ToolResult]]] = []

    def judge(
        self,
        rxn_smiles: str,
        results: list[ToolResult],
    ) -> ReactionGrade:
        captured_results = list(results)
        self.calls.append((rxn_smiles, captured_results))
        return ReactionGrade(
            final_grade=self.final_grade,
            short_reason=self.short_reason,
            results=captured_results,
            comment=self.comment,
        )


class SpyChecker(ReactionChecker):
    """A reaction checker that returns result of handler function and records how many times it's called"""

    def __init__(self, name: str, handler, *, stops_on_fail: bool = False) -> None:
        self.name = name
        self._handler = handler
        self.stops_on_fail = stops_on_fail
        self.calls: list[str] = []

    def run(self, rxn_smiles: str, context: dict[str, ToolResult]) -> ToolResult:
        self.calls.append(rxn_smiles)
        return self._handler(rxn_smiles, context)


@pytest.fixture
def install_mock_llm_services(monkeypatch):
    def _install(
        *, fix: ReactionFix = None, final_grade: FinalGrade = FinalGrade.POSSIBLE
    ):
        if fix is None:
            fix = ReactionFix(
                fixed_reaction_smiles="CCO.O>>CC.O",
                added_reactants=[],
                added_products=[],
                removed_agents=[],
                reasoning_summary="Fake fix",
            )
        fixer = RecordingReactionFixer(fix)
        judge = RecordingJudge(final_grade=final_grade)

        monkeypatch.setattr(
            "flask_tools.pipette.pipeline.LLMReactionFixer.from_config",
            lambda config: fixer,
        )
        monkeypatch.setattr(
            "flask_tools.pipette.pipeline.LLMJudge.from_config", lambda config: judge
        )
        return fixer, judge

    return _install
