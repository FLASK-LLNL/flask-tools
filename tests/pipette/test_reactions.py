from __future__ import annotations

import sys
import pytest

from flask_tools.pipette import ToolResult
from flask_tools.pipette.constants import FinalGrade, ReactionGrade
from flask_tools.pipette.grade_rxn import grade_reaction, main
from flask_tools.pipette.config import load_config, ConfigType
from flask_tools.pipette.reaction_fixer import ReactionFix
from flask_tools.pipette.verifiers import ChargeConservationChecker

# Caffeine
ORIGINAL_REACTION_SMILES = "Cn1cnc2c1c(=O)[nH]c(=O)n2C.CI>>CN1C=NC2=C1C(=O)N(C(=O)N2C)C"
# Balance with [I-] on right
FIXED_REACTION_SMILES = "CI.Cn1cnc2c1c(=O)[nH]c(=O)n2C>>Cn1c(=O)c2c(ncn2C)n(C)c1=O.[I-]"


def _mock_caffeine() -> ReactionFix:
    return ReactionFix(
        fixed_reaction_smiles=FIXED_REACTION_SMILES,
        removed_agents=[],
        added_reactants=[],
        added_products=["[I-]"],
        reasoning_summary="Added iodide to the product side.",
    )


@pytest.mark.parametrize(
    "config_name",
    [
        # ConfigType.RULES_NO_DFT,
        pytest.param(ConfigType.LLM_JUDGE_NO_DFT, marks=pytest.mark.llm_query),
    ],
)
def test_calls_fixer_caffeine_llm_judge(install_mock_llm_services, config_name) -> None:
    # peggy: should end to end test have own test file?

    config = load_config(config_name)
    results = grade_reaction([ORIGINAL_REACTION_SMILES], config=config)
    assert len(results) == 1
    result: ReactionGrade = results[0]
    tool_results: list[ToolResult] = result.results

    if config_name == ConfigType.LLM_JUDGE_NO_DFT:
        assert result.final_grade is FinalGrade.LIKELY
        assert result.short_reason.startswith(
            "ai."
        )  # Prompt requests this format for the short reason
        assert False  # debugging
    elif config_name == ConfigType.RULES_NO_DFT:
        assert result.final_grade is FinalGrade.IMPOSSIBLE
    else:
        raise ValueError(f"Unknown config_name: {config_name}")

    tool_names = [tool_result.name for tool_result in result.results]
    assert "llm_reaction_fix" in tool_names

    fix_result = next(tr for tr in tool_results if tr.name == "llm_reaction_fix")
    assert fix_result.data["original_reaction_smiles"] == ORIGINAL_REACTION_SMILES
    assert fix_result.data["fixed_reaction_smiles"] == FIXED_REACTION_SMILES
    assert fix_result.data["added_products"] == ["[I-]"]


def test_verify_charge():
    rxn0 = "CI.Cn1cnc2c1c(=O)[nH]c(=O)n2C>>Cn1c(=O)c2c(ncn2C)n(C)c1=O.I"
    rxn1 = "CI.Cn1cnc2c1c(=O)[nH]c(=O)n2C>>Cn1c(=O)c2c(ncn2C)n(C)c1=O.[I-]"
    checker = ChargeConservationChecker()
    r0 = checker.run(rxn0, {})
    r1 = checker.run(rxn1, {})
    pass
    # debugging
