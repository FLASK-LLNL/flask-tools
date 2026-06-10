from __future__ import annotations

import sys
import pytest

from flask_tools.pipette.constants import FinalGrade
from flask_tools.pipette.grade_rxn import grade_reaction, main
from flask_tools.pipette.config import load_config, ConfigType
from flask_tools.pipette.reaction_fixer import ReactionFix


ORIGINAL_REACTION_SMILES = "Cn1cnc2c1c(=O)[nH]c(=O)n2C.CI>>CN1C=NC2=C1C(=O)N(C(=O)N2C)C"
FIXED_REACTION_SMILES = "CI.Cn1cnc2c1c(=O)[nH]c(=O)n2C>>Cn1c(=O)c2c(ncn2C)n(C)c1=O.[I-]"


def _mock_caffeine() -> ReactionFix:
    return ReactionFix(
        fixed_reaction_smiles=FIXED_REACTION_SMILES,
        removed_agents=[],
        added_reactants=[],
        added_products=["[I-]"],
        reasoning_summary="Added iodide to the product side.",
    )


# Todo add a non-mocked version that actually calls an LLM. but will need to be behind some flag to not run too often.
@pytest.mark.parametrize(
    "config_name",
    [
        ConfigType.LLM_JUDGE_NO_DFT,
        ConfigType.RULES_NO_DFT,
    ],
)
def test_calls_fixer_caffeine_llm_judge(install_mock_llm_services, config_name) -> None:
    # peggy: keep, this will be the end to end test
    fixer, judge = install_mock_llm_services(
        fix=_mock_caffeine(),
        final_grade=FinalGrade.POSSIBLE,
    )

    config = load_config(config_name)
    results = grade_reaction([ORIGINAL_REACTION_SMILES], config=config)

    assert len(fixer.calls) == 1
    assert fixer.calls[0][0] == ORIGINAL_REACTION_SMILES

    result = results[0]

    if config_name == ConfigType.LLM_JUDGE_NO_DFT:
        assert result.final_grade is FinalGrade.POSSIBLE
        assert len(judge.calls) == 1
        assert judge.calls[0][0] == FIXED_REACTION_SMILES
    elif config_name == ConfigType.RULES_NO_DFT:
        assert result.final_grade is FinalGrade.IMPOSSIBLE
        assert len(judge.calls) == 0
    else:
        raise ValueError(f"Unknown config_name: {config_name}")

    tool_names = [tool_result.name for tool_result in result.results]
    assert "llm_reaction_fix" in tool_names

    fix_result = next(
        tool_result
        for tool_result in result.results
        if tool_result.name == "llm_reaction_fix"
    )
    assert fix_result.data["original_reaction_smiles"] == ORIGINAL_REACTION_SMILES
    assert fix_result.data["fixed_reaction_smiles"] == FIXED_REACTION_SMILES
    assert fix_result.data["added_products"] == ["[I-]"]
