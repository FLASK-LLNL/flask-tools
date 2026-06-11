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
ORIGINAL_REACTION_SMI = "Cn1cnc2c1c(=O)[nH]c(=O)n2C.CI>>CN1C=NC2=C1C(=O)N(C(=O)N2C)C"
# Balance with I on right
FIXED_REACTION_SMIS = (
    "CI.Cn1cnc2c1c(=O)[nH]c(=O)n2C>>Cn1c(=O)c2c(ncn2C)n(C)c1=O.I",
    "CI.Cn1cnc2c1c(=O)[nH]c(=O)n2C>>Cn1c(=O)c2c(ncn2C)n(C)c1=O.[H+].[I-]",
)


@pytest.mark.llm_query
@pytest.mark.parametrize(
    "config_name",
    [
        ConfigType.LLM_JUDGE_NO_DFT,
    ],
)
def test_calls_fixer_caffeine_llm_judge(install_mock_llm_services, config_name) -> None:
    # peggy: should end to end test have own test file?

    config = load_config(config_name)
    results = grade_reaction([ORIGINAL_REACTION_SMI], config=config)
    assert len(results) == 1
    result: ReactionGrade = results[0]
    tool_results: list[ToolResult] = result.results
    tool_names = [tool_result.name for tool_result in result.results]
    assert "llm_reaction_fix" in tool_names

    fix_result = next(tr for tr in tool_results if tr.name == "llm_reaction_fix")
    assert fix_result.data["original_reaction_smiles"] == ORIGINAL_REACTION_SMI
    assert fix_result.data["fixed_reaction_smiles"] in FIXED_REACTION_SMIS, fix_result
    assert fix_result.data["added_products"] in (["I"], ["H+", "I-"]), fix_result

    if config_name == ConfigType.LLM_JUDGE_NO_DFT:
        assert result.final_grade is FinalGrade.LIKELY, (result.final_grade, fix_result)
        assert result.short_reason.startswith(
            "ai."
        )  # Prompt requests this format for the short reason
    else:
        raise ValueError(f"Unknown config_name: {config_name}")
