###############################################################################
## Copyright 2025-2026 Lawrence Livermore National Security, LLC.
## See the top-level LICENSE file for details.
##
## SPDX-License-Identifier: Apache-2.0
###############################################################################

"""Integration tests"""

from __future__ import annotations

import dataclasses
import json
import sys
from collections import namedtuple
from copy import copy
from dataclasses import asdict
from typing import Sequence

import pytest
from onnxruntime.tools.ort_format_model.ort_flatbuffers_py.fbs import SequenceType
from rdkit.Chem.ChemUtils import TemplateExpand

from flask_tools.pipette import ToolResult, ToolStatus, FinalGrade
from flask_tools.pipette.constants import FinalGrade, ReactionGrade
from flask_tools.pipette.grade_rxn import grade_reaction, main
from flask_tools.pipette.config import load_config, ConfigType, PipetteConfig
from flask_tools.pipette.pipeline import GradingPipeline
from flask_tools.pipette.reaction_fixer import ReactionFix
from flask_tools.pipette.verifiers import ChargeConservationChecker, ReactionChecker
from conftest import SpyChecker

# Caffeine
CAFFEINE_ORIGINAL_REACTION_SMI = (
    "Cn1cnc2c1c(=O)[nH]c(=O)n2C.CI>>CN1C=NC2=C1C(=O)N(C(=O)N2C)C"
)
CAFFEINE_FIXED_REACTION_SMIS = (
    "CI.Cn1cnc2c1c(=O)[nH]c(=O)n2C>>Cn1c(=O)c2c(ncn2C)n(C)c1=O.I",
    "CI.Cn1cnc2c1c(=O)[nH]c(=O)n2C>>Cn1c(=O)c2c(ncn2C)n(C)c1=O.[H+].[I-]",
)


@dataclasses.dataclass()
class RxnToTest:
    should_pass: bool
    orig_smi: str
    possible_fixed_smis: (
        Sequence[str] | None
    )  # None if it's not meant to be balanceable
    accepted_grades: Sequence[FinalGrade]


rxns_to_test = {
    (CAFFEINE := "caffeine"): RxnToTest(
        True,
        orig_smi=CAFFEINE_ORIGINAL_REACTION_SMI,
        possible_fixed_smis=CAFFEINE_FIXED_REACTION_SMIS,
        accepted_grades=[FinalGrade.LIKELY],
    ),
    # mislabeled rxn: common solvent as product pulled from flask-copilot retrosynthesis
    (TOLUENE := "toluene"): RxnToTest(
        False,
        orig_smi="Cl[SiH](C)C.BrCCCCCCO[SiH](C)C>>Cc1ccccc1",
        possible_fixed_smis=None,
        accepted_grades=[FinalGrade.IMPOSSIBLE],
    ),
}


@pytest.mark.llm_query
@pytest.mark.parametrize(
    "rxn_name",
    [
        CAFFEINE,
        TOLUENE,
    ],
)
def test_calls_fixer_caffeine_llm_judge(rxn_name: str) -> None:
    # def test_calls_fixer_caffeine_llm_judge(install_mock_llm_services, config_name) -> None:
    rxn_to_test = rxns_to_test[rxn_name]

    # Short tests that checks fixed smiles and final grade
    config = load_config(ConfigType.LLM_JUDGE_NO_DFT)
    results = grade_reaction([rxn_to_test.orig_smi], config=config)
    assert len(results) == 1
    result: ReactionGrade = results[0]
    tool_results: list[ToolResult] = result.results
    tool_names = [tool_result.name for tool_result in result.results]
    assert "llm_reaction_fix" in tool_names

    fix_result = next(tr for tr in tool_results if tr.name == "llm_reaction_fix")
    assert fix_result.data["original_reaction_smiles"] == rxn_to_test.orig_smi
    if rxn_to_test.possible_fixed_smis:
        assert (
            fix_result.data["fixed_reaction_smiles"] in rxn_to_test.possible_fixed_smis
        ), fix_result
        if rxn_name == CAFFEINE:
            assert fix_result.data["added_products"] in (
                ["I"],
                ["[H+]", "[I-]"],
            ), fix_result

    assert (
        result.final_grade in rxn_to_test.accepted_grades
    ), f"{(result.final_grade, fix_result)}"
    assert result.short_comment.startswith(
        "ai."
    )  # Prompt requests this format for the short reason


@pytest.mark.llm_query
def test_pipeline_fixed_reaction(
    tests_relative_path,
) -> None:
    # In-depth test that checks that ever single expected tool is called
    original = "CCO>>C=C"
    fixed = "CCO>>C=C.O"

    smiles_validation = SpyChecker(
        "basic_smiles_validation",
        lambda rxn_smiles, _: ToolResult(
            name="basic_smiles_validation",
            status=ToolStatus.PASS,
            comment=f"parsed {rxn_smiles}",
        ),
    )
    exact = SpyChecker(
        "exact_match",
        lambda rxn_smiles, _: ToolResult(
            name="exact_match",
            status=ToolStatus.UNKNOWN,
            data={"found": None},
            comment="No exact match.",
        ),
    )
    charge = SpyChecker(
        "charge_conservation",
        lambda rxn_smiles, _: ToolResult(
            name="charge_conservation",
            status=ToolStatus.PASS,
            data={"charge_difference": 0},
            comment="Charge is conserved.",
        ),
        stops_on_fail=True,
    )
    mass = SpyChecker(
        "mass_conservation",
        lambda rxn_smiles, _: ToolResult(
            name="mass_conservation",
            status=ToolStatus.FAIL if rxn_smiles == original else ToolStatus.PASS,
            comment="Mass failed." if rxn_smiles == original else "Mass passed.",
        ),
    )

    reaction_energy = SpyChecker(
        "reaction_energy",
        lambda rxn_smiles, _: ToolResult(
            name="reaction_energy",
            status=ToolStatus.PASS,
            comment="Energy passed.",
        ),
    )

    class StubReactionFixer:
        # Have a fixed LLM fixer step to better test LLM judge step
        def __init__(self) -> None:
            self.calls: list[str] = []

        def fix(self, rxn_smiles: str, results: list[ToolResult]) -> ReactionFix:
            self.calls.append(rxn_smiles)
            assert [result.name for result in results] == [
                "basic_smiles_validation",
                "exact_match",
                # "charge_conservation",
                # "mass_conservation",
            ]
            return ReactionFix(
                fixed_reaction_smiles=fixed,
                removed_agents=[],
                added_reactants=[],
                removed_products=[],
                added_products=["O"],
                reasoning_summary="Removed the agent and balanced both sides with water.",
            )

    fixer = StubReactionFixer()
    pipeline = GradingPipeline(
        checkers=[smiles_validation, exact, charge, mass, reaction_energy],
        config=PipetteConfig(mode="exact"),
        reaction_fixer=fixer,  # noqa
    )

    result = next(pipeline.grade([original]))

    # Manual sanity check (run test twice): two runs can have all the same tool results, but different final grade
    # dbg_file = tests_relative_path / "tool_res.json"
    # result2 = copy(result)
    # result2.final_grade = None
    # result2.short_comment = None
    # result2.short_comment = None
    # result2.comment = None
    # res_dict = asdict(result2)
    # # Remove the llm judge answer
    # if not dbg_file.exists():
    #     prev_res_dict = None
    #     with open(dbg_file, "w") as f:
    #         json.dump(res_dict, f, indent=2)
    # else:
    #     with open(dbg_file, "r") as f:
    #         prev_res_dict = json.load(f)
    #
    # if prev_res_dict is not None:
    #     # assert res_dict == prev_res_dict
    #     assert json.loads(json.dumps(res_dict)) == prev_res_dict

    assert smiles_validation.calls == [original, fixed]
    assert exact.calls == [original, fixed]
    assert fixer.calls == [original]
    assert charge.calls == [fixed]
    assert mass.calls == [fixed]
    assert reaction_energy.calls == [fixed]
    assert result.short_comment.startswith("ai.")
    assert [tool.name for tool in result.results] == [
        "basic_smiles_validation",
        "exact_match",
        "llm_reaction_fix",
        "basic_smiles_validation",
        "exact_match",
        "charge_conservation",
        "mass_conservation",
        "reaction_energy",
    ]
    llm_fix_i = 2
    assert result.results[llm_fix_i].data["original_reaction_smiles"] == original
    assert result.results[llm_fix_i].data["fixed_reaction_smiles"] == fixed
    assert result.final_grade == FinalGrade.LIKELY, result
