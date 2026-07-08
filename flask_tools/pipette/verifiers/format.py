###############################################################################
## Copyright 2025-2026 Lawrence Livermore National Security, LLC.
## See the top-level LICENSE file for details.
##
## SPDX-License-Identifier: Apache-2.0
###############################################################################

from __future__ import annotations

from .base import ReactionChecker
from ..constants import (
    FinalGrade,
    ToolResult,
    ToolStatus,
    ToolResultsDict,
    ToolResultDetails,
)
from ..smiles import parse_reaction_smi


def validate_smiles(rxn_smiles: str) -> bool:
    parse_reaction_smi(rxn_smiles)
    return True


class SmilesValidationResultDetails(ToolResultDetails):
    reactant_count: int
    product_count: int


class BasicSmilesValidationChecker(ReactionChecker):
    """Checks if reaction SMILES is valid
    ToolResult example:
        ToolResult(
            name="basic_smiles_validation",
            status=ToolStatus.PASS,
            data=SmilesValidationResultDetails(
                reactant_count = len(reactants),
                product_count = len(products),
            ),
            comment="Reaction SMILES parsed successfully.",
        )
    """

    name = "basic_smiles_validation"
    stops_on_fail = True

    def run(self, rxn_smiles: str, context: ToolResultsDict) -> ToolResult:
        fail_res = ToolResult(
            name=self.name,
            status=ToolStatus.FAIL,
            data=None,
            comment="filled-later",
        )
        try:
            reactants, _, products = parse_reaction_smi(rxn_smiles)
        except Exception as exc:
            fail_res.comment = f"Invalid reaction SMILES: {exc}"
            return fail_res

        if not reactants or not products:
            fail_res.comment = (
                "Reaction must contain at least one reactant and one product."
            )
            return fail_res

        return ToolResult(
            name=self.name,
            status=ToolStatus.PASS,
            data=SmilesValidationResultDetails(
                reactant_count=len(reactants),
                product_count=len(products),
            ),
            comment="Reaction SMILES parsed successfully.",
        )
