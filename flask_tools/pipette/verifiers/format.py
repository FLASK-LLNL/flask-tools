from __future__ import annotations

from .base import ReactionChecker
from ..constants import FinalGrade, ToolResult, ToolStatus
from ..smiles import parse_reaction_smi


def validate_smiles(rxn_smiles: str) -> bool:
    parse_reaction_smi(rxn_smiles)
    return True


class BasicSmilesValidationChecker(ReactionChecker):
    name = "basic_smiles_validation"
    stops_on_fail = True

    def run(self, rxn_smiles: str, context: dict[str, ToolResult]) -> ToolResult:
        fail_res = ToolResult(
            name=self.name,
            status=ToolStatus.FAIL,
            comment="tbd",
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
            data={
                "reactant_count": len(reactants),
                "product_count": len(products),
            },
            comment="Reaction SMILES parsed successfully.",
        )
