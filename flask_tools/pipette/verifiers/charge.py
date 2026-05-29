from __future__ import annotations

from .base import ReactionChecker
from ..constants import FinalGrade, ToolResult, ToolStatus
from ..smiles import parse_reaction_smi


def total_charge(molecules: list[object]) -> int:
    charge = 0
    for molecule in molecules:
        for atom in molecule.GetAtoms():
            charge += atom.GetFormalCharge()
    return charge


def charge_difference(rxn_smiles: str) -> int:
    reactants, _, products = parse_reaction_smi(rxn_smiles)
    return total_charge(products) - total_charge(reactants)


def check_charge_conservation(rxn_smiles: str) -> tuple[bool, str]:
    try:
        diff = charge_difference(rxn_smiles)
    except Exception as exc:
        return False, f"Error while parsing reaction SMILES: {exc}"

    if diff != 0:
        return False, f"Net difference in charge: {diff}"
    return True, "Reaction is charge-conserved."


class ChargeConservationChecker(ReactionChecker):
    name = "charge_conservation"
    stops_on_fail = True

    def run(self, rxn_smiles: str, context: dict[str, ToolResult]) -> ToolResult:
        try:
            diff = charge_difference(rxn_smiles)
        except Exception as exc:
            return ToolResult(
                name=self.name,
                status=ToolStatus.ERROR,
                grade_hint=FinalGrade.IMPOSSIBLE,
                comment=f"Charge conservation could not be evaluated: {exc}",
            )

        if diff != 0:
            return ToolResult(
                name=self.name,
                status=ToolStatus.FAIL,
                grade_hint=FinalGrade.IMPOSSIBLE,
                data={"charge_difference": diff},
                comment=f"Charge is not conserved. Product minus reactant charge: {diff}.",
            )

        return ToolResult(
            name=self.name,
            status=ToolStatus.PASS,
            grade_hint=FinalGrade.LIKELY,
            data={"charge_difference": 0},
            comment="Charge is conserved.",
        )
