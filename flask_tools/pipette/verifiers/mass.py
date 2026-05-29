from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

from rdkit import Chem

from pipette.verifiers.base import ReactionChecker
from pipette.config import PipetteConfig
from pipette.constants import FinalGrade, ToolResult, ToolStatus, MissingProductRule
from pipette.smiles import parse_reaction_smi


def atom_bag(
    molecules: list[object], *, explicit_hydrogens: bool = False
) -> dict[str, int]:
    bag: defaultdict[str, int] = defaultdict(int)
    prepared = [Chem.AddHs(mol) if explicit_hydrogens else mol for mol in molecules]
    for molecule in prepared:
        for atom in molecule.GetAtoms():
            bag[atom.GetSymbol()] += 1
    return dict(bag)


def bag_difference(
    products: dict[str, int], reactants: dict[str, int]
) -> dict[str, int]:
    delta: defaultdict[str, int] = defaultdict(int)
    for symbol, count in products.items():
        delta[symbol] += count
    for symbol, count in reactants.items():
        delta[symbol] -= count
    return {symbol: count for symbol, count in delta.items() if count}


def formula_mass(formula: dict[str, int]) -> float:
    table = Chem.GetPeriodicTable()
    return sum(
        table.GetAtomicWeight(symbol) * count for symbol, count in formula.items()
    )


def formula_from_smiles(smiles: str) -> dict[str, int]:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"Invalid solvent SMILES: {smiles!r}")
    return atom_bag([molecule], explicit_hydrogens=True)


def element_delta(rxn_smiles: str, explicit_hydrogens: bool = False) -> dict[str, int]:
    """Products - Reactants."""
    reactants, _, products = parse_reaction_smi(rxn_smiles)
    reactant_bag = atom_bag(reactants, explicit_hydrogens=explicit_hydrogens)
    product_bag = atom_bag(products, explicit_hydrogens=explicit_hydrogens)
    return bag_difference(product_bag, reactant_bag)


def check_mass_conservation(
    rxn_smiles: str, explicit_hydrogens: bool = False
) -> tuple[bool, str]:
    try:
        delta = element_delta(rxn_smiles, explicit_hydrogens=explicit_hydrogens)
    except Exception as exc:
        return False, f"Error while parsing reaction SMILES: {exc}"

    if delta:
        return False, f"Net difference in atom counts: {delta}"
    return True, "Reaction is mass-conserved."


def find_basic_missing(
    delta: dict[str, int],
    rules: list[MissingProductRule],
) -> list[dict[str, object]]:
    """Basic exceptions like missing hydrogens."""
    matches: list[dict[str, object]] = []
    if len(delta) == 1 and "H" in delta:
        matches.append(
            {
                "name": "hydrogen",
                "missing_side": "products" if delta["H"] > 0 else "reactants",
                "mass_amu": round(abs(delta["H"]) * 1.00794, 4),
                "confidence": "high",
            }
        )
    return matches


def find_missing_product_matches(
    delta: dict[str, int],
    rules: list[MissingProductRule],
) -> list[dict[str, object]]:
    matches: list[dict[str, object]] = []
    for rule in rules:
        negative_formula = {element: -count for element, count in rule.formula.items()}
        if delta == negative_formula:
            missing_side = "products"
        elif delta == rule.formula:
            missing_side = "reactants"
        else:
            continue
        matches.append(
            {
                "name": rule.name,
                "smiles": rule.smiles,
                "confidence": rule.confidence,
                "missing_side": missing_side,
                "mass_amu": round(formula_mass(rule.formula), 4),
            }
        )
    return matches


def load_solvent_commonness(path: Path) -> dict[str, str]:
    commonness: dict[str, str] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            smiles = (row.get("smiles") or "").strip()
            tier = (row.get("commonness") or "").strip().lower()
            if not smiles or not tier:
                continue
            commonness[smiles] = tier
    return commonness


def load_solvent_rules(
    solvents_path: Path, commonness_path: Path
) -> list[MissingProductRule]:
    commonness_by_smiles = load_solvent_commonness(commonness_path)
    rules: list[MissingProductRule] = []

    with solvents_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            smiles = (row.get("smiles") or "").strip()
            name = (row.get("common_name") or "").strip()
            if not smiles or not name:
                continue

            rules.append(
                MissingProductRule(
                    name=name,
                    smiles=smiles,
                    formula=formula_from_smiles(smiles),
                    confidence=commonness_by_smiles.get(smiles, "low"),
                )
            )

    return rules


class MassConservationChecker(ReactionChecker):
    name = "mass_conservation"

    def __init__(self, config: PipetteConfig) -> None:
        self.config = config
        self.missing_product_rules = load_solvent_rules(
            config.solvent_catalog_path,
            config.solvent_commonness_path,
        )

    def run(self, rxn_smiles: str, context: dict[str, ToolResult]) -> ToolResult:
        try:
            delta = element_delta(rxn_smiles, explicit_hydrogens=True)
        except Exception as exc:
            return ToolResult(
                name=self.name,
                status=ToolStatus.ERROR,
                grade_hint=FinalGrade.IMPOSSIBLE,
                comment=f"Mass conservation could not be evaluated: {exc}",
            )

        if not delta:
            return ToolResult(
                name=self.name,
                status=ToolStatus.PASS,
                grade_hint=FinalGrade.LIKELY,
                data={
                    "mass_difference_amu": 0.0,
                    "element_difference": {},
                    "possible_missing_products": [],
                    "missing_product_confidence": None,
                    "closest_stoich": None,
                },
                comment="Element counts are conserved.",
            )

        matches = find_basic_missing(delta, self.missing_product_rules)
        if not matches:
            matches = find_missing_product_matches(delta, self.missing_product_rules)
        if matches:
            best_match = matches[0]
            return ToolResult(
                name=self.name,
                status=ToolStatus.POTENTIAL,
                grade_hint=FinalGrade.POSSIBLE,
                data={
                    "mass_difference_amu": best_match["mass_amu"],
                    "element_difference": delta,
                    "possible_missing_products": matches,
                    "missing_product_confidence": best_match["confidence"],
                    "closest_stoich": None,
                },
                comment=(
                    "Element counts are not conserved, but the difference matches a "
                    f"common omitted species or solvent: {best_match['name']}."
                ),
            )

        mass_difference = round(
            abs(
                formula_mass({element: abs(count) for element, count in delta.items()})
            ),
            4,
        )
        return ToolResult(
            name=self.name,
            status=ToolStatus.FAIL,
            grade_hint=FinalGrade.IMPOSSIBLE,
            data={
                "mass_difference_amu": mass_difference,
                "element_difference": delta,
                "possible_missing_products": [],
                "missing_product_confidence": None,
                "closest_stoich": None,
            },
            comment="Element counts are not conserved and do not match a configured common omission.",
        )
