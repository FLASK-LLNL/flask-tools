###############################################################################
## Copyright 2025-2026 Lawrence Livermore National Security, LLC.
## See the top-level LICENSE file for details.
##
## SPDX-License-Identifier: Apache-2.0
###############################################################################

from dataclasses import dataclass
from typing import Literal
from typing_extensions import TypedDict

from rdkit import Chem
from rdkit.Chem import AllChem


StrategyName = Literal[
    "vinyl",
    "acrylate",
    "rop_thf",
    "rop_epoxide",
    "ketene",
    "cond_alpha_hydroxy_acid",
    "cond_diphenol",
    "rop_lactam",
    "cond_omega_amino_acid",
    "alkyne",
    "romp_bicyclic_alkene",
    "cond_polyester",
    "cond_polyamide",
    "polyacetylene",
]


class SuggestionDict(TypedDict):
    strategy: str
    confidence: float
    reason: str


class PolymerizeResult(TypedDict):
    repeat_smiles: str
    strategy: str
    rationale: str


class InputAssessment(TypedDict):
    input_smiles: str
    normalized_smiles: str
    canonical_smiles: str | None
    input_kind: str
    contains_wildcards: bool
    has_bigsmiles_wrapper: bool
    notes: list[str]


class RetrosynthesisCheckResult(TypedDict):
    is_match: bool
    target_repeat_smiles: str
    canonical_target_repeat_smiles: str | None
    candidate_monomer_smiles: str
    predicted_repeat_smiles: str | None
    canonical_predicted_repeat_smiles: str | None
    strategy_used: str | None
    rationale: str | None
    notes: list[str]


@dataclass
class Suggestion:
    strategy: str
    confidence: float
    reason: str


GUIDANCE_ONLY_STRATEGIES = frozenset(
    {"needs_comonomer", "multifunctional_hydroxy_diacid"}
)

SPECIFICITY_ORDER = {
    "multifunctional_hydroxy_diacid": 95,
    "rop_epoxide": 90,
    "rop_lactam": 88,
    "cond_omega_amino_acid": 86,
    "rop_thf": 80,
    "cond_alpha_hydroxy_acid": 75,
    "romp_bicyclic_alkene": 74,
    "alkyne": 72,
    "acrylate": 70,
    "cond_diphenol": 65,
    "ketene": 60,
    "vinyl": 50,
    "needs_comonomer": 0,
}


# ---------------------------------------------------------------------------
# Reaction SMARTS
# ---------------------------------------------------------------------------

RXN_VINYL_HT = AllChem.ReactionFromSmarts(
    "[CH2:1]=[C:2]>>[*]-[CH2:1]-[C:2](-[*])"
)

RXN_ACRYLATE_HT = AllChem.ReactionFromSmarts(
    "[CH2:1]=[CH:2]-[C:3](=O)[O,N:4]>>[*]-[CH2:1]-[CH:2](-[*])-[C:3](=O)[O,N:4]"
)

RXN_METHACRYLATE_HT = AllChem.ReactionFromSmarts(
    "[CH2:1]=[C:2]([#6:5])-[C:3](=O)[O,N:4]>>[*]-[CH2:1]-[C:2]([#6:5])(-[*])-[C:3](=O)[O,N:4]"
)

RXN_ROP_THF = AllChem.ReactionFromSmarts(
    "[O;X2;R:1]1-[C;R:2]-[C;R:3]-[C;R:4]-[C;R:5]-1>>[*]-[O:1]-[C:2]-[C:3]-[C:4]-[C:5]-[*]"
)

RXN_ROP_EPOXIDE = AllChem.ReactionFromSmarts(
    "[O;X2;R:1]1[C;R:2][C;R:3]1>>[*]-[O:1]-[C:2]-[C:3]-[*]"
)

RXN_KETENE_TO_PEO = AllChem.ReactionFromSmarts(
    "[C:1]=[C:2]=[O:3]>>[*]-[CH2:1]-[CH2:2]-[O:3]-[*]"
)

RXN_COND_ALPHA_HA = AllChem.ReactionFromSmarts(
    "[O;H1:1]-[C:2]-[C:3](=O)[O;H1]>>[*]-[O:1]-[C:2]-[C:3](=O)-[*]"
)

RXN_DIPHENOL_HQ = AllChem.ReactionFromSmarts(
    "[O;H1:1]c1ccc([O;H1:2])cc1>>[*]-[O:1]-c1ccc([O:2]-[*])cc1"
)

RXN_DIPHENOL_GENERIC_ONE_END = AllChem.ReactionFromSmarts(
    "[O;H1:1]-[a:2]>>[*]-[O:1]-[a:2]"
)

RXN_LACTAM_R7_TO_N6 = AllChem.ReactionFromSmarts(
    "O=[C:2]1[N:1][C:3][C:4][C:5][C:6][C:7]1>>[*]-[N:1]-[C:3]-[C:4]-[C:5]-[C:6]-[C:7]-[C:2](=O)-[*]"
)

RXN_LACTAM_GENERIC = AllChem.ReactionFromSmarts(
    "[N;R:1]-[C;R:2](=O)>>[*]-[N:1]-[C:2](=O)-[*]"
)

RXN_OMEGA_AMINO_ACID_N6 = AllChem.ReactionFromSmarts(
    "[N:1][C:2][C:3][C:4][C:5][C:6][C:7](=O)[O;H1]>>[*]-[N:1]-[C:2]-[C:3]-[C:4]-[C:5]-[C:6]-[C:7](=O)-[*]"
)

RXN_OMEGA_AMINO_ACID_GENERIC = AllChem.ReactionFromSmarts(
    "[N:1]-[CH2:2]-[CH2:3]-[CH2:4]-[CH2:5]-[CH2:6]-[C:7](=O)[O;H1]>>[*]-[N:1]-[CH2:2]-[CH2:3]-[CH2:4]-[CH2:5]-[CH2:6]-[C:7](=O)-[*]"
)

RXN_ALKYNE_TO_POLYALKENYLENE = AllChem.ReactionFromSmarts(
    "[C:1]#[C:2]>>[*]-[C:1]=[C:2]-[*]"
)

RXN_IMIDE_FUSED_BICYCLOALKENE_ROMP = AllChem.ReactionFromSmarts(
    "[O:1]=[C:2]1[N:3][C:4](=[O:5])[C:6]2[C:7]3[C:8]=[C:9][C:10]([C:11]3)[C:12]12>>[O:1]=[C:2]1[N:3][C:4](=[O:5])[C:6]2[C:7]([C:8]=[*])[C:10][C:11]([C:9]=[*])[C:12]12"
)

# ---------------------------------------------------------------------------
# Pattern SMARTS
# ---------------------------------------------------------------------------

PATT_TERMINAL_VINYL = Chem.MolFromSmarts("[CH2]=[C]")
PATT_METHACRYLATE = Chem.MolFromSmarts("C=C([#6])-[C](=O)[O,N]")
PATT_ACRYLATE = Chem.MolFromSmarts("[CH2]=[CH]-[C](=O)[O,N]")
PATT_EPOXIDE = Chem.MolFromSmarts("[O;X2;R]1[C;R][C;R]1")
PATT_THF = Chem.MolFromSmarts("[O;X2;R]1[C;R][C;R][C;R][C;R]1")
PATT_KETENE = Chem.MolFromSmarts("C=C=O")
PATT_ALPHA_HA = Chem.MolFromSmarts("[O;H1]-[C]-C(=O)[O;H1]")
PATT_PHENOL = Chem.MolFromSmarts("[O;H1]-a")
PATT_CARBOXYLIC_ACID = Chem.MolFromSmarts("[CX3](=O)[OX2H1]")
PATT_DIOL_1_2_3_ = Chem.MolFromSmarts("[O;H1]-[CX4]-[CX4]-[O;H1]")
PATT_ITACONATE = Chem.MolFromSmarts("[CH2]=[C](-C(=O)O[#6])C(=O)O[#6]")
PATT_LACTAM_R7_EXACT = Chem.MolFromSmarts("O=C1NCCCCC1")
PATT_LACTAM_GENERIC = Chem.MolFromSmarts("[N;R]-[C;R](=O)")
PATT_IMIDE_RING = Chem.MolFromSmarts("[N;R](-[C;R](=O))-[C;R](=O)")
PATT_OMEGA_AMINO_ACID_N6 = Chem.MolFromSmarts("NCCCCCC(=O)O")
PATT_OMEGA_AMINO_ACID_GENERIC = Chem.MolFromSmarts(
    "N-[CH2]-[CH2]-[CH2]-[CH2]-[CH2,$([CH2][CH2])]-C(=O)O"
)
PATT_AMINE_NON_AMIDE = Chem.MolFromSmarts("[N;H1,H2;!$(N-C=O)]")
PATT_ALKYNE_ANY = Chem.MolFromSmarts("[#6]#[#6]")
PATT_IMIDE_FUSED_BICYCLOALKENE_ROMP = Chem.MolFromSmarts(
    "O=C1NC(=O)C2[C]3[C]=[C][C]([C]3)C12"
)
PATT_BICYCLOALKENE_ROMP = Chem.MolFromSmarts("[C]1=[C][C]2[C][C][C]1[C]2")
PATT_ALIPHATIC_DIOL_END = Chem.MolFromSmarts("[O;H1]-[CX4]")
PATT_DIAMINE_END = Chem.MolFromSmarts("[N;H1,H2]-[#6]")
PATT_DIACID_END = Chem.MolFromSmarts("[C:1](=[O:3])[O;H1:2]")
PATT_DIACID_CHLORIDE_END = Chem.MolFromSmarts("[C:1](=[O:3])[Cl:2]")
PATT_ANHYDRIDE_END = Chem.MolFromSmarts("[C:1](=[O:3])[O:2][C:4](=[O:5])")


# ---------------------------------------------------------------------------
# Input normalization and diagnostics
# ---------------------------------------------------------------------------


def _strip_bigsmiles_wrapper(smiles: str) -> str:
    normalized = smiles.strip()
    if normalized.startswith("{") and normalized.endswith("}"):
        return normalized[1:-1].strip()
    return normalized


def _parse_smiles(smiles: str) -> Chem.Mol:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES.")
    return mol


def assess_input(smiles: str) -> InputAssessment:
    normalized = _strip_bigsmiles_wrapper(smiles)
    contains_wildcards = "*" in normalized
    has_bigsmiles_wrapper = smiles.strip().startswith("{") and smiles.strip().endswith("}")

    notes: list[str] = []
    canonical_smiles: str | None = None

    mol = Chem.MolFromSmiles(normalized)
    if mol is None:
        notes.append("RDKit could not parse this SMILES string.")
        input_kind = "invalid"
    else:
        canonical_smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
        if contains_wildcards or has_bigsmiles_wrapper:
            input_kind = "polymer_repeat_or_fragment"
            notes.append(
                "Contains wildcard connection points or a BigSMILES-like wrapper, so this looks like a polymer repeat or fragment rather than a monomer."
            )
        else:
            input_kind = "monomer_candidate"

    return {
        "input_smiles": smiles,
        "normalized_smiles": normalized,
        "canonical_smiles": canonical_smiles,
        "input_kind": input_kind,
        "contains_wildcards": contains_wildcards,
        "has_bigsmiles_wrapper": has_bigsmiles_wrapper,
        "notes": notes,
    }


def _require_monomer_like_input(smiles: str, tool_name: str) -> Chem.Mol:
    assessment = assess_input(smiles)
    if assessment["input_kind"] == "invalid":
        raise ValueError("Invalid SMILES.")
    if assessment["input_kind"] != "monomer_candidate":
        raise ValueError(
            f"`{tool_name}` expects a monomer candidate, but this input looks like a polymer repeat or fragment. Remove wildcard connection points such as `*` before calling this tool."
        )
    return _parse_smiles(assessment["normalized_smiles"])


def _canonicalize_smiles(smiles: str) -> str:
    mol = _parse_smiles(smiles)
    return Chem.MolToSmiles(mol, isomericSmiles=True)


def canonicalize_repeat_smiles(smiles: str) -> str:
    return _canonicalize_smiles(_strip_bigsmiles_wrapper(smiles))


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def first_valid_product(mol: Chem.Mol, rxn: AllChem.ChemicalReaction) -> Chem.Mol | None:
    """Run a reaction and return the first sanitized product that succeeds."""
    for products in rxn.RunReactants((mol,)):
        try:
            product = Chem.Mol(products[0])
            Chem.SanitizeMol(product)
            return product
        except Exception:
            continue
    return None


def count_phenolic_oh(mol: Chem.Mol) -> int:
    return len(mol.GetSubstructMatches(PATT_PHENOL))


def count_carboxylic_acids(mol: Chem.Mol) -> int:
    return len(mol.GetSubstructMatches(PATT_CARBOXYLIC_ACID))


def count_non_amide_amines(mol: Chem.Mol) -> int:
    return len(mol.GetSubstructMatches(PATT_AMINE_NON_AMIDE))


def _unique_match_atom_indices(mol: Chem.Mol, patt: Chem.Mol, atom_pos: int) -> list[int]:
    return sorted({match[atom_pos] for match in mol.GetSubstructMatches(patt)})


def _dummy_neighbor_idx(mol: Chem.Mol, atom_idx: int) -> int:
    dummies = [
        neighbor.GetIdx()
        for neighbor in mol.GetAtomWithIdx(atom_idx).GetNeighbors()
        if neighbor.GetAtomicNum() == 0
    ]
    if len(dummies) != 1:
        raise ValueError("Expected exactly one wildcard endpoint on the reactive atom.")
    return dummies[0]


def _prepare_nucleophile_fragment(
    mol: Chem.Mol, patt: Chem.Mol, atom_pos: int
) -> tuple[Chem.Mol, list[int]] | None:
    sites = _unique_match_atom_indices(mol, patt, atom_pos)
    if len(sites) != 2:
        return None

    rw = Chem.RWMol(Chem.Mol(mol))
    for atom_idx in sites:
        rw.AddBond(atom_idx, rw.AddAtom(Chem.Atom(0)), Chem.BondType.SINGLE)

    fragment = rw.GetMol()
    Chem.SanitizeMol(fragment)
    return fragment, sites


def _prepare_acyl_fragment(
    mol: Chem.Mol, *, allow_anhydride: bool = False
) -> tuple[Chem.Mol, list[int]] | None:
    acid_sites = {match[0]: match[2] for match in mol.GetSubstructMatches(PATT_DIACID_END)}
    chloride_sites = {
        match[0]: match[2] for match in mol.GetSubstructMatches(PATT_DIACID_CHLORIDE_END)
    }
    acyl_sites = {**acid_sites, **chloride_sites}
    if len(acyl_sites) != 2:
        if allow_anhydride:
            return _prepare_anhydride_fragment(mol)
        return None

    rw = Chem.RWMol(Chem.Mol(mol))
    carbonyl_sites = sorted(acyl_sites)
    leaving_indices = sorted(set(acyl_sites.values()), reverse=True)

    for carbonyl_idx in carbonyl_sites:
        rw.AddBond(carbonyl_idx, rw.AddAtom(Chem.Atom(0)), Chem.BondType.SINGLE)

    adjusted_sites = carbonyl_sites[:]
    for leaving_idx in leaving_indices:
        rw.RemoveAtom(leaving_idx)
        adjusted_sites = [
            site_idx - 1 if leaving_idx < site_idx else site_idx
            for site_idx in adjusted_sites
        ]

    fragment = rw.GetMol()
    Chem.SanitizeMol(fragment)
    if len(Chem.GetMolFrags(fragment)) != 1:
        return None
    return fragment, adjusted_sites


def _prepare_anhydride_fragment(mol: Chem.Mol) -> tuple[Chem.Mol, list[int]] | None:
    matches = mol.GetSubstructMatches(PATT_ANHYDRIDE_END)
    if not matches:
        return None

    carbonyl_sites = sorted({match[0] for match in matches} | {match[3] for match in matches})
    bridge_oxygens = {match[2] for match in matches}
    if len(carbonyl_sites) != 2 or len(bridge_oxygens) != 1:
        return None

    rw = Chem.RWMol(Chem.Mol(mol))
    bridge_idx = next(iter(bridge_oxygens))
    for carbonyl_idx in carbonyl_sites:
        rw.RemoveBond(carbonyl_idx, bridge_idx)

    for carbonyl_idx in carbonyl_sites:
        rw.AddBond(carbonyl_idx, rw.AddAtom(Chem.Atom(0)), Chem.BondType.SINGLE)

    rw.RemoveAtom(bridge_idx)
    adjusted_sites = [site_idx - 1 if bridge_idx < site_idx else site_idx for site_idx in carbonyl_sites]

    fragment = rw.GetMol()
    Chem.SanitizeMol(fragment)
    if len(Chem.GetMolFrags(fragment)) != 1:
        return None
    return fragment, adjusted_sites


def _join_bifunctional_fragments(
    nucleophile_fragment: Chem.Mol,
    nucleophile_sites: list[int],
    acyl_fragment: Chem.Mol,
    acyl_sites: list[int],
) -> Chem.Mol:
    combo = Chem.CombineMols(nucleophile_fragment, acyl_fragment)
    rw = Chem.RWMol(combo)
    offset = nucleophile_fragment.GetNumAtoms()

    nucleophile_idx = nucleophile_sites[0]
    acyl_idx = acyl_sites[0] + offset
    nucleophile_dummy = _dummy_neighbor_idx(rw, nucleophile_idx)
    acyl_dummy = _dummy_neighbor_idx(rw, acyl_idx)

    rw.AddBond(nucleophile_idx, acyl_idx, Chem.BondType.SINGLE)
    for atom_idx in sorted((nucleophile_dummy, acyl_dummy), reverse=True):
        rw.RemoveAtom(atom_idx)

    product = rw.GetMol()
    Chem.SanitizeMol(product)
    return product


def _build_step_growth_repeat(
    monomer_a: Chem.Mol,
    monomer_b: Chem.Mol,
    nucleophile_patt: Chem.Mol,
    nucleophile_name: str,
    acyl_partner_name: str,
    *,
    allow_anhydride: bool = False,
) -> Chem.Mol:
    attempts = (
        (
            _prepare_nucleophile_fragment(monomer_a, nucleophile_patt, 0),
            _prepare_acyl_fragment(monomer_b, allow_anhydride=allow_anhydride),
        ),
        (
            _prepare_nucleophile_fragment(monomer_b, nucleophile_patt, 0),
            _prepare_acyl_fragment(monomer_a, allow_anhydride=allow_anhydride),
        ),
    )

    for nucleophile, acyl in attempts:
        if nucleophile is None or acyl is None:
            continue
        return _join_bifunctional_fragments(
            nucleophile[0], nucleophile[1], acyl[0], acyl[1]
        )

    raise ValueError(
        f"This strategy requires one {nucleophile_name} and one {acyl_partner_name} comonomer."
    )


def same_aromatic_component(mol: Chem.Mol) -> bool:
    """Check whether two phenolic OH groups sit on the same aromatic component."""
    phenol_matches = mol.GetSubstructMatches(PATT_PHENOL)
    if len(phenol_matches) != 2:
        return False

    _, atom_a = phenol_matches[0]
    _, atom_b = phenol_matches[1]
    aromatic_atoms = {idx for (idx,) in mol.GetSubstructMatches(Chem.MolFromSmarts("a"))}

    visited = {atom_a}
    stack = [atom_a]
    while stack:
        current = stack.pop()
        if current == atom_b:
            return True
        atom = mol.GetAtomWithIdx(current)
        for neighbor in atom.GetNeighbors():
            neighbor_idx = neighbor.GetIdx()
            if neighbor_idx in aromatic_atoms and neighbor_idx not in visited:
                visited.add(neighbor_idx)
                stack.append(neighbor_idx)
    return False


def wrap_bigsmiles_like(smiles_with_wildcards: str) -> str:
    return "{" + smiles_with_wildcards + "}"


def _normalize_strategy_name(strategy: str) -> str:
    if strategy == "polyacetylene":
        return "alkyne"
    return strategy


def comonomers_to_repeat_smiles(
    monomer_a_smiles: str, monomer_b_smiles: str, strategy: str
) -> str:
    """Build a repeat unit from an explicit two-monomer step-growth strategy."""
    monomer_a = _require_monomer_like_input(
        monomer_a_smiles, "comonomers_to_repeat_smiles"
    )
    monomer_b = _require_monomer_like_input(
        monomer_b_smiles, "comonomers_to_repeat_smiles"
    )
    strategy = _normalize_strategy_name(strategy)

    if strategy == "cond_polyester":
        product = _build_step_growth_repeat(
            monomer_a,
            monomer_b,
            PATT_ALIPHATIC_DIOL_END,
            "diol",
            "diacid, diacid chloride, or suitable cyclic anhydride",
            allow_anhydride=True,
        )
    elif strategy == "cond_polyamide":
        product = _build_step_growth_repeat(
            monomer_a,
            monomer_b,
            PATT_DIAMINE_END,
            "diamine",
            "diacid or diacid chloride",
        )
    else:
        raise NotImplementedError(f"Unknown two-monomer strategy '{strategy}'.")

    return Chem.MolToSmiles(product, isomericSmiles=True)


# ---------------------------------------------------------------------------
# Strategy application
# ---------------------------------------------------------------------------


def run_acrylate_head_to_tail(mol: Chem.Mol) -> Chem.Mol | None:
    if mol.HasSubstructMatch(PATT_METHACRYLATE):
        product = first_valid_product(mol, RXN_METHACRYLATE_HT)
        if product is not None:
            return product

    if mol.HasSubstructMatch(PATT_ACRYLATE):
        product = first_valid_product(mol, RXN_ACRYLATE_HT)
        if product is not None:
            return product

    return None


def _apply_rop_lactam(mol: Chem.Mol) -> Chem.Mol | None:
    if mol.HasSubstructMatch(PATT_IMIDE_RING):
        return None
    if mol.HasSubstructMatch(PATT_LACTAM_R7_EXACT):
        product = first_valid_product(mol, RXN_LACTAM_R7_TO_N6)
        if product is not None:
            return product
    if mol.HasSubstructMatch(PATT_LACTAM_GENERIC):
        return first_valid_product(mol, RXN_LACTAM_GENERIC)
    return None


def _apply_romp_bicyclic_alkene(mol: Chem.Mol) -> Chem.Mol | None:
    if mol.HasSubstructMatch(PATT_IMIDE_FUSED_BICYCLOALKENE_ROMP):
        product = first_valid_product(mol, RXN_IMIDE_FUSED_BICYCLOALKENE_ROMP)
        if product is not None:
            return product

    for match in mol.GetSubstructMatches(PATT_BICYCLOALKENE_ROMP):
        atom_1, atom_2, *_ = match
        rw = Chem.RWMol(Chem.Mol(mol))
        alkene_bond = rw.GetBondBetweenAtoms(atom_1, atom_2)
        if alkene_bond is None or alkene_bond.GetBondType() != Chem.BondType.DOUBLE:
            continue

        rw.RemoveBond(atom_1, atom_2)
        rw.AddBond(atom_1, rw.AddAtom(Chem.Atom(0)), Chem.BondType.DOUBLE)
        rw.AddBond(atom_2, rw.AddAtom(Chem.Atom(0)), Chem.BondType.DOUBLE)

        product = rw.GetMol()
        try:
            Chem.SanitizeMol(product)
        except Exception:
            continue
        return product
    return None


def _apply_cond_omega_amino_acid(mol: Chem.Mol) -> Chem.Mol | None:
    if mol.HasSubstructMatch(PATT_OMEGA_AMINO_ACID_N6):
        product = first_valid_product(mol, RXN_OMEGA_AMINO_ACID_N6)
        if product is not None:
            return product
    return first_valid_product(mol, RXN_OMEGA_AMINO_ACID_GENERIC)


def _apply_cond_diphenol(mol: Chem.Mol) -> Chem.Mol | None:
    product = first_valid_product(mol, RXN_DIPHENOL_HQ)
    if product is not None:
        return product

    if count_phenolic_oh(mol) == 2 and same_aromatic_component(mol):
        working = Chem.Mol(mol)
        for _ in range(2):
            product = first_valid_product(working, RXN_DIPHENOL_GENERIC_ONE_END)
            if product is None:
                return None
            working = product
        if count_phenolic_oh(working) == 0:
            return working
    return None


def monomer_to_repeat_smiles(monomer_smiles: str, strategy: str) -> str:
    """
    Transform a monomer into a polymer repeat-unit SMILES with wildcard endpoints `*`.
    """
    mol = _require_monomer_like_input(monomer_smiles, "monomer_to_repeat_smiles")
    strategy = _normalize_strategy_name(strategy)

    transformers = {
        "vinyl": lambda molecule: first_valid_product(molecule, RXN_VINYL_HT),
        "acrylate": run_acrylate_head_to_tail,
        "alkyne": lambda molecule: first_valid_product(molecule, RXN_ALKYNE_TO_POLYALKENYLENE),
        "romp_bicyclic_alkene": _apply_romp_bicyclic_alkene,
        "rop_thf": lambda molecule: first_valid_product(molecule, RXN_ROP_THF),
        "rop_epoxide": lambda molecule: first_valid_product(molecule, RXN_ROP_EPOXIDE),
        "ketene": lambda molecule: first_valid_product(molecule, RXN_KETENE_TO_PEO),
        "cond_alpha_hydroxy_acid": lambda molecule: first_valid_product(molecule, RXN_COND_ALPHA_HA),
        "rop_lactam": _apply_rop_lactam,
        "cond_omega_amino_acid": _apply_cond_omega_amino_acid,
        "cond_diphenol": _apply_cond_diphenol,
    }

    if strategy not in transformers:
        valid = ", ".join(sorted(transformers))
        raise NotImplementedError(f"Unknown strategy '{strategy}'. Valid strategies: {valid}.")

    product = transformers[strategy](mol)
    if product is None:
        raise ValueError(f"No applicable polymerization site found for strategy '{strategy}'.")

    return Chem.MolToSmiles(product, isomericSmiles=True)


# ---------------------------------------------------------------------------
# Rule suggestion and automatic selection
# ---------------------------------------------------------------------------


def suggest_polymerization_rules(monomer_smiles: str) -> list[Suggestion]:
    """
    Inspect a monomer and suggest plausible single-monomer polymerization rules.
    """
    mol = _require_monomer_like_input(monomer_smiles, "suggest_rules")

    phenolic_oh_count = count_phenolic_oh(mol)
    carboxylic_acid_count = count_carboxylic_acids(mol)
    amine_count = count_non_amide_amines(mol)

    suggestions: list[Suggestion] = []

    if mol.HasSubstructMatch(PATT_ITACONATE):
        suggestions.append(
            Suggestion(
                "acrylate",
                0.93,
                "Itaconate diester detected; vinyl head-to-tail polymerization at the CH2 is the most likely single-monomer transform.",
            )
        )

    if mol.HasSubstructMatch(PATT_EPOXIDE):
        suggestions.append(
            Suggestion(
                "rop_epoxide",
                0.95,
                "Epoxide ring detected; ring-opening polymerization gives a poly(alkylene oxide)-like repeat.",
            )
        )

    if mol.HasSubstructMatch(PATT_LACTAM_GENERIC) and not mol.HasSubstructMatch(PATT_IMIDE_RING):
        suggestions.append(
            Suggestion(
                "rop_lactam",
                0.92,
                "Lactam ring detected; ring-opening polymerization to a polyamide is plausible.",
            )
        )

    if mol.HasSubstructMatch(PATT_OMEGA_AMINO_ACID_GENERIC):
        suggestions.append(
            Suggestion(
                "cond_omega_amino_acid",
                0.88,
                "Omega-amino acid detected; A-B self-condensation to a polyamide is plausible.",
            )
        )

    if mol.HasSubstructMatch(PATT_THF):
        suggestions.append(
            Suggestion(
                "rop_thf",
                0.85,
                "A saturated 5-member cyclic ether was detected; ring-opening polymerization is plausible.",
            )
        )

    if mol.HasSubstructMatch(PATT_ALKYNE_ANY):
        suggestions.append(
            Suggestion(
                "alkyne",
                0.88,
                "An alkyne was detected; a poly(alkenylene)-like repeat is a plausible conceptual transform.",
            )
        )

    if mol.HasSubstructMatch(PATT_BICYCLOALKENE_ROMP):
        suggestions.append(
            Suggestion(
                "romp_bicyclic_alkene",
                0.94,
                "A strained bicyclic alkene motif was detected; ring-opening metathesis polymerization is plausible.",
            )
        )

    if mol.HasSubstructMatch(PATT_METHACRYLATE):
        suggestions.append(
            Suggestion(
                "acrylate",
                0.91,
                "A methacrylate-like alpha,beta-unsaturated carbonyl was detected; chain-growth polymerization at the alkene is plausible.",
            )
        )

    if mol.HasSubstructMatch(PATT_ACRYLATE):
        suggestions.append(
            Suggestion(
                "acrylate",
                0.90,
                "An alpha,beta-unsaturated carbonyl was detected; chain-growth polymerization at the alkene is plausible.",
            )
        )

    if mol.HasSubstructMatch(PATT_TERMINAL_VINYL):
        suggestions.append(
            Suggestion(
                "vinyl",
                0.70,
                "A terminal vinyl group is present; generic chain-growth polymerization is plausible.",
            )
        )

    if mol.HasSubstructMatch(PATT_KETENE):
        suggestions.append(
            Suggestion(
                "ketene",
                0.80,
                "A ketene fragment was detected; a conceptual oxyethylene-like repeat can be generated.",
            )
        )

    if mol.HasSubstructMatch(PATT_ALPHA_HA):
        suggestions.append(
            Suggestion(
                "cond_alpha_hydroxy_acid",
                0.75,
                "An alpha-hydroxy acid motif was detected; self-condensation to a polyester-like repeat is plausible.",
            )
        )

    if phenolic_oh_count == 2 and same_aromatic_component(mol):
        suggestions.append(
            Suggestion(
                "cond_diphenol",
                0.70,
                "Two phenolic OH groups are present on the same aromatic component; a self-condensed arylene ether-like repeat is plausible.",
            )
        )

    if phenolic_oh_count >= 2 and carboxylic_acid_count >= 2:
        suggestions.append(
            Suggestion(
                "multifunctional_hydroxy_diacid",
                0.97,
                "An aromatic hydroxy-diacid was detected. It can self-condense without a separate comonomer, but the multiple OH and COOH groups imply branching or network formation rather than one unique linear repeat unit.",
            )
        )

    if carboxylic_acid_count >= 2 and phenolic_oh_count < 2:
        suggestions.append(
            Suggestion(
                "needs_comonomer",
                0.96,
                "A diacid-like motif was detected. This usually needs a diol or diamine comonomer to define a linear polyester or polyamide repeat.",
            )
        )

    if amine_count >= 2 and carboxylic_acid_count == 0:
        suggestions.append(
            Suggestion(
                "needs_comonomer",
                0.95,
                "A diamine-like motif was detected. This usually needs a diacid comonomer to define a linear polyamide repeat.",
            )
        )

    if mol.HasSubstructMatch(PATT_DIOL_1_2_3_):
        suggestions.append(
            Suggestion(
                "needs_comonomer",
                0.95,
                "A diol-like motif was detected. This usually needs a diacid, diisocyanate, or dihalide comonomer to define a linear repeat.",
            )
        )

    best_by_strategy: dict[str, Suggestion] = {}
    for suggestion in suggestions:
        current = best_by_strategy.get(suggestion.strategy)
        if current is None or suggestion.confidence > current.confidence:
            best_by_strategy[suggestion.strategy] = suggestion

    return sorted(best_by_strategy.values(), key=lambda item: item.confidence, reverse=True)


def suggest_copolymerization_rules(
    monomer_a_smiles: str, monomer_b_smiles: str
) -> list[Suggestion]:
    """Inspect a monomer pair and suggest explicit two-monomer step-growth rules."""
    monomer_a = _require_monomer_like_input(monomer_a_smiles, "suggest_copolymer_rules")
    monomer_b = _require_monomer_like_input(monomer_b_smiles, "suggest_copolymer_rules")

    suggestions: list[Suggestion] = []
    has_diol = _prepare_nucleophile_fragment(monomer_a, PATT_ALIPHATIC_DIOL_END, 0) or _prepare_nucleophile_fragment(monomer_b, PATT_ALIPHATIC_DIOL_END, 0)
    has_diamine = _prepare_nucleophile_fragment(monomer_a, PATT_DIAMINE_END, 0) or _prepare_nucleophile_fragment(monomer_b, PATT_DIAMINE_END, 0)
    has_polyester_acyl = _prepare_acyl_fragment(monomer_a, allow_anhydride=True) or _prepare_acyl_fragment(monomer_b, allow_anhydride=True)
    has_polyamide_acyl = _prepare_acyl_fragment(monomer_a) or _prepare_acyl_fragment(monomer_b)

    if has_diol and has_polyester_acyl:
        suggestions.append(
            Suggestion(
                "cond_polyester",
                0.96,
                "Diol plus diacid, diacid chloride, or suitable cyclic anhydride pair detected; step-growth polyester condensation is plausible.",
            )
        )

    if has_diamine and has_polyamide_acyl:
        suggestions.append(
            Suggestion(
                "cond_polyamide",
                0.95,
                "Diamine plus diacid or diacid chloride pair detected; step-growth polyamide condensation is plausible.",
            )
        )

    return suggestions


def choose_strategy_auto(
    monomer_smiles: str,
    min_confidence: float = 0.80,
    allow_fallback_to_lower_confidence: bool = True,
) -> tuple[str, str]:
    """
    Choose the most plausible single-monomer polymerization strategy.
    """
    _require_monomer_like_input(monomer_smiles, "polymerize_auto")
    ranked = suggest_polymerization_rules(monomer_smiles)

    candidates = [suggestion for suggestion in ranked if suggestion.confidence >= min_confidence]
    if not candidates and allow_fallback_to_lower_confidence:
        candidates = [suggestion for suggestion in ranked if suggestion.confidence >= 0.65]

    if not candidates:
        if ranked:
            summary = ", ".join(
                f"{suggestion.strategy}({suggestion.confidence:.2f})" for suggestion in ranked[:3]
            )
            raise ValueError(
                f"No single-monomer rule met the confidence threshold. Top suggestions were: {summary}."
            )
        raise ValueError(
            "No supported single-monomer polymerization motif was detected. If this is a step-growth monomer, you may need to reason about a comonomer pair instead of a single monomer transform."
        )

    top_confidence = candidates[0].confidence
    nearly_tied = [
        suggestion for suggestion in candidates if abs(suggestion.confidence - top_confidence) < 0.05
    ]
    real_rules = [
        suggestion
        for suggestion in nearly_tied
        if suggestion.strategy not in GUIDANCE_ONLY_STRATEGIES
    ]
    if not real_rules and candidates[0].strategy in GUIDANCE_ONLY_STRATEGIES:
        raise ValueError(candidates[0].reason)

    candidates = [
        suggestion
        for suggestion in candidates
        if suggestion.strategy not in GUIDANCE_ONLY_STRATEGIES
    ] or candidates

    best_specificity = max(SPECIFICITY_ORDER.get(suggestion.strategy, 0) for suggestion in candidates)
    specific = [
        suggestion
        for suggestion in candidates
        if SPECIFICITY_ORDER.get(suggestion.strategy, 0) == best_specificity
    ]
    specific.sort(key=lambda suggestion: suggestion.confidence, reverse=True)

    if len(specific) > 1 and abs(specific[0].confidence - specific[1].confidence) < 0.03:
        names = ", ".join(
            f"{suggestion.strategy}({suggestion.confidence:.2f})" for suggestion in specific[:3]
        )
        raise ValueError(
            f"Ambiguous polymerization class candidates: {names}. Please specify a strategy explicitly."
        )

    winner = specific[0]
    return winner.strategy, winner.reason


def monomer_to_repeat_auto(
    monomer_smiles: str,
    min_confidence: float = 0.80,
    allow_fallback_to_lower_confidence: bool = True,
) -> tuple[str, str, str]:
    strategy, reason = choose_strategy_auto(
        monomer_smiles,
        min_confidence=min_confidence,
        allow_fallback_to_lower_confidence=allow_fallback_to_lower_confidence,
    )
    repeat = monomer_to_repeat_smiles(monomer_smiles, strategy=strategy)
    return repeat, strategy, reason


def choose_pair_strategy_auto(
    monomer_a_smiles: str,
    monomer_b_smiles: str,
    min_confidence: float = 0.80,
    allow_fallback_to_lower_confidence: bool = True,
) -> tuple[str, str]:
    ranked = suggest_copolymerization_rules(monomer_a_smiles, monomer_b_smiles)
    candidates = [suggestion for suggestion in ranked if suggestion.confidence >= min_confidence]
    if not candidates and allow_fallback_to_lower_confidence:
        candidates = [suggestion for suggestion in ranked if suggestion.confidence >= 0.65]

    if not candidates:
        raise ValueError(
            "No suitable two-monomer polymerization rule detected for this monomer set."
        )

    top_confidence = candidates[0].confidence
    top_like = [
        suggestion for suggestion in candidates if abs(suggestion.confidence - top_confidence) < 0.05
    ]
    if len(top_like) > 1 and abs(top_like[0].confidence - top_like[1].confidence) < 0.03:
        names = ", ".join(
            f"{suggestion.strategy}({suggestion.confidence:.2f})" for suggestion in top_like[:3]
        )
        raise ValueError(
            f"Ambiguous two-monomer polymerization candidates: {names}. Please specify a strategy explicitly."
        )

    winner = candidates[0]
    return winner.strategy, winner.reason


def monomer_pair_to_repeat_auto(
    monomer_a_smiles: str,
    monomer_b_smiles: str,
    min_confidence: float = 0.80,
    allow_fallback_to_lower_confidence: bool = True,
) -> tuple[str, str, str]:
    strategy, reason = choose_pair_strategy_auto(
        monomer_a_smiles,
        monomer_b_smiles,
        min_confidence=min_confidence,
        allow_fallback_to_lower_confidence=allow_fallback_to_lower_confidence,
    )
    repeat = comonomers_to_repeat_smiles(
        monomer_a_smiles, monomer_b_smiles, strategy=strategy
    )
    return repeat, strategy, reason


def _normalize_monomer_inputs(monomer_smiles: str | list[str]) -> list[str]:
    if isinstance(monomer_smiles, str):
        return [monomer_smiles]
    if not isinstance(monomer_smiles, list) or not all(
        isinstance(item, str) for item in monomer_smiles
    ):
        raise ValueError("`monomer_smiles` must be a SMILES string or a list of SMILES strings.")
    if not monomer_smiles:
        raise ValueError("`monomer_smiles` must contain one or two SMILES strings.")
    if len(monomer_smiles) > 2:
        raise ValueError("`monomer_smiles` must contain at most two SMILES strings.")
    return monomer_smiles


# ---------------------------------------------------------------------------
# Agent-facing tool functions
# ---------------------------------------------------------------------------


def polymerize_auto(
    monomer_smiles: str | list[str],
    min_confidence: float = 0.80,
    allow_fallback_to_lower_confidence: bool = True,
    bigsmiles_wrap: bool = False,
) -> PolymerizeResult:
    monomer_inputs = _normalize_monomer_inputs(monomer_smiles)
    if len(monomer_inputs) == 1:
        rep, strategy, rationale = monomer_to_repeat_auto(
            monomer_inputs[0],
            min_confidence=min_confidence,
            allow_fallback_to_lower_confidence=allow_fallback_to_lower_confidence,
        )
    else:
        rep, strategy, rationale = monomer_pair_to_repeat_auto(
            monomer_inputs[0],
            monomer_inputs[1],
            min_confidence=min_confidence,
            allow_fallback_to_lower_confidence=allow_fallback_to_lower_confidence,
        )
    repeat_out = wrap_bigsmiles_like(rep) if bigsmiles_wrap else rep
    return {
        "repeat_smiles": repeat_out,
        "strategy": strategy,
        "rationale": rationale,
    }


def polymerize_explicit(
    monomer_smiles: str,
    strategy: StrategyName,
    comonomer_smiles: str | None = None,
    bigsmiles_wrap: bool = False,
) -> str:
    normalized_strategy = _normalize_strategy_name(strategy)
    if normalized_strategy in {"cond_polyester", "cond_polyamide"}:
        if comonomer_smiles is None:
            raise ValueError(f"Strategy '{normalized_strategy}' requires a second monomer.")
        repeat = comonomers_to_repeat_smiles(
            monomer_smiles, comonomer_smiles, strategy=normalized_strategy
        )
    else:
        repeat = monomer_to_repeat_smiles(monomer_smiles, strategy=normalized_strategy)
    return wrap_bigsmiles_like(repeat) if bigsmiles_wrap else repeat


def suggest_rules(monomer_smiles: str | list[str], top_k: int = 5) -> list[SuggestionDict]:
    monomer_inputs = _normalize_monomer_inputs(monomer_smiles)
    if len(monomer_inputs) == 1:
        ranked = suggest_polymerization_rules(monomer_inputs[0])
    else:
        ranked = suggest_copolymerization_rules(monomer_inputs[0], monomer_inputs[1])
    return [
        {
            "strategy": suggestion.strategy,
            "confidence": float(suggestion.confidence),
            "reason": suggestion.reason,
        }
        for suggestion in ranked[:top_k]
    ]


def suggest_copolymer_rules(
    monomer_a_smiles: str, monomer_b_smiles: str, top_k: int = 5
) -> list[SuggestionDict]:
    ranked = suggest_copolymerization_rules(monomer_a_smiles, monomer_b_smiles)
    return [
        {
            "strategy": suggestion.strategy,
            "confidence": float(suggestion.confidence),
            "reason": suggestion.reason,
        }
        for suggestion in ranked[:top_k]
    ]


def check_retrosynthesis_candidate(
    target_polymer_smiles: str,
    candidate_monomer_smiles: str,
    strategy: str | None = None,
    min_confidence: float = 0.80,
    allow_fallback_to_lower_confidence: bool = True,
) -> RetrosynthesisCheckResult:
    """
    Check whether a candidate monomer reproduces a target polymer repeat.

    This is designed for agent workflows: given a target polymer repeat and a
    proposed monomer, return a structured match/no-match result instead of making
    the caller infer success from exceptions.
    """
    notes: list[str] = []
    target_assessment = assess_input(target_polymer_smiles)
    candidate_assessment = assess_input(candidate_monomer_smiles)

    target_canonical: str | None = None
    predicted_repeat: str | None = None
    predicted_canonical: str | None = None
    strategy_used: str | None = None
    rationale: str | None = None

    try:
        target_canonical = canonicalize_repeat_smiles(target_polymer_smiles)
    except Exception as exc:
        notes.append(f"Could not parse target polymer repeat: {exc}")

    if target_assessment["input_kind"] == "monomer_candidate":
        notes.append(
            "The target input does not contain wildcard connection points. Matching is more reliable when the target is a repeat unit with `*` endpoints."
        )

    try:
        if strategy:
            strategy_used = _normalize_strategy_name(strategy)
            predicted_repeat = polymerize_explicit(
                candidate_monomer_smiles,
                strategy=strategy_used,
                bigsmiles_wrap=False,
            )
            rationale = f"Applied the explicit strategy '{strategy_used}'."
        else:
            auto_result = polymerize_auto(
                candidate_monomer_smiles,
                min_confidence=min_confidence,
                allow_fallback_to_lower_confidence=allow_fallback_to_lower_confidence,
                bigsmiles_wrap=False,
            )
            predicted_repeat = auto_result["repeat_smiles"]
            strategy_used = auto_result["strategy"]
            rationale = auto_result["rationale"]
    except Exception as exc:
        notes.append(f"Candidate monomer could not be polymerized into a comparable repeat: {exc}")

    if predicted_repeat is not None:
        try:
            predicted_canonical = canonicalize_repeat_smiles(predicted_repeat)
        except Exception as exc:
            notes.append(f"Could not canonicalize predicted repeat: {exc}")

    is_match = (
        target_canonical is not None
        and predicted_canonical is not None
        and target_canonical == predicted_canonical
    )

    if predicted_repeat is not None and target_canonical and predicted_canonical and not is_match:
        notes.append(
            "The predicted repeat does not match the target repeat after canonicalization."
        )

    if candidate_assessment["input_kind"] != "monomer_candidate":
        notes.append(
            "The candidate input already looks like a polymer repeat or fragment, so it is not a valid monomer hypothesis for this checker."
        )

    return {
        "is_match": is_match,
        "target_repeat_smiles": _strip_bigsmiles_wrapper(target_polymer_smiles),
        "canonical_target_repeat_smiles": target_canonical,
        "candidate_monomer_smiles": candidate_monomer_smiles,
        "predicted_repeat_smiles": predicted_repeat,
        "canonical_predicted_repeat_smiles": predicted_canonical,
        "strategy_used": strategy_used,
        "rationale": rationale,
        "notes": notes,
    }


if __name__ == "__main__":
    tests = [
        ("C=CC1=CC=CC=C1", "styrene"),
        ("CC(=C)C(=O)OC", "methyl methacrylate"),
        ("C1CCOC1", "THF"),
        ("C1CO1", "ethylene oxide"),
        ("C=C=O", "ketene"),
        ("CC(O)C(=O)O", "lactic acid"),
        ("Oc1ccc(O)cc1", "hydroquinone"),
        ("C=C(CC(=O)OCCCC)C(=O)OCCCC", "itaconate"),
        ("O=C1NCCCCC1", "caprolactam"),
        ("NCCCCCC(=O)O", "6-aminohexanoic acid"),
        ("C#Cc1ccccc1", "aryl alkyne"),
        ("C1(C=C2)CC2CC1", "bicyclic alkene"),
        ("O=C1C2C(C(N1)=O)C3C=CC2C3", "imide-fused bicyclic alkene"),
        ("OCCO", "ethylene glycol"),
        (
            "O=C(O)c1c(O)c(C(=O)O)c(O)c([N+](=O)[O-])c1O",
            "multifunctional hydroxy-diacid",
        ),
    ]

    pair_tests = [
        ("OCCO", "O=C(O)c1ccc(C(=O)O)cc1", "cond_polyester", "PET-like"),
        ("OCCO", "O=C(Cl)c1ccc(C(=O)Cl)cc1", "cond_polyester", "acid chloride polyester"),
        ("OCCO", "O=C1OC(=O)c2ccccc21", "cond_polyester", "anhydride polyester"),
        ("NCCN", "O=C(O)c1ccc(C(=O)O)cc1", "cond_polyamide", "diacid polyamide"),
        ("NCCN", "O=C(Cl)c1ccc(C(=O)Cl)cc1", "cond_polyamide", "aramid-like"),
    ]

    for smiles, name in tests:
        try:
            result = polymerize_auto(smiles)
            print(
                f"{name:32s} -> {result['repeat_smiles']:28s} via {result['strategy']:24s} | {result['rationale']}"
            )
        except Exception as exc:
            print(f"{name:32s} -> ERROR: {exc}")

    for left, right, strategy, name in pair_tests:
        try:
            repeat = polymerize_explicit(
                left, strategy=strategy, comonomer_smiles=right, bigsmiles_wrap=False
            )
            print(f"{name:32s} -> {repeat:28s} via {strategy}")
        except Exception as exc:
            print(f"{name:32s} -> ERROR: {exc}")
