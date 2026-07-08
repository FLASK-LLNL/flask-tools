###############################################################################
## Copyright 2025-2026 Lawrence Livermore National Security, LLC.
## See the top-level LICENSE file for details.
##
## SPDX-License-Identifier: Apache-2.0
###############################################################################

from __future__ import annotations

from textwrap import dedent

from rdkit.Chem import MolFromSmiles
from rdkit.Chem.inchi import MolToInchiKey, MolToInchi

from flask_tools.pipette.verifiers.reaction_energy import (
    MoleculeEnergyStore,
)


# Should this be removed completely b/c DFT calculation is not implemented rn?
def test_molecule_energy_store_from_csv(tmp_path) -> None:
    smiles_to_inchi = lambda x: MolToInchi(MolFromSmiles(x))
    path = tmp_path / "fake_molecule_energies.tsv"
    ethanol_inchi = smiles_to_inchi(eth_smi := "CCO")
    acetaldehyde_inchi = smiles_to_inchi(acetal_smi := "CC=O")
    path.write_text(
        dedent(
            f"""
            inchi\tenergy_ev_mol
            {ethanol_inchi}\t-7.0
            {acetaldehyde_inchi}\t-10.0
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )

    store = MoleculeEnergyStore.from_csv(path)
    reactants_energy = store.lookup(eth_smi).energy_ev
    products_energy = store.lookup(acetal_smi).energy_ev
    assert reactants_energy == -7.0
    assert products_energy - reactants_energy == -3.0
