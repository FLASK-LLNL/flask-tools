from __future__ import annotations

from textwrap import dedent

from rdkit.Chem import MolFromSmiles
from rdkit.Chem.inchi import MolToInchiKey

from flask_tools.pipette.verifiers.reaction_energy import (
    MoleculeEnergyStore,
)


# Should this be removed completely b/c DFT calculation is not implemented rn?
def test_molecule_energy_store_from_csv(tmp_path) -> None:
    smiles_to_inchi_key = lambda x: MolToInchiKey(MolFromSmiles(x))
    path = tmp_path / "fake_molecule_energies.csv"
    ethanol_inchi = smiles_to_inchi_key(eth_smi := "CCO")
    acetaldehyde_inchi = smiles_to_inchi_key(acetal_smi := "CC=O")
    path.write_text(
        dedent(
            f"""
            inchi_key,energy_ev_mol
            {ethanol_inchi},-7.0
            {acetaldehyde_inchi},-10.0
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
