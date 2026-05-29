from __future__ import annotations

from textwrap import dedent

from pipette.verifiers.reaction_energy import (
    MoleculeEnergyStore,
    ReactionEnergyChecker,
    ReactionEnergyRecord,
    MoleculeEnergyRecord,
)
from pipette.config import PipetteConfig, RulesConfig
from pipette.constants import ToolStatus


class EmptyDFTExecutor:
    """Used in palce of FakeDFTExecutor b/c reaction_energy.py has special logic for returning passing values for
    reactions when using FakeDFTExecutor"""

    def run(self, rxn_smiles: str) -> MoleculeEnergyRecord:
        raise ValueError(
            "Test case should not even get to DFT call, all mols should be cached"
        )


def _make_config() -> PipetteConfig:
    return PipetteConfig()


def _config_with_fake_dft_enabled() -> PipetteConfig:
    return PipetteConfig(rules=RulesConfig(enable_fake_dft=True))


def test_csv_molecule_energy_store_looks_up_reaction_energy(
    monkeypatch, tmp_path
) -> None:
    path = tmp_path / "fake_molecule_energies.csv"
    path.write_text(
        dedent(
            """
            inchi_key,energy_ev_mol
            ETHANOL,-7.0
            ACETALDEHYDE,-10.0
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "pipette.verifiers.reaction_energy.smiles_to_inchi_key",
        lambda smiles: {
            "CCO": "ETHANOL",
            "CC=O": "ACETALDEHYDE",
        }[smiles],
    )

    store = MoleculeEnergyStore.from_csv(path)
    reactants_energy = store.lookup("CCO").energy_ev
    products_energy = store.lookup("CC=O").energy_ev
    assert reactants_energy == -7.0

    assert products_energy - reactants_energy == -3.0


def test_reaction_energy_checker_uses_csv_database_path(monkeypatch, tmp_path) -> None:
    path = tmp_path / "fake_molecule_energies.csv"
    path.write_text(
        dedent(
            """
            inchi_key,energy_ev_mol
            ETHANOL,-7.0
            ACETALDEHYDE,-10.0
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "pipette.verifiers.reaction_energy.smiles_to_inchi_key",
        lambda smiles: {
            "CCO": "ETHANOL",
            "CC=O": "ACETALDEHYDE",
        }[smiles],
    )

    checker = ReactionEnergyChecker(
        _make_config(),
        database=str(path),
        dft_executor=EmptyDFTExecutor(),
    )
    result = checker.run("CCO>>CC=O", {})

    assert checker.database == str(path)
    assert result.status is ToolStatus.PASS, result
    assert result.data["energy_difference_ev_mol"] == -3.0


def test_reaction_energy_checker_uses_fake_dft_fallback(monkeypatch) -> None:
    sleep_calls: list[float] = []
    monkeypatch.setattr(
        "pipette.verifiers.reaction_energy.time.sleep",
        lambda seconds: sleep_calls.append(seconds),
    )

    checker = ReactionEnergyChecker(_config_with_fake_dft_enabled())
    result = checker.run("CCO>>CC=O", {})

    assert (
        sleep_calls == []
    ), "Using fake dft should by pass molecule energy DFT calculations"
    assert result.status is ToolStatus.PASS
    assert result.data["energy_difference_ev_mol"] == -100
    assert result.data["source"] == None


def test_molecule_energy_store_save_updates_csv_cache(tmp_path) -> None:
    path = tmp_path / "fake_molecule_energies.csv"
    path.write_text(
        dedent(
            """
            inchi_key,smiles,energy_ev_mol
            EXISTING,, -1.0
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )

    store = MoleculeEnergyStore.from_csv(path)
    smi_inchis = [
        (
            "CCO",
            "LFQSCWFLJHTTHZ-UHFFFAOYSA-N",
        ),
        (
            "CC=C",
            "QQONPFPTGQHPMA-UHFFFAOYSA-N",
        ),
    ]
    store.save(
        [
            MoleculeEnergyRecord(
                smi=smi_inchis[0][0],
                energy_ev=-1.0,
                source="molecule_energy_csv",
            ),
            MoleculeEnergyRecord(
                smi=smi_inchis[1][0],
                energy_ev=-2.0,
                source="molecule_energy_csv",
            ),
        ]
    )

    assert store.energies_by_inchi_key[smi_inchis[0][1]] == -1.0
    assert store.energies_by_inchi_key[smi_inchis[1][1]] == -2.0
    rows = path.read_text(encoding="utf-8").splitlines()
    assert rows == [
        "inchi_key,smiles,energy_ev_mol",
        "EXISTING,,-1.0",
        f"{smi_inchis[0][1]},CCO,-1.0",
        f"{smi_inchis[1][1]},CC=C,-2.0",
    ]
