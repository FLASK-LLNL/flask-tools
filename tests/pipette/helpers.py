from flask_tools.pipette.config import PipetteConfig
from flask_tools.pipette.verifiers.reaction_energy import (
    DFTExecutor,
    MoleculeEnergyRecord,
)
import time


def apply_no_dft_to_config(config: PipetteConfig) -> PipetteConfig:
    config.rules.use_dft = False
    return config


class FakeDFTExecutor(DFTExecutor):
    def run(self, rxn_smiles: str) -> MoleculeEnergyRecord:
        time.sleep(2)
        return MoleculeEnergyRecord(smi=rxn_smiles, energy_ev=100.0, source="fake_dft")
