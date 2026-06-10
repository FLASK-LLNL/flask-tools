from flask_tools.pipette.config import PipetteConfig


def apply_no_dft_to_config(config: PipetteConfig) -> PipetteConfig:
    config.rules.use_dft = False
    return config
