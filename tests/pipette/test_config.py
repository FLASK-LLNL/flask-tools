from __future__ import annotations

from textwrap import dedent

from flask_tools.pipette.config import PipetteConfig


def test_pipette_config_from_yaml_loads_nested_sections(tmp_path) -> None:
    # Test of the yaml to config conversion is ok, and that paths are relative to the yaml file
    config_path = tmp_path / "custom.yaml"
    config_path.write_text(
        dedent(
            """
            mode: ai
            tool_list:
              - basic_smiles_validation
              - reaction_energy
            tools_setting:
              reaction_energy:
                database: fake_molecule_energies.csv
            llm_judge:
              allow_fail:
                - reaction_energy
              url: https://example.test/v1/chat/completions
              model: custom-model
              api_key: sk-test
              prompt_path: prompt.txt
            llm_reaction_fixer:
              enabled: true
              model: gpt-5.5
              api_key: sk-fix
            rules:
              stop_on_hard_fail: false
              mass_tolerance_atoms: 2
              reaction_energy_max_ev_mol: 9.5
              enable_fake_dft: true
            solvent_catalog_path: solvents.csv
            solvent_commonness_path: commonness.csv
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )

    config = PipetteConfig.from_yaml(config_path)

    assert config.mode == "ai"
    assert config.tool_list == ["basic_smiles_validation", "reaction_energy"]
    assert (
        config.tools_setting.reaction_energy.database
        == (tmp_path / "fake_molecule_energies.csv").resolve()
    )
    assert config.llm_judge.allow_fail == ["reaction_energy"]
    assert config.llm_judge.url == "https://example.test/v1/chat/completions"
    assert config.llm_judge.model == "custom-model"
    assert config.llm_judge.api_key == "sk-test"
    assert config.llm_judge.prompt_path == (tmp_path / "prompt.txt").resolve()
    assert config.llm_reaction_fixer.enabled is True
    assert config.llm_reaction_fixer.model == "gpt-5.5"
    assert config.llm_reaction_fixer.api_key == "sk-fix"
    assert config.rules.stop_on_hard_fail is False
    assert config.rules.mass_tolerance_atoms == 2
    assert config.rules.reaction_energy_max_ev_mol == 9.5
    assert config.rules.enable_fake_dft is True
    assert config.solvent_catalog_path == (tmp_path / "solvents.csv").resolve()
    assert config.solvent_commonness_path == (tmp_path / "commonness.csv").resolve()


def test_default_yaml_configs_load() -> None:
    # Checks loading the default assets (from yaml files in this package) works
    exact = PipetteConfig.from_default_exact_yaml()
    ai = PipetteConfig.from_default_ai_yaml()

    assert exact.mode == "exact"
    assert exact.tool_list == "all"
    assert exact.tools_setting.reaction_energy.database is not None
    assert ai.mode == "ai"
    assert ai.tool_list == "all"
    assert ai.tools_setting.reaction_energy.database is not None
    assert ai.llm_judge.allow_fail == ["exact_match"]
    assert ai.llm_judge.model == "gpt-5.4"
    assert ai.llm_judge.url == "https://livai-api.llnl.gov/v1/chat/completions"
    assert ai.llm_judge.prompt_path.name == "prompt.txt"
    assert ai.llm_reaction_fixer.enabled is True
    assert ai.llm_reaction_fixer.model == "gpt-5.4"
