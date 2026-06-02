from __future__ import annotations

from textwrap import dedent

from flask_tools.pipette.config import PipetteConfig
from flask_tools.pipette.judge import LLMJudge
from flask_tools.pipette.reaction_fixer import LLMReactionFixer


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
              url: https://example.test/v1/
              model: custom-model
              api_key: sk-test
              prompt_path: judge-prompt.txt
              prompt: inline judge prompt
            llm_reaction_fixer:
              enabled: true
              model: gpt-5.5
              api_key: sk-fix
              prompt_path: fixer-prompt.txt
              prompt: inline fixer prompt
            rules:
              stop_on_hard_fail: false
              mass_tolerance_atoms: 2
              reaction_energy_max_ev_mol: 9.5
              enable_fake_dft: true
            solvent_catalog_path: solvents.csv
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
    assert config.llm_judge.url == "https://example.test/v1/"
    assert config.llm_judge.model == "custom-model"
    assert config.llm_judge.api_key == "sk-test"
    assert config.llm_judge.prompt_path == (tmp_path / "judge-prompt.txt").resolve()
    assert config.llm_judge.prompt == "inline judge prompt"
    assert config.llm_reaction_fixer.enabled is True
    assert config.llm_reaction_fixer.model == "gpt-5.5"
    assert config.llm_reaction_fixer.api_key == "sk-fix"
    assert (
        config.llm_reaction_fixer.prompt_path
        == (tmp_path / "fixer-prompt.txt").resolve()
    )
    assert config.llm_reaction_fixer.prompt == "inline fixer prompt"
    assert config.rules.stop_on_hard_fail is False
    assert config.rules.mass_tolerance_atoms == 2
    assert config.rules.reaction_energy_max_ev_mol == 9.5
    assert config.rules.enable_fake_dft is True
    assert config.solvent_catalog_path == (tmp_path / "solvents.csv").resolve()

    judge = LLMJudge(
        url=config.llm_judge.url,
        model=config.llm_judge.model,
        api_key="sk-test",
        prompt_path=config.llm_judge.prompt_path,
        prompt=config.llm_judge.prompt,
    )
    fixer = LLMReactionFixer(
        url=config.llm_judge.url,
        model=config.llm_reaction_fixer.model,
        api_key="sk-test",
        prompt_path=config.llm_reaction_fixer.prompt_path,
        prompt=config.llm_reaction_fixer.prompt,
    )


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
    assert ai.llm_judge.url == "https://livai-api.llnl.gov/v1/"
    assert ai.llm_judge.prompt_path.name == "judge-prompt.txt"
    assert ai.llm_judge.prompt is None
    assert ai.llm_reaction_fixer.enabled is True
    assert ai.llm_reaction_fixer.model == "gpt-5.4"
    assert ai.llm_reaction_fixer.prompt_path.name == "fixer-prompt.txt"
    assert ai.llm_reaction_fixer.prompt is None
