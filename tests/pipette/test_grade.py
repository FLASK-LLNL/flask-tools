from __future__ import annotations

import pytest

from flask_tools.pipette.config import PipetteConfig, ConfigType
from flask_tools.pipette.grade_rxn import (
    _apply_debug_overrides,
    grade_reaction,
    load_reaction_smiles_file,
)
from flask_tools.pipette.constants import FinalGrade, ReactionGrade


def test_grade_reaction_uses_default_exact_yaml_and_returns_batch(monkeypatch) -> None:
    loaded = PipetteConfig(mode="exact", tool_list=["basic_smiles_validation"])
    captured: dict[str, object] = {}

    @classmethod
    def fake_from_default_exact_yaml(cls) -> PipetteConfig:
        captured["loaded_default"] = True
        return loaded

    def fake_build_default_pipeline(*, config, judge):
        captured["config"] = config
        captured["judge"] = judge

        class StubPipeline:
            def grade(self, rxn_smiles_list: list[str]):
                for rxn_smiles in rxn_smiles_list:
                    captured.setdefault("rxn_smiles", []).append(rxn_smiles)
                    yield ReactionGrade(
                        final_grade=FinalGrade.UNCERTAIN,
                        short_reason=f"test.{rxn_smiles}",
                        results=[],
                        comment="stub",
                    )

        return StubPipeline()

    monkeypatch.setattr(
        PipetteConfig, "from_default_exact_yaml", fake_from_default_exact_yaml
    )
    monkeypatch.setattr(
        "flask_tools.pipette.grade_rxn.build_default_pipeline",
        fake_build_default_pipeline,
    )

    result = grade_reaction(["CCO>>CC=O", "CC>>C=C"], ConfigType.DEFAULT_EXACT)

    assert [item.short_reason for item in result] == ["test.CCO>>CC=O", "test.CC>>C=C"]
    assert captured["loaded_default"] is True
    assert captured["config"] is loaded
    assert captured["judge"] is None
    assert captured["rxn_smiles"] == ["CCO>>CC=O", "CC>>C=C"]


def test_load_reaction_smiles_file_reads_plaintext_lines(tmp_path) -> None:
    path = tmp_path / "rxns.txt"
    path.write_text("CCO>>CC=O\n\nCC>>C=C\n", encoding="utf-8")

    assert load_reaction_smiles_file(path) == ["CCO>>CC=O", "CC>>C=C"]


def test_load_reaction_smiles_file_reads_csv_column(test_data_path) -> None:
    path = test_data_path / "rxns.csv"

    result = load_reaction_smiles_file(path)

    assert len(result) == 3
    assert result[0].startswith("COC(=O)[C@H](CC=C)NC(=O)")
    assert result[1] == "CC(=O)CN1C(C[C](C1=O)NC)=O>>CC(=O)CN1C(C[C@H](C1=O)NC)=O"


def test_load_reaction_smiles_file_rejects_csv_without_smiles_column(tmp_path) -> None:
    path = tmp_path / "rxns.csv"
    path.write_text("id,notes\n1,missing\n", encoding="utf-8")

    with pytest.raises(ValueError, match="reaction SMILES columns"):
        load_reaction_smiles_file(path)
