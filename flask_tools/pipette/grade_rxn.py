from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

from pipette.config import PipetteConfig, load_config, ConfigType
from pipette.constants import ReactionGrade, ToolResult
from pipette.pipeline import build_default_pipeline

if TYPE_CHECKING:
    from pipette.judges import LLMJudge

REACTION_SMILES_COLUMNS = (
    "rxn_smiles",
    "reaction_smiles",
    "rxn_smi",
    "smiles",
)


def _get_possible_fixed_rxn_smi(reaction_grade: ReactionGrade) -> str | None:
    """Return a fixed reaction SMILES string when the fixer tool produced one."""
    tool_res: ToolResult
    for tool_res in reaction_grade.results:
        if tool_res.name == "llm_reaction_fix":
            return tool_res.data["fixed_reaction_smiles"]
    return None


def _grade_reactions(
    rxn_smiles_list: list[str],
    config: PipetteConfig | None = None,
    judge: LLMJudge | None = None,
) -> list[ReactionGrade]:
    resolved_config = config or PipetteConfig.from_default_exact_yaml()
    pipeline = build_default_pipeline(config=resolved_config, judge=judge)
    return list(pipeline.grade(rxn_smiles_list))


def _build_output_records(
    rxn_smiles_list: list[str],
    results: list[ReactionGrade],
) -> list[dict]:
    return [
        {
            "rxn_smiles": rxn_smiles,
            "cleaned_rxn_smiles": _get_possible_fixed_rxn_smi(result) or rxn_smiles,
            "grade": asdict(result),
        }
        for rxn_smiles, result in zip(rxn_smiles_list, results, strict=True)
    ]


def grade_reaction(
    rxn_smiles_list: list[str],
    config: PipetteConfig | None = None,
    judge: LLMJudge | None = None,
) -> list[ReactionGrade]:
    res = _grade_reactions(rxn_smiles_list, config=config, judge=judge)
    for rxn_smiles, result in zip(rxn_smiles_list, res, strict=True):
        print(f"{rxn_smiles}:\n{result}\n\n")
    return res


def grade_reaction_json(
    rxn_smiles: str,
    config: PipetteConfig | None = None,
    judge: LLMJudge | None = None,
) -> dict:
    """Return single-reaction JSON object with full information."""
    return grade_reactions_json([rxn_smiles], config=config, judge=judge)[0]


def grade_reactions_json(
    rxn_smiles_list: list[str],
    config: PipetteConfig | None = None,
    judge: LLMJudge | None = None,
) -> list[dict]:
    """Return the full information JSON payload for one or more reactions."""
    results = _grade_reactions(rxn_smiles_list, config=config, judge=judge)
    return _build_output_records(rxn_smiles_list, results)


def load_reaction_smiles_file(path: str | Path) -> list[str]:
    file_path = Path(path).expanduser().resolve()
    if file_path.suffix.lower() == ".csv":
        return _load_reaction_smiles_csv(file_path)
    return _load_reaction_smiles_lines(file_path)


def _load_reaction_smiles_csv(path: Path) -> list[str]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            return []

        column_name = next(
            (name for name in REACTION_SMILES_COLUMNS if name in reader.fieldnames),
            None,
        )
        if column_name is None:
            raise ValueError(
                "CSV file must contain one of these reaction SMILES columns: "
                + ", ".join(REACTION_SMILES_COLUMNS)
            )

        return [
            (row.get(column_name) or "").strip()
            for row in reader
            if (row.get(column_name) or "").strip()
        ]


def _load_reaction_smiles_lines(path: Path) -> list[str]:
    with path.open(encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def _apply_debug_overrides(config: PipetteConfig) -> PipetteConfig:
    config.rules.enable_fake_dft = True
    return config


def main() -> list[dict]:
    parser = argparse.ArgumentParser(
        description="Grade one or more reaction SMILES strings."
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--rxn-smi",
        nargs="+",
        dest="rxn_smi",
        help="One or more reaction SMILES strings, for example 'CCO>>CC=O'.",
    )
    input_group.add_argument(
        "-f",
        "--file",
        dest="file",
        help="Reaction SMILES file, one per line or a CSV with a reaction SMILES column.",
    )
    parser.add_argument(
        "--config",
        default="default-exact",
        help=f"Path to a Pipette YAML config file, or the special values '{ConfigType.LLM_JUDGE}', "
        f"'{ConfigType.LLM_JUDGE_NO_DFT}, or '{ConfigType.DEFAULT_EXACT}'.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Set rules.enable_fake_dft=true to allow the fake DFT fallback.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Prints out json object",
    )
    args = parser.parse_args()

    rxn_smiles_list = args.rxn_smi or load_reaction_smiles_file(args.file)
    config = load_config(args.config)
    if args.debug:
        config = _apply_debug_overrides(config)
    results = grade_reaction(rxn_smiles_list, config=config)
    output = _build_output_records(rxn_smiles_list, results)
    if args.verbose:
        print("Full output, json formatted:")
        print(json.dumps(output, indent=2))
    return output


if __name__ == "__main__":
    raise SystemExit(0 if bool(main()) else 1)
