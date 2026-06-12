# Pipette

Pipette grades predicted chemical reactions by running a sequence of validation
tools and combining their outputs into a final reaction grade.

The current pipeline includes:

- reaction SMILES parsing
- basic SMILES validation
- exact-match checker interfaces for reaction databases
- reaction fixing LLM call
  - Run if no exact match is found
  - If new reaction is returned, goes back to start. Only allowed to run once in a pipeline
- charge conservation
- mass conservation with simple missing-product heuristics and solvent catalogs
- (Planned) reaction-energy checker interfaces for cached DFT results or external runs
- Grading:
  - an exact rule-based grader
  - an AI-judge pipeline interface that can run selected tools and hand the results
    to an external LLM judge

You must have one of `FLASK_ORCHESTRATOR_API_KEY, OPENAI_API_KEY, or PIPETTE_API_KEY` set in your environment.
Optional: `FLASK_ORCHESTRATOR_MODEL, FLASK_ORCHESTRATOR_URL`


## Install

```bash
pip install -e .
```

For development and tests:

```bash
pip install -e .[dev]
```

## Quick start

```shell
python -m flask_tools.pipette.grade_rxn --rxn-smi 'Cn1cnc2c1c(=O)[nH]c(=O)n2C.CI>>CN1C=NC2=C1C(=O)N(C(=O)N2C)C'
```
Note, the default config is `llm-judge` (no DFT) and may be left out in the CLI and in the python code.

```python
from flask_tools.pipette.grade_rxn import grade_reaction
from flask_tools.pipette.config import load_config

result = grade_reaction(["Cn1cnc2c1c(=O)[nH]c(=O)n2C.CI>>CN1C=NC2=C1C(=O)N(C(=O)N2C)C"]) # Optionally, config='llm-judge' or "file.yaml"

# Or
config = load_config('llm-judge')
result = grade_reaction(["Cn1cnc2c1c(=O)[nH]c(=O)n2C.CI>>CN1C=NC2=C1C(=O)N(C(=O)N2C)C"], config=config)
```
This loads the config from `pipette/assets/llm-judge.yaml`.

Configs can also be a custom yaml file.


Example human-readable output
```aiignore
Cn1cnc2c1c(=O)[nH]c(=O)n2C.CI>>CN1C=NC2=C1C(=O)N(C(=O)N2C)C:
ReactionGrade(final_grade=likely, short_reason=ai.plausible_n_methylation)
comment: The reaction is chemically plausible: methyl iodide can N-methylate the xanthine NH to give the trimethylated product, with HI represented as [H+] and [I-]. The SMILES parses correctly, and both charge and mass are conserved.
tool_results:
  - basic_smiles_validation: pass - Reaction SMILES parsed successfully.
    {
      "reactant_count": 2,
      "product_count": 1
    }
  - exact_match: unknown - No reaction database backend is configured.
  - llm_reaction_fix: pass - N-methylation of the xanthine NH with methyl iodide requires HI as the byproduct, represented as [H+] and [I-]. No agents were present to remove.
    {
      "original_reaction_smiles": "Cn1cnc2c1c(=O)[nH]c(=O)n2C.CI>>CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
      "fixed_reaction_smiles": "CI.Cn1cnc2c1c(=O)[nH]c(=O)n2C>>Cn1c(=O)c2c(ncn2C)n(C)c1=O.[H+].[I-]",
      "removed_agents": [],
      "added_reactants": [],
      "added_products": [
        "[H+]",
        "[I-]"
      ]
    }
  - basic_smiles_validation: pass - Reaction SMILES parsed successfully.
    {
      "reactant_count": 2,
      "product_count": 3
    }
  - exact_match: unknown - No reaction database backend is configured.
  - charge_conservation: pass - Charge is conserved.
    {
      "charge_difference": 0
    }
  - mass_conservation: pass - Element counts are conserved.
    {
      "mass_difference_amu": 0.0,
      "element_difference": {},
      "possible_missing_products": [],
      "missing_product_confidence": null,
      "closest_stoich": null
    }
```

#### With DFT Reaction Energy Set up
Not fully implemented

Todo: config details (how to specify DFT runner if necessary)


## Reaction Fixing
Unbalanced reactions can be fixed by adding missing byproducts or reactants. This is a default part of the pipeline
and uses an LLM. An example of an unbalanced reaction is:

```smiles
Not balanced
Cn1cnc2c1c(=O)[nH]c(=O)n2C.CI>>CN1C=NC2=C1C(=O)N(C(=O)N2C)C
Balanced
Cn1cnc2c1c(=O)[nH]c(=O)n2C.CI>>CN1C=NC2=C1C(=O)N(C(=O)N2C)C.[OH3+].[I-]
```

## Full JSON Output
`results` from `grade_reactions()` is a list of `ReactionGrade` objects, which is a series of `ToolResults` and a final grade:
```
[
  {
    "rxn_smiles": "Cn1cnc2c1c(=O)[nH]c(=O)n2C.CI>>CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
    "cleaned_rxn_smiles": "CI.Cn1cnc2c1c(=O)[nH]c(=O)n2C>>Cn1c(=O)c2c(ncn2C)n(C)c1=O.[H+].[I-]",
    "grade": {
      "final_grade": "likely",
      "short_reason": "ai.plausible_n_methylation",
      "results": [
        {
          "name": "basic_smiles_validation",
          "status": "pass",
          "data": {
            "reactant_count": 2,
            "product_count": 1
          },
          "comment": "Reaction SMILES parsed successfully.",
          "skipped_reason": null
        },
        {
          "name": "exact_match",
          "status": "unknown",
          "data": {},
          "comment": "No reaction database backend is configured.",
          "skipped_reason": null
        },
        {
          "name": "llm_reaction_fix",
          "status": "pass",
          "data": {
            "original_reaction_smiles": "Cn1cnc2c1c(=O)[nH]c(=O)n2C.CI>>CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
            "fixed_reaction_smiles": "CI.Cn1cnc2c1c(=O)[nH]c(=O)n2C>>Cn1c(=O)c2c(ncn2C)n(C)c1=O.[H+].[I-]",
            "removed_agents": [],
            "added_reactants": [],
            "added_products": [
              "[H+]",
              "[I-]"
            ]
          },
          "comment": "N-methylation of the xanthine NH with methyl iodide requires HI as the byproduct, represented as [H+] and [I-]. No agents were present to remove.",
          "skipped_reason": null
        },
        {
          "name": "basic_smiles_validation",
          "status": "pass",
          "data": {
            "reactant_count": 2,
            "product_count": 3
          },
          "comment": "Reaction SMILES parsed successfully.",
          "skipped_reason": null
        },
        {
          "name": "exact_match",
          "status": "unknown",
          "data": {},
          "comment": "No reaction database backend is configured.",
          "skipped_reason": null
        },
        {
          "name": "charge_conservation",
          "status": "pass",
          "data": {
            "charge_difference": 0
          },
          "comment": "Charge is conserved.",
          "skipped_reason": null
        },
        {
          "name": "mass_conservation",
          "status": "pass",
          "data": {
            "mass_difference_amu": 0.0,
            "element_difference": {},
            "possible_missing_products": [],
            "missing_product_confidence": null,
            "closest_stoich": null
          },
          "comment": "Element counts are conserved.",
          "skipped_reason": null
        }
      ],
      "comment": "The reaction is chemically plausible: methyl iodide can N-methylate the xanthine NH to give the trimethylated product, with HI represented as [H+] and [I-]. The SMILES parses correctly, and both charge and mass are conserved."
    }
  }
]

```

By default, `grade_reaction(...)` loads the packaged "llm-config-no-dft" rule based config YAML by from
`pipette/assets/ai_judge_no_dft.yaml`.

To restrict the pipeline to specific tools, set `PipetteConfig.tool_list` to
either `"all"` or an explicit list of tool names such as
`["basic_smiles_validation", "charge_conservation"]`.

To load your own config YAML config,

```shell
python -m flask_tools.pipette.grade_rxn --rxn-smi 'CCO>>CC=O' --config my-config.yaml
```
Or
```python
from flask_tools.pipette.config import PipetteConfig

config = PipetteConfig.from_yaml("my-config.yaml")
```

# Tests

`pytest`
Or
`pytest -m llm_query` to run the tests that use LLM
