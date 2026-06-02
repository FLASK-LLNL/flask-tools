# Pipette

Pipette grades predicted chemical reactions by running a sequence of validation
tools and combining their outputs into a final reaction grade.

The current pipeline includes:

- reaction SMILES parsing
- basic SMILES validation
- exact-match checker interfaces for reaction databases
- charge conservation
- mass conservation with simple missing-product heuristics and solvent catalogs
- reaction fixing LLM call
  - Run if mass conservation fails
  - If new reaction is returned, goes back to start. Only allowed to run once in a pipeline
- reaction-energy checker interfaces for cached DFT results or external runs
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
There are two modes, "exact" which is deterministic but could fail on edge cases and "ai" where we ask an LLM for a
final decision.

### AI-Judge Mode

```shell
python -m flask_tools.pipette.grade_rxn --rxn-smi '[HH].FF>>F.F' --config llm-judge-no-dft
# Or with --debug which uses fake DFT reaction energies
python -m flask_tools.pipette.grade_rxn --rxn-smi '[HH].FF>>F.F' --config llm-judge --debug
```

```python
from flask_tools.pipette.grade_rxn import grade_reaction
from flask_tools.pipette.config import load_config

result = grade_reaction(["[HH].FF>>F.F"], config='llm-judge-no-dft')

# Or
config = load_config('llm-judge')
config.rules.enable_fake_dft = True  # Reaction energy will always return a passing value
result = grade_reaction(["[HH].FF>>F.F"], config=config)
```
This loads the config from `pipette/assets/llm-judge.yaml`.

### Exact Rule-based Grader
Rule based. For example, if Mass Conservation says that imbalance is caused by a common or uncommon solvent,
and reaction energies check out afterward, then it'll pass if the solvent was common, and fail otherwise.

The "ai" mode will send to an LLM similar tool results to what the exact mode would have used to make a decision.

#### Without DFT Reaction Energy Set up

```shell
python -m flask_tools.pipette.grade_rxn --rxn-smi 'CCO>>CC=O' --config default-exact --debug  # --debug uses fake DFT reaction energies
```
Or

```python
from flask_tools.pipette.grade_rxn import grade_reaction
from flask_tools.pipette.config import load_config

config = load_config('default-exact')  # Rule-based decision
config.rules.enable_fake_dft = True  # Reaction energy will always return a passing value
results = grade_reaction(["[HH].FF>>F.F"], config=config)
for result in results:
  print(result.final_grade.value)
  for tool_result in result.results:
    print(tool_result.name, tool_result.status.value, tool_result.comment)
```

Example output
```aiignore
[HH].FF>>F.F:
ReactionGrade(final_grade=likely, short_reason=exact.mass_and_energy_pass)
comment: Mass conservation and reaction energy both passed.
tool_results:
  - basic_smiles_validation: pass (possible) - Reaction SMILES parsed successfully.
    {
      "reactant_count": 2,
      "product_count": 2
    }
  - exact_match: unknown - No reaction database backend is configured.
  - charge_conservation: pass (likely) - Charge is conserved.
    {
      "charge_difference": 0
    }
  - mass_conservation: pass (likely) - Element counts are conserved.
    {
      "mass_difference_amu": 0.0,
      "element_difference": {},
      "possible_missing_products": [],
      "missing_product_confidence": null,
      "closest_stoich": null
    }
  - reaction_energy: pass (likely) - Reaction energy is within the allowed threshold.
    {
      "energy_difference_ev_mol": -542.0,
      "source": {
        "[HH]": "cache",
        "FF": "cache",
        "F": "cache"
      },
      "metadata": {
        "reactants": {
          "[HH]": 0.0,
          "FF": 0.0
        },
        "products": {
          "F": -271.0
        }
      }
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

## Output Format
### Exact Rule-based Grader Results
`results` is a list of `ReactionGrade` objects, which is a series of `ToolResults` and a final grade:
```
[ReactionGrade(final_grade=<FinalGrade.POSSIBLE: 'possible'>,
               short_reason='exact.mass_potential-high.energy_pass',
               results=[ToolResult(name='basic_smiles_validation',
                                   status=<ToolStatus.PASS: 'pass'>,
                                   grade_hint=<FinalGrade.POSSIBLE: 'possible'>,
                                   data={'product_count': 1,
                                         'reactant_count': 1},
                                   comment='Reaction SMILES parsed '
                                           'successfully.',
                                   skipped_reason=None),
                        ToolResult(name='exact_match',
                                   status=<ToolStatus.UNKNOWN: 'unknown'>,
                                   grade_hint=None,
                                   data={},
                                   comment='No reaction database backend is '
                                           'configured.',
                                   skipped_reason=None),
                        ToolResult(name='charge_conservation',
                                   status=<ToolStatus.PASS: 'pass'>,
                                   grade_hint=<FinalGrade.LIKELY: 'likely'>,
                                   data={'charge_difference': 0},
                                   comment='Charge is conserved.',
                                   skipped_reason=None),
                        ToolResult(name='mass_conservation',
                                   status=<ToolStatus.POTENTIAL: 'potential'>,
                                   grade_hint=<FinalGrade.POSSIBLE: 'possible'>,
                                   data={'closest_stoich': None,
                                         'element_difference': {'H': -2},
                                         'mass_difference_amu': 2.0159,
                                         'missing_product_confidence': 'high',
                                         'possible_missing_products': [{'confidence': 'high',
                                                                        'mass_amu': 2.0159,
                                                                        'missing_side': 'reactants',
                                                                        'name': 'hydrogen'}]},
                                   comment='Element counts are not conserved, '
                                           'but the difference matches a '
                                           'common omitted species or solvent: '
                                           'hydrogen.',
                                   skipped_reason=None),
                        ToolResult(name='reaction_energy',
                                   status=<ToolStatus.PASS: 'pass'>,
                                   grade_hint=<FinalGrade.LIKELY: 'likely'>,
                                   data={'energy_difference_ev_mol': 0,
                                         'metadata': {'products': {'CC=O': -50.2},
                                                      'reactants': {'CCO': -56.4}},
                                         'source': 'dft'},
                                   comment='Reaction energy is within the '
                                           'allowed threshold.',
                                   skipped_reason=None)],
               comment='Reaction depends on a possible omitted species to '
                       'satisfy mass balance, reaction energy passed.')]

```

By default, `grade_reaction(...)` loads the packaged "exact" rule based config YAML by from
`pipette/assets/exact.yaml`.

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
