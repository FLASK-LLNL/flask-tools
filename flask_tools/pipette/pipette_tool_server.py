###############################################################################
## Copyright 2025-2026 Lawrence Livermore National Security, LLC.
## See the top-level LICENSE file for details.
##
## SPDX-License-Identifier: Apache-2.0
###############################################################################

################################################################################
## Copyright 2025 Lawrence Livermore National Security, LLC.
## See the top-level LICENSE file for details.
##
## SPDX-License-Identifier: Apache-2.0
################################################################################

"""Necessary env vars for LLM steps:
An api key, either:
    "FLASK_ORCHESTRATOR_API_KEY"
    "PIPETTE_API_KEY"
    "OPENAI_API_KEY"

Optional:
LLM Url, either:
    "FLASK_ORCHESTRATOR_URL"
    "PIPETTE_LLM_BASE_URL"
LLM Model:
    "FLASK_ORCHESTRATOR_MODEL"

Run with (default) --config llm-judge to skip dft, llm-judge-with-dft to use dft, or rules to use a rule based assesment instead of a final LLM judge

Example rxn for caffeine, Cn1cnc2c1c(=O)[nH]c(=O)n2C.CI>>CN1C=NC2=C1C(=O)N(C(=O)N2C)C which will require LLM fixing
to balance
"""
import click
from fastmcp import FastMCP
from loguru import logger

from lc_conductor.tool_registration import register_tool_server
from flask_tools.utils.server_utils import get_hostname
from flask_tools.pipette.grade_rxn import grade_reaction_json
from flask_tools.pipette.config import load_config, ConfigType


@click.command()
@click.option(
    "--config",
    default=ConfigType.LLM_JUDGE_NO_DFT,
    help="Path to config yaml file, or llm-judge for default settings",
)
@click.option(
    "--transport",
    type=click.Choice(["stdio", "streamable-http"]),
    help="MCP transport type",
    default="streamable-http",
)
@click.option("--port", type=int, default=8130, help="Port to run the server on")
@click.option("--host", type=str, default=None, help="Host to run the server on")
@click.option(
    "--name", type=str, default="pipette_tools", help="Name of the MCP server"
)
@click.option(
    "--copilot-port", type=int, default=8001, help="Port to the running copilot backend"
)
@click.option(
    "--copilot-host", type=str, default=None, help="Host to the running copilot backend"
)
def main(
    config: str,
    transport: str,
    port: int,
    host: str | None,
    name: str,
    copilot_port: int,
    copilot_host: str | None,
):
    if host is None:
        _, host = get_hostname()

    try:
        register_tool_server(port, host, name, copilot_port, copilot_host)
    except Exception:
        logger.info(
            f"{name} could not connect to server for registration -- requires manual registration"
        )

    mcp = FastMCP(
        "Pipette reaction grading MCP Server",
        instructions=(
            "Grade a single reaction SMILES string and return a "
            "JSON dictionary with a final_grade, and a list of verification tool results for that reaction."
        ),
    )

    resolved_config = load_config(config)

    @mcp.tool()
    def grade_reaction(
        rxn_smi: str,
    ) -> dict:
        """Grade one reaction smiles with the format "reactants>>products" and return a result dictionary with analysis
        in the "grade" sub dictionary.
        The input reaction smiles should not include reagents, only reactants and products.
        The keys of the result dictionary are:
            - rxn_smiles: the smiles of the reaction
            - cleaned_rxn_smiles: rxn_smiles after balancing, which may or may not be different
            - grade: a dictionary of the final grade. The keys are:
                - final_grade: The reaction's grade. Possible values are `likely`|`possible but unlikely`|`impossible`|`uncertain`
                - short_comment: a short comment explaining the grade
                - comment: a comment explaining the grade
                - results: a list of results from sub tools. Each result is a dictionary. Important keys in the individual dictionaries are:
                    - name: the name of the tool
                    - status: The status. Possible values are:
                        - pass | fail | unknown | not_run | error
                    - comment: a short comment about the tool result
                    - data: a dictionary with extra information pertaining to that tool

        Example result:
        {
        "rxn_smiles": "Cn1cnc2c1c(=O)[nH]c(=O)n2C.CI>>CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
        "cleaned_rxn_smiles": "CI.Cn1cnc2c1c(=O)[nH]c(=O)n2C>>Cn1c(=O)c2c(ncn2C)n(C)c1=O.[H+].[I-]",
        "grade": {
          "final_grade": "likely",
          "short_comment": "ai.plausible_n_methylation",
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
                "imbalanced_molecules": [],
                "imbalanced_molecule_confidence": null,
                "closest_stoich": null
              },
              "comment": "Element counts are conserved.",
              "skipped_reason": null
            }
          ],
          "comment": "The reaction is chemically plausible: methyl iodide can N-methylate the xanthine NH to give the trimethylated product, with HI represented as [H+] and [I-]. The SMILES parses correctly, and both charge and mass are conserved."
         }
        }
        """
        return grade_reaction_json(rxn_smi, config=resolved_config)

    mcp.run(
        transport=transport,
        host=host,
        port=port,
        path=f"/{name}/mcp",
    )


if __name__ == "__main__":
    main()
