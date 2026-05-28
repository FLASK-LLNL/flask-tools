################################################################################
## Copyright 2025 Lawrence Livermore National Security, LLC.
## See the top-level LICENSE file for details.
##
## SPDX-License-Identifier: Apache-2.0
################################################################################

"""Necessary env vars for LLM steps:
One of:
    "API_KEY",
    "JUDGE_API_KEY",
    "PIPETTE_API_KEY",
    "PIPETTE_LLM_API_KEY",
    "OPENAI_API_KEY",
Optionally one of:
    "LLM_BASE_URL"
    "PIPETTE_LLM_BASE_URL"

Run with (default) --config llm-judge-no-dft to skip dft, llm-judge to use dft, or default-exact to use a rule based assesment instead of a final LLM judge

Example rxn for caffeine, Cn1cnc2c1c(=O)[nH]c(=O)n2C.CI>>CN1C=NC2=C1C(=O)N(C(=O)N2C)C which will require LLM fixing
to balance
"""
import click
from fastmcp import FastMCP
from loguru import logger

from lc_conductor.tool_registration import register_tool_server
from flask_tools.utils.server_utils import get_hostname
from pipette.grade_rxn import grade_reaction_json, _apply_debug_overrides
from pipette.config import load_config


@click.command()
@click.option(
    "--config",
    default="llm-judge-no-dft",
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
@click.option(
    "--debug", type=bool, help="Enable debug mode which uses a fake, always-passing DFT"
)
def main(
    config: str,
    transport: str,
    port: int,
    host: str | None,
    name: str,
    copilot_port: int,
    copilot_host: str | None,
    debug: bool,
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
            "Grade a single reaction SMILES string with Pipette and return a "
            "JSON list of verification tool results for that reaction."
        ),
    )

    resolved_config = load_config(config)
    if debug:
        resolved_config = _apply_debug_overrides(resolved_config)

    @mcp.tool()
    def grade_reaction(
        rxn_smi: str,
    ) -> dict:
        """Grade one reaction smiles with the format "reactants>>products" and return a result dictionary with analysis
        in the "grade" sub dictionary.
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
