################################################################################
## Copyright 2025 Lawrence Livermore National Security, LLC. and Binghamton University.
## See the top-level LICENSE file for details.
##
## SPDX-License-Identifier: Apache-2.0
################################################################################

import os
import click
import sys
from loguru import logger
from fastmcp import FastMCP

from flask_tools.chemistry import smiles_utils
from flask_tools.utils.server_utils import update_mcp_network, get_hostname
from lc_conductor.tool_registration import register_tool_server


@click.command()
@click.option(
    "--transport",
    type=click.Choice(["stdio", "streamable-http"]),
    help="MCP transport type",
    default="streamable-http",
)
@click.option("--port", type=int, default=8124, help="Port to run the server on")
@click.option("--host", type=str, default=None, help="Host to run the server on")
@click.option("--name", type=str, default="lmo_tools", help="Name of the MCP server")
@click.option(
    "--copilot-port", type=int, default=8001, help="Port to the running copilot backend"
)
@click.option(
    "--copilot-host", type=str, default=None, help="Host to the running copilot backend"
)
@click.option(
    "--model",
    type=str,
    default="gpt-5.1",
    help="Model to use for the LMO tool server diagnose functions",
)
@click.option(
    "--backend",
    type=str,
    default="openai",
    help="Backend to use for the LMO tool server diagnose functions",
)
@click.option(
    "--api-key", type=str, default=None, help="API key for the LMO tool server"
)
@click.option(
    "--base-url", type=str, default=None, help="Base URL for the LMO tool server"
)
@click.option(
    "--json-file",
    type=str,
    default="known_molecules.json",
    help="Path to known molecules JSON",
)
def main(
    transport: str,
    port,
    host,
    name,
    copilot_port,
    copilot_host,
    api_key,
    base_url,
    model,
    backend,
    json_file,
):
    if host is None:
        _, host = get_hostname()

    try:
        register_tool_server(port, host, name, copilot_port, copilot_host)
    except:
        logger.info(
            f"{name} could not connect to server for registration -- requires manual registration"
        )

    mcp = FastMCP(
        "SMILES Diagnosis and retrieval MCP Server",
    )

    import flask_tools.lmo.lmo_tools as LMO_MCP

    LMO_MCP.setup_autogen_pool(
        api_key=api_key,
        base_url=base_url,
        model=model,
        backend=backend,
    )

    # This should be convierted to a shared database instance
    # to ensure the MCP and the backend use the same known molecules

    if not os.path.exists(json_file):
        abs_path = os.path.abspath(json_file)
        with open(abs_path, "w") as f:
            pass

    LMO_MCP.JSON_FILE_PATH = json_file

    mcp.tool()(LMO_MCP.diagnose_smiles)
    mcp.tool()(LMO_MCP.is_already_known)
    mcp.tool()(LMO_MCP.calculate_property)

    # Add the SMILES utility functions as MCP tools
    mcp.tool()(smiles_utils.canonicalize_smiles)
    mcp.tool()(smiles_utils.verify_smiles)

    logger.info(f"Using model: {model} on backend: {backend}")
    logger.info(f"Using known molecules database at: {LMO_MCP.JSON_FILE_PATH}")

    # Run MCP server
    mcp.run(
        transport=transport,
        host=host,
        port=port,
        path=f"/lmo_tools/mcp",
    )


if __name__ == "__main__":
    main()
