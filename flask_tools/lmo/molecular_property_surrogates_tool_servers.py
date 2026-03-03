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

from flask_tools.utils.server_utils import update_mcp_network, get_hostname
from lc_conductor.tool_registration import register_tool_server
from fastmcp import FastMCP

from flask_tools.lmo.molecular_property_utils import calculate_property_hf


@click.command()
@click.option(
    "--transport",
    type=click.Choice(["stdio", "streamable-http"]),
    help="MCP transport type",
    default="streamable-http",
)
@click.option("--port", type=int, default=8126, help="Port to run the server on")
@click.option("--host", type=str, default=None, help="Host to run the server on")
@click.option(
    "--name", type=str, default="mol_prop_surrogates", help="Name of the MCP server"
)
@click.option(
    "--copilot-port", type=int, default=8001, help="Port to the running copilot backend"
)
@click.option(
    "--copilot-host", type=str, default=None, help="Host to the running copilot backend"
)
def main(
    transport,
    port,
    host,
    name,
    copilot_port,
    copilot_host,
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
        "Computationally expensive surrogate models for molecular properties MCP Server",
    )
    mcp.tool()(calculate_property_hf)

    mcp.run(
        transport=transport,
        host=host,
        port=port,
        path=f"/mol_prop_tools/mcp",
    )


if __name__ == "__main__":
    main()
