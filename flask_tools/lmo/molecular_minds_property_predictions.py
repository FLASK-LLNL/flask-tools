###############################################################################
## Copyright 2025-2026 Lawrence Livermore National Security, LLC.
## See the top-level LICENSE file for details.
##
## SPDX-License-Identifier: Apache-2.0
###############################################################################

#!/usr/bin/env python3
"""
Molecular Minds: Multi-Property Predictions for Materials

Usage:
    from molecular_minds_property_predictions import predict_hof, predict_density

    hof = predict_hof('CCO')
    density = predict_density('CCO')
"""

import sys
import os
from typing import Optional, Dict
import click
import uvicorn
from loguru import logger

from fastmcp import FastMCP

mcp = FastMCP(
    "Molecular Minds Property Predictor",
)

# Add path to find Molecular_Minds module
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
grandparent_dir = os.path.dirname(parent_dir)
molecular_minds_path = os.path.join(grandparent_dir, "examples")

if molecular_minds_path not in sys.path:
    sys.path.insert(0, molecular_minds_path)

# Import from Molecular_Minds package
from Molecular_Minds import predict_smiles, get_default_predictor, set_default_model
import Molecular_Minds.molecular_minds_predictor

from lc_conductor.tool_registration import register_tool_server, get_asgi_app
from flask_tools.utils.server_utils import update_mcp_network, get_hostname


@mcp.tool()
def predict_hof(smiles: str) -> float:
    """
    Predict Heat of Formation (hof_s) for a molecule.

    Args:
        smiles: SMILES string of the molecule
        predictor: Optional. Dictionary returned by load_model(). If None, uses default model.

    Returns:
        Predicted hof_s value in kcal/mol

    Example:
        >>> hof = predict_hof('CCO')
        >>> print(f"HOF: {hof:.2f} kcal/mol")
    """
    predictor = get_default_predictor()
    results = predict_smiles(smiles, predictor)
    return results["hof_s"]


@mcp.tool()
def predict_density(smiles: str) -> float:
    """
    Predict density for a molecule.

    Args:
        smiles: SMILES string of the molecule
        predictor: Optional. Dictionary returned by load_model(). If None, uses default model.

    Returns:
        Predicted density in g/cc

    Example:
        >>> density = predict_density('CCO')
        >>> print(f"Density: {density:.3f} g/cc")
    """
    predictor = get_default_predictor()
    results = predict_smiles(smiles, predictor)
    return results["density"]


@mcp.tool()
def predict_bp(smiles: str) -> float:
    """
    Predict boiling point (bp) for a molecule.

    Args:
        smiles: SMILES string of the molecule
        predictor: Optional. Dictionary returned by load_model(). If None, uses default model.

    Returns:
        Predicted boiling point in °C

    Example:
        >>> bp = predict_bp('CCO')
        >>> print(f"Boiling Point: {bp:.1f} °C")
    """
    predictor = get_default_predictor()
    results = predict_smiles(smiles, predictor)
    return results["bp"]


@mcp.tool()
def predict_dh50(smiles: str) -> float:
    """
    Predict log(dh50) for a molecule.

    Args:
        smiles: SMILES string of the molecule
        predictor: Optional. Dictionary returned by load_model(). If None, uses default model.

    Returns:
        Predicted log(dh50) value in cm

    Example:
        >>> dh50 = predict_dh50('CCO')
        >>> print(f"log(DH50): {dh50:.3f}")
    """
    predictor = get_default_predictor()
    results = predict_smiles(smiles, predictor)
    return results["log(dh50)"]


@mcp.tool()
def predict_mp(smiles: str) -> float:
    """
    Predict melting point (mp) for a molecule.

    Args:
        smiles: SMILES string of the molecule
        predictor: Optional. Dictionary returned by load_model(). If None, uses default model.

    Returns:
        Predicted melting point in °C

    Example:
        >>> mp = predict_mp('CCO')
        >>> print(f"Melting Point: {mp:.1f} °C")
    """
    predictor = get_default_predictor()
    results = predict_smiles(smiles, predictor)
    return results["mp"]


@mcp.tool()
def predict_vp(smiles: str) -> float:
    """
    Predict log vapor pressure (logvp) for a molecule.

    Args:
        smiles: SMILES string of the molecule
        predictor: Optional. Dictionary returned by load_model(). If None, uses default model.

    Returns:
        Predicted logvp value in Pa

    Example:
        >>> vp = predict_vp('CCO')
        >>> print(f"Log Vapor Pressure: {vp:.3f} log(Pa)")
    """
    predictor = get_default_predictor()
    results = predict_smiles(smiles, predictor)
    return results["logvp"]


@click.command()
@click.option(
    "--transport",
    type=click.Choice(["stdio", "streamable-http"]),
    help="MCP transport type",
    default="streamable-http",
)
@click.option("--port", type=int, default=8128, help="Port to run the server on")
@click.option("--host", type=str, default=None, help="Host to run the server on")
@click.option(
    "--name",
    type=str,
    default="flask_molecular_minds_tool",
    help="Name of the MCP server",
)
@click.option(
    "--copilot-port", type=int, default=8001, help="Port to the running copilot backend"
)
@click.option(
    "--copilot-host", type=str, default=None, help="Host to the running copilot backend"
)
@click.option(
    "--checkpoint-path",
    envvar="FLASK_MOLECULAR_MINDS_CHECKPOINT_PATH",
    required=True,
    type=str,
    help="Path to model checkpoint path",
)
@click.pass_context
def main(
    ctx,
    transport: str,
    port: int,
    host: str,
    name: str,
    copilot_port: int,
    copilot_host: int,
    checkpoint_path: str,
):
    print("\n".join(f"{k} = {v}" for k, v in ctx.params.items()))
    model_path = checkpoint_path
    Molecular_Minds.molecular_minds_predictor.DEFAULT_MODEL_PATH = model_path

    if host is None:
        _, host = get_hostname()

    try:
        register_tool_server(port, host, name, copilot_port, copilot_host)
    except:
        logger.info(
            f"{name} could not connect to server for registration -- requires manual registration"
        )

    # Run MCP server
    mcp.run(
        transport=transport,
        host=host,
        port=port,
        path=f"/flask_molecular_minds_tool/mcp",
        json_response=True,
    )


# ===== SCRIPT =====
if __name__ == "__main__":
    main()
