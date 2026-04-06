################################################################################
## Copyright 2025 Lawrence Livermore National Security, LLC. and Binghamton University.
## See the top-level LICENSE file for details.
##
## SPDX-License-Identifier: Apache-2.0
################################################################################

import click
import sys
from loguru import logger
from typing import Optional
from fastmcp import FastMCP

try:
    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
        LlamaForCausalLM,
        PreTrainedTokenizer,
    )
    from peft import PeftModel
    from trl import apply_chat_template
    import torch

    HAS_FLASKV2 = True
except (ImportError, ModuleNotFoundError) as e:
    HAS_FLASKV2 = False
    logger.warning(
        "Please install the flask support packages to use this module."
        "Install it with: pip install charge[flask]",
    )

import flask_tools.retrosynthesis.flaskv2_reactions as flask
from flask_tools.utils.server_utils import update_mcp_network, get_hostname
from lc_conductor.tool_registration import register_tool_server


@click.command()
@click.option(
    "--model-dir-fwd", envvar="FLASKV2_MODEL_FWD", help="Path to flaskv2 model"
)
@click.option(
    "--model-dir-retro",
    envvar="FLASKV2_MODEL_RETRO",
    help="Path to flaskv2 model for retrosynthesis",
)
@click.option("--adapter-weights-fwd", help="LoRA adapter weights, if used")
@click.option(
    "--adapter-weights-retro",
    help="LoRA adapter weights for retrosynthesis model, if used",
)
@click.option(
    "--transport",
    type=click.Choice(["stdio", "streamable-http"]),
    help="MCP transport type",
    default="streamable-http",
)
@click.option("--port", type=int, default=8125, help="Port to run the server on")
@click.option("--host", type=str, default=None, help="Host to run the server on")
@click.option(
    "--name", type=str, default="flaskv2_tools", help="Name of the MCP server"
)
@click.option(
    "--copilot-port", type=int, default=8001, help="Port to the running copilot backend"
)
@click.option(
    "--copilot-host", type=str, default=None, help="Host to the running copilot backend"
)
def main(
    model_dir_fwd: str,
    model_dir_retro: str,
    adapter_weights_fwd: str,
    adapter_weights_retro: str,
    transport: str,
    port: str,
    host: str,
    name: str,
    copilot_port: str,
    copilot_host: str,
):
    if not HAS_FLASKV2:
        raise ImportError(
            "Please install the [flask] optional packages to use this module."
        )
    if not model_dir_fwd and not model_dir_retro:
        raise ValueError("At least one model has to be given to the MCP server")

    if host is None:
        _, host = get_hostname()

    # Init MCP server
    mcp = FastMCP(
        "FLASKv2 Reaction Predictor",
    )

    # HF models and tokenizer
    fwd_model = None
    retro_model = None
    tokenizer = None

    # Load tokenizer and models
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir_fwd or model_dir_retro, padding_side="left"
    )
    tokenizer.add_special_tokens({"pad_token": "<|finetune_right_pad_id|>"})
    if model_dir_fwd:
        fwd_model = AutoModelForCausalLM.from_pretrained(
            model_dir_fwd,
            device_map="cuda",
            torch_dtype=torch.bfloat16,
        )
        if adapter_weights_fwd is not None:
            fwd_model = PeftModel.from_pretrained(fwd_model, adapter_weights_fwd)
            fwd_model = fwd_model.merge_and_unload()
    if model_dir_retro:
        retro_model = AutoModelForCausalLM.from_pretrained(
            model_dir_retro,
            device_map="cuda",
            torch_dtype=torch.bfloat16,
        )
        if adapter_weights_retro is not None:
            retro_model = PeftModel.from_pretrained(retro_model, adapter_weights_retro)
            retro_model = retro_model.merge_and_unload()

    # Enable model optimizations
    if fwd_model is not None:
        fwd_model.eval()
        if hasattr(fwd_model, "config") and hasattr(fwd_model.config, "use_cache"):
            fwd_model.config.use_cache = True  # enable KV caching
    if retro_model is not None:
        retro_model.eval()
        if hasattr(retro_model, "config") and hasattr(retro_model.config, "use_cache"):
            retro_model.config.use_cache = True  # enable KV caching

    # Dynamic tool creation based on input models
    available_tools = []
    if fwd_model is not None:
        available_tools.append("Forward Prediction")

        @mcp.tool()
        def predict_reaction_products(reactants: list[str]) -> list[str]:
            """
            Given a set of reactant molecules, predict the likely product molecule(s).

            Args:
                reactants (list[str]): a list of reactant molecules in SMILES representation.
            Returns:
                list[str]: a list of predictions, each of which is a json string listing the predicted product molecule(s) in SMILES.
            """
            logger.debug("Calling `predict_reaction_products`")
            return flask.predict_reaction_internal(
                reactants, False, fwd_model, retro_model, tokenizer
            )

    if retro_model is not None:
        available_tools.append("Single-Step Retrosynthesis")

        @mcp.tool()
        def predict_reaction_reactants(products: list[str]) -> list[str]:
            """
            Given a product molecule, predict the likely reactants and other chemical species (e.g., agents, solvents).

            Args:
                products (list[str]): a list of product molecules in SMILES representation.
            Returns:
                list[str]: a list of predictions, each of which is a json string listing the predicted reactant molecule(s) in SMILES,
                    as well as potential (re)agents and solvents used in the reaction.
            """
            logger.debug("Calling `predict_reaction_reactants`")
            return flask.predict_reaction_internal(
                products, True, fwd_model, retro_model, tokenizer
            )

    logger.info(f"Available tools: {', '.join(available_tools)}")

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
        path=f"/flaskv2_tools/mcp",
    )


if __name__ == "__main__":
    main()
