###############################################################################
## Copyright 2025-2026 Lawrence Livermore National Security, LLC.
## See the top-level LICENSE file for details.
##
## SPDX-License-Identifier: Apache-2.0
###############################################################################

import click
from loguru import logger
import json
from mcp.server.fastmcp import FastMCP

# from fastmcp import FastMCP
from typing import Optional

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

from flask_tools.utils.server_utils import update_mcp_network, get_hostname

REAGENT_KEYS = ["reactants", "agents", "solvents", "catalysts", "atmospheres"]
PRODUCT_KEYS = ["products"]


def format_rxn_prompt(data: dict, forward: bool) -> dict:
    required_keys = [
        "reactants",
        "products",
        "agents",
        "solvents",
        "catalysts",
        "atmospheres",
    ]
    non_product_keys = [k for k in required_keys if k != "products"]
    if forward:
        d = {k: data[k] for k in non_product_keys if data.get(k, None)}
        prompt = json.dumps(d)
    else:
        d = {"products": data["products"]}
        prompt = json.dumps(d)
    data["prompt"] = [{"role": "user", "content": prompt}]
    return data


def predict_reaction_internal(
    molecules: list[str],
    retrosynthesis: bool,
    fwd_model: Optional[AutoModelForCausalLM],
    retro_model: Optional[AutoModelForCausalLM],
    tokenizer: AutoTokenizer,
) -> list[str]:
    if not HAS_FLASKV2:
        raise ImportError(
            "Please install the [flask] optional packages to use this module."
        )
    model = retro_model if retrosynthesis else fwd_model
    data = {"products": molecules} if retrosynthesis else {"reactants": molecules}
    with torch.inference_mode():
        prompt = format_rxn_prompt(data, forward=(not retrosynthesis))
        prompt = apply_chat_template(prompt, tokenizer=tokenizer)
        inputs = tokenizer(prompt["prompt"], return_tensors="pt", padding="longest").to(
            "cuda"
        )
        prompt_length = inputs["input_ids"].size(1)
        outputs = model.generate(
            **inputs,
            max_new_tokens=2048,
            num_return_sequences=3,
            # do_sample=True,
            num_beams=3,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,  # enable KV cache
        )
        processed_outputs = [
            tokenizer.decode(out[prompt_length:], skip_special_tokens=True)
            for out in outputs
        ]
    logger.debug(f'Model input: {prompt["prompt"]}')
    processed_outs = "\n".join(processed_outputs)
    logger.debug(f"Model output: {processed_outs}")
    return processed_outputs
