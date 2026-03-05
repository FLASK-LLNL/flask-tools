################################################################################
## Copyright 2025 Lawrence Livermore National Security, LLC. and Binghamton University.
## See the top-level LICENSE file for details.
##
## SPDX-License-Identifier: Apache-2.0
################################################################################

from loguru import logger

try:
    from rdkit import Chem

    HAS_RDKIT = True
except (ImportError, ModuleNotFoundError) as e:
    HAS_RDKIT = False
    logger.warning(
        "Please install the rdkit support packages to use this module."
        "Install it with: pip install flask-mcp-servers[rdkit]",
    )

import json
import os
from charge.tasks.task import Task
from flask_tools.utils.server_utils import add_server_arguments, update_mcp_network

from charge.clients.autogen import AutoGenBackend
from charge.clients.client import Client
import asyncio
from flask_tools.chemistry import smiles_utils
from flask_tools.lmo.molecular_property_utils import get_density
import argparse
from typing import Optional, Literal, Tuple
from flask_tools.lmo.molecular_property_utils import PropertyType


JSON_FILE_PATH = f"{os.getcwd()}/known_molecules.json"
AGENT_BACKEND: AutoGenBackend | None = None


class DiagnoseSMILESTask(Task):
    def __init__(self, smiles: str, *args, **kwargs):
        system_prompt = (
            "You are a world-class chemist. Your task is to diagnose and evaluate "
            "the quality of the provided SMILES strings. You will be given invalid"
            " SMILES strings, and your task is to identify the issues with them, "
            "correct them if possible. Return concise explanations for each SMILE string."
        )

        user_prompt = (
            f"Diagnose the followig SMILES string {smiles}. Give it a short and concise "
            "explanation of what is wrong with it, and if possible, provide a corrected "
            "version of the SMILES string. If the SMILES string is valid, simply state "
            "'The SMILES string is valid.'"
        )
        super().__init__(
            system_prompt=system_prompt, user_prompt=user_prompt, *args, **kwargs
        )


def diagnose_smiles(smiles: str) -> str:
    """
    Diagnose a SMILES string. Returns a diagnosis of the SMILES string.

    Args:
        smiles (str): The input SMILES string.
    Returns:
        str: The diagnosis of the SMILES string.
    """
    if not HAS_RDKIT:
        raise ImportError(
            "Please install the rdkit support packages to use this module."
        )
    logger.info(f"Diagnosing SMILES string: {smiles}")
    task = DiagnoseSMILESTask(smiles=smiles)

    global AGENT_BACKEND
    assert (
        AGENT_BACKEND is not None
    ), "Agent pool is not initialized. Diagnoise Tool not available."

    diagnose_agent = AGENT_BACKEND.create_agent(task=task)

    try:
        response = asyncio.run(diagnose_agent.run())
        assert response is not None
        assert len(response.messages) > 0  # type: ignore
        assert response.messages[-1] is not None  # type: ignore

        diagnoses = response.messages[-1].content  # type: ignore
        logger.info(f"Diagnosis: {diagnoses}")
        return f"SMILES diagnoses: {diagnoses}"  # type: ignore

    except Exception as e:
        logger.error(f"An error occurred: {e}")
        return "Error: Unable to process the SMILES string at this time."


def is_already_known(smiles: str) -> bool:
    """
    Check if a SMILES string provided is already known. Only provide
    valid SMILES strings. Returns True if the SMILES string is valid, and
    already in the database, False otherwise.
    Args:
        smiles (str): The input SMILES string.
    Returns:
        bool: True if the SMILES string is valid and known, False otherwise.

    Raises:
        ValueError: If the SMILES string is invalid.
    """
    if not HAS_RDKIT:
        raise ImportError(
            "Please install the rdkit support packages to use this module."
        )
    if not Chem.MolFromSmiles(smiles):
        raise ValueError("Invalid SMILES string.")

    try:
        canonical_smiles = smiles_utils.canonicalize_smiles(smiles)

        try:
            with open(JSON_FILE_PATH) as f:
                known_mols = json.load(f)
                known_smiles = [mol["smiles"] for mol in known_mols]

        except FileNotFoundError:
            logger.warning(f"{JSON_FILE_PATH} not found. Creating a new one.")
            known_mols = []

    except Exception as e:
        raise ValueError("Error in canonicalizing SMILES string.") from e

    # Check if the SMILES string is already known (in the database)
    # This is a placeholder for the actual database check
    return canonical_smiles in known_smiles


def calculate_property(
    smiles: str, property: Literal["density", "synthesizability"]
) -> Tuple[PropertyType, float]:
    """
    Get a molecular property given its SMILES string.

    Args:
        smiles (str): The input SMILES string.
        property (str): The property to calculate ("density" or "synthesizability").
    Returns:
    str:
        The property to predict. Must be one of the valid property names listed above.
    float: The requested property of the molecule.
    """
    if not HAS_RDKIT:
        raise ImportError(
            "Please install the rdkit support packages to use this module."
        )
    if property == "density":
        _, density = get_density(smiles, property)
        logger.info(f"Density for SMILES {smiles}: {density}")
        return property, density
    elif property == "synthesizability":
        synth_score = smiles_utils.get_synthesizability(smiles)
        logger.info(f"Synthesizability for SMILES {smiles}: {synth_score}")
        return property, synth_score
    else:
        raise ValueError(f"Unknown property: {property}")


def setup_autogen_pool(
    model: str, backend: str, api_key: Optional[str], base_url: Optional[str]
):
    global AGENT_BACKEND
    AGENT_BACKEND = AutoGenBackend(
        model=model, backend=backend, api_key=api_key, base_url=base_url
    )
