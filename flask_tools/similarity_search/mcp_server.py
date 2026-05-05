###############################################################################
## Copyright 2025-2026 Lawrence Livermore National Security, LLC.
## See the top-level LICENSE file for details.
##
## SPDX-License-Identifier: Apache-2.0
###############################################################################

import click
from flask_tools.similarity_search import SmilesEmbedder, FaissDataRetriever
from flask_tools.similarity_search.tokenizers import ChemformerTokenizer
from lc_conductor.tool_registration import register_tool_server
from flask_tools.utils.server_utils import update_mcp_network, get_hostname

import logging
from loguru import logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

from fastmcp import FastMCP
mcp = FastMCP("RAG")


@mcp.tool()
def retrieve_similar_reactions(smiles: str, forward: bool, k: int) -> list[dict]:
    """Retrieve similar reactions based on input smiles.

    Given a SMILES string, this function retrieves reactions with similar molecules
    from some chemical reaction database.

    Args:
        smiles (str): SMILES string that the similarity metric is based on.
            Must be either reactant(s) or product(s) of a chemical reaction.
            Note that a SMILES string can possibly represent multiple molecules.
            Examples:
                CN1C=NC2=C1C(=O)N(C(=O)N2C)C
                Fc1cc(I)ccn1.OB(O)c1ccc2ccccc2c1
        forward (bool): should be specified to True if `smiles` are reactants, False otherwise.
        k (int): number of samples to retrieve, i.e., top-k similar reactions.

    Returns:
        A list of similar reactions, each in dictionary/json format.
    """
    retriever = forward_retriever if forward else retro_retriever
    embedder = forward_embedder if forward else retro_embedder
    query_emb = embedder.embed_smiles([smiles])
    similar_dist, similar_idx, similar = retriever.search_similar(query_emb, k=k)
    similar = similar[0] # list[list[Any]] -> list[Any]
    # Clean up the retrieved similar reactions
    similar_ = [
        {k: v for k, v in rxn.items() if v not in ([], '', [''], None) and k != 'id'}
        for rxn in similar
    ]
    return similar_


forward_embedder: SmilesEmbedder = None
retro_embedder: SmilesEmbedder = None
forward_retriever: FaissDataRetriever = None
retro_retriever: FaissDataRetriever = None


@click.command()
@click.option(
    "--transport",
    type=click.Choice(["stdio", "streamable-http", "sse"]),
    help="MCP transport type",
    default="streamable-http",
)
@click.option("--port", type=int, default=8127, help="Port to run the server on")
@click.option("--host", type=str, default=None, help="Host to run the server on")
@click.option("--name", type=str, default="similarity_search_tools", help="Name of the MCP server")
@click.option("--copilot-port", type=int, default=8001, help="Port to the running copilot backend")
@click.option("--copilot-host", type=str, default=None, help="Host to the running copilot backend")
@click.option("--database-path", type=str, help="Path to database file for similarity search")
@click.option("--forward-embedder-path", type=str, help="Path to embedding model for reactants")
@click.option("--retro-embedder-path", type=str, help="Path to embedding model for products")
@click.option("--embedder-vocab-path", type=str, help="Path to embedding model vocab json file")
@click.option("--forward-embedding-path", type=str, help="Path to embedding file (NPY) for database reactants")
@click.option("--retro-embedding-path", type=str, help="Path to embedding file (NPY) for database products")
@click.pass_context
def main(
    ctx,
    transport: str,
    port: int,
    host: str,
    name: str,
    copilot_port: int,
    copilot_host: int,
    database_path: str,
    forward_embedder_path: str,
    retro_embedder_path: str,
    embedder_vocab_path: str,
    forward_embedding_path: str,
    retro_embedding_path: str,
):
    global forward_embedder, retro_embedder, forward_retriever, retro_retriever, tokenizer
    print("\n".join(f"{k} = {v}" for k, v in ctx.params.items()))

    # Init components for similarity search
    forward_embedder = SmilesEmbedder(
        model_path=forward_embedder_path,
        tokenizer=ChemformerTokenizer(
            vocab_path=embedder_vocab_path,
        ),
    )
    retro_embedder = SmilesEmbedder(
        model_path=retro_embedder_path,
        tokenizer=ChemformerTokenizer(
            vocab_path=embedder_vocab_path,
        ),
    )
    forward_retriever = FaissDataRetriever(
        data_path=database_path,
        emb_path=forward_embedding_path,
    )
    retro_retriever = FaissDataRetriever(
        data_path=database_path,
        emb_path=retro_embedding_path,
    )

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
        path="/flask_similarity_search_tools/mcp",
        json_response=True,
    )


if __name__ == "__main__":
    main()
