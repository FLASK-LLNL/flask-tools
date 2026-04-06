# server.py
# pip install "mcp[cli]" rdkit-pypi
import click
from typing_extensions import TypedDict
from loguru import logger
from typing import Literal, List
from fastmcp import FastMCP

# import your existing module (the one we’ve been building)
import flask_tools.chemistry.polymerizer as pr


# ----- Structured outputs for nicer tool schemas -----
class Suggestion(TypedDict):
    strategy: str
    confidence: float
    reason: str


class PolymerizeResult(TypedDict):
    repeat_smiles: str
    strategy: str
    rationale: str


# ----- Tools -----

from flask_tools.utils.server_utils import get_hostname
from lc_conductor.tool_registration import register_tool_server


@click.command()
@click.option(
    "--transport",
    type=click.Choice(["stdio", "streamable-http"]),
    help="MCP transport type",
    default="streamable-http",
)
@click.option("--port", type=int, default=8129, help="Port to run the server on")
@click.option("--host", type=str, default=None, help="Host to run the server on")
@click.option(
    "--name", type=str, default="polymerizer_tools", help="Name of the MCP server"
)
@click.option(
    "--copilot-port", type=int, default=8001, help="Port to the running copilot backend"
)
@click.option(
    "--copilot-host", type=str, default=None, help="Host to the running copilot backend"
)
def main(
    transport: str,
    port: str,
    host: str,
    name: str,
    copilot_port: str,
    copilot_host: str,
):
    if host is None:
        _, host = get_hostname()

    # Init MCP server
    mcp = FastMCP(
        "Polymerizer",
        instructions=(
            "Generate polymer repeat-unit SMILES from one monomer or from a "
            "two-monomer monomer set, and suggest supported forward "
            "polymerization rules for the corresponding monomer-to-polymer "
            "transformation."
        ),
        # description=(
        # "Expose monomer→polymer repeat transforms via MCP tools. "
        # "Tools: polymerize_explicit, polymerize_auto, suggest_rules."
    )

    @mcp.tool()
    def polymerize_explicit(
        monomer_smiles: str,
        strategy: str,
        comonomer_smiles: str | None = None,
        bigsmiles_wrap: bool = False,
    ) -> str:
        """
        Apply a specific forward polymerization strategy and return the polymer
        repeat-unit SMILES.

        Use `comonomer_smiles` for supported two-monomer step-growth cases such
        as polyester and polyamide formation. Single-monomer cases include
        chain-growth, ring-opening, and supported bicyclic ROMP transforms.
        """
        return pr.polymerize_explicit(
            monomer_smiles=monomer_smiles,
            strategy=strategy,
            comonomer_smiles=comonomer_smiles,
            bigsmiles_wrap=bigsmiles_wrap,
        )

    @mcp.tool()
    def suggest_rules(monomer_smiles: str | list[str], top_k: int = 5) -> List[Suggestion]:
        """
        Suggest ranked forward polymerization rules for converting one monomer
        or a two-monomer monomer set into a polymer repeat unit.
        """
        return pr.suggest_rules(monomer_smiles=monomer_smiles, top_k=top_k)

    @mcp.tool()
    def polymerize_auto(
        monomer_smiles: str | list[str],
        min_confidence: float = 0.80,
        allow_fallback_to_lower_confidence: bool = True,
        bigsmiles_wrap: bool = False,
    ) -> PolymerizeResult:
        """
        Auto-select a polymerization rule and return the repeat-unit SMILES.

        Pass either one monomer SMILES string or a list containing one or two
        monomer SMILES strings. Two-monomer inputs are used for supported
        step-growth sets such as diol/diacid, diol/anhydride, and
        diamine/diacid.
        """
        return pr.polymerize_auto(
            monomer_smiles=monomer_smiles,
            min_confidence=min_confidence,
            allow_fallback_to_lower_confidence=allow_fallback_to_lower_confidence,
            bigsmiles_wrap=bigsmiles_wrap,
        )

    @mcp.tool(name="classify_polymer_input")
    def assess_input(smiles: str) -> dict:
        """Classify whether a SMILES string looks like a monomer candidate or a polymer repeat/fragment."""
        return pr.assess_input(smiles)

    @mcp.tool(name="verify_monomer_candidate_for_polymer")
    def check_retrosynthesis_candidate(
        target_polymer_smiles: str,
        candidate_monomer_smiles: str,
        strategy: str | None = None,
        min_confidence: float = 0.80,
        allow_fallback_to_lower_confidence: bool = True,
    ) -> dict:
        """
        Verify whether a candidate monomer reproduces a target polymer repeat.

        This is a forward-validation check for polymer retrosynthesis workflows.
        For two-monomer hypotheses, use `polymerize_auto` or `polymerize_explicit`
        on the monomer set directly.
        """
        return pr.check_retrosynthesis_candidate(
            target_polymer_smiles=target_polymer_smiles,
            candidate_monomer_smiles=candidate_monomer_smiles,
            strategy=strategy,
            min_confidence=min_confidence,
            allow_fallback_to_lower_confidence=allow_fallback_to_lower_confidence,
        )

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
        path=f"/{name}/mcp",
        json_response=True,
    )


# ----- Run the server -----
if __name__ == "__main__":
    main()
