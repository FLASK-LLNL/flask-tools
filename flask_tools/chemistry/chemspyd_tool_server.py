################################################################################
## Copyright 2025 Lawrence Livermore National Security, LLC. and Binghamton University.
## See the top-level LICENSE file for details.
##
## SPDX-License-Identifier: Apache-2.0
################################################################################

import click
from typing import List, Optional, Union
from typing_extensions import TypedDict
from loguru import logger
from fastmcp import FastMCP

from flask_tools.utils.server_utils import get_hostname
from lc_conductor.tool_registration import register_tool_server


class TransferSolidResult(TypedDict):
    """Result from solid transfer operation"""

    success: bool
    dispensed_weights: List[float]
    message: str


class MeasureLevelResult(TypedDict):
    """Result from level measurement"""

    success: bool
    levels: List[float]
    zone: str


class StatusResult(TypedDict):
    """Result from status read operation"""

    success: bool
    status: Union[dict, float]


@click.command()
@click.option(
    "--transport",
    type=click.Choice(["stdio", "streamable-http"]),
    help="MCP transport type",
    default="streamable-http",
)
@click.option("--port", type=int, default=8130, help="Port to run the server on")
@click.option("--host", type=str, default=None, help="Host to run the server on")
@click.option(
    "--name", type=str, default="chemspyd_tools", help="Name of the MCP server"
)
@click.option(
    "--copilot-port", type=int, default=8001, help="Port to the running copilot backend"
)
@click.option(
    "--copilot-host", type=str, default=None, help="Host to the running copilot backend"
)
@click.option(
    "--cmd-folder",
    type=str,
    required=True,
    help="Path to the command folder for ChemSpeed communication",
)
@click.option(
    "--element-config",
    type=str,
    required=True,
    help="Path to element_config.json file",
)
@click.option(
    "--system-liquids",
    type=str,
    required=True,
    help="Path to system_liquids.json file",
)
@click.option(
    "--statuses",
    type=str,
    required=True,
    help="Path to statuses.json file",
)
@click.option(
    "--simulation",
    is_flag=True,
    default=False,
    help="Run in simulation mode (no hardware commands sent)",
)
@click.option(
    "--verbosity",
    type=int,
    default=2,
    help="Logging verbosity level (0-3)",
)
def main(
    transport: str,
    port: int,
    host: str,
    name: str,
    copilot_port: int,
    copilot_host: str,
    cmd_folder: str,
    element_config: str,
    system_liquids: str,
    statuses: str,
    simulation: bool,
    verbosity: int,
):
    """ChemSpeed MCP Server - Control ChemSpeed robotic chemistry platforms via MCP"""

    if host is None:
        _, host = get_hostname()

    # Initialize ChemSpeed Controller
    try:
        from chemspyd import Controller
        from chemspyd import routines

        controller = Controller(
            cmd_folder=cmd_folder,
            element_config=element_config,
            system_liquids=system_liquids,
            statuses=statuses,
            verbosity=verbosity,
            simulation=simulation,
        )
        logger.info(f"ChemSpeed Controller initialized (simulation={simulation})")
    except Exception as e:
        logger.error(f"Failed to initialize ChemSpeed Controller: {e}")
        raise

    # Init MCP server
    mcp = FastMCP(
        "ChemSpeed Control",
        instructions=(
            "Control ChemSpeed robotic chemistry platforms for automated synthesis. "
            "Includes liquid/solid transfer, temperature control, stirring, vacuum, "
            "reflux, vial transport, and high-level chemistry routines."
        ),
    )

    # ====== Core Controller Tools ======

    @mcp.tool()
    def transfer_liquid(
        source: str,
        destination: str,
        volume: float,
        needle: int,
        src_flow: float = 20.0,
        dst_flow: float = 40.0,
        rinse_volume: float = 2.0,
    ) -> dict:
        """
        Transfer liquid from source to destination zone volumetrically.

        Args:
            source: Source zone identifier (e.g., "RACKL:1")
            destination: Destination zone identifier (e.g., "RACKR:1")
            volume: Volume to transfer in mL
            needle: Needle number to use (0 means all needles)
            src_flow: Draw speed at source in mL/min (default 20.0)
            dst_flow: Dispense speed at destination in mL/min (default 40.0)
            rinse_volume: Needle rinsing volume after action in mL (default 2.0)

        Returns:
            Result dict with success status and message
        """
        try:
            controller.transfer_liquid(
                source=source,
                destination=destination,
                volume=volume,
                needle=needle,
                src_flow=src_flow,
                dst_flow=dst_flow,
                rinse_volume=rinse_volume,
            )
            return {
                "success": True,
                "message": f"Transferred {volume} mL from {source} to {destination}",
            }
        except Exception as e:
            logger.error(f"transfer_liquid failed: {e}")
            return {"success": False, "message": f"Error: {str(e)}"}

    @mcp.tool()
    def transfer_solid(
        source: str,
        destination: str,
        weight: float,
        height: float = 0.0,
        chunk: float = 0.1,
        equilib: float = 5.0,
    ) -> TransferSolidResult:
        """
        Transfer solid from source to destination zone gravimetrically.

        Args:
            source: Source zone identifier
            destination: Destination zone identifier
            weight: Target mass to dispense in mg
            height: Dispense height relative to vial top in mm (default 0.0)
            chunk: Rough dispensing chunk size in mg (default 0.1)
            equilib: Equilibration time for balance in seconds (default 5.0)

        Returns:
            Result dict with success status, dispensed weights, and message
        """
        try:
            dispensed_weights = controller.transfer_solid(
                source=source,
                destination=destination,
                weight=weight,
                height=height,
                chunk=chunk,
                equilib=equilib,
            )
            return {
                "success": True,
                "dispensed_weights": dispensed_weights,
                "message": f"Dispensed solid to {destination}. Target: {weight} mg",
            }
        except Exception as e:
            logger.error(f"transfer_solid failed: {e}")
            return {
                "success": False,
                "dispensed_weights": [],
                "message": f"Error: {str(e)}",
            }

    @mcp.tool()
    def set_temperature(
        temp_zone: str,
        state: str,
        temperature: float = 25.0,
        ramp: float = 0.0,
    ) -> dict:
        """
        Control heating/cooling zones.

        Args:
            temp_zone: Temperature zone identifier
            state: "on" to activate heating/cooling, "off" to deactivate
            temperature: Target temperature in Celsius (default 25.0)
            ramp: Temperature ramp rate in C/min (default 0.0 = no ramping)

        Returns:
            Result dict with success status and message
        """
        try:
            controller.set_temperature(
                temp_zone=temp_zone,
                state=state,
                temperature=temperature,
                ramp=ramp,
            )
            return {
                "success": True,
                "message": f"Set {temp_zone} to {state} at {temperature}°C",
            }
        except Exception as e:
            logger.error(f"set_temperature failed: {e}")
            return {"success": False, "message": f"Error: {str(e)}"}

    @mcp.tool()
    def set_stir(stir_zone: str, state: str, rpm: float = 0.0) -> dict:
        """
        Control stirring in specified zones.

        Args:
            stir_zone: Stir zone identifier
            state: "on" to start stirring, "off" to stop
            rpm: Stirring speed in rotations per minute (default 0.0)

        Returns:
            Result dict with success status and message
        """
        try:
            controller.set_stir(stir_zone=stir_zone, state=state, rpm=rpm)
            return {
                "success": True,
                "message": f"Set stirring in {stir_zone} to {state} at {rpm} rpm",
            }
        except Exception as e:
            logger.error(f"set_stir failed: {e}")
            return {"success": False, "message": f"Error: {str(e)}"}

    @mcp.tool()
    def set_vacuum(vac_zone: str, state: str, vacuum: float = 1000.0) -> dict:
        """
        Control vacuum pressure in specified zones.

        Args:
            vac_zone: Vacuum zone identifier
            state: "on" to activate vacuum, "off" to deactivate
            vacuum: Target vacuum pressure in mbar (default 1000.0)

        Returns:
            Result dict with success status and message
        """
        try:
            controller.set_vacuum(vac_zone=vac_zone, state=state, vacuum=vacuum)
            return {
                "success": True,
                "message": f"Set vacuum in {vac_zone} to {state} at {vacuum} mbar",
            }
        except Exception as e:
            logger.error(f"set_vacuum failed: {e}")
            return {"success": False, "message": f"Error: {str(e)}"}

    @mcp.tool()
    def set_reflux(reflux_zone: str, state: str, temperature: float = 0.0) -> dict:
        """
        Control reflux chilling in specified zones.

        Args:
            reflux_zone: Reflux zone identifier
            state: "on" to activate reflux cooling, "off" to deactivate
            temperature: Target cooling temperature in Celsius (default 0.0)

        Returns:
            Result dict with success status and message
        """
        try:
            controller.set_reflux(
                reflux_zone=reflux_zone, state=state, temperature=temperature
            )
            return {
                "success": True,
                "message": f"Set reflux in {reflux_zone} to {state} at {temperature}°C",
            }
        except Exception as e:
            logger.error(f"set_reflux failed: {e}")
            return {"success": False, "message": f"Error: {str(e)}"}

    @mcp.tool()
    def set_drawer(zone: str, state: str, environment: str = "none") -> dict:
        """
        Control ISYNTH drawers (open/close).

        Args:
            zone: Drawer zone identifier
            state: "open" to open drawer, "close" to close drawer
            environment: Environment type - "none", "argon", or other (default "none")

        Returns:
            Result dict with success status and message
        """
        try:
            controller.set_drawer(zone=zone, state=state, environment=environment)
            return {
                "success": True,
                "message": f"Set drawer {zone} to {state} (environment: {environment})",
            }
        except Exception as e:
            logger.error(f"set_drawer failed: {e}")
            return {"success": False, "message": f"Error: {str(e)}"}

    @mcp.tool()
    def vial_transport(source: str, destination: str, tool: str = "4FTG") -> dict:
        """
        Transport vials between zones.

        Args:
            source: Source zone identifier
            destination: Destination zone identifier
            tool: Tool type to use for transport (default "4FTG")

        Returns:
            Result dict with success status and message
        """
        try:
            controller.vial_transport(source=source, destination=destination, tool=tool)
            return {
                "success": True,
                "message": f"Transported vial from {source} to {destination}",
            }
        except Exception as e:
            logger.error(f"vial_transport failed: {e}")
            return {"success": False, "message": f"Error: {str(e)}"}

    @mcp.tool()
    def measure_level(zone: str) -> MeasureLevelResult:
        """
        Measure material level in wells.

        Args:
            zone: Zone identifier to measure

        Returns:
            Result dict with success status, level measurements, and zone
        """
        try:
            levels = controller.measure_level(zone=zone)
            return {
                "success": True,
                "levels": levels,
                "zone": zone,
            }
        except Exception as e:
            logger.error(f"measure_level failed: {e}")
            return {
                "success": False,
                "levels": [],
                "zone": zone,
            }

    @mcp.tool()
    def wait(duration: float) -> dict:
        """
        Wait for specified duration.

        Args:
            duration: Wait time in seconds

        Returns:
            Result dict with success status and message
        """
        try:
            controller.wait(duration=duration)
            return {"success": True, "message": f"Waited for {duration} seconds"}
        except Exception as e:
            logger.error(f"wait failed: {e}")
            return {"success": False, "message": f"Error: {str(e)}"}

    @mcp.tool()
    def unmount_all() -> dict:
        """
        Unmount all tools from the robotic arm.

        Returns:
            Result dict with success status and message
        """
        try:
            controller.unmount_all()
            return {"success": True, "message": "All tools unmounted"}
        except Exception as e:
            logger.error(f"unmount_all failed: {e}")
            return {"success": False, "message": f"Error: {str(e)}"}

    @mcp.tool()
    def read_status(key: Optional[str] = None) -> StatusResult:
        """
        Read instrument status.

        Args:
            key: Specific status key to read (optional). If None, returns all status.

        Returns:
            Result dict with success status and status data
        """
        try:
            status = controller.read_status(key=key)
            return {"success": True, "status": status}
        except Exception as e:
            logger.error(f"read_status failed: {e}")
            return {"success": False, "status": {}}

    @mcp.tool()
    def stop_manager() -> dict:
        """
        Stop the ChemSpeed manager safely.

        Returns:
            Result dict with success status and message
        """
        try:
            controller.stop_manager()
            return {"success": True, "message": "Manager stopped successfully"}
        except Exception as e:
            logger.error(f"stop_manager failed: {e}")
            return {"success": False, "message": f"Error: {str(e)}"}

    # ====== High-Level Routine Tools ======

    @mcp.tool()
    def prime_pumps(pump: int, volume: float) -> dict:
        """
        Prime ChemSpeed pumps.

        Args:
            pump: Pump number to prime
            volume: Volume to use for priming in mL

        Returns:
            Result dict with success status and message
        """
        try:
            routines.prime_pumps(chmspd=controller, pump=pump, volume=volume)
            return {
                "success": True,
                "message": f"Primed pump {pump} with {volume} mL",
            }
        except Exception as e:
            logger.error(f"prime_pumps failed: {e}")
            return {"success": False, "message": f"Error: {str(e)}"}

    @mcp.tool()
    def inject_to_hplc(
        source: str,
        destination: str,
        volume: float,
        needle: int,
        src_flow: float = 10.0,
        dst_flow: float = 0.5,
        equib_dst: float = 30.0,
    ) -> dict:
        """
        Inject liquid to HPLC injection ports.

        Args:
            source: Source zone identifier
            destination: HPLC injection port zone identifier
            volume: Volume to inject in mL
            needle: Needle number to use
            src_flow: Draw speed at source in mL/min (default 10.0)
            dst_flow: Dispense speed at destination in mL/min (default 0.5)
            equib_dst: Equilibration time at destination in seconds (default 30.0)

        Returns:
            Result dict with success status and message
        """
        try:
            routines.inject_to_hplc(
                chmspd=controller,
                source=source,
                destination=destination,
                volume=volume,
                needle=needle,
                src_flow=src_flow,
                dst_flow=dst_flow,
                equib_dst=equib_dst,
            )
            return {
                "success": True,
                "message": f"Injected {volume} mL from {source} to HPLC port {destination}",
            }
        except Exception as e:
            logger.error(f"inject_to_hplc failed: {e}")
            return {"success": False, "message": f"Error: {str(e)}"}

    @mcp.tool()
    def do_schlenk_cycles(
        wells: str,
        evac_time: float,
        backfill_time: float,
        cycles: int,
    ) -> dict:
        """
        Perform Schlenk cycles (evacuation and backfill with inert gas).

        Args:
            wells: Well zone identifier
            evac_time: Evacuation time in seconds
            backfill_time: Backfill time in seconds
            cycles: Number of cycles to perform

        Returns:
            Result dict with success status and message
        """
        try:
            routines.do_schlenk_cycles(
                chmspd=controller,
                wells=wells,
                evac_time=evac_time,
                backfill_time=backfill_time,
                cycles=cycles,
            )
            return {
                "success": True,
                "message": f"Completed {cycles} Schlenk cycles on {wells}",
            }
        except Exception as e:
            logger.error(f"do_schlenk_cycles failed: {e}")
            return {"success": False, "message": f"Error: {str(e)}"}

    @mcp.tool()
    def heat_under_reflux(
        wells: str,
        stir_rate: float,
        temperature: float,
        duration: float,
    ) -> dict:
        """
        Heat reaction under reflux with cooling and stirring.

        Args:
            wells: Well zone identifier
            stir_rate: Stirring speed in rpm
            temperature: Target temperature in Celsius
            duration: Heating duration in seconds

        Returns:
            Result dict with success status and message
        """
        try:
            routines.heat_under_reflux(
                chmspd=controller,
                wells=wells,
                stir_rate=stir_rate,
                temperature=temperature,
                duration=duration,
            )
            return {
                "success": True,
                "message": f"Heated {wells} under reflux at {temperature}°C for {duration}s",
            }
        except Exception as e:
            logger.error(f"heat_under_reflux failed: {e}")
            return {"success": False, "message": f"Error: {str(e)}"}

    @mcp.tool()
    def filter_liquid(
        source: str,
        filtration_zone: str,
        destination: str,
        volume: float,
    ) -> dict:
        """
        Filter liquid samples through filtration zone.

        Args:
            source: Source zone identifier
            filtration_zone: Filtration zone identifier
            destination: Destination zone identifier for filtered liquid
            volume: Volume to filter in mL

        Returns:
            Result dict with success status and message
        """
        try:
            routines.filter_liquid(
                chmspd=controller,
                source=source,
                filtration_zone=filtration_zone,
                destination=destination,
                volume=volume,
            )
            return {
                "success": True,
                "message": f"Filtered {volume} mL from {source} through {filtration_zone} to {destination}",
            }
        except Exception as e:
            logger.error(f"filter_liquid failed: {e}")
            return {"success": False, "message": f"Error: {str(e)}"}

    @mcp.tool()
    def set_isynth_drawers(state: str, environment: str = "none") -> dict:
        """
        Set all ISYNTH drawers to the same state simultaneously.

        Args:
            state: "open" to open all drawers, "close" to close all drawers
            environment: Environment type - "none", "argon", or other (default "none")

        Returns:
            Result dict with success status and message
        """
        try:
            routines.set_isynth_drawers(
                chmspd=controller, state=state, environment=environment
            )
            return {
                "success": True,
                "message": f"Set all ISYNTH drawers to {state} (environment: {environment})",
            }
        except Exception as e:
            logger.error(f"set_isynth_drawers failed: {e}")
            return {"success": False, "message": f"Error: {str(e)}"}

    # Register server with copilot backend
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


if __name__ == "__main__":
    main()
