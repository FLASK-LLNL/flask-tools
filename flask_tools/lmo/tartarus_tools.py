import os, sys
from pathlib import Path
import shutil
import tempfile
import time
import click
import subprocess
from subprocess import DEVNULL
import rdkit
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, RDConfig
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer
import os
from fastmcp import FastMCP
import numpy as np

#Check that xtb and CREST path are set correctly
# 1. Define the root directory (Equivalent to XTB_HOME)
xtb_home = os.environ.get('XTB_HOME')

# Sanity check: Ensure the variable exists
if not xtb_home:
    print("Error: XTB_HOME is not set. Please run 'export XTB_HOME=/path/to/xtb' before starting.")

# 2. Update the environment dictionary
env = os.environ.copy()

# Set XTBPATH (Crucial for parameter files)
env["XTBPATH"] = os.path.join(xtb_home, "share", "xtb")

# Update PATH (So Python can find the 'xtb' executable)
env["PATH"] = os.path.join(xtb_home, "bin") + os.pathsep + env.get("PATH", "")

# Update LD_LIBRARY_PATH (So the system can find the .so files in /lib)
env["LD_LIBRARY_PATH"] = os.path.join(xtb_home, "lib") + os.pathsep + env.get("LD_LIBRARY_PATH", "")

# 3. Test it by running a version check
try:
    result = subprocess.run(
        ["xtb", "--version"], 
        env=env, 
        capture_output=True, 
        text=True
    )
    print("xtb is working! Version output:")
    print(result.stdout)
except FileNotFoundError:
    print("Error: Could not find the xtb executable. Check your paths.")

#Also check for the crest path
crest_path=os.environ.get("CREST_HOME")

# We append the new path to the existing PATH using the system's path separator (:)
os.environ["PATH"] = os.environ.get("PATH", "") + os.pathsep + crest_path
# 3. Verify it works by calling crest
try:
    # This will now find 'crest' because you updated the PATH
    subprocess.run(["crest", "--version"], check=True)
except FileNotFoundError:
    print("Error: 'crest' executable not found in the provided path.")

def log_progress(message):
    timestamp = time.strftime('%H:%M:%S')
    print(f'[{timestamp}] {message}', flush=True)


def run_command(command, verbose, step_name=None):
    started_at = time.time()
    if step_name:
        log_progress(f'Starting: {step_name}')

    run_kwargs = {
        'shell': True,
        'env': globals().get('env')
    }

    if verbose:
        result = subprocess.run(command, **run_kwargs)
    else:
        result = subprocess.run(command, stdout=DEVNULL, stderr=DEVNULL, **run_kwargs)

    elapsed = time.time() - started_at
    if result.returncode != 0:
        if step_name:
            log_progress(f'Failed: {step_name} (exit code {result.returncode}, {elapsed:.1f}s)')
        raise RuntimeError(f"Command failed with exit code {result.returncode}: {command}")

    if step_name:
        log_progress(f'Finished: {step_name} ({elapsed:.1f}s)')

# Surrogate model for Jsc
def gaussian(x, A, B):
    return A * np.exp(-x** 2 / B)


def require_executables(*names):
    missing = [name for name in names if shutil.which(name) is None]
    if missing:
        raise RuntimeError(
            "Missing required executable(s): {}. Install them and make sure they are on PATH.".format(
                ", ".join(missing)
            )
        )

mcp = FastMCP("Tartarus MCP Tools")
@mcp.tool()
def get_OPV_properties(smile, verbose=True, scratch: str='/tmp'): 
    '''
    Return fitness functions for the design of organic photovoltaics molecules.

    :param smile: `str` representing molecule
    :param verbose: `bool` turn on print statements for debugging
    :param scratch: `str` temporary directory

    :returns: 
        - dipm - `float` dipole moment
        - gap - `float` homo lumo gap
        - lumo - `float` lumo energy
        - combined - `float` combined objective (HL_gap - LUMO + dipole moment)
        - pce_pcbm_sas - `float` Power Conversion Efficiency (PCE) for phenyl-C61-butyric acid methyl ester (PCBM) acceptor minus synthetic accessibility score (SAS).
        - pce_pcdtbt_sas - `float` Power Conversion Efficiency (PCE) for poly[N-90-heptadecanyl-2,7-carbazole-alt-5,5-(40,70-di-2-thienyl-20,10,30-benzothiadiazole)] (PCDTBT) donor
        minus synthetic accessibility score (SAS).
    '''

    require_executables('obabel', 'xtb', 'crest')

    # Create and switch to temporary directory
    owd = Path.cwd()
    scratch_path = Path(scratch)
    tmp_dir = tempfile.TemporaryDirectory(dir=scratch_path)
    os.chdir(tmp_dir.name)
    log_progress(f'Working directory: {tmp_dir.name}')

    try:
        # Create mol object
        mol = Chem.MolFromSmiles(smile)
        mol = Chem.AddHs(mol)
        if mol == None:
            return "INVALID"
        charge = Chem.rdmolops.GetFormalCharge(mol)
        atom_number = mol.GetNumAtoms()

        sas = sascorer.calculateScore(mol)

        with open('test.smi', 'w') as f:
            f.writelines([smile])

        # Prepare the input file:
        run_command('obabel test.smi --gen3D -O test.xyz', verbose, 'Generate 3D coordinates')

        # Run the preliminary xtb:
        command_pre = 'CHARGE={};xtb {} --gfn 0 --opt normal -c $CHARGE --iterations 4000'.format(charge, 'test.xyz')
        run_command(command_pre, verbose, 'Preliminary xTB optimization')
        run_command("rm -f ./gfnff_charges ./gfnff_topo", verbose, 'Remove preliminary GFN-FF files')

        # Run crest conformer ensemble
        crest_executable = os.path.join(os.environ.get("CREST_HOME", ""), "crest")
        command_crest = (
    'OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 '
    'CHARGE={};{} {} -gff -mquick -chrg $CHARGE --noreftopo --T 1'
).format(charge, crest_executable, 'xtbopt.xyz')
        run_command(command_crest, verbose, 'CREST conformer search')
        run_command('rm -f ./gfnff_charges ./gfnff_topo', verbose, 'Remove CREST GFN-FF files')
        run_command('head -n {} crest_conformers.xyz > crest_best.xyz'.format(atom_number+2), verbose, 'Extract best conformer')

        # Run the calculation:
        command = 'CHARGE={};xtb {} --opt normal -c $CHARGE --iterations 4000 > out_dump'.format(charge, 'crest_best.xyz')
        run_command(command, verbose, 'Final xTB optimization')

        # Read the output:
        log_progress('Parsing xTB output')
        with open('./out_dump', 'r') as f:
            text_content = f.readlines()

        output_index = [i for i in range(len(text_content)) if 'Property Printout' in text_content[i]]
        if not output_index:
            raise RuntimeError(
                "Failed to parse xTB output from 'out_dump': missing 'Property Printout'. "
                "Run get_OPV_properties(..., verbose=True) to inspect the external tool output."
            )

        text_content = text_content[output_index[0]:]
        homo_data = [x for x in text_content if '(HOMO)' in x]
        lumo_data = [x for x in text_content if '(LUMO)' in x]
        homo_lumo_gap = [x for x in text_content if 'HOMO-LUMO GAP' in x]
        mol_dipole = [text_content[i:i+4] for i, x in enumerate(text_content) if 'molecular dipole:' in x]

        if not homo_data or not lumo_data or not homo_lumo_gap or not mol_dipole:
            raise RuntimeError(
                "Failed to parse all expected properties from xTB output. "
                "Run get_OPV_properties(..., verbose=True) to inspect the external tool output."
            )

        lumo_val = float(lumo_data[0].split(' ')[-2])
        homo_val = float(homo_data[0].split(' ')[-2])
        homo_lumo_val = float(homo_lumo_gap[0].split(' ')[-5])
        mol_dipole_val = float(mol_dipole[0][-1].split(' ')[-1])

        # Determine value of custom function for optimization
        HL_range_rest = homo_lumo_val # Good range for the HL gap: 0.8856-3.2627
        if 0.8856 <= HL_range_rest <= 3.2627:
            HL_range_rest = 1.0
        elif HL_range_rest < 0.8856:
            HL_range_rest = 0.1144 + homo_lumo_val
        else:
            HL_range_rest = 4.2627 - HL_range_rest
        combined = mol_dipole_val + HL_range_rest - lumo_val # Maximize this function

        # Compute calibrated homo and lumo levels
        homo_cal = homo_val * 0.8051030400316004 + 2.5376777453204133
        lumo_cal = lumo_val * 0.8787863933542347 + 3.7912767464357200

        # Define parameters for Scharber model
        A = 433.11633173034136
        B = 2.3353220382662894
        Pin = 900.1393292842149

        # Scharber model objective 1: Optimization of donor for phenyl-C61-butyric acid methyl ester (PCBM) acceptor
        voc_1 = (abs(homo_cal) - abs(-4.3)) - 0.3
        if voc_1 < 0.0:
            voc_1 = 0.0
        lumo_offset_1 = lumo_cal + 4.3
        if lumo_offset_1 < 0.3:
            pce_1 = 0.0
        else:
            jsc_1 = gaussian(lumo_cal - homo_cal, A, B)
            if jsc_1 > 415.22529811760637:
                jsc_1 = 415.22529811760637
            pce_1 = 100 * voc_1 * 0.65 * jsc_1 / Pin

        # Scharber model objective 2: Optimization of acceptor for poly[N-90-heptadecanyl-2,7-carbazole-alt-5,5-(40,70-di-2-thienyl-20,10,30-benzothiadiazole)] (PCDTBT) donor
        voc_2 = (abs(-5.5) - abs(lumo_cal)) - 0.3
        if voc_2 < 0.0:
            voc_2 = 0.0
        lumo_offset_2 = -3.6 - lumo_cal
        if lumo_offset_2 < 0.3:
            pce_2 = 0.0
        else:
            jsc_2 = gaussian(lumo_cal - homo_cal, A, B)
            if jsc_2 > 415.22529811760637:
                jsc_2 = 415.22529811760637
            pce_2 = 100 * voc_2 * 0.65 * jsc_2 / Pin
    finally:
        os.chdir(owd)
        tmp_dir.cleanup()

    # assign values
    pce_pcbm_sas = pce_1 - sas
    pce_pcdtbt_sas = pce_2 - sas
    log_progress(
        f'Completed property calculation: PCE_PCBM-SAS={pce_pcbm_sas:.6f}, '
        f'PCE_PCDTBT-SAS={pce_pcdtbt_sas:.6f}'
    )

    return pce_pcbm_sas, pce_pcdtbt_sas

@click.command()
@click.option(
    "--transport",
    type=click.Choice(["stdio", "streamable-http"]),
    help="MCP transport type",
    default="streamable-http",
)
@click.option("--port", type=int, default=8124, help="Port to run the server on")
@click.option("--host", type=str, default=None, help="Host to run the server on")
@click.option("--name", type=str, default="Tartarus MCP Tools", help="Name of the MCP server")
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
):
    # Run MCP server
    mcp.run(
        transport=transport,
        host=host,
        port=port,
        path=f"/tartarus_tools/mcp",
    )

if __name__ == '__main__':
    main()
