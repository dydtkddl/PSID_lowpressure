import os
import argparse
import subprocess
from joblib import Parallel, delayed
from tqdm import tqdm

pressure = "5"
temp = "298"
gas = "CO2"
base_input = "00base_template.input"
CUTOFFVDW = "12.5"
CUTOFFCHARGECHARGE = "12.50"
CUTOFFCHARGEBONDDIPOLE = "12.50"
CUTOFFBONDDIPOLEBONDDIPOLE = "12.50"

NumberOfCycles = "20000"
NumberOfInitializationCycles = "1000"
PrintEvery = "1000"

def run_simulation(mof):
    try:
        subprocess.run([
            "python", "01make_simulation.py",
            "--mof", mof,
            "--pressure", pressure,
            "--temp", temp,
            "--gas", gas,
            "--base_input", base_input,
            "--cutoffvdw", CUTOFFVDW,
            "--cutoffchargecharge", CUTOFFCHARGECHARGE,
            "--cutoffchargebonddipole", CUTOFFCHARGEBONDDIPOLE,
            "--cutoffbonddipolebonddipole", CUTOFFBONDDIPOLEBONDDIPOLE,
            "--NumberOfCycles", NumberOfCycles,
            "--NumberOfInitializationCycles", NumberOfInitializationCycles,
            "--PrintEvery", PrintEvery,
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error while executing simulation for {mof}: {e}")
    except Exception as e:
        print(f"Unexpected error for {mof}: {e}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n_jobs", 
        type=int, 
        default=1,
        help="Number of parallel jobs to run simultaneously (default=1)"
    )
    args = parser.parse_args()

    cif_list_path = "04cif_list.txt"
    with open(cif_list_path, "r") as file:
        cif_list = [line.strip() for line in file if line.strip()]

    print(f"Found {len(cif_list)} CIFs. Starting parallel processing with n_jobs={args.n_jobs} ...")

    Parallel(n_jobs=args.n_jobs)(
        delayed(run_simulation)(mof) 
        for mof in tqdm(cif_list, desc="Simulations")
    )

if __name__ == "__main__":
    main()
