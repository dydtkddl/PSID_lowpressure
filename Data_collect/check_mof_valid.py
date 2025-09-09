#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import logging
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

from pymatgen.core import Structure

def setup_logger(logfile="compare_mofs.log"):
    logging.basicConfig(
        filename=logfile,
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s"
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    console.setFormatter(formatter)
    logging.getLogger().addHandler(console)

def count_atoms(cif_path):
    try:
        s = Structure.from_file(cif_path)
        return len(s.sites)
    except Exception as e:
        logging.error(f"Failed to parse {cif_path}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Compare atom counts between MOFs and CoreMOFs")
    parser.add_argument("--txt", required=True, help="Path to txt file with CIF base names (no extension)")
    parser.add_argument("--basepath", required=True, help="Base path to MOF CIF files")
    parser.add_argument("--coremofdir", required=True, help="Directory containing CoreMOF CIF files")
    parser.add_argument("--n_jobs", type=int, default=1, help="Number of parallel jobs")
    parser.add_argument("--out", default="compare_atoms.csv", help="Output CSV file")
    args = parser.parse_args()

    setup_logger()

    # 1. Load list from txt
    with open(args.txt, "r") as f:
        mof_names = [line.strip() for line in f if line.strip()]
    logging.info(f"Loaded {len(mof_names)} MOF names from {args.txt}")

    # 2. Build CoreMOF dict
    core_cifs = [f for f in os.listdir(args.coremofdir) if f.endswith(".cif")]
    core_dict = {}
    for cif in tqdm(core_cifs, desc="Reading CoreMOFs"):
        name = os.path.splitext(cif)[0]
        cif_path = os.path.join(args.coremofdir, cif)
        core_dict[name] = count_atoms(cif_path)
    logging.info(f"Parsed {len(core_dict)} CoreMOF CIFs from {args.coremofdir}")

    # 3. Compare base vs core
    results = []

    def process_one(name):
        base_path = os.path.join(args.basepath, f"{name}.cif")
        base_atoms = count_atoms(base_path)
        core_atoms = core_dict.get(name, None)
        return {"mof_name": name, "base_atoms": base_atoms, "core_atoms": core_atoms}

    with ProcessPoolExecutor(max_workers=args.n_jobs) as ex:
        futures = {ex.submit(process_one, name): name for name in mof_names}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Comparing"):
            res = fut.result()
            results.append(res)
            logging.info(f"{res['mof_name']}: base={res['base_atoms']} | core={res['core_atoms']}")

    # 4. Save CSV
    df = pd.DataFrame(results)
    df.to_csv(args.out, index=False)
    logging.info(f"Saved results to {args.out}")

if __name__ == "__main__":
    main()

