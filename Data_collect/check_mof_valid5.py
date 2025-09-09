#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import logging
import pandas as pd
from multiprocessing import get_context, cpu_count
from tqdm import tqdm
from pymatgen.core import Structure  # 직접 호출

def setup_logger(logfile="compare_mofs.log"):
    logging.basicConfig(
        filename=logfile,
        filemode="w",
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s"
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    console.setFormatter(formatter)
    logging.getLogger().addHandler(console)

def count_atoms(cif_path):
    """pymatgen 직접 호출해서 원자 개수 세기"""
    if not os.path.exists(cif_path):
        return None
    try:
        s = Structure.from_file(cif_path)
        return len(s.sites)
    except Exception as e:
        logging.error(f"Failed to parse {cif_path}: {e}")
        return None

def process_one(name_base_core):
    """MOF 이름과 경로 받아서 base+core atom count"""
    name, basepath, coremofdir = name_base_core
    base_cif = os.path.join(basepath, f"{name}.cif")
    core_cif = os.path.join(coremofdir, f"{name}.cif")

    base_atoms = count_atoms(base_cif)
    core_atoms = count_atoms(core_cif)

    logging.info(f"{name}: base={base_atoms}, core={core_atoms}")
    return {"mof_name": name, "base_atoms": base_atoms, "core_atoms": core_atoms}

def main():
    parser = argparse.ArgumentParser(description="Compare atom counts in MOFs and CoREMOFs")
    parser.add_argument("--txt", required=True, help="Text file with MOF names (no extension)")
    parser.add_argument("--basepath", required=True, help="Directory containing MOF CIFs")
    parser.add_argument("--coremofdir", required=True, help="Directory containing CoREMOF CIFs")
    parser.add_argument("--n_cpus", type=int, default=1, help="Number of parallel processes")
    parser.add_argument("--out", default="compare_results.csv", help="Output CSV file")
    args = parser.parse_args()

    setup_logger()
    logging.info("Start MOF-CoREMOF comparison")

    with open(args.txt, "r") as f:
        mof_names = [line.strip() for line in f if line.strip()]
    logging.info(f"Loaded {len(mof_names)} MOF names from {args.txt}")

    tasks = [(name, args.basepath, args.coremofdir) for name in mof_names]
    n_jobs = min(args.n_cpus, cpu_count())

    results = []
    with get_context("spawn").Pool(processes=n_jobs) as pool:
        for res in tqdm(pool.imap_unordered(process_one, tasks), total=len(tasks), desc="Processing"):
            results.append(res)

    df = pd.DataFrame(results)
    df.to_csv(args.out, index=False)
    logging.info(f"Saved results to {args.out}")
    logging.info("Done!")

if __name__ == "__main__":
    main()

