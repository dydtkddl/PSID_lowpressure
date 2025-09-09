#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import logging
import pandas as pd
import subprocess
from tqdm import tqdm
from queue import Queue
from threading import Thread

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

def run_one(cif_path):
    """subprocess로 pymatgen 호출해서 원자 개수 세기"""
    if not os.path.exists(cif_path):
        return None
    try:
        cmd = [
            "python3", "-c",
            (
                "from pymatgen.core import Structure; "
                f"s=Structure.from_file('{cif_path}'); "
                "print(len(s.sites))"
            )
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return int(result.stdout.strip())
    except Exception as e:
        logging.error(f"Failed to parse {cif_path}: {e}")
        return None

def worker(q, results, basepath, core_dict):
    while True:
        item = q.get()
        if item is None:
            break
        name = item
        base_path = os.path.join(basepath, f"{name}.cif")
        base_atoms = run_one(base_path)
        core_atoms = core_dict.get(name, None)
        results.append({"mof_name": name, "base_atoms": base_atoms, "core_atoms": core_atoms})
        logging.info(f"{name}: base={base_atoms}, core={core_atoms}")
        q.task_done()

def main():
    parser = argparse.ArgumentParser(description="Compare atom counts between MOFs and CoreMOFs (subprocess + threading)")
    parser.add_argument("--txt", required=True, help="Path to txt file with CIF base names (no extension)")
    parser.add_argument("--basepath", required=True, help="Base path to MOF CIF files")
    parser.add_argument("--coremofdir", required=True, help="Directory containing CoreMOF CIF files")
    parser.add_argument("--n_cpus", type=int, default=1, help="Number of parallel workers")
    parser.add_argument("--out", default="compare_atoms.csv", help="Output CSV file")
    args = parser.parse_args()

    setup_logger()
    logging.info("Start MOF-CoreMOF comparison")

    # 1. txt에서 MOF 이름 읽기
    with open(args.txt, "r") as f:
        mof_names = [line.strip() for line in f if line.strip()]
    logging.info(f"Loaded {len(mof_names)} MOF names")

    # 2. CoreMOF 디렉토리 스캔
    core_dict = {}
    core_cifs = [f for f in os.listdir(args.coremofdir) if f.endswith(".cif")]
    for cif in tqdm(core_cifs, desc="Reading CoreMOFs"):
        name = os.path.splitext(cif)[0]
        cif_path = os.path.join(args.coremofdir, cif)
        core_dict[name] = run_one(cif_path)
        logging.info(f"[CoreMOF] {name}: {core_dict[name]} atoms")

    logging.info(f"Parsed {len(core_dict)} CoreMOF CIFs")

    # 3. 병렬 처리 (Queue + Threads + subprocess)
    q = Queue()
    results = []
    threads = []
    for _ in range(args.n_cpus):
        t = Thread(target=worker, args=(q, results, args.basepath, core_dict))
        t.start()
        threads.append(t)

    for name in mof_names:
        q.put(name)

    q.join()

    for _ in threads:
        q.put(None)
    for t in threads:
        t.join()

    # 4. CSV 저장
    df = pd.DataFrame(results)
    df.to_csv(args.out, index=False)
    logging.info(f"Saved results to {args.out}")
    logging.info("Done!")

if __name__ == "__main__":
    main()

