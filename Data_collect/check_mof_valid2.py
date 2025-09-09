#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import logging
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import subprocess

# ── Logger ──
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

# ── subprocess로 원자수 세기 ──
def count_atoms_with_subprocess(cif_path):
    """pymatgen Structure.from_file을 별도 subprocess에서 실행"""
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
        return None

# ── 개별 MOF 처리 ──
def process_one(name, basepath, core_dict):
    base_path = os.path.join(basepath, f"{name}.cif")
    base_atoms = count_atoms_with_subprocess(base_path) if os.path.exists(base_path) else None
    core_atoms = core_dict.get(name, None)
    return {"mof_name": name, "base_atoms": base_atoms, "core_atoms": core_atoms}

# ── 메인 ──
def main():
    parser = argparse.ArgumentParser(description="Compare atom counts between MOFs and CoreMOFs (subprocess parallel)")
    parser.add_argument("--txt", required=True, help="Path to txt file with CIF base names (no extension)")
    parser.add_argument("--basepath", required=True, help="Base path to MOF CIF files")
    parser.add_argument("--coremofdir", required=True, help="Directory containing CoreMOF CIF files")
    parser.add_argument("--n_cpus", type=int, default=1, help="Number of parallel subprocesses")
    parser.add_argument("--out", default="compare_atoms.csv", help="Output CSV file")
    args = parser.parse_args()

    setup_logger()
    logging.info("Start MOF-CoreMOF comparison (subprocess mode)")

    # 1. txt 파일 읽기
    with open(args.txt, "r") as f:
        mof_names = [line.strip() for line in f if line.strip()]
    logging.info(f"Loaded {len(mof_names)} MOF names")

    # 2. CoreMOF 원자 수 dict 만들기
    core_dict = {}
    core_cifs = [f for f in os.listdir(args.coremofdir) if f.endswith(".cif")]
    for cif in tqdm(core_cifs, desc="Reading CoreMOFs"):
        name = os.path.splitext(cif)[0]
        cif_path = os.path.join(args.coremofdir, cif)
        core_dict[name] = count_atoms_with_subprocess(cif_path)
    logging.info(f"Parsed {len(core_dict)} CoreMOF CIFs")

    # 3. subprocess 병렬 실행
    results = []
    with ThreadPoolExecutor(max_workers=args.n_cpus) as ex:
        futures = [ex.submit(process_one, name, args.basepath, core_dict) for name in mof_names]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Comparing"):
            res = fut.result()
            results.append(res)
            logging.info(f"{res['mof_name']}: base={res['base_atoms']} | core={res['core_atoms']}")

    # 4. 결과 저장
    df = pd.DataFrame(results)
    df.to_csv(args.out, index=False)
    logging.info(f"Saved results to {args.out}")
    logging.info("Done!")

if __name__ == "__main__":
    main()

