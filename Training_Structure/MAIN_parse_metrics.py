import argparse 
import numpy as np 
import os 
import pandas as pd 
import logging
from pathlib import Path
from tqdm import tqdm

def process_dirname(dirname):
    SPL = dirname.split("_")
    GAS = SPL[0]
    TEMP = SPL[1]
    LOW = SPL[2]
    HIGH = SPL[4]
    INPUT = SPL[6]
    SAMPLING = "_".join(SPL[8:-2])
    MODEL = SPL[-1]
    return {
        "GAS": GAS,
        "TEMP": TEMP,
        "LOW": LOW,
        "HIGH": HIGH,
        "INPUT": INPUT,
        "SAMPLING": SAMPLING,
        "MODEL": MODEL,
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_dir", default="./")
    parser.add_argument("--outcsv", default="METRICS_OUTPUT.csv")
    parser.add_argument("--n_trial", default=5)
    args = parser.parse_args()

    # logging 설정
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )

    S_PATH = Path(args.target_dir)
    N_TRIAL = int(args.n_trial)
    LIS = []

    # 상위 디렉토리 순회
    for P in tqdm([x for x in os.listdir(S_PATH) if os.path.isdir(S_PATH/x)], desc="Processing directories"):
        logging.info(f"Processing directory: {P}")
        DICT = process_dirname(P)

        TRIALS = [f"trial_{i:03d}" for i in range(1, N_TRIAL + 1)]
        R2_LIS, MAE_LIS = [], []

        for TRIAL in tqdm(TRIALS, desc=f"Trials for {P}", leave=False):
            T = TRIAL.replace("_", "")
            CSV = S_PATH / P / TRIAL / f"metrics_holdout_{T}.csv"
            if not CSV.exists():
                logging.warning(f"Missing file: {CSV}")
                continue

            df = pd.read_csv(CSV)
            R2 = df["R2"].values[0]
            MAE = df["MAE"].values[0]
            R2_LIS.append(R2)
            MAE_LIS.append(MAE)
            logging.info(f"[{P} | {TRIAL}] R2={R2:.4f}, MAE={MAE:.4f}")

        if R2_LIS:
            DICT["R2_MEAN"] = np.mean(R2_LIS)
            DICT["R2_STD"] = np.std(R2_LIS)
            DICT["MAE_MEAN"] = np.mean(MAE_LIS)
            DICT["MAE_STD"] = np.std(MAE_LIS)
            LIS.append(DICT)
            logging.info(f"[{P}] R2_MEAN={DICT['R2_MEAN']:.4f}, MAE_MEAN={DICT['MAE_MEAN']:.4f}")

    out_df = pd.DataFrame(LIS)
    out_df.to_csv(args.outcsv, index=False)
    logging.info(f"Saved results → {args.outcsv}")

if __name__ == "__main__":
    main()
