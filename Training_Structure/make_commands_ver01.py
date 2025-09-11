import logging
import sys
from pathlib import Path
from tqdm import tqdm

# === 설정 ===
DATA_ROOT = Path("../Data_collect/DataSet")  # 데이터셋 루트
OUT_ROOT = Path("./try05")
OUT_ROOT.mkdir(exist_ok=True)

GASES = ["Ar", "He", "N2", "O2", "CO2", "H2"]
TEMPS = [273, 293, 313]

INPUTS = ["Henry", 0.01, 0.05, 0.1, 0.5]
OUTPUTS = [1, 5, 15]

FEATURES_SAMPLER_COMBOS = [
    ("struct", "random_struct"),
    ("struct+input", "random_struct"),
    ("struct+input", "qt_then_rd"),
]
MODELS = ["rf", "gbm", "cat"]
TRIALS = range(1, 6)  # 1 ~ 5
SEED = 42
N_BINS = 20

# === 로깅 설정 ===
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

# === 명령어 생성 ===
all_cmds = []
for gas in GASES:
    for temp in TEMPS:
        new_DATA_PATH = DATA_ROOT / f"{gas}_{temp}K"

        for inp in INPUTS:
            for out in OUTPUTS:
                # dataset 파일명 규칙
                dataset_name = f"{gas}_{temp}K_{gas}_{temp}_{inp}_to_{gas}_{temp}_{out}_dataset.csv"
                data_path = new_DATA_PATH / dataset_name

                for feat, sampler in tqdm(FEATURES_SAMPLER_COMBOS,
                                          desc=f"{gas}_{temp}K Input={inp}, Output={out}"):
                    for model in MODELS:
                        outdir = OUT_ROOT / f"{gas}_{temp}_{inp}_to_{out}__{feat}__{sampler}__{model}"
                        outdir.mkdir(parents=True, exist_ok=True)

                        for trial in TRIALS:
                            cmd = [
                            "python", "pipeline_single_trial.py",
                                "--data", str(data_path),
                                "--outdir", str(outdir),
                                "--trial", str(trial),
                                "--seed-base", str(SEED),
                                "--features", feat,
                                "--sampler", sampler,
                                "--model", model,
                                "--n-bins", str(N_BINS),
                            ]
                            all_cmds.append(" ".join(cmd))

logging.info(f"총 생성된 commands 수: {len(all_cmds)}")

# === 결과 저장 ===
cmds_file = OUT_ROOT / "run_commands.txt"
with open(cmds_file, "w") as f:
    f.write("\n".join(all_cmds))

logging.info(f"명령어 리스트 저장 완료: {cmds_file}")

