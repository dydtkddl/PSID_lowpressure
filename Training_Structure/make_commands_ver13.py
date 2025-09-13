import logging
from tqdm import tqdm
import argparse
import math

def generate_commands():
    TEMPS = [273, 293, 313]
    GASES = ["He", "Ar"]
    TRAINRATIOS = [0.05, 0.1 , 0.15 , 0.2 , 0.25 ,0.3 ,0.35 ,0.4 ,0.45, 0.5 ,0.55 ,0.6, 0.65, 0.7, 0.75, 0.8]
    SAMPLINGS = ["qt_then_rd", "random_struct"]
    SAMPLINGS = ["random_struct"]
    MODELS = ["cat", "rf", "gbm"]
    SEED = 52
    TRIALS = [1, 2, 3, 4, 5]

    commands = []

    total_combos = len(TEMPS) * len(GASES) * len(TRAINRATIOS) * len(SAMPLINGS) * len(MODELS) * len(TRIALS)
    logging.info(f"총 조합 개수: {total_combos}")

    for gas in tqdm(GASES, desc="Gases loop"):
        for temp in TEMPS:
            for train_ratio in TRAINRATIOS:
                for sampling in SAMPLINGS:
                    for model in MODELS:
                        for trial in TRIALS:
                            cmd = (
                                f"python pipeline_single_trial_QT_LOGSAMPLE.py "
                                f"--data ../Data_collect/DataSet/{gas}_{temp}K/{gas}_{temp}K_{gas}_{temp}_0.01_to_{gas}_{temp}_1_dataset.csv "
                                f"--outdir try09/{gas}_{temp}_0.01_to_1__struct+input__{sampling}__{model}_TRAIN_RATIO{train_ratio}_QTFRAC_{train_ratio} "
                                f"--seed-base {SEED} --features struct+input --sample {sampling} "
                                f"--model {model} --n-bins 200 --train-ratio {train_ratio} --qt-frac {train_ratio} --trial {trial}"
                            )
                            commands.append(cmd)
    return commands


def save_split(commands, split, prefix="commands_part"):
    n = len(commands)
    chunk_size = math.ceil(n / split)
    for i in range(split):
        start = i * chunk_size
        end = min((i+1) * chunk_size, n)
        filename = f"{prefix}{i+1}.txt"
        with open(filename, "w") as f:
            for cmd in commands[start:end]:
                f.write(cmd + "\n")
        logging.info(f"{filename} 저장 완료 ({end-start} 개)")


def main():
    parser = argparse.ArgumentParser(description="Generate and split commands list")
    parser.add_argument("--split", type=int, default=2, help="나눌 파일 개수 (기본=2)")
    args = parser.parse_args()

    # 로깅 설정
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )

    commands = generate_commands()
    save_split(commands, args.split)


if __name__ == "__main__":
    main()
