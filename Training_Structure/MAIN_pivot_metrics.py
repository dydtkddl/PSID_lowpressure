import pandas as pd
import logging
from tqdm import tqdm
import argparse
import sys

def process_metrics(input_file: str, output_file: str):
    # 로깅 설정
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()]
    )

    # 데이터 불러오기
    df = pd.read_csv(input_file)
    logging.info(f"Data loaded with {len(df)} rows and {len(df.columns)} columns from {input_file}")

    # 정렬 기준
    sort_cols = ["GAS", "TEMP", "MODEL", "SAMPLING", "HIGH", "LOW"]

    # 정렬 (tqdm 사용)
    for col in tqdm(sort_cols, desc="Sorting step-by-step"):
        df = df.sort_values(by=col, kind="mergesort")
        logging.info(f"Sorted by {col}")

    df_sorted = df.sort_values(by=sort_cols, ascending=True, kind="mergesort").reset_index(drop=True)
    logging.info("Final sorting completed.")

    # HIGH를 컬럼으로 pivot
    df_pivot = df_sorted.pivot(
        index=["GAS", "TEMP", "MODEL", "SAMPLING", "LOW"],
        columns="HIGH",
        values=["R2_MEAN", "R2_STD", "MAE_MEAN", "MAE_STD", "INPUT"]
    )

    # 컬럼명 정리
    df_pivot.columns = [f"{val}_HIGH_{col}" for val, col in df_pivot.columns]
    df_pivot = df_pivot.reset_index()

    # 저장
    df_pivot.to_csv(output_file, index=False)
    logging.info(f"Pivoted file saved as {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Process METRICS_OUTPUT.csv and pivot HIGH column.")
    parser.add_argument(
        "--input", "-i", 
        type=str, 
        default="METRICS_OUTPUT.csv",
        help="Input CSV file (default: METRICS_OUTPUT.csv)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="METRICS_OUTPUT_pivot.csv",
        help="Output CSV file (default: METRICS_OUTPUT_pivot.csv)"
    )

    args = parser.parse_args()

    process_metrics(args.input, args.output)


if __name__ == "__main__":
    main()

