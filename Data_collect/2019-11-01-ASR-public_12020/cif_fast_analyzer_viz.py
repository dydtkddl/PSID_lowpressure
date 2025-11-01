import os
import argparse
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from pymatgen.io.cif import CifParser
import matplotlib.pyplot as plt

plt.rcParams["font.size"] = 11
plt.rcParams["axes.grid"] = True
plt.rcParams["figure.figsize"] = (14, 10)

# ───────────────────────────────────────────────
# 함수 정의
# ───────────────────────────────────────────────
def process_cif(path_tuple):
    """하나의 CIF 파일에서 원자 수 및 격자 길이만 추출"""
    folder, fname = path_tuple
    path = os.path.join(folder, fname)
    try:
        parser = CifParser(path)
        structure = parser.get_structures()[0]
        cell = structure.lattice.abc
        return {
            "folder": os.path.basename(folder),
            "filename": fname,
            "num_atoms": len(structure.sites),
            "a": cell[0],
            "b": cell[1],
            "c": cell[2],
        }
    except Exception:
        return None


def collect_cif_paths(root_dirs):
    """주어진 폴더 목록에서 모든 CIF 파일 경로를 수집"""
    cif_paths = []
    for folder in root_dirs:
        if not os.path.exists(folder):
            print(f"⚠️ 폴더 없음: {folder}")
            continue
        files = [f for f in os.listdir(folder) if f.endswith(".cif")]
        print(f"📁 {folder} - {len(files)}개 파일 발견")
        cif_paths += [(folder, f) for f in files]
    return cif_paths


def plot_statistics(df, output_path):
    """CIF 데이터 통계 요약을 시각화"""
    fig, axs = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("CIF Dataset Statistical Summary", fontsize=16, weight="bold")

    # 1️⃣ 원자수 히스토그램
    axs[0, 0].hist(df["num_atoms"], bins=40, color="skyblue", edgecolor="black")
    axs[0, 0].set_title("Number of Atoms per Structure")
    axs[0, 0].set_xlabel("Atoms")
    axs[0, 0].set_ylabel("Frequency")

    # 2️⃣ 격자 길이 분포 (a,b,c)
    axs[0, 1].hist(df["a"], bins=40, alpha=0.6, label="a", color="r")
    axs[0, 1].hist(df["b"], bins=40, alpha=0.6, label="b", color="g")
    axs[0, 1].hist(df["c"], bins=40, alpha=0.6, label="c", color="b")
    axs[0, 1].set_title("Lattice Parameters Distribution")
    axs[0, 1].set_xlabel("Lattice Length (Å)")
    axs[0, 1].legend()

    # 3️⃣ aspect ratio (a/b, a/c, b/c)
    axs[0, 2].hist(df["aspect_ratio_ab"], bins=40, alpha=0.7, label="a/b")
    axs[0, 2].hist(df["aspect_ratio_ac"], bins=40, alpha=0.7, label="a/c")
    axs[0, 2].hist(df["aspect_ratio_bc"], bins=40, alpha=0.7, label="b/c")
    axs[0, 2].set_title("Aspect Ratio Distribution")
    axs[0, 2].set_xlabel("Ratio")
    axs[0, 2].legend()

    # 4️⃣ a-b 산점도
    axs[1, 0].scatter(df["a"], df["b"], s=5, alpha=0.6, color="purple")
    axs[1, 0].set_xlabel("a (Å)")
    axs[1, 0].set_ylabel("b (Å)")
    axs[1, 0].set_title("a vs b")

    # 5️⃣ a-c 산점도
    axs[1, 1].scatter(df["a"], df["c"], s=5, alpha=0.6, color="teal")
    axs[1, 1].set_xlabel("a (Å)")
    axs[1, 1].set_ylabel("c (Å)")
    axs[1, 1].set_title("a vs c")

    # 6️⃣ 원자수 상자그림
    axs[1, 2].boxplot(df["num_atoms"], vert=True, patch_artist=True, boxprops=dict(facecolor="lightblue"))
    axs[1, 2].set_title("Boxplot of Atom Counts")
    axs[1, 2].set_ylabel("Atom Count")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"📊 시각화 결과 저장 완료 → {output_path}")


def main():
    # ───────────────────────────────────────────────
    # 1️⃣ argparse 설정
    # ───────────────────────────────────────────────
    parser = argparse.ArgumentParser(description="CIF 병렬 분석 및 시각화 엔진")
    parser.add_argument(
        "--dirs",
        nargs="+",
        default=[
            "./2019-11-01-ASR-public_12020/disorder_1877/",
            "./2019-11-01-ASR-public_12020/structure_10143/",
        ],
        help="분석할 CIF 폴더 경로들 (공백으로 구분)",
    )
    parser.add_argument(
        "--n_cpus",
        type=int,
        default=max(2, cpu_count() - 2),
        help="병렬 처리에 사용할 CPU 코어 수 (기본값: 전체-2)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./",
        help="CSV 및 Figure 저장 디렉토리",
    )
    args = parser.parse_args()

    # ───────────────────────────────────────────────
    # 2️⃣ 파일 수집
    # ───────────────────────────────────────────────
    print("\n🚀 CIF 구조 파일 상세 분석 시작")
    print("=" * 90)
    cif_paths = collect_cif_paths(args.dirs)
    print(f"\n총 {len(cif_paths):,}개 CIF 처리 예정, CPU {args.n_cpus}개 사용")
    print("=" * 90)

    # ───────────────────────────────────────────────
    # 3️⃣ 병렬 파싱
    # ───────────────────────────────────────────────
    with Pool(args.n_cpus) as pool:
        results = list(
            tqdm(pool.imap_unordered(process_cif, cif_paths), total=len(cif_paths), desc="병렬 CIF 파싱 중")
        )

    records = [r for r in results if r is not None]
    print(f"\n✅ 성공적으로 파싱된 CIF: {len(records):,}/{len(cif_paths):,}")
    print("=" * 90)

    DF = pd.DataFrame(records)
    if len(DF) == 0:
        print("❌ 유효한 CIF 데이터가 없습니다.")
        return

    # ───────────────────────────────────────────────
    # 4️⃣ 통계 및 저장
    # ───────────────────────────────────────────────
    DF["aspect_ratio_ab"] = DF["a"] / DF["b"]
    DF["aspect_ratio_ac"] = DF["a"] / DF["c"]
    DF["aspect_ratio_bc"] = DF["b"] / DF["c"]

    os.makedirs(args.output_dir, exist_ok=True)
    df_path = os.path.join(args.output_dir, "CIF_fast_analysis.csv")
    DF.to_csv(df_path, index=False)

    summary = DF.groupby("folder").agg(
        {"num_atoms": ["mean", "std", "min", "max"], "a": ["mean", "std", "min", "max"], "b": ["mean", "std", "min", "max"], "c": ["mean", "std", "min", "max"]}
    ).round(3)
    summary_path = os.path.join(args.output_dir, "CIF_fast_summary.csv")
    summary.to_csv(summary_path)

    print(f"\n💾 CSV 결과 저장 완료:")
    print(f" ├─ {df_path}")
    print(f" └─ {summary_path}")

    # ───────────────────────────────────────────────
    # 5️⃣ 시각화 실행
    # ───────────────────────────────────────────────
    fig_path = os.path.join(args.output_dir, "CIF_statistics_summary.png")
    plot_statistics(DF, fig_path)

    print("=" * 90)
    print("✅ 전체 분석 및 시각화 완료!")


if __name__ == "__main__":
    main()


