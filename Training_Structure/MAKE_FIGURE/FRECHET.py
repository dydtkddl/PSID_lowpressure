import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from tqdm.auto import tqdm
from pathlib import Path
from joblib import Parallel, delayed
import similaritymeasures
import shutil

# ==============================================================================
# 섹션 0: Isotherm 모델 및 헬퍼 함수 정의
# ==============================================================================

# --- Isotherm 모델 함수 ---
def sips(P, K1, K2, K3):
    P = np.asarray(P); PK3 = P**K3; return (K1 * K2 * PK3) / (1 + K2 * PK3)
def langmuir(P, K1, K2):
    return (K1 * K2 * P) / (1 + K2 * P)
def freundlich(P, K1, K2):
    return K1 * (np.asarray(P)**K2)
def quadratic(P, K1, K2, K3):
    P = np.asarray(P); num = K1 * (K2 * P + K3 * P**2); den = 1 + K2 * P + 2 * K3 * P**2; return num / den
def peleg(P, K1, K2, K3, K4):
    P = np.asarray(P); return K1 * (P**K2) + K3 * (P**K4)

# 모델 이름을 실제 함수와 매핑하는 딕셔너리
MODEL_FUNCTIONS = {
    'Sips': sips, 'Langmuir': langmuir, 'Freundlich': freundlich,
    'Quadratic': quadratic, 'Peleg': peleg
}

# --- 곡선 생성 함수 ---
def generate_isotherm_curve(model_name, params):
    """주어진 모델과 파라미터로 등온선 곡선(좌표 배열)을 생성합니다."""
    if model_name not in MODEL_FUNCTIONS:
        return None
    model_func = MODEL_FUNCTIONS[model_name]
    P_curve = np.linspace(0.01, 15, 100)
    q_curve = model_func(P_curve, **params)
    return np.column_stack((P_curve, q_curve))

# --- ✅ [수정] 병렬 처리를 위한 작업자 함수 ---
# 전체 딕셔너리 대신, 딱 필요한 MOF의 데이터만 받도록 수정
def calculate_frechet_for_mof(ref_mof_data, current_mof_data):
    """단일 MOF에 대해 Fréchet 거리를 계산합니다."""
    try:
        ref_curve = generate_isotherm_curve(ref_mof_data['model'], ref_mof_data['params'])
        current_curve = generate_isotherm_curve(current_mof_data['model'], current_mof_data['params'])
        
        if ref_curve is not None and current_curve is not None:
            return similaritymeasures.frechet_dist(ref_curve, current_curve)
        else:
            return None
    except Exception:
        return None

# ==============================================================================
# 메인 스크립트 실행 블록
# ==============================================================================
if __name__ == "__main__":
    # --- 섹션 1: 데이터 로딩 및 준비 ---
    print("Step 1: Loading and preparing data...")
    file_path = Path('./best_fit_isotherm_results.pkl')
    try:
        with open(file_path, "rb") as f:
            all_results_by_fraction = pickle.load(f)
    except FileNotFoundError:
        print(f"❌ Error: File not found at {file_path}.")
        exit()

    REFERENCE_FRAC = 1.0
    if REFERENCE_FRAC not in all_results_by_fraction:
        print(f"❌ Error: Reference fraction {REFERENCE_FRAC} not found.")
        exit()

    ref_isotherms = all_results_by_fraction[REFERENCE_FRAC]['best_fit_params']
    print(f"Reference data (Fraction=1.0) loaded for {len(ref_isotherms)} MOFs.")

    # --- 섹션 2: Fraction별 Fréchet 거리 계산 및 통계치 집계 ---
    print("\nStep 2: Calculating Fréchet distance for each fraction...")
    fractions_for_plot = []
    mean_distances = []
    std_distances = []
    target_fractions = sorted([f for f in all_results_by_fraction.keys() ])
    print(f"Total {len(target_fractions)} fractions to process.")

    for frac in target_fractions:
        print(f"\n----- Processing Fraction: {frac} -----")
        current_isotherms = all_results_by_fraction[frac]['best_fit_params']
        common_mofs = sorted(list(ref_isotherms.keys() & current_isotherms.keys()))
        
        if len(common_mofs) < 2:
            print(f"⚠️ Warning: Not enough common MOFs for fraction {frac}. Skipping.")
            continue
        
        print(f"Found {len(common_mofs)} common MOFs to compare.")
        
        # ✅ [수정] Parallel 호출 시, 각 MOF에 해당하는 작은 데이터 조각만 전달하여 병목 현상 해결
        distances_for_fraction = Parallel(n_jobs=30)(
            delayed(calculate_frechet_for_mof)(ref_isotherms[mof], current_isotherms[mof])
            for mof in tqdm(common_mofs, desc=f"  Comparing curves for Frac {frac}")
        )
        
        valid_distances = [d for d in distances_for_fraction if d is not None]
        
        if valid_distances:
            mean_dist = np.mean(valid_distances)
            std_dist = np.std(valid_distances)
            fractions_for_plot.append(frac)
            mean_distances.append(mean_dist)
            std_distances.append(std_dist)
            print(f"  -> Calculated Stats: Mean Dist = {mean_dist:.4f}, Std Dev = {std_dist:.4f}")
        else:
            print("  -> No valid distances were calculated for this fraction.")

    print("\nCalculations complete for all fractions.")

    # --- 섹션 3: 결과 CSV 파일로 저장 ---
    print("\nStep 3: Saving detailed results to CSV...")
    results_df = pd.DataFrame({
        'Fraction': fractions_for_plot,
        'Mean_Frechet_Distance': mean_distances,
        'Std_Dev_Frechet_Distance': std_distances
    })
    output_csv_path = 'frechet_distance_summary.csv'
    results_df.to_csv(output_csv_path, index=False, float_format='%.6f')
    print(f"✅ Detailed results saved to: {output_csv_path}")

    # --- 섹션 4: 평균/표준편차 그래프 시각화 ---
    print("\nStep 4: Generating the plot...")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.errorbar(fractions_for_plot, mean_distances, yerr=std_distances,
                marker='o', linestyle='-', capsize=5, ecolor='lightcoral', label='Mean Fréchet Distance')
    ax.set_xlabel('Training Data Fraction', fontsize=14)
    ax.set_ylabel('Mean Fréchet Distance', fontsize=14)
    ax.set_title('Isotherm Curve Similarity vs. Ground Truth (Fraction=1.0)', fontsize=16)
    ax.legend(fontsize=12)
    ax.set_ylim(bottom=0)
    ax.tick_params(axis='both', which='major', labelsize=12)
    fig.tight_layout()
    output_plot_path = 'frechet_distance_by_fraction.png'
    plt.savefig(output_plot_path, dpi=300)
    plt.show()
    print(f"\n✅ Plot saved to: {output_plot_path}")
