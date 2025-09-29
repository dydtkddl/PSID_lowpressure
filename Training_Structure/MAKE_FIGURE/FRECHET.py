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

MODEL_FUNCTIONS = {
    'Sips': sips, 'Langmuir': langmuir, 'Freundlich': freundlich,
    'Quadratic': quadratic, 'Peleg': peleg
}

# --- 곡선 생성 함수 ---
def generate_isotherm_curve(model_name, params):
    if model_name not in MODEL_FUNCTIONS: return None
    model_func = MODEL_FUNCTIONS[model_name]
    P_curve = np.linspace(0.01, 15, 100)
    q_curve = model_func(P_curve, **params)
    return np.column_stack((P_curve, q_curve))

# --- 병렬 처리를 위한 작업자 함수 ---
def calculate_frechet_for_mof(mof, ref_mof_data, current_mof_data):
    """단일 MOF에 대해 Fréchet 거리를 계산하고 MOF 이름과 함께 반환합니다."""
    try:
        ref_curve = generate_isotherm_curve(ref_mof_data['model'], ref_mof_data['params'])
        current_curve = generate_isotherm_curve(current_mof_data['model'], current_mof_data['params'])
        if ref_curve is not None and current_curve is not None:
            dist = similaritymeasures.frechet_dist(ref_curve, current_curve)
            return (mof, dist)
        else: return (mof, None)
    except Exception: return (mof, None)

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
        print(f"❌ Error: File not found at {file_path}."); exit()

    REFERENCE_FRAC = 1.0
    if REFERENCE_FRAC not in all_results_by_fraction:
        print(f"❌ Error: Reference fraction {REFERENCE_FRAC} not found."); exit()

    ref_isotherms = all_results_by_fraction[REFERENCE_FRAC]['best_fit_params']
    print(f"Reference data (Fraction=1.0) loaded for {len(ref_isotherms)} MOFs.")

    # --- 섹션 2: Fraction별 Fréchet 거리 계산 ---
    print("\nStep 2: Calculating Fréchet distance for each fraction...")
    
    # ✅ [수정] 모든 MOF의 거리 값을 저장할 딕셔너리 초기화
    all_distances_dict = {}

    target_fractions = sorted([f for f in all_results_by_fraction.keys() if f != REFERENCE_FRAC])
    print(f"Total {len(target_fractions)} fractions to process.")

    for frac in target_fractions:
        print(f"\n----- Processing Fraction: {frac} -----")
        current_isotherms = all_results_by_fraction[frac]['best_fit_params']
        common_mofs = sorted(list(ref_isotherms.keys() & current_isotherms.keys()))
        
        if len(common_mofs) < 2:
            print(f"⚠️ Warning: Not enough common MOFs. Skipping."); continue
        
        print(f"Found {len(common_mofs)} common MOFs to compare.")
        
        # 병렬 계산 실행 (결과는 [(mof, dist), (mof, dist), ...])
        distance_results = Parallel(n_jobs=30)(
            delayed(calculate_frechet_for_mof)(mof, ref_isotherms[mof], current_isotherms[mof])
            for mof in tqdm(common_mofs, desc=f"  Comparing curves for Frac {frac}")
        )
        
        # 결과를 딕셔너리로 변환하여 저장
        frac_distances = {mof: dist for mof, dist in distance_results if dist is not None}
        all_distances_dict[frac] = frac_distances
        print(f"  -> Successfully calculated distances for {len(frac_distances)} MOFs.")

    print("\nCalculations complete for all fractions.")

    # --- ✅ 섹션 3: 결과 DataFrame 생성 및 CSV 저장 ---
    print("\nStep 3: Creating detailed DataFrame and saving to CSV...")
    
    # 딕셔너리를 DataFrame으로 변환 (행: MOF, 열: Fraction)
    df_distances = pd.DataFrame(all_distances_dict)
    df_distances.index.name = 'MOF_Name'
    
    # 통계치 계산
    mean_distances = df_distances.mean()
    std_distances = df_distances.std()
    
    # 통계 DataFrame 생성
    df_summary = pd.DataFrame({
        'Mean_Frechet_Distance': mean_distances,
        'Std_Dev_Frechet_Distance': std_distances
    })
    df_summary.index.name = 'Fraction'

    # CSV 파일로 저장
    output_detailed_csv_path = 'frechet_distance_by_mof.csv'
    output_summary_csv_path = 'frechet_distance_summary.csv'
    df_distances.to_csv(output_detailed_csv_path, float_format='%.6f')
    df_summary.to_csv(output_summary_csv_path, float_format='%.6f')

    print(f"✅ Detailed results for each MOF saved to: {output_detailed_csv_path}")
    print(f"✅ Summary statistics saved to: {output_summary_csv_path}")

    # --- 섹션 4: 평균/표준편차 그래프 시각화 ---
    print("\nStep 4: Generating the plot...")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 7))

    ax.errorbar(df_summary.index, df_summary['Mean_Frechet_Distance'], yerr=df_summary['Std_Dev_Frechet_Distance'],
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
