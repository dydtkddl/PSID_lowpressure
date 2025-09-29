import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from tqdm.auto import tqdm
from pathlib import Path
from joblib import Parallel, delayed
import pickle

# ==============================================================================
# 섹션 0: 핵심 함수 및 변수 정의 (스크립트 최상단에 위치)
# ==============================================================================

# --- 경로 변수 ---
# CATBOOST_DIR 경로는 사용자 환경에 맞게 확인해주세요.
CATBOOST_DIR = Path("../try15_Ar313_0.01_15bar_Isotherm/")

# --- Isotherm 모델 함수 ---
def sips(P, K1, K2, K3):
    """Sips 등온흡착식 모델"""
    P = np.asarray(P)
    PK3 = P**K3
    return (K1 * K2 * PK3) / (1 + K2 * PK3)

def langmuir(P, K1, K2):
    """Langmuir 등온흡착식 모델"""
    return (K1 * K2 * P) / (1 + K2 * P)

# --- 데이터 로딩 함수 ---
def Make_Uptake_list(ml_pressure, target_fraction, col):
    """지정된 압력과 분율에 대한 ML 결과를 읽어 Uptake 리스트를 생성합니다."""
    if target_fraction != 1.0:
        dir_name = f'Ar_313_0.01_to_{ml_pressure}__struct+input__qt_then_rd__cat_TRAIN_RATIO{target_fraction}_QTFRAC_{target_fraction}'
        base_path = CATBOOST_DIR / dir_name / 'trial_001'
        train_df = pd.read_csv(base_path / 'df_train.csv')
        pred_df = pd.read_csv(base_path / 'predictions_holdout_trial001.csv')
        pred_mofs = set(pred_df['filename'])
        train_mofs_gcmc = set(train_df['filename']) - pred_mofs
        train_UPTAKES = train_df[train_df['filename'].isin(train_mofs_gcmc)][["filename", "Output"]]
        test_UPTAKES = pred_df[["filename", "y_pred"]]
        UPTAKE_LIST = pd.concat([train_UPTAKES.rename(columns={"Output": "Uptake"}), test_UPTAKES.rename(columns={"y_pred": "Uptake"})])
    else: # target_fraction == 1.0 (Ground Truth)
        dir_name = f'Ar_313_0.01_to_{ml_pressure}__struct+input__qt_then_rd__cat_TRAIN_RATIO0.05_QTFRAC_0.05'
        base_path = CATBOOST_DIR / dir_name / 'trial_001'
        train_df = pd.read_csv(base_path / 'df_train.csv')
        UPTAKE_LIST = train_df[["filename", "Output"]].rename(columns={"Output": "Uptake"})
    return UPTAKE_LIST

# --- 병렬 처리를 위한 작업자 함수 ---
def fit_isotherm_models(mof_name, q_data):
    """MOF 하나에 대해 Sips, Langmuir 모델 순서로 피팅을 시도하고 결과를 반환합니다."""
    P_data = np.array([0.01, 1.0, 5.0, 15.0])
    try: # Sips 시도
        popt, _ = curve_fit(sips, P_data, q_data, bounds=(0, np.inf), p0=[np.max(q_data), 0.1, 1.0], maxfev=5000, ftol=1e-6, xtol=1e-6)
        return (mof_name, 'Sips', {'K1': popt[0], 'K2': popt[1], 'K3': popt[2]})
    except RuntimeError:
        try: # Langmuir 시도
            popt, _ = curve_fit(langmuir, P_data, q_data, bounds=(0, np.inf), p0=[np.max(q_data), 0.1], maxfev=5000, ftol=1e-6, xtol=1e-6)
            return (mof_name, 'Langmuir', {'K1': popt[0], 'K2': popt[1]})
        except RuntimeError: # 모두 실패
            return (mof_name, 'Failed', None)

# ==============================================================================
# 섹션 1: 모든 프로세스를 실행하는 메인 함수
# ==============================================================================
def run_isotherm_fitting_for_fraction(target_fraction):
    """주어진 target_fraction에 대해 전체 등온선 피팅 프로세스를 실행합니다."""
    print(f"\n{'='*25} Running for Fraction: {target_fraction} {'='*25}")
    
    # --- 데이터 준비 및 병합 ---
    print("Step 1: Preparing and merging uptake data...")
    try:
        UPTAKE_LIST_0_01 = pd.read_csv(CATBOOST_DIR / "Ar_313_0.01_to_15__struct+input__qt_then_rd__cat_TRAIN_RATIO0.05_QTFRAC_0.05/trial_001/df_train.csv")[["filename", "Input"]]
        UPTAKE_LIST_1bar = Make_Uptake_list(1, target_fraction, 'time_below_50_3_13_1')
        UPTAKE_LIST_5bar = Make_Uptake_list(5, target_fraction, '5bar_AR_313_CORE')
        UPTAKE_LIST_15bar = Make_Uptake_list(15, target_fraction, '15bar_AR_313_CORE')
        
        df_isotherm_data = pd.concat([
            UPTAKE_LIST_0_01.rename(columns={"Input": "0.01bar"}).set_index('filename'),
            UPTAKE_LIST_1bar.rename(columns={"Uptake": "1bar"}).set_index('filename'),
            UPTAKE_LIST_5bar.rename(columns={"Uptake": "5bar"}).set_index('filename'),
            UPTAKE_LIST_15bar.rename(columns={"Uptake": "15bar"}).set_index('filename')
        ], axis=1)

        print(f"Data shape before cleaning: {df_isotherm_data.shape}")
        df_isotherm_data_clean = df_isotherm_data.dropna()
        print(f"Data shape after cleaning (dropna): {df_isotherm_data_clean.shape}")
        print("-" * 30)
    except FileNotFoundError as e:
        print(f"\n❌ Error loading data for fraction {target_fraction}: {e}")
        return None, None, None

    # --- 등온선 피팅 (병렬 처리) ---
    print(f"\nStep 2: Fitting isotherm models for {len(df_isotherm_data_clean)} MOFs...")
    results = Parallel(n_jobs=30)(
        delayed(fit_isotherm_models)(mof_name, row.values) 
        for mof_name, row in tqdm(df_isotherm_data_clean.iterrows(), total=len(df_isotherm_data_clean))
    )

    # --- 결과 정리 ---
    sips_params_dict, langmuir_params_dict, failed_mofs = {}, {}, []
    for mof_name, model_name, params in results:
        if model_name == 'Sips':
            sips_params_dict[mof_name] = params
        elif model_name == 'Langmuir':
            langmuir_params_dict[mof_name] = params
        else:
            failed_mofs.append(mof_name)

    # --- 최종 피팅 결과 요약 리포트 ---
    print("-" * 30)
    print(f"\n✅ Fitting process complete for Fraction: {target_fraction}")
    print(f"  - ✔️ Sips fits:     {len(sips_params_dict)}")
    print(f"  - ✔️ Langmuir fits: {len(langmuir_params_dict)}")
    print(f"  - ❌ Total Failed:  {len(failed_mofs)}")
    print("-" * 30)
    
    # --- 실패 원인 진단을 위한 시각화 ---
    if failed_mofs:
        num_to_plot = min(len(failed_mofs), 5)
        fig, axes = plt.subplots(1, num_to_plot, figsize=(num_to_plot * 4, 4), squeeze=False)
        axes = axes.flatten()
        for i in range(num_to_plot):
            mof_to_plot = failed_mofs[i]
            q_data_failed = df_isotherm_data_clean.loc[mof_to_plot].values
            ax = axes[i]
            ax.scatter(np.array([0.01, 1.0, 5.0, 15.0]), q_data_failed, color='red', zorder=5, label='Data Points')
            ax.plot(np.array([0.01, 1.0, 5.0, 15.0]), q_data_failed, 'r--', alpha=0.7)
            ax.set_title(f"Failed Fit: {mof_to_plot[:25]}...", fontsize=10)
            ax.set_xlabel("Pressure (bar)")
            ax.set_ylabel("Uptake")
            ax.grid(True, linestyle='--', alpha=0.6)
            ax.legend()
        plt.tight_layout()
        # plt.show() # 루프 중에는 주석 처리하여 자동으로 창이 뜨는 것을 방지할 수 있습니다.

    return sips_params_dict, langmuir_params_dict, failed_mofs

# ==============================================================================
# 메인 스크립트 실행 블록
# ==============================================================================
if __name__ == "__main__":
    
    TRAIN_RATIOS = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 1.0]
    all_results_by_fraction = {}

    for fraction in TRAIN_RATIOS:
        sips_res, lang_res, failed = run_isotherm_fitting_for_fraction(fraction)
        if sips_res is not None:
            all_results_by_fraction[fraction] = {
                'sips_params': sips_res,
                'langmuir_params': lang_res,
                'failed_mofs': failed
            }
            
    print(f"\n\n{'='*20} All Analyses Complete {'='*20}")
    
    # --- 최종 딕셔너리를 파일로 저장 ---
    output_path = Path('./isotherm_fitting_results.pkl')
    
    print(f"Saving all results to: {output_path}")
    with open(output_path, 'wb') as f:
        pickle.dump(all_results_by_fraction, f)
        
    print("✅ Results successfully saved.")
