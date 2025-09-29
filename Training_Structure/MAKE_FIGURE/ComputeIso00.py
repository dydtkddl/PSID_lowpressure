import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from tqdm.auto import tqdm
from pathlib import Path
from joblib import Parallel, delayed
import pickle

# ==============================================================================
# 섹션 0: 핵심 함수 및 전역 변수 정의
# ==============================================================================

# --- 경로 변수 ---
CATBOOST_DIR = Path("../try15_Ar313_0.01_15bar_Isotherm/")

# --- Isotherm 모델 함수 ---
def sips(P, K1, K2, K3):
    P = np.asarray(P); PK3 = P**K3; return (K1 * K2 * PK3) / (1 + K2 * PK3)

def langmuir(P, K1, K2):
    return (K1 * K2 * P) / (1 + K2 * P)

# --- 데이터 로딩 함수 ---
def Make_Uptake_list(ml_pressure, target_fraction, col):
    # (기존과 동일)
    if target_fraction != 1.0:
        dir_name = f'Ar_313_0.01_to_{ml_pressure}__struct+input__qt_then_rd__cat_TRAIN_RATIO{target_fraction}_QTFRAC_{target_fraction}'
        base_path = CATBOOST_DIR / dir_name / 'trial_001'
        train_df = pd.read_csv(base_path / 'df_train.csv'); pred_df = pd.read_csv(base_path / 'predictions_holdout_trial001.csv')
        pred_mofs = set(pred_df['filename']); train_mofs_gcmc = set(train_df['filename']) - pred_mofs
        train_UPTAKES = train_df[train_df['filename'].isin(train_mofs_gcmc)][["filename", "Output"]]
        test_UPTAKES = pred_df[["filename", "y_pred"]]
        return pd.concat([train_UPTAKES.rename(columns={"Output": "Uptake"}), test_UPTAKES.rename(columns={"y_pred": "Uptake"})])
    else:
        dir_name = f'Ar_313_0.01_to_{ml_pressure}__struct+input__qt_then_rd__cat_TRAIN_RATIO0.05_QTFRAC_0.05'
        base_path = CATBOOST_DIR / dir_name / 'trial_001'; train_df = pd.read_csv(base_path / 'df_train.csv')
        return train_df[["filename", "Output"]].rename(columns={"Output": "Uptake"})

# --- 병렬 처리를 위한 작업자 함수 (모든 정보 포함) ---
def fit_isotherm_models(mof_name, q_data):
    """
    MOF 하나에 대해 Sips -> Langmuir 순으로 피팅을 시도하고,
    성공 시 모델 정보, 파라미터, 원본 데이터, 에러 메트릭을 포함한 결과를 반환합니다.
    """
    P_data = np.array([0.01, 1.0, 5.0, 15.0])
    
    # --- 피팅 및 메트릭 계산을 위한 내부 함수 ---
    def fit_and_evaluate(model_func, model_name, p0):
        popt, _ = curve_fit(model_func, P_data, q_data, bounds=(0, np.inf), p0=p0, maxfev=5000, ftol=1e-6, xtol=1e-6)
        
        q_pred = model_func(P_data, *popt)
        
        # 에러(메트릭) 계산
        sse = np.sum((q_data - q_pred)**2)
        rmse = np.sqrt(sse / len(q_data))
        ss_tot = np.sum((q_data - np.mean(q_data))**2)
        r2 = 1 - (sse / ss_tot) if ss_tot > 0 else 0
        
        param_names = model_func.__code__.co_varnames[1:len(popt)+1]
        params = {name: val for name, val in zip(param_names, popt)}
        
        return {
            'status': 'success',
            'model': model_name,
            'params': params,
            'original_data': q_data.tolist(),
            'metrics': {'rmse': rmse, 'r2': r2}
        }

    # --- 메인 피팅 로직 ---
    try: # Sips 시도
        result = fit_and_evaluate(sips, 'Sips', p0=[np.max(q_data), 0.1, 1.0])
        return (mof_name, result)
    except RuntimeError:
        try: # Langmuir 시도
            result = fit_and_evaluate(langmuir, 'Langmuir', p0=[np.max(q_data), 0.1])
            return (mof_name, result)
        except RuntimeError: # 모두 실패
            return (mof_name, {'status': 'failed', 'original_data': q_data.tolist()})

# ==============================================================================
# 섹션 1: 모든 프로세스를 실행하는 메인 함수
# ==============================================================================
def run_isotherm_fitting_for_fraction(target_fraction):
    """주어진 target_fraction에 대해 전체 등온선 피팅 프로세스를 실행하고 통합된 결과를 반환합니다."""
    print(f"\n{'='*25} Running for Fraction: {target_fraction} {'='*25}")
    try:
        # (데이터 준비 및 병합 로직)
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
        df_isotherm_data_clean = df_isotherm_data.dropna()
        print(f"Data shape after cleaning (dropna): {df_isotherm_data_clean.shape}")
    except FileNotFoundError as e:
        print(f"\n❌ Error loading data for fraction {target_fraction}: {e}"); return None
    
    # (병렬 처리 피팅)
    results = Parallel(n_jobs=30)(
        delayed(fit_isotherm_models)(mof_name, row.values) 
        for mof_name, row in tqdm(df_isotherm_data_clean.iterrows(), total=len(df_isotherm_data_clean))
    )
    
    # (결과 정리 -> 단일 딕셔너리로 통합)
    unified_results = {mof_name: result_dict for mof_name, result_dict in results}
    
    # (결과 요약 리포트)
    sips_count = sum(1 for res in unified_results.values() if res.get('model') == 'Sips')
    langmuir_count = sum(1 for res in unified_results.values() if res.get('model') == 'Langmuir')
    failed_count = sum(1 for res in unified_results.values() if res['status'] == 'failed')
    
    print("-" * 30); print(f"\n✅ Fitting process complete for Fraction: {target_fraction}")
    print(f"  - ✔️ Sips fits:     {sips_count}")
    print(f"  - ✔️ Langmuir fits: {langmuir_count} (Sips failed, Langmuir succeeded)")
    print(f"  - ❌ Total Failed:  {failed_count}")
    print("-" * 30)

    return unified_results

# ==============================================================================
# 메인 스크립트 실행 블록
# ==============================================================================
if __name__ == "__main__":
    
    TRAIN_RATIOS = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 1.0]
    all_results_by_fraction = {}

    for fraction in TRAIN_RATIOS:
        # 이제 함수는 결과 딕셔너리 하나만 반환합니다.
        fraction_results = run_isotherm_fitting_for_fraction(fraction)
        if fraction_results is not None:
            all_results_by_fraction[fraction] = fraction_results
            
    print(f"\n\n{'='*20} All Analyses Complete {'='*20}")
    
    # --- 최종 딕셔너리를 파일로 저장 ---
    output_path = Path('./isotherm_fitting_results_detailed.pkl')
    
    print(f"Saving all results to: {output_path}")
    with open(output_path, 'wb') as f:
        pickle.dump(all_results_by_fraction, f)
        
    print("✅ Results successfully saved.")

    # --- 저장된 파일 다시 불러와서 구조 확인 (참고용) ---
    if 0.1 in all_results_by_fraction:
        print("\n--- Example of saved data structure for fraction 0.1 ---")
        example_mof = list(all_results_by_fraction[0.1].keys())[0]
        print(f"Data for MOF '{example_mof}':")
        print(all_results_by_fraction[0.1][example_mof])
