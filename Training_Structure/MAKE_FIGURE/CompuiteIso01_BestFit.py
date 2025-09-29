import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, OptimizeWarning
from tqdm.auto import tqdm
from pathlib import Path
from joblib import Parallel, delayed
import pickle
import warnings

# ==============================================================================
# 섹션 0: 핵심 함수 및 전역 변수 정의
# ==============================================================================

# --- ✅ 1. Warning 메시지 무시 설정 ---
# curve_fit에서 발생하는 RuntimeWarning, OptimizeWarning을 무시하여 출력창을 깔끔하게 유지
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=OptimizeWarning)

# --- 경로 변수 ---
CATBOOST_DIR = Path("../try15_Ar313_0.01_15bar_Isotherm/")

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

# --- 시도할 모델 목록 (전역 변수) ---
MODELS_TO_TRY = [
    {'name': 'Sips',       'func': sips,       'k': 3, 'p0_func': lambda q: [np.max(q), 0.1, 1.0]},
    {'name': 'Langmuir',   'func': langmuir,   'k': 2, 'p0_func': lambda q: [np.max(q), 0.1]},
    {'name': 'Freundlich', 'func': freundlich, 'k': 2, 'p0_func': lambda q: [np.max(q), 0.1]},
    {'name': 'Quadratic',  'func': quadratic,  'k': 3, 'p0_func': lambda q: [np.max(q), 0.1, 0.1]},
    {'name': 'Peleg',      'func': peleg,      'k': 4, 'p0_func': lambda q: [np.max(q)/2, 0.5, np.max(q)/2, 1.0]}
]

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

# --- ✅ 2. 병렬 처리를 위한 작업자 함수 (상세 정보 포함) ---
def find_best_isotherm_model(mof_name, q_data):
    """모든 Isotherm 모델을 시도하고, 최적 모델의 상세 정보를 포함한 결과를 반환합니다."""
    P_data = np.array([0.01, 1.0, 5.0, 15.0]); n = len(P_data)
    fit_results = []
    # ✅ [수정] 각 병렬 프로세스(Worker) 내부에서 Warning을 무시하도록 설정
    warnings.filterwarnings("ignore", category=OptimizeWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)
        
    for model in MODELS_TO_TRY:
        try:
            popt, _ = curve_fit(model['func'], P_data, q_data, bounds=(0, np.inf), p0=model['p0_func'](q_data), maxfev=5000)
            sse = np.sum((q_data - model['func'](P_data, *popt))**2)
            if sse == 0: sse = 1e-20
            aic = n * np.log(sse / n) + 2 * model['k']
            param_names = model['func'].__code__.co_varnames[1:model['k']+1]
            params = {name: val for name, val in zip(param_names, popt)}
            fit_results.append({'name': model['name'], 'aic': aic, 'params': params, 'popt': popt, 'func': model['func']})
        except (RuntimeError, ValueError):
            continue

    if not fit_results:
        return (mof_name, {'status': 'failed', 'original_data': q_data.tolist()})
        
    best_model = min(fit_results, key=lambda x: x['aic'])
    
    # 최적 모델에 대한 에러(메트릭) 계산
    q_pred = best_model['func'](P_data, *best_model['popt'])
    sse = np.sum((q_data - q_pred)**2)
    rmse = np.sqrt(sse / n)
    ss_tot = np.sum((q_data - np.mean(q_data))**2)
    r2 = 1 - (sse / ss_tot) if ss_tot > 0 else 0
    
    final_result = {
        'status': 'success',
        'model': best_model['name'],
        'params': best_model['params'],
        'original_data': q_data.tolist(),
        'metrics': {'rmse': rmse, 'r2': r2}
    }
    return (mof_name, final_result)

# ==============================================================================
# 섹션 1: 모든 프로세스를 실행하는 메인 함수
# ==============================================================================
def run_isotherm_fitting_for_fraction(target_fraction):
    """주어진 target_fraction에 대해 최적 등온선 모델을 찾고 상세 결과를 반환합니다."""
    print(f"\n{'='*25} Running for Fraction: {target_fraction} {'='*25}")
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
        
        df_isotherm_data_clean = df_isotherm_data.dropna()
        print(f"Data shape after cleaning (dropna): {df_isotherm_data_clean.shape}")
    except FileNotFoundError as e:
        print(f"\n❌ Error loading data for fraction {target_fraction}: {e}"); return None
    
    results = Parallel(n_jobs=-1)(
        delayed(find_best_isotherm_model)(mof_name, row.values) 
        for mof_name, row in tqdm(df_isotherm_data_clean.iterrows(), total=len(df_isotherm_data_clean))
    )
    
    # ✅ 3. 결과를 하나의 통합 딕셔너리로 바로 변환
    unified_results = {mof_name: result_dict for mof_name, result_dict in results}
    
    # 결과 요약
    model_counts = {model['name']: 0 for model in MODELS_TO_TRY}
    failed_count = 0
    for res in unified_results.values():
        if res['status'] == 'success':
            model_counts[res['model']] += 1
        else:
            failed_count += 1

    print("-" * 30); print(f"\n✅ Fitting process complete for Fraction: {target_fraction}")
    print("--- Best Model Selection Counts ---")
    for name, count in model_counts.items():
        print(f"  - {name:<10}: {count} times")
    print(f"  - {'Failed':<10}: {failed_count} times")
    print("-" * 30)

    return unified_results

# ==============================================================================
# 메인 스크립트 실행 블록
# ==============================================================================
if __name__ == "__main__":
    TRAIN_RATIOS = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 1.0]
    all_results_by_fraction = {}

    for fraction in TRAIN_RATIOS:
        fraction_results = run_isotherm_fitting_for_fraction(fraction)
        if fraction_results is not None:
            all_results_by_fraction[fraction] = fraction_results
            
    print(f"\n\n{'='*20} All Analyses Complete {'='*20}")
    
    output_path = Path('./best_fit_isotherm_results_detailed.pkl')
    print(f"Saving all results to: {output_path}")
    with open(output_path, 'wb') as f:
        pickle.dump(all_results_by_fraction, f)
    print("✅ Results successfully saved.")

    # 저장된 파일 구조 확인 예시
    if 0.1 in all_results_by_fraction:
        print("\n--- Example of saved data structure for fraction 0.1 ---")
        example_mof = list(all_results_by_fraction[0.1].keys())[0]
        print(f"Data for MOF '{example_mof}':")
        print(all_results_by_fraction[0.1][example_mof])
