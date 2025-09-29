'''
This script visualizes the GCMC computation time for different training ratios and pressure conditions.
It generates a stacked bar chart with a line plot overlay to show the trend of total computation time.
'''
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# --- Configuration ---
TARGER_PRESSURES = [1, 5, 15]
TRAIN_RATIOS = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]
CATBOOST_DIR = Path("../try15_Ar313_0.01_15bar_Isotherm/")
COMPTIMES_PATH = Path("./ComputationTimeSimulation/Ar_313K_TIME_20250929/Final_Computation_Time_Pivot_Table_Reordered.csv")

# --- Data Loading ---
try:
    COMPTIMES = pd.read_csv(COMPTIMES_PATH)
except FileNotFoundError:
    print(f"Error: Computation time file not found at {COMPTIMES_PATH}. Cannot proceed.")
    raise

COMPTIMES_INDEXED = COMPTIMES.set_index('MOF')

# --- Data Processing ---
results = []
for pressure in TARGER_PRESSURES:
    if pressure == 1:
        TARGET_COL = 'time_below_50_313_1'
    else:
        TARGET_COL = f'{pressure}bar_AR_313_CORE'

    if TARGET_COL not in COMPTIMES_INDEXED.columns:
        print(f"Warning: Column {TARGET_COL} not found in {COMPTIMES_PATH}. Skipping pressure {pressure} bar.")
        continue

    # GCMC time for each training ratio
    for train_ratio in TRAIN_RATIOS:
        dir_name = f'Ar_313_0.01_to_{pressure}__struct+input__qt_then_rd__cat_TRAIN_RATIO{train_ratio}_QTFRAC_{train_ratio}'
        base_path = CATBOOST_DIR / dir_name / 'trial_001'

        try:
            train_df = pd.read_csv(base_path / 'df_train.csv')
            pred_df = pd.read_csv(base_path / 'predictions_holdout_trial001.csv')
        except FileNotFoundError:
            print(f"Warning: ML files for TRAIN_RATIO {train_ratio} at pressure {pressure} not found in {base_path}. Skipping.")
            continue

        pred_mofs = set(pred_df['filename'])
        train_mofs_gcmc = set(train_df['filename']) - pred_mofs

        if not train_mofs_gcmc:
            gcmc_sampling_time = 0.0
        else:
            df_gcmc_mofs = pd.DataFrame(list(train_mofs_gcmc), columns=['MOF']).set_index('MOF')
            df_merged = df_gcmc_mofs.join(COMPTIMES_INDEXED, how='inner')
            gcmc_sampling_time = df_merged[TARGET_COL].sum()

        gcmc_sampling_time_h = gcmc_sampling_time / 3600

        results.append({
            'TRAIN_RATIO': train_ratio,
            'PRESSURE': f'{pressure} bar',
            'GCMC_SAMPLING_TIME_H': gcmc_sampling_time_h
        })

    # Total GCMC time (ratio = 1.0)
    total_gcmc_time_s = COMPTIMES_INDEXED[TARGET_COL].sum()
    total_gcmc_time_h = total_gcmc_time_s / 3600
    results.append({
        'TRAIN_RATIO': 1.0,
        'PRESSURE': f'{pressure} bar',
        'GCMC_SAMPLING_TIME_H': total_gcmc_time_h
    })

df_results = pd.DataFrame(results)

# --- Plotting ---
sns.set_theme(style="whitegrid") # Corrected style setting
fig, ax = plt.subplots(figsize=(16, 10))

# Pivot data for stacking
df_pivot = df_results.pivot(index='TRAIN_RATIO', columns='PRESSURE', values='GCMC_SAMPLING_TIME_H')

# Ensure the columns are in the desired order for stacking
pressure_order = [f'{p} bar' for p in TARGER_PRESSURES if f'{p} bar' in df_pivot.columns]
df_pivot = df_pivot.reindex(columns=pressure_order)
df_pivot = df_pivot.sort_index()


# Define colors for each pressure
colors = {'1 bar': '#DDAA33', '5 bar': '#BB5566', '15 bar': '#004488'}

# Plot stacked bars
df_pivot.plot(kind='bar', stacked=True, ax=ax, color=[colors[col] for col in pressure_order], width=0.7)

# Plot line on top of the stacked bars
df_pivot['Total'] = df_pivot.sum(axis=1)
# The x-axis for the line plot needs to match the bar plot's x-axis, which are strings.
x_ticks = [str(r) for r in df_pivot.index]
ax.plot(x_ticks, df_pivot['Total'], marker='o', color='black', linestyle='-', label='Total Time')

# --- Aesthetics ---
ax.set_title('GCMC Computation Time vs. Training Data Ratio', fontsize=20, fontweight='bold')
ax.set_xlabel('Training Data Ratio', fontsize=16)
ax.set_ylabel('Total GCMC Time (Hours)', fontsize=16)

ax.tick_params(axis='x', labelsize=12, rotation=45)
ax.tick_params(axis='y', labelsize=12)

# Legend
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, title='Pressure', loc='upper left', fontsize=12, title_fontsize=14)

ax.grid(True, which='major', linestyle='--', linewidth=0.5)
ax.set_ylim(0)

plt.tight_layout()
plt.savefig("GCMC_Computation_Time.png", dpi=300)
plt.show()

print("\nVisualization saved as GCMC_Computation_Time.png")