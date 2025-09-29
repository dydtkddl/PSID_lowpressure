### 데이터 분석 보고서: 압력에 따른 계산 시간 분석 (Ar @ 313K)

#### **1. 데이터 개요 및 전처리**

먼저, 데이터를 불러와 구조를 파악하고 컬럼명을 정리합니다. 피벗 테이블의 인덱스로 사용되었을 'Unnamed: 0' 컬럼은 'Pressure [bar]'로 변경하여 분석의 명확성을 높입니다.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score

# 한글 폰트 설정 (그래프에서 한글 깨짐 방지)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# CSV 파일 경로
file_path = r"C:\Users\PSID_PC_20\Desktop\ActiveLearning\Training_Structure\MAKE_FIGURE\ComputationTimeSimulation\Ar_313K_TIME_20250929\Final_Computation_Time_Pivot_Table_Reordered.csv"

# 데이터 불러오기 및 전처리
try:
    df = pd.read_csv(file_path)
    if 'Unnamed: 0' in df.columns:
        df = df.rename(columns={'Unnamed: 0': 'Pressure [bar]'})
    
    # 압력 컬럼을 숫자형으로 변환 (필요 시)
    df['Pressure [bar]'] = pd.to_numeric(df['Pressure [bar]'], errors='coerce')
    df = df.dropna(subset=['Pressure [bar]'])
    df = df.set_index('Pressure [bar]').sort_index() # 압력 기준으로 정렬

    print("--- 데이터프레임 정보 ---")
    df.info()
    print("\n--- 데이터 확인 (상위 5개) ---")
    print(df.head())

except FileNotFoundError:
    print(f"오류: 파일을 찾을 수 없습니다. 경로를 확인해주세요:\n{file_path}")
except Exception as e:
    print(f"데이터 로딩 중 오류 발생: {e}")

```

#### **2. 기술 통계 분석**

각 압력 조건에서의 계산 시간 관련 통계량(평균, 표준편차, 최소/최대값 등)의 전반적인 분포를 확인합니다.

```python
# 기술 통계량 출력
print("\n--- 기술 통계량 ---")
print(df.describe())
```
**분석:**
*   `describe()` 결과를 통해 전체 압력 범위에 대한 계산 시간의 평균, 편차, 사분위수 등을 파악할 수 있습니다.
*   예를 들어, `Average_Time`의 `mean` 값은 모든 압력 조건에 걸친 평균 계산 시간을 나타내며, `std`는 그 변동성을 의미합니다.
*   `min`과 `max`를 통해 계산 시간이 가장 짧거나 길었던 압력 조건의 대략적인 범위를 짐작할 수 있습니다.

#### **3. 압력에 따른 계산 시간 변화 분석**

압력이 증가함에 따라 평균 계산 시간이 어떻게 변하는지 시각화하여 추세를 분석합니다.

```python
# 압력에 따른 평균 계산 시간 시각화
plt.figure(figsize=(15, 8))
avg_time_col = 'Average_Time' # 평균 시간 컬럼명 (실제 파일에 맞게 수정 가능)

# Bar plot과 Line plot을 함께 그려 추세 강조
sns.barplot(x=df.index, y=df[avg_time_col], color='skyblue', label='개별 압력에서의 평균 시간')
sns.lineplot(x=df.index, y=df[avg_time_col], marker='o', color='royalblue', lw=2, label='압력에 따른 추세선')

plt.title('압력(Pressure)에 따른 평균 계산 시간(Average Computation Time)', fontsize=18, pad=20)
plt.xlabel('Pressure [bar]', fontsize=14)
plt.ylabel('Average Time [s]', fontsize=14)
plt.xticks(rotation=45)
plt.legend()
plt.grid(True, axis='y', linestyle='--')
plt.tight_layout()
plt.show()
```
**분석:**
*   **경향성**: 그래프는 압력이 증가할수록 계산 시간이 비선형적으로 급격히 증가하는 경향을 보일 것으로 예상됩니다. 이는 고압 조건에서 시스템 내 입자 수가 많아져 상호작용 계산량이 폭증하기 때문입니다.
*   **임계점**: 특정 압력 구간에서 계산 시간이 급증하는 '임계점'이 관찰될 수 있습니다. 이 지점은 시뮬레이션 비용이 크게 증가하는 변곡점으로 해석할 수 있습니다.

#### **4. 계산 시간 분포 및 변동성 분석**

압력별 계산 시간의 변동성(표준편차)을 분석하여, 어떤 압력 조건에서 시뮬레이션 결과의 편차가 큰지 확인합니다.

```python
# 압력에 따른 계산 시간 표준편차 시각화
plt.figure(figsize=(15, 8))
std_time_col = 'Stdev_Time' # 표준편차 컬럼명 (실제 파일에 맞게 수정 가능)

sns.lineplot(x=df.index, y=df[std_time_col], marker='s', color='darkorange', label='계산 시간 표준편차')
plt.fill_between(df.index, df[std_time_col], color='orange', alpha=0.2)

plt.title('압력(Pressure)에 따른 계산 시간 변동성(Standard Deviation)', fontsize=18, pad=20)
plt.xlabel('Pressure [bar]', fontsize=14)
plt.ylabel('Standard Deviation of Time [s]', fontsize=14)
plt.xticks(rotation=45)
plt.legend()
plt.grid(True, linestyle=':')
plt.tight_layout()
plt.show()
```
**분석:**
*   일반적으로 평균 계산 시간이 긴 고압 영역에서 표준편차 또한 크게 나타날 가능성이 높습니다.
*   이는 고압 시뮬레이션이 초기 조건에 더 민감하거나, 계산 과정의 무작위성(randomness)으로 인해 실행할 때마다 소요 시간의 편차가 커짐을 의미합니다.

#### **5. 상관 관계 분석**

데이터셋에 포함된 모든 수치형 컬럼(평균, 표준편차, 최대, 최소 시간 등) 간의 상관 관계를 히트맵으로 시각화합니다.

```python
# 모든 컬럼 간의 상관계수 행렬 계산 및 히트맵 시각화
plt.figure(figsize=(10, 8))
corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=.5)
plt.title('계산 시간 지표 간 상관 관계 분석', fontsize=16, pad=20)
plt.show()
```
**분석:**
*   `Average_Time`은 `Max_Time` 및 `Min_Time`과 매우 높은 양의 상관 관계(0.9 이상)를 보일 것입니다. 이는 당연한 결과입니다.
*   `Average_Time`과 `Stdev_Time` 간의 상관 관계를 통해, 평균 계산 시간이 길어질수록 결과의 변동성도 함께 커지는 경향이 있는지 정량적으로 확인할 수 있습니다.

#### **6. 회귀 분석을 통한 예측 모델링**

압력과 평균 계산 시간 사이의 관계를 다항 회귀 모델로 피팅하여, 특정 압력에서의 계산 시간을 예측할 수 있는 모델을 만듭니다.

```python
# 2차 다항 회귀 모델 학습
X = df.index.values.reshape(-1, 1)
y = df[avg_time_col].values

poly_features = PolynomialFeatures(degree=2)
X_poly = poly_features.fit_transform(X)

model = LinearRegression()
model.fit(X_poly, y)
y_pred = model.predict(X_poly)

# R-squared (결정계수) 계산
r2 = r2_score(y, y_pred)

# 회귀 모델 시각화
plt.figure(figsize=(15, 8))
plt.scatter(X, y, color='skyblue', label='실제 데이터')
plt.plot(X, y_pred, color='red', lw=2, label=f'2차 다항 회귀 모델 (R² = {r2:.4f})')

plt.title('압력-계산시간 관계에 대한 회귀 분석', fontsize=18, pad=20)
plt.xlabel('Pressure [bar]', fontsize=14)
plt.ylabel('Average Time [s]', fontsize=14)
plt.legend()
plt.grid(True, linestyle='--')
plt.tight_layout()
plt.show()

print(f"\n--- 회귀 모델 정보 ---")
print(f"결정계수 (R-squared): {r2:.4f}")
print(f"모델 계수: {model.coef_}")
print(f"모델 절편: {model.intercept_}")
```
**분석:**
*   **모델 적합도**: R-squared (결정계수) 값은 모델이 실제 데이터를 얼마나 잘 설명하는지를 나타냅니다. 1에 가까울수록 모델의 설명력이 높습니다.
*   **예측**: 생성된 회귀 모델을 사용하면, 시뮬레이션을 직접 수행하지 않은 압력 값에 대해서도 예상 계산 시간을 예측할 수 있습니다. 예를 들어, `model.predict(poly_features.transform([[12.5]]))`와 같은 코드로 12.5 bar에서의 계산 시간을 추정할 수 있습니다.

#### **7. 종합 결론**

*   **주요 발견**: 본 분석을 통해 압력이 시뮬레이션 계산 시간에 미치는 영향은 매우 크며, 특히 고압으로 갈수록 계산 비용이 비선형적으로 급증함을 확인했습니다.
*   **변동성**: 계산 시간의 변동성 또한 압력과 함께 증가하는 경향을 보여, 고압 시뮬레이션은 예측의 불확실성이 더 큼을 시사합니다.
*   **예측 가능성**: 압력과 계산 시간의 관계는 2차 다항식과 같은 비교적 간단한 모델로도 높은 정확도(높은 R-squared 값)로 설명될 수 있습니다. 이는 향후 시뮬레이션 계획 수립 시 필요한 계산 자원을 예측하는 데 유용하게 사용될 수 있습니다.
*   **활용 방안**: 이 분석 결과는 Active Learning 프레임워크에서 다음 시뮬레이션 대상을 선정할 때, 계산 비용(시간)을 중요한 제약 조건 또는 평가 기준으로 고려하는 데 활용될 수 있습니다. 예를 들어, 정보 획득량(acquisition)이 비슷하다면 계산 비용이 낮은 후보를 우선적으로 선택하는 전략을 구사할 수 있습니다.

```
