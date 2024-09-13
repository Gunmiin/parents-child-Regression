
import time
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score,mean_absolute_error
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
import xgboost as xgb
import lightgbm as lgb
start_time = time.time()
def remove_outliers(df, columns):
    df_cleaned = df.copy()
    for column in columns:
        Q1 = df_cleaned[column].quantile(0.25)
        Q3 = df_cleaned[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df_cleaned = df_cleaned[(df_cleaned[column] >= lower_bound) & (df_cleaned[column] <= upper_bound)]
    return df_cleaned
mse_list=[]
r_list=[]
mae_list = []
# CSV 파일 읽기
df1 = pd.read_csv('Asian_train_new.csv')
df2 = pd.read_csv('Asian_height.csv')
dfm=pd.read_csv('American_height.csv')

columns_to_check = ['father', 'mother', 'childHeight']
df1_cleaned = remove_outliers(df1, columns_to_check)
df2_cleaned = remove_outliers(df2, columns_to_check)
dfm_cleaned = remove_outliers(dfm, columns_to_check)

def augment_data(df, noise_level=0.01):
    augmented_df = df.copy()
    
    # 랜덤 노이즈 생성
    noise = np.random.normal(0, noise_level, df[['father', 'mother']].shape)
    
    # 데이터에 노이즈 추가
    augmented_df[['father', 'mother']] += noise
    
    return pd.concat([df, augmented_df], ignore_index=True)

dfm_cleaned['country']=1
dfm_1 = augment_data(dfm_cleaned, noise_level=0.01)
dfm_1 = augment_data(dfm_1, noise_level=0.01)
# print("Number of rows in dfm_1:", dfm_1.shape[0])
df4 = pd.concat([df1_cleaned, df2_cleaned], ignore_index=True)
df4 = augment_data(df4, noise_level=0.01)
# print("Number of rows in dfm_1:", df4.shape[0])
df4['country'] = 0
df3 = pd.concat([df4, dfm_1], ignore_index=True)
df3['gender'] = df3['gender'].map({'M': 1, 'F': 0})
X = df3[['father', 'mother', 'gender','country']]
y = df3['childHeight']
df3_augmented = augment_data(df3, noise_level=0.01)
df3_augmented = augment_data(df3_augmented, noise_level=0.01)
def evaluate_model_performance(X_train, X_test, y_train, y_test):
    # XGBoost 모델 초기화 및 학습
    xgb_model = XGBRegressor(
        objective='reg:squarederror',
        n_estimators=552,
        learning_rate=0.01678456310062124,
        max_depth=10,
        min_child_weight=1,
        subsample=0.9970510933233787,
        colsample_bytree=0.8023174805231743,
        gamma=4.962865023385216e-07,
        reg_alpha=0.7333266269251827,
        reg_lambda=1.480742588439296
    )
    xgb_model.fit(X_train, y_train)
    y_pred = xgb_model.predict(X_test)

    # 성능 지표 계산
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return mse, mae, r2

# 증강 단계별 성능 저장 리스트 초기화
mse_augmented = []
mae_augmented = []
r2_augmented = []

# 원본 데이터로 평가
X = df3[['father', 'mother', 'gender', 'country']]
y = df3['childHeight']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
mse, mae, r2 = evaluate_model_performance(X_train, X_test, y_train, y_test)
mse_augmented.append(mse)
mae_augmented.append(mae)
r2_augmented.append(r2)

num_augmentations = 10
df_augmented = df3.copy()

for i in range(1, num_augmentations + 1):
    df_augmented = augment_data(df_augmented, noise_level=0.01)
    X = df_augmented[['father', 'mother', 'gender', 'country']]
    y = df_augmented['childHeight']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    mse, mae, r2 = evaluate_model_performance(X_train, X_test, y_train, y_test)
    mse_augmented.append(mse)
    mae_augmented.append(mae)
    r2_augmented.append(r2)
# 그래프 그리기
augmentation_levels = ['No Aug', '1 Aug', '2 Aug','3 Aug','4 Aug','5 Aug','6 Aug','7 Aug','8 Aug','9 Aug','10 Aug']

plt.figure(figsize=(30, 6))

plt.subplot(1, 3, 1)
plt.plot(augmentation_levels, mse_augmented, marker='o', color='b', label='MSE')
plt.title('Mean Squared Error (MSE)')
plt.xlabel('Augmentation Level')
plt.ylabel('MSE')
plt.grid(True)

plt.subplot(1, 3, 2)
plt.plot(augmentation_levels, mae_augmented, marker='o', color='g', label='MAE')
plt.title('Mean Absolute Error (MAE)')
plt.xlabel('Augmentation Level')
plt.ylabel('MAE')
plt.grid(True)

plt.subplot(1, 3, 3)
plt.plot(augmentation_levels, r2_augmented, marker='o', color='r', label='R²')
plt.title('R² Score')
plt.xlabel('Augmentation Level')
plt.ylabel('R² Score')
plt.grid(True)

plt.tight_layout()
plt.show()

