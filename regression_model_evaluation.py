
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
# df3_augmented = augment_data(df3_augmented, noise_level=0.01)
# df3_augmented = augment_data(df3_augmented, noise_level=0.01)
X = df3_augmented[['father', 'mother', 'gender', 'country']]
y = df3_augmented['childHeight']
print(df3_augmented.shape[0])


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#선형 회귀 모델
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mse_Linear = mean_squared_error(y_test, y_pred)
mae_Linear = mean_absolute_error(y_test, y_pred)
r2_Linear = r2_score(y_test, y_pred)
mse_list.append(mse_Linear)
mae_list.append(mae_Linear)
r_list.append(r2_Linear)

# 서포트 벡터 회귀 모델
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

svr_model = SVR(kernel='rbf')
svr_model.fit(X_train_scaled, y_train)
y_pred_svr = svr_model.predict(X_test_scaled)
mse_svr = mean_squared_error(y_test, y_pred_svr)
mae_svr = mean_absolute_error(y_test, y_pred_svr)
r2_svr = r2_score(y_test, y_pred_svr)
mse_list.append(mse_svr)
mae_list.append(mae_Linear)
r_list.append(r2_svr)

# Decision Tree 모델 생성 및 학습
decision_tree_model = DecisionTreeRegressor(random_state=42)
decision_tree_model.fit(X_train, y_train)

y_pred_tree = decision_tree_model.predict(X_test)
mse_tree = mean_squared_error(y_test, y_pred_tree)
mae_tree = mean_absolute_error(y_test, y_pred_tree)
r2_tree = r2_score(y_test, y_pred_tree)

mse_list.append(mse_tree)
r_list.append(r2_tree)
mae_list.append(mae_Linear)
# XGBoost 모델
# xgboost_model = XGBRegressor(objective='reg:squarederror', n_estimators=200, learning_rate=0.3, max_depth=3, random_state=42)
xgboost_model = XGBRegressor(
    objective='reg:squarederror',
   
    n_estimators=552,         # 트리 개수
    learning_rate=0.01678456310062124,       # 학습률
    max_depth=10,              # 최대 깊이
    min_child_weight=1,       # 리프 노드에서 필요한 최소 가중치
    subsample=0.9970510933233787,            # 데이터 샘플 비율
    colsample_bytree=0.8023174805231743,     # 특성 샘플 비율
    gamma=4.962865023385216e-07,                  # 최소 손실 감소
    reg_alpha=0.7333266269251827,            # L1 정규화
    reg_lambda=1.480742588439296            # L2 정규화
)
xgboost_model.fit(X_train, y_train)
y_pred_xgb = xgboost_model.predict(X_test)
mse_xgboost = mean_squared_error(y_test, y_pred_xgb)
mae_xgboost = mean_absolute_error(y_test, y_pred_xgb)
r2_xgboost = r2_score(y_test, y_pred_xgb)
mse_list.append(mse_xgboost)
mae_list.append(mae_xgboost)
r_list.append(r2_xgboost)
print(mse_xgboost)
# 엘라스틱넷 모델
elastic_net = ElasticNet(alpha=0.001, l1_ratio=0.5, random_state=42)
elastic_net.fit(X_train_scaled, y_train)
y_pred_elastic = elastic_net.predict(X_test_scaled)
mse_elastic = mean_squared_error(y_test, y_pred_elastic)
mae_elastic = mean_absolute_error(y_test, y_pred_elastic)
r2_elastic = r2_score(y_test, y_pred_elastic)
mse_list.append(mse_elastic)
mae_list.append(mae_elastic)
r_list.append(r2_elastic)
# 랜덤 포레스트 회귀 모델
random_forest_model = RandomForestRegressor(n_estimators=100, random_state=42)
random_forest_model.fit(X_train, y_train)

y_pred_rf = random_forest_model.predict(X_test)
mse_rf = mean_squared_error(y_test, y_pred_rf)
mae_rf = mean_absolute_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)
mse_list.append(mse_rf)
mae_list.append(mae_rf)
r_list.append(r2_rf)

# LightGBM
lgbm_model = lgb.LGBMRegressor(
    objective='regression',
    n_estimators=135,
    learning_rate=0.05,
    max_depth=-1,
    num_leaves=90,
    boosting_type='gbdt',
    random_state=42
)
lgbm_model.fit(X_train, y_train)
y_pred_lgbm = lgbm_model.predict(X_test)
mse_lgbm = mean_squared_error(y_test, y_pred_lgbm)
mae_lgbm = mean_absolute_error(y_test, y_pred_lgbm)
r2_lgbm = r2_score(y_test, y_pred_lgbm)
mse_list.append(mse_lgbm)
mae_list.append(mae_lgbm)
r_list.append(r2_lgbm)

# 모델 성능 평가
print("\nmse가 가장 낮은 모델의 번호:", mse_list.index(min(mse_list)) + 1)
print("R계수가 가장 높은 모델의 번호:", r_list.index(max(r_list)) + 1)
print("mae가 가장 낮은 모델의 번호:", mae_list.index(min(mae_list)) + 1)

# R^2가 가장 높은 모델을 선택
best_model_index = r_list.index(max(r_list))
if best_model_index == 0:
    best_model = model
    print("\n가장 좋은 성능의 모델: Linear Regression:" + str(max(r_list)))
    print("mse는 " +str(min(mse_list)))
    print("mae는 " +str(min(mae_list)))
elif best_model_index == 1:
    best_model = svr_model
    print("\n가장 좋은 성능의 모델: Support Vector Regression:" + str(max(r_list)))
    print("mse는 " +str(min(mse_list)))
    print("mae는 " +str(min(mae_list)))
elif best_model_index==2:
    best_model = decision_tree_model
    print("\n가장 좋은 성능의 모델: Decision Tree Regression:" + str(max(r_list)))
    print("mse는 " +str(min(mse_list)))
    print("mae는 " +str(min(mae_list)))
elif best_model_index == 3:
    best_model = xgboost_model
    print("\n가장 좋은 성능의 모델: XGBoost Regression:" + str(max(r_list)))
    print("mse는 " +str(min(mse_list)))
    print("mae는 " +str(min(mae_list)))
elif best_model_index == 4:
    best_model = elastic_net
    print("\n가장 좋은 성능의 모델: ElasticNet Regression:" + str(max(r_list)))
    print("mse는 " +str(min(mse_list)))
    print("mae는 " +str(min(mae_list)))
elif best_model_index == 5:
    best_model = random_forest_model
    print("\n가장 좋은 성능의 모델: Random Forest Regression:" + str(max(r_list)))
    print("mse는 " + str(min(mse_list)))
    print("mae는 " +str(min(mae_list)))
elif best_model_index == 6:
    best_model = lgbm_model
    print("\n가장 좋은 성능의 모델: lgbm_model Regression:" + str(max(r_list)))
    print("mse는 " + str(min(mse_list)))
    print("mae는 " +str(min(mae_list)))
end_time = time.time()
timing=end_time-start_time
print("총 걸린 시간은:",timing,"입니다.")
#예측
def predict_child_height(father_height, mother_height, gender, country):

    gender = 1 if gender in ['M', 'm'] else 0
    country = 1 if country in ['USA', 'usa'] else 0
    input_data = pd.DataFrame([[father_height, mother_height, gender, country]], columns=['father', 'mother', 'gender', 'country'])
    
    if best_model in ['Support Vector Regression', 'ElasticNet Regression']:
        input_data_scaled = scaler.transform(input_data)
        prediction = best_model.predict(input_data_scaled)
    else:
        prediction = best_model.predict(input_data)
    
    return prediction[0]

# 예시 입력
father_height = float(input("Father's height (cm): "))
mother_height = float(input("Mother's height (cm): "))
gender = input("Child's gender (M/F): ")
country = input("Country (Asian/USA): ")

predicted_height = predict_child_height(father_height, mother_height, gender, country)
print(f"Predicted child's height: {predicted_height:.2f} cm")



father_height = float(input("Father's height (cm): "))
mother_height = float(input("Mother's height (cm): "))
gender = input("Child's gender (M/F): ")
country = input("Country (Asian/USA): ")

predicted_height = predict_child_height(father_height, mother_height, gender, country)
print(f"Predicted child's height: {predicted_height:.2f} cm")
end_time = time.time()
timing=end_time-start_time
print("총 걸린 시간은:",timing,"입니다.")
