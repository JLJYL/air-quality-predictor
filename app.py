# app.py - 供 Render 部署使用 (已移除耗時的訓練邏輯)

# =================================================================
# 導入所有必要的庫
# =================================================================
import requests
import pandas as pd
import datetime
import random
import re
import os
import warnings
import numpy as np
import xgboost as xgb
import json # 新增：用於讀取元數據
from datetime import timedelta, timezone
from flask import Flask, render_template

# 忽略警告
warnings.filterwarnings('ignore')

# 模型與元數據路徑
MODELS_DIR = 'models'
META_PATH = os.path.join(MODELS_DIR, 'model_meta.json')

# =================================================================
# 全域變數 - 改為從檔案載入
# =================================================================
TRAINED_MODELS = {} 
LAST_OBSERVATION = None 
FEATURE_COLUMNS = []
POLLUTANT_PARAMS = [] # 實際找到並訓練的模型參數
HOURS_TO_PREDICT = 24

# =================================================================
# 常數設定 (僅保留與預測相關的常數)
# =================================================================
LOCAL_TZ = "Asia/Taipei"
LAG_HOURS = [1, 2, 3, 6, 12, 24] # 預測遞迴需要這些參數
ROLLING_WINDOWS = [6, 12, 24] # 預測遞迴需要這些參數
POLLUTANT_TARGETS = ["pm25", "pm10", "o3", "no2", "so2", "co"] # 用於 AQI 計算

# 簡化的 AQI 分級表 (基於小時值和 US EPA 標準的常用數值)
AQI_BREAKPOINTS = {
    "pm25": [(0.0, 12.0, 0, 50), (12.1, 35.4, 51, 100), (35.5, 55.4, 101, 150), (55.5, 150.4, 151, 200)],
    "pm10": [(0, 54, 0, 50), (55, 154, 51, 100), (155, 254, 101, 150), (255, 354, 151, 200)],
    "o3": [(0, 54, 0, 50), (55, 70, 51, 100), (71, 85, 101, 150), (86, 105, 151, 200)],
    "co": [(0.0, 4.4, 0, 50), (4.5, 9.4, 51, 100), (9.5, 12.4, 101, 150), (12.5, 15.4, 151, 200)],
    "no2": [(0, 100, 0, 50), (101, 360, 51, 100), (361, 649, 101, 150), (650, 1249, 151, 200)],
    "so2": [(0, 35, 0, 50), (36, 75, 51, 100), (76, 185, 101, 150), (186, 304, 151, 200)],
}

# =================================================================
# 輔助函式: AQI 計算 (從原 app.py 複製過來，必須保留)
# =================================================================

def calculate_aqi_sub_index(param: str, concentration: float) -> float:
    """計算單一污染物濃度對應的 AQI 子指數 (I)"""
    if pd.isna(concentration) or concentration < 0:
        return 0

    breakpoints = AQI_BREAKPOINTS.get(param)
    if not breakpoints:
        return 0

    for C_low, C_high, I_low, I_high in breakpoints:
        if C_low <= concentration <= C_high:
            if C_high == C_low:
                return I_high
            I = ((I_high - I_low) / (C_high - C_low)) * (concentration - C_low) + I_low
            return np.round(I)

        if concentration > breakpoints[-1][1]:
            I_low, I_high = breakpoints[-1][2], breakpoints[-1][3]
            C_low, C_high = breakpoints[-1][0], breakpoints[-1][1]
            if C_high == C_low:
                return I_high
            I_rate = (I_high - I_low) / (C_high - C_low)
            I = I_high + I_rate * (concentration - C_high)
            return np.round(I)

    return 0

def calculate_aqi(row: pd.Series, params: list) -> int:
    """根據多個污染物濃度計算最終 AQI (取最大子指數)"""
    sub_indices = []
    for p in params:
        col_name = f'{p}_pred' if f'{p}_pred' in row else f'{p}_value'
        if col_name in row and not pd.isna(row[col_name]):
            sub_index = calculate_aqi_sub_index(p, row[col_name])
            sub_indices.append(sub_index)

    if not sub_indices:
        return np.nan

    return int(np.max(sub_indices))

# =================================================================
# 預測函式 (從原 app.py 複製過來，必須保留)
# =================================================================

def predict_future_multi(models, last_data, feature_cols, pollutant_params, hours=24):
    """預測未來 N 小時的多個目標污染物 (遞迴預測) 並計算 AQI"""
    predictions = []

    # last_data 現在是單行 DataFrame，需要先轉換時間格式
    last_data['datetime'] = pd.to_datetime(last_data['datetime']).dt.tz_localize('UTC')
    last_datetime_aware = last_data['datetime'].iloc[0]
    # 注意：這裡使用 to_dict() 創建一個可變字典副本作為迭代的基礎
    current_data_dict = last_data[feature_cols].iloc[0].to_dict() 

    weather_feature_names_base = ['temperature', 'humidity', 'pressure']
    weather_feature_names = [col for col in weather_feature_names_base if col in feature_cols]
    has_weather = bool(weather_feature_names)

    for h in range(hours):
        future_time = last_datetime_aware + timedelta(hours=h + 1)
        pred_features = current_data_dict.copy()

        # 1. 更新時間特徵
        pred_features['hour'] = future_time.hour
        pred_features['day_of_week'] = future_time.dayofweek
        pred_features['month'] = future_time.month
        pred_features['day_of_year'] = future_time.timetuple().tm_yday # 使用 day_of_year
        pred_features['is_weekend'] = int(future_time.dayofweek in [5, 6])
        pred_features['hour_sin'] = np.sin(2 * np.pi * future_time.hour / 24)
        pred_features['hour_cos'] = np.cos(2 * np.pi * future_time.hour / 24)
        pred_features['day_sin'] = np.sin(2 * np.pi * pred_features['day_of_year'] / 365)
        pred_features['day_cos'] = np.cos(2 * np.pi * pred_features['day_of_year'] / 365)

        # 2. 模擬未來天氣變化 (使用前一小時的天氣值進行隨機擾動)
        if has_weather:
            np.random.seed(future_time.hour + future_time.day + 42)
            for w_col in weather_feature_names:
                base_value = current_data_dict.get(w_col)
                if base_value is not None and not np.isnan(base_value):
                    # 模擬輕微隨機變化
                    new_weather_value = base_value + np.random.normal(0, 0.5) 
                    pred_features[w_col] = new_weather_value
                    # 將新的天氣值更新到 current_data_dict，以便下一小時使用
                    current_data_dict[w_col] = new_weather_value

        current_prediction_row = {'datetime': future_time}
        new_pollutant_values = {}

        # 3. 預測所有污染物
        for param in pollutant_params:
            model = models[param]
            # 確保輸入特徵的順序與模型訓練時一致
            pred_input = np.array([pred_features[col] for col in feature_cols]).reshape(1, -1)
            pred = model.predict(pred_input)[0]
            pred = max(0, pred) # 濃度不能小於 0

            current_prediction_row[f'{param}_pred'] = pred
            new_pollutant_values[param] = pred

        # 4. 計算預測的 AQI
        predicted_aqi = calculate_aqi(pd.Series(current_prediction_row), pollutant_params)
        current_prediction_row['aqi_pred'] = predicted_aqi
        new_pollutant_values['aqi'] = predicted_aqi

        predictions.append(current_prediction_row)

        # 5. 更新滯後特徵 (用當前預測值填充 Lag_1h，並將其他 Lag 向後移動)
        for param in pollutant_params + ['aqi']:
            # 從最大的 Lag 開始更新，避免覆蓋
            for i in range(len(LAG_HOURS) - 1, 0, -1):
                lag_current = LAG_HOURS[i]
                lag_prev = LAG_HOURS[i-1]
                lag_current_col = f'{param}_lag_{lag_current}h'
                lag_prev_col = f'{param}_lag_{lag_prev}h'

                if lag_current_col in current_data_dict and lag_prev_col in current_data_dict:
                    current_data_dict[lag_current_col] = current_data_dict[lag_prev_col]

            # 更新 1 小時滯後特徵為當前預測值
            if f'{param}_lag_1h' in current_data_dict and param in new_pollutant_values:
                current_data_dict[f'{param}_lag_1h'] = new_pollutant_values[param]

        # 6. 滾動平均/標準差特徵無法在遞迴中準確更新，這裡保持省略
        
    return pd.DataFrame(predictions)


# =================================================================
# 模型載入邏輯 (取代 initialize_app_data)
# =================================================================

def load_models_and_metadata():
    global TRAINED_MODELS, LAST_OBSERVATION, FEATURE_COLUMNS, POLLUTANT_PARAMS

    if not os.path.exists(META_PATH):
        print("🚨 [Load] 找不到模型元數據檔案 (model_meta.json)，無法載入模型。")
        return

    try:
        # 1. 載入元數據
        with open(META_PATH, 'r', encoding='utf-8') as f:
            metadata = json.load(f)

        POLLUTANT_PARAMS = metadata.get('pollutant_params', [])
        FEATURE_COLUMNS = metadata.get('feature_columns', [])
        
        # 將最後一筆數據的 JSON 轉換回 DataFrame
        if 'last_observation_json' in metadata:
            # 從 JSON 讀取時，日期會變成字串，之後會在 predict_future_multi 處理
            LAST_OBSERVATION = pd.read_json(metadata['last_observation_json'], orient='records')

        # 2. 載入 XGBoost 模型
        TRAINED_MODELS = {}
        for param in POLLUTANT_PARAMS:
            model_path = os.path.join(MODELS_DIR, f'{param}_model.json')
            if os.path.exists(model_path):
                model = xgb.XGBRegressor()
                model.load_model(model_path)
                TRAINED_MODELS[param] = model
            else:
                print(f"❌ [Load] 找不到 {param} 的模型檔案: {model_path}")
                del POLLUTANT_PARAMS[POLLUTANT_PARAMS.index(param)] # 移除缺失模型的參數
        
        if TRAINED_MODELS:
            print(f"✅ [Load] 成功載入 {len(TRAINED_MODELS)} 個模型。")
        else:
            print("🚨 [Load] 未載入任何模型。")


    except Exception as e:
        print(f"❌ [Load] 模型載入失敗: {e}") 
        TRAINED_MODELS = {} 
        LAST_OBSERVATION = None
        FEATURE_COLUMNS = []
        POLLUTANT_PARAMS = []

# =================================================================
# Flask 應用程式設定與啟動
# =================================================================
app = Flask(__name__)

# 應用程式啟動時，立即執行模型載入 (快速)
with app.app_context():
    load_models_and_metadata() 

@app.route('/')
def index():
    city_name = "高雄"
    
    # 檢查模型是否成功載入
    if TRAINED_MODELS and LAST_OBSERVATION is not None:
        try:
            # 僅執行快速的預測邏輯 (predict_future_multi)
            future_predictions = predict_future_multi(
                TRAINED_MODELS,
                LAST_OBSERVATION,
                FEATURE_COLUMNS,
                POLLUTANT_PARAMS,
                hours=HOURS_TO_PREDICT
            )

            # 格式化結果
            future_predictions['datetime_local'] = future_predictions['datetime'].dt.tz_convert(LOCAL_TZ)
            max_aqi = int(future_predictions['aqi_pred'].max())

            aqi_predictions = [
                {'time': item['datetime_local'].strftime('%Y-%m-%d %H:%M'), 'aqi': int(item['aqi_pred'])}
                for item in future_predictions.to_dict(orient='records')
            ]
            
        except Exception as e:
            max_aqi = "N/A"
            aqi_predictions = []
            print(f"❌ [Request] 預測執行失敗: {e}") 
    else:
        max_aqi = "N/A"
        aqi_predictions = []
        print("🚨 [Request] 模型或數據尚未初始化，無法進行預測。")

    return render_template('index.html', max_aqi=max_aqi, aqi_predictions=aqi_predictions, city_name=city_name)

if __name__ == '__main__':
    # 在本地環境運行時使用
    app.run(debug=True)
