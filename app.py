# app.py - Open-Meteo Weather Integration Revision

# =================================================================
# Import all necessary libraries 
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
import json
from datetime import timedelta, timezone
from flask import Flask, render_template, request
# 引入 Open-Meteo 相關函式庫
import openmeteo_requests
import requests_cache

# Ignore warnings
warnings.filterwarnings('ignore')

# Model and metadata paths
MODELS_DIR = 'models'
META_PATH = os.path.join(MODELS_DIR, 'model_meta.json')

# =================================================================
# OpenAQ API Constants
# =================================================================
# ⚠️ Replace with your own API Key
API_KEY = "fb579916623e8483cd85344b14605c3109eea922202314c44b87a2df3b1fff77" 
HEADERS = {"X-API-Key": API_KEY}
# BASE V3
BASE = "https://api.openaq.org/v3"

# Target geographical coordinates (Default for initial load)
TARGET_LAT = 22.6324 
TARGET_LON = 120.2954

# Initial/Default Location (These will be updated by initialize_location)
DEFAULT_LOCATION_ID = 2395624 # Default: Kaohsiung-Qianjin
DEFAULT_LOCATION_NAME = "Kaohsiung-Qianjin"
DEFAULT_LOCATION_DISPLAY = "高雄市 前金區"
# 確保 location_list 與您 train_and_save.py 中的一致
LOCATION_LIST = [
    {"name": "Kaohsiung-Qianjin", "display": "高雄市 前金區", "id": 2395624, "lat": 22.6324, "lon": 120.2954},
    {"name": "Taichung-Qingshui", "display": "台中市 清水區", "id": 2404099, "lat": 24.2691, "lon": 120.5902},
    {"name": "Taoyuan-Guanyin", "display": "桃園市 觀音區", "id": 2401188, "lat": 25.0410, "lon": 121.0504},
]


# =================================================================
# Feature and Model Constants (從 model_meta.json 載入)
# =================================================================
POLLUTANT_TARGETS = []
FEATURE_COLUMNS = []
LAG_HOURS = []
ROLLING_WINDOWS = []
CURRENT_OBSERVATION_AQI = "N/A"
CURRENT_OBSERVATION_TIME = "N/A"
LAST_OBSERVATION = None
TRAINED_MODELS = {}

# OpenAQ Pollutant Conversion (for V3 endpoint to V2/V1 compatible)
POLLUTANT_MAPPING = {
    'pm25': 'pm25', 'pm10': 'pm10', 'o3': 'o3', 'no2': 'no2', 'so2': 'so2', 'co': 'co'
}

# AQI Calculation helper (US EPA standard simplified)
# You should adapt this to your training standard if different
AQI_BREAKPOINTS = {
    'pm25': [(0.0, 12.0, 0, 50), (12.1, 35.4, 51, 100), (35.5, 55.4, 101, 150), 
             (55.5, 150.4, 151, 200), (150.5, 250.4, 201, 300), (250.5, 350.4, 301, 400), 
             (350.5, 500.4, 401, 500)],
    'o3': [(0.000, 0.054, 0, 50), (0.055, 0.070, 51, 100), (0.071, 0.085, 101, 150), 
           (0.086, 0.105, 151, 200), (0.106, 0.200, 201, 300)], # Units in ppm
    'pm10': [(0, 54, 0, 50), (55, 154, 51, 100), (155, 254, 101, 150), 
             (255, 354, 151, 200), (355, 424, 201, 300), (425, 504, 301, 400), 
             (505, 604, 401, 500)],
}

# I_high - I_low       C_obs - C_low
# -------------- = -----------------
# C_high - C_low       I_high - I_low
def calculate_aqi(pollutant_name, concentration):
    if pd.isna(concentration):
        return np.nan
    for C_low, C_high, I_low, I_high in AQI_BREAKPOINTS.get(pollutant_name, []):
        if C_low <= concentration <= C_high:
            if C_high == C_low: # Avoid division by zero
                return I_low 
            aqi = ((I_high - I_low) / (C_high - C_low)) * (concentration - C_low) + I_low
            return round(aqi)
    return np.nan

def get_overall_aqi(row):
    # pm25 in ug/m3, o3 in ppm, pm10 in ug/m3
    pm25_aqi = calculate_aqi('pm25', row.get('pm25_value'))
    o3_aqi = calculate_aqi('o3', row.get('o3_value'))
    pm10_aqi = calculate_aqi('pm10', row.get('pm10_value'))

    aqi_values = [v for v in [pm25_aqi, o3_aqi, pm10_aqi] if pd.notna(v)]
    return max(aqi_values) if aqi_values else np.nan


# =================================================================
# 輔助函式 (Helper Functions)
# =================================================================

def load_models_and_metadata():
    """載入所有模型和元數據"""
    global POLLUTANT_TARGETS, FEATURE_COLUMNS, LAG_HOURS, ROLLING_WINDOWS, TRAINED_MODELS, LAST_OBSERVATION
    
    if os.path.exists(META_PATH):
        try:
            with open(META_PATH, 'r') as f:
                meta_data = json.load(f)

            POLLUTANT_TARGETS = meta_data.get('pollutant_targets', [])
            LAG_HOURS = meta_data.get('lag_hours', [])
            ROLLING_WINDOWS = meta_data.get('rolling_windows', [])
            FEATURE_COLUMNS = meta_data.get('feature_columns', [])
            
            # 載入 LAST_OBSERVATION
            last_observation_json = meta_data.get('last_observation_json')
            if last_observation_json:
                LAST_OBSERVATION = pd.DataFrame([last_observation_json])
            
            # 載入 TRAINED_MODELS
            TRAINED_MODELS = {}
            for param in POLLUTANT_TARGETS:
                model_path = os.path.join(MODELS_DIR, f'{param}_model.json')
                if os.path.exists(model_path):
                    xgb_model = xgb.XGBRegressor()
                    xgb_model.load_model(model_path)
                    TRAINED_MODELS[param] = xgb_model
                    
            print(f"✅ [Init] 成功載入 {len(TRAINED_MODELS)} 個模型及元數據。")

        except Exception as e:
            print(f"❌ [Init] 載入模型或元數據時發生錯誤: {e}")
            
    else:
        print(f"❌ [Init] 找不到元數據檔案: {META_PATH}")


def initialize_location(location_id=None):
    """根據 ID 設定目標地點的經緯度"""
    global TARGET_LAT, TARGET_LON, DEFAULT_LOCATION_ID, DEFAULT_LOCATION_NAME, DEFAULT_LOCATION_DISPLAY
    
    selected_loc = next((loc for loc in LOCATION_LIST if str(loc['id']) == str(location_id)), None)
    
    if selected_loc:
        TARGET_LAT = selected_loc['lat']
        TARGET_LON = selected_loc['lon']
        DEFAULT_LOCATION_ID = selected_loc['id']
        DEFAULT_LOCATION_NAME = selected_loc['name']
        DEFAULT_LOCATION_DISPLAY = selected_loc['display']
    
    print(f"🌍 [Location] 已設定地點: {DEFAULT_LOCATION_DISPLAY} (ID: {DEFAULT_LOCATION_ID})")


def fetch_latest_observation_data(location_id):
    """從 OpenAQ 取得單一地點的最新觀測值"""
    global CURRENT_OBSERVATION_AQI, CURRENT_OBSERVATION_TIME
    
    CURRENT_OBSERVATION_AQI = "N/A"
    CURRENT_OBSERVATION_TIME = "N/A"
    
    url = f"{BASE}/locations/{location_id}/latest"
    print(f"⏳ [OpenAQ] 正在取得最新觀測值...")

    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if not data.get('results'):
            print("🚨 [OpenAQ] 找不到觀測數據。")
            return pd.DataFrame()

        latest_data = data['results'][0]['latest']
        
        # 將數據轉換為 DataFrame 的單行格式
        row = {'aqi': np.nan}
        valid_count = 0
        latest_time = None
        
        for item in latest_data:
            param = item['parameter']
            value = item['value']
            # 確保只處理我們需要的污染物
            if param in POLLUTANT_MAPPING and pd.notna(value):
                row[f'{param}_value'] = value
                row[f'{param}_unit'] = item['unit']
                valid_count += 1
                
                # 更新觀測時間（取最新的時間）
                time_str = item['date']['utc']
                if time_str:
                    current_time = pd.to_datetime(time_str)
                    if latest_time is None or current_time > latest_time:
                        latest_time = current_time

        if latest_time:
            # 轉換為本地時間
            row['datetime_utc'] = latest_time 
            row['datetime_local'] = latest_time.tz_convert(timezone(timedelta(hours=8))).strftime('%Y-%m-%d %H:%M')
            CURRENT_OBSERVATION_TIME = row['datetime_local']
            
            # 計算整體 AQI
            row['aqi_value'] = get_overall_aqi(row)
            if pd.notna(row['aqi_value']):
                row['aqi'] = int(row['aqi_value'])
                CURRENT_OBSERVATION_AQI = row['aqi']
            
            print(f"✅ [OpenAQ] 觀測時間: {CURRENT_OBSERVATION_TIME}, AQI: {CURRENT_OBSERVATION_AQI}")
            return pd.DataFrame([row])

    except requests.RequestException as e:
        print(f"❌ [OpenAQ] 取得數據失敗: {e}")
    except Exception as e:
        print(f"❌ [OpenAQ] 處理數據失敗: {e}")

    return pd.DataFrame()


def fetch_weather_forecast(lat, lon):
    """從 Open-Meteo 取得未來 24 小時的天氣預報"""
    print("⏳ [Weather] 正在取得未來 24 小時天氣預報...")
    # Setup Open-Meteo client
    cache_session = requests_cache.CachedSession('.openmeteo_cache', expire_after=-1)
    openmeteo = openmeteo_requests.Client(session=cache_session)
    
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ["temperature_2m", "relative_humidity_2m", "surface_pressure", "wind_speed_10m"],
        "forecast_hours": 24, # 只需要未來 24 小時
        "timezone": "Asia/Taipei"
    }
    
    try:
        response = openmeteo.weather_api(url, params=params)[0]
        hourly = response.Hourly()
        
        hourly_data = {
            "datetime_local": pd.to_datetime(hourly.Time(), unit="s").tz_convert(timezone(timedelta(hours=8))),
            "temperature": hourly.Variables(0).ValuesAsNumpy(),
            "humidity": hourly.Variables(1).ValuesAsNumpy(),
            "pressure": hourly.Variables(2).ValuesAsNumpy(),
            "wind_speed": hourly.Variables(3).ValuesAsNumpy(),
        }
        
        forecast_df = pd.DataFrame(hourly_data)
        forecast_df['datetime_local'] = forecast_df['datetime_local'].dt.strftime('%Y-%m-%d %H:%M')
        
        print(f"✅ [Weather] 成功取得 {len(forecast_df)} 筆天氣預報。")
        return forecast_df
        
    except Exception as e:
        print(f"❌ [Weather] 取得天氣預報失敗: {e}")
        return pd.DataFrame()


def prepare_final_data(observation_df, weather_df):
    """將觀測值與未來 24 小時的天氣預報合併"""
    if observation_df.empty or weather_df.empty:
        return pd.DataFrame()
    
    # 取出觀測時間
    obs_time = observation_df['datetime_local'].iloc[0]
    
    # 找到天氣預報中第一個在觀測時間之後或等於觀測時間的時刻
    weather_start_idx = weather_df['datetime_local'].searchsorted(obs_time, side='left')

    # 由於天氣預報是從現在開始，我們取觀測時間之後的 24 個預測時間
    # 我們需要從 weather_df 中取出與預測時間點對齊的 24 筆數據
    
    # 確保觀測時間點在預報中存在
    if weather_start_idx >= len(weather_df):
        print("🚨 [Data Prep] 觀測時間晚於所有天氣預報時間。")
        return pd.DataFrame()
        
    # 我們只需要觀測時間對應的時刻及之後的 23 個時刻 (總共 24 筆)
    # 這裡假設觀測值對應天氣預報的第一筆數據
    weather_df_24h = weather_df.iloc[weather_start_idx:weather_start_idx + 24].reset_index(drop=True)

    if len(weather_df_24h) < 24:
        print(f"🚨 [Data Prep] 僅取得 {len(weather_df_24h)} 筆天氣數據，不足 24 小時。")
        return pd.DataFrame()

    # 將觀測值與未來 24 小時的天氣預報進行合併
    final_df = weather_df_24h.copy()

    # 將觀測值複製到所有 24 個時間步
    # 這是為了確保在遞迴預測開始時，觀測數據（非預測值）是可用的
    for col in [c for c in observation_df.columns if c not in ['datetime_local', 'datetime_utc']]:
        final_df[col] = observation_df[col].iloc[0]

    return final_df.reset_index(drop=True)


def predict_future_multi(initial_data, models, feature_cols, hours_to_predict=24):
    """遞迴地預測未來 24 小時的污染物數值"""
    
    # 檢查核心模型是否存在
    if not all(param in models for param in POLLUTANT_TARGETS):
        print("❌ [Predict] 模型載入不完整，無法進行預測。")
        return pd.DataFrame()

    # 複製初始數據 (包含觀測值和天氣預報)
    prediction_data = initial_data.copy()
    
    # 創建一個字典，儲存當前時間步的模型輸入數據
    current_data_dict = prediction_data.iloc[0].to_dict()
    
    # 存放 24 小時預測結果的 DataFrame
    predictions_list = []

    for t in range(hours_to_predict):
        # 1. 準備當前時間步的特徵輸入
        # 確保字典中的鍵與模型的 feature_cols 一致
        current_features = {k: v for k, v in current_data_dict.items() if k in feature_cols}
        
        # 轉為 DataFrame，用於模型預測 (只有一行)
        X_current = pd.DataFrame([current_features])

        # 2. 進行多污染物預測
        new_pollutant_values = {}
        for param, model in models.items():
            try:
                # 預測該污染物下一小時的數值 (因為我們的模型是預測 t+1 的值)
                predicted_value = model.predict(X_current)[0]
                new_pollutant_values[param] = max(0.0, predicted_value) # 確保非負
            except Exception as e:
                # 若預測失敗，使用前一小時的預測值或觀測值作為 fallback
                print(f"⚠️ [Predict] 預測 {param} 失敗: {e}")
                new_pollutant_values[param] = current_data_dict.get(f'{param}_value', 0.0)

        # 3. 記錄當前時間步的結果
        result_row = {
            'datetime_local': prediction_data.iloc[t]['datetime_local'],
            'hour': t + 1,
            # 記錄預測數值
            **{f'{param}_pred': new_pollutant_values[param] for param in POLLUTANT_TARGETS},
            # 記錄天氣特徵
            'temperature': current_data_dict['temperature'],
            'humidity': current_data_dict['humidity'],
            'pressure': current_data_dict['pressure'],
            'wind_speed': current_data_dict['wind_speed'],
        }
        
        # 計算預測的綜合 AQI
        predicted_aqi_values = {f'{p}_value': new_pollutant_values[p] for p in POLLUTANT_TARGETS}
        result_row['aqi_pred'] = get_overall_aqi(predicted_aqi_values)
        predictions_list.append(result_row)
        
        # 4. 準備下一時間步 (t+1) 的輸入數據
        if t < hours_to_predict - 1:
            # 複製 t+1 的天氣預報
            next_weather = prediction_data.iloc[t + 1].to_dict()
            for key in ['temperature', 'humidity', 'pressure', 'wind_speed']:
                current_data_dict[key] = next_weather.get(key, 0)

            # 更新滯後特徵 (Lagged Features)
            for param in POLLUTANT_TARGETS:
                # 污染物數值（t+1 的當前值）
                current_data_dict[f'{param}_value'] = new_pollutant_values[param]
                
                # 更新 lag_1h: t+1 的 {param}_lag_1h 是 t 的預測值
                if f'{param}_lag_1h' in current_data_dict:
                    current_data_dict[f'{param}_lag_1h'] = new_pollutant_values[param]
                
                # 更新其他滯後特徵 (t+1 的 lag_2h 是 t 的 lag_1h, 以此類推)
                for i in range(len(LAG_HOURS) - 1):
                    lag_hour_next = LAG_HOURS[i+1] # 2, 3, 6, 12, 24
                    lag_hour_current = LAG_HOURS[i] # 1, 2, 3, 6, 12
                    
                    col_next = f'{param}_lag_{lag_hour_next}h'
                    col_current = f'{param}_lag_{lag_hour_current}h'
                    
                    if col_next in current_data_dict and col_current in current_data_dict:
                        current_data_dict[col_next] = current_data_dict[col_current]

                # 更新滾動平均 (Rolling Mean)
                # 這是一個近似的遞迴更新，真正的滾動平均需要歷史序列
                # 這裡我們只更新最長的滾動平均作為一個粗略的近似
                longest_window = max(ROLLING_WINDOWS)
                mean_col = f'{param}_rolling_mean_{longest_window}h'
                std_col = f'{param}_rolling_std_{longest_window}h'
                
                if mean_col in current_data_dict:
                    # 簡單地將新的預測值納入平均值，作為一個近似
                    current_mean = current_data_dict[mean_col]
                    new_mean = (current_mean * (longest_window - 1) + new_pollutant_values[param]) / longest_window
                    current_data_dict[mean_col] = new_mean
                
                # 滾動標準差保持不變或設為一個小值
                if std_col in current_data_dict:
                    current_data_dict[std_col] = current_data_dict.get(std_col, 0.0)

    return pd.DataFrame(predictions_list)


# =================================================================
# Flask 應用程式設定
# =================================================================
app = Flask(__name__)

# 應用程式啟動時載入模型和元數據
load_models_and_metadata()

@app.route('/', methods=['GET', 'POST'])
def index():
    """主頁面路由，處理地點選擇與預測"""
    global CURRENT_OBSERVATION_AQI, CURRENT_OBSERVATION_TIME, LAST_OBSERVATION
    
    # ========== 1️⃣ 處理輸入地點選擇 ==========
    selected_id = request.values.get('location_id')
    if selected_id:
        initialize_location(selected_id)
    else:
        # 確保初始載入時使用預設地點
        initialize_location(DEFAULT_LOCATION_ID)

    is_fallback_mode = True
    aqi_predictions = []
    max_aqi = 50

    # ========== 2️⃣ 取得最新觀測值 ==========
    current_observation_raw = fetch_latest_observation_data(DEFAULT_LOCATION_ID)

    # ========== 3️⃣ 取得未來 24 小時天氣預報 ==========
    weather_forecast_df = fetch_weather_forecast(TARGET_LAT, TARGET_LON)

    # ========== 4️⃣ 資料整合與前處理 ==========
    data_for_prediction = prepare_final_data(current_observation_raw, weather_forecast_df)
    
    # ========== 5️⃣ 進行 24 小時遞迴預測 ==========
    if not data_for_prediction.empty and LAST_OBSERVATION is not None and not LAST_OBSERVATION.empty and TRAINED_MODELS:
        try:
            # 複製 LAST_OBSERVATION 作為模型輸入的基礎（包含所有靜態的滯後特徵）
            observation_for_prediction = LAST_OBSERVATION.iloc[:1].copy()
            latest_row = current_observation_raw.iloc[0]
            
            # (1) 更新非滯後特徵 (觀測值和天氣預報)
            for col in latest_row.index:
                if col in observation_for_prediction.columns and not any(s in col for s in ['lag_', 'rolling_']):
                    observation_for_prediction[col] = latest_row[col]
            
            for col in ['temperature', 'humidity', 'pressure', 'wind_speed']:
                if col in observation_for_prediction.columns and col in data_for_prediction.columns:
                    observation_for_prediction[col] = data_for_prediction.iloc[0][col]


            # =========================================================================
            # ⭐️ 修正核心問題：強制滯後特徵與當前觀測值對齊 (解決預測值恆定在 41 的問題)
            # =========================================================================
            latest_aqi = observation_for_prediction.get('aqi', 41).iloc[0] 
            
            # 取得當前所有污染物最新觀測值 (若有缺失則假設為 0)
            latest_pollutants = {
                p: observation_for_prediction.get(f'{p}_value', 0).iloc[0] 
                for p in POLLUTANT_TARGETS if f'{p}_value' in observation_for_prediction.columns
            }

            # 1. 更新所有 AQI 滯後特徵
            if pd.notna(latest_aqi):
                for lag_hour in LAG_HOURS:
                    aqi_lag_col = f'aqi_lag_{lag_hour}h'
                    if aqi_lag_col in observation_for_prediction.columns:
                        observation_for_prediction[aqi_lag_col] = latest_aqi
            
            # 2. 更新所有污染物滯後特徵和滾動平均特徵
            for param, latest_value in latest_pollutants.items():
                if pd.notna(latest_value):
                    # 更新滯後特徵 (e.g., pm25_lag_1h, pm25_lag_24h)
                    for lag_hour in LAG_HOURS:
                        lag_col = f'{param}_lag_{lag_hour}h'
                        if lag_col in observation_for_prediction.columns:
                            observation_for_prediction[lag_col] = latest_value
                    
                    # 更新滾動平均特徵 (e.g., pm25_rolling_mean_6h)
                    for window in ROLLING_WINDOWS:
                        mean_col = f'{param}_rolling_mean_{window}h'
                        std_col = f'{param}_rolling_std_{window}h'
                        if mean_col in observation_for_prediction.columns:
                            observation_for_prediction[mean_col] = latest_value
                        # 假設過去穩定，標準差為 0
                        if std_col in observation_for_prediction.columns:
                            observation_for_prediction[std_col] = 0.0
            
            # =========================================================================
            # 修正結束
            # =========================================================================

            # 進行預測
            prediction_df = predict_future_multi(
                data_for_prediction, 
                TRAINED_MODELS, 
                FEATURE_COLUMNS
            )
            
            # 處理輸出
            predictions_df = prediction_df[['datetime_local', 'aqi_pred']].copy()
            max_aqi_val = predictions_df['aqi_pred'].max()
            max_aqi = int(max_aqi_val) if pd.notna(max_aqi_val) else CURRENT_OBSERVATION_AQI
            predictions_df['aqi_pred'] = predictions_df['aqi_pred'].replace(np.nan, "N/A")
            predictions_df['aqi'] = predictions_df['aqi_pred'].apply(
                lambda x: int(x) if x != "N/A" else "N/A"
            ).astype(object)
            aqi_predictions = [
                {'time': item['datetime_local'], 'aqi': item['aqi']}
                for item in predictions_df.to_dict(orient='records')
            ]
            
            # 將當前觀測值添加到預測列表的最前面
            if CURRENT_OBSERVATION_AQI != "N/A":
                aqi_predictions.insert(0, {
                    'time': CURRENT_OBSERVATION_TIME,
                    'aqi': CURRENT_OBSERVATION_AQI,
                    'is_obs': True # 標記為觀測值而非預測值
                })

            if aqi_predictions:
                is_fallback_mode = False
                print("✅ [Request] Prediction successful!")
        except Exception as e:
            print(f"❌ [Predict] Error: {e}")

    if is_fallback_mode:
        print("🚨 [Fallback Mode] Showing latest observed AQI only.")
        if CURRENT_OBSERVATION_AQI != "N/A":
            aqi_predictions = [{
                'time': CURRENT_OBSERVATION_TIME,
                'aqi': CURRENT_OBSERVATION_AQI,
                'is_obs': True
            }]

    # ========== 6️⃣ 輸出頁面 ==========
    return render_template(
        'index.html',
        max_aqi=max_aqi,
        aqi_predictions=aqi_predictions,
        current_location_id=DEFAULT_LOCATION_ID,
        current_location_name=DEFAULT_LOCATION_DISPLAY,
        current_aqi=CURRENT_OBSERVATION_AQI,
        current_time=CURRENT_OBSERVATION_TIME,
        location_list=LOCATION_LIST
    )

# Run the app
if __name__ == '__main__':
    # 這裡可以設置 host='0.0.0.0' 以允許外部訪問 (例如在 Render 上運行時)
    app.run(debug=True, host='0.0.0.0')
