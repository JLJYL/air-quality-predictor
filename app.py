# app.py - Open-Meteo Weather Integration Revision (with Traceback Debugging)

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
import traceback # <-- Added for debugging
from datetime import timedelta, timezone
from flask import Flask, render_template, request
# 引入 Open-Meteo 相關函式庫
import openmeteo_requests
import requests_cache
from retry import retry

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
DEFAULT_CITY = "Kaohsiung City"
DEFAULT_COUNTRY = "TW"

# Timezone setting for localization
LOCAL_TZ = "Asia/Taipei"

# Pollutants used for modeling
POLLUTANT_TARGETS = ["pm25", "pm10", "o3", "no2", "so2", "co"]
# Feature columns must match train_and_save.py
FEATURE_COLUMNS = [] # Will be loaded from model_meta.json

# Globals to store loaded model/data
TRAINED_MODELS = {}
LAST_OBSERVATION = None 
INITIAL_AQI_INFO = None
LOCATION_LIST = []

# Open-Meteo Setup
# Setup the cache for Open-Meteo requests
cache_session = requests_cache.CachedSession('.openmeteo_cache', expire_after = 3600)  # Cache for 1 hour
openmeteo = openmeteo_requests.Client(session=cache_session)
WEATHER_URL = "https://api.open-meteo.com/v1/forecast"


# =================================================================
# AQI Calculation (Referencing US EPA Standard)
# =================================================================
# AQI Breakpoints (Simplified for main pollutants)
AQI_BREAKPOINTS = {
    'pm25': [(0.0, 12.0, 0, 50), (12.1, 35.4, 51, 100), (35.5, 55.4, 101, 150), 
             (55.5, 150.4, 151, 200), (150.5, 250.4, 201, 300), (250.5, 500.4, 301, 500)],
    'pm10': [(0, 54, 0, 50), (55, 154, 51, 100), (155, 254, 101, 150), 
             (255, 354, 151, 200), (355, 424, 201, 300), (425, 604, 301, 500)],
    # O3, NO2, SO2, CO... (Simplified for core pollutants)
}

def aqi_from_conc(pollutant, conc):
    """計算單一污染物濃度對應的 AQI"""
    if pd.isna(conc) or conc < 0:
        return np.nan
    
    # 查找對應的污染物區間
    breakpoints = AQI_BREAKPOINTS.get(pollutant, [])
    
    for bp_low, bp_high, aqi_low, aqi_high in breakpoints:
        if bp_low <= conc <= bp_high:
            # 使用線性插值公式
            if bp_low == bp_high: # 避免除以零
                return aqi_low
            
            aqi = ((aqi_high - aqi_low) / (bp_high - bp_low)) * (conc - bp_low) + aqi_low
            return int(round(aqi))
    
    # 超出最高區間
    if breakpoints and conc > breakpoints[-1][1]:
         # 簡單地返回最高區間的最高 AQI 或 NaN
        return np.nan # 保持一致性，如果超出，則讓 max() 忽略
        
    return np.nan

def calculate_aqi(row):
    """計算觀測數據的整體 AQI,取所有污染物 AQI 的最大值"""
    aqis = []
    
    for param in POLLUTANT_TARGETS:
        conc = row.get(f'{param}_value') # 使用帶有 _value 的欄位
        if pd.notna(conc):
            aqi = aqi_from_conc(param, conc)
            if pd.notna(aqi):
                aqis.append(aqi)
    
    # 注意: 這裡計算的是觀測或預測後的 'aqi' 欄位，不應該讀取 'aqi_pred'
    # 'aqi_pred' 應只在 predict_future_multi 中用於最終輸出。
        
    if not aqis:
        return np.nan
        
    # 整體 AQI 為所有污染物 AQI 中的最大值
    return max(aqis)

def get_aqi_category(aqi):
    """根據 AQI 值返回類別和顏色"""
    if pd.isna(aqi) or aqi == "N/A": return "N/A", "gray"
    aqi = int(aqi)
    
    if 0 <= aqi <= 50:
        return "良好", "bg-emerald-500"
    elif 51 <= aqi <= 100:
        return "中等", "bg-yellow-500"
    elif 101 <= aqi <= 150:
        return "對敏感族群不健康", "bg-orange-500"
    elif 151 <= aqi <= 200:
        return "不健康", "bg-red-600"
    elif 201 <= aqi <= 300:
        return "非常不健康", "bg-purple-600"
    else:
        return "危險", "bg-gray-800"


# =================================================================
# OpenAQ Data Fetching Functions
# =================================================================

@retry(tries=3, delay=2, backoff=2, exceptions=(requests.exceptions.Timeout, requests.exceptions.HTTPError))
def fetch_location_list(country_id=DEFAULT_COUNTRY):
    """獲取國家/地區內的測站列表"""
    try:
        url = f"{BASE}/locations"
        params = {
            'country_id': country_id,
            'limit': 1000,
            'order_by': 'name'
        }
        response = requests.get(url, headers=HEADERS, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        locations = []
        for loc in data.get('results', []):
            loc_id = loc.get('id')
            name = loc.get('name')
            # 確保有必要的資訊
            if loc_id and name:
                 locations.append({
                    'id': loc_id,
                    'name': name,
                    'city': loc.get('city', 'N/A'),
                    'latitude': loc.get('coordinates', {}).get('latitude'),
                    'longitude': loc.get('coordinates', {}).get('longitude')
                 })
                 
        # 僅保留有經緯度的測站
        LOCATION_LIST.extend([loc for loc in locations if loc['latitude'] is not None and loc['longitude'] is not None])
        print(f"✅ [Location] Loaded {len(LOCATION_LIST)} locations.")

    except Exception as e:
        print(f"❌ [Location] Error fetching locations: {e}")
        traceback.print_exc()

@retry(tries=3, delay=2, backoff=2, exceptions=(requests.exceptions.Timeout, requests.exceptions.HTTPError))
def fetch_latest_observation(location_id):
    """獲取單一測站的最新觀測數據"""
    try:
        url = f"{BASE}/latest"
        params = {
            'location_id': location_id,
            'limit': 100, # 獲取所有污染物
            'parameter_id': [f"pm25", "pm10", "o3", "no2", "so2", "co"]
        }
        response = requests.get(url, headers=HEADERS, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        # 檢查是否有結果或測量數據
        if not data.get('results') or not data['results'][0].get('measurements'):
            print(f"⚠️ [OpenAQ] No measurements found for location ID: {location_id}.")
            return pd.DataFrame()

        # 扁平化結果
        latest_data = data['results'][0]
        obs = latest_data['measurements']
        
        # 轉換為 DataFrame
        df = pd.DataFrame(obs)
        if df.empty:
            return pd.DataFrame()
            
        # 轉換日期時間。API 返回的時間是 UTC，但沒有時區標記，我們假設它是一個 'Z' 結尾的 UTC 時間
        df['datetime'] = pd.to_datetime(df['datetime'])
        # 將 UTC 時間轉換為本地時區，然後移除時區資訊 (Naive Local Time)
        df['datetime'] = df['datetime'].dt.tz_convert(LOCAL_TZ).dt.tz_localize(None)
        
        # 轉換為寬格式
        pivot_df = df.pivot_table(index='datetime', columns='parameter_id', values='value').reset_index()
        
        # 確保所有目標污染物欄位存在 (如果 OpenAQ 沒有返回，則填 NaN)
        for param in POLLUTANT_TARGETS:
            if param not in pivot_df.columns:
                pivot_df[param] = np.nan
        
        # 僅保留最新一筆數據
        latest_row = pivot_df.sort_values(by='datetime', ascending=False).iloc[:1].copy()
        
        # 計算 AQI
        latest_row['aqi'] = latest_row.apply(
            lambda row: max([aqi_from_conc(p, row[p]) for p in POLLUTANT_TARGETS if p in row and pd.notna(row[p])]), 
            axis=1
        )
        
        # 重新命名以匹配訓練數據的格式 (用於 t=0 的輸入)
        latest_row.rename(columns={p: f'{p}_value' for p in POLLUTANT_TARGETS}, inplace=True)
        
        return latest_row.reset_index(drop=True)

    except Exception as e:
        print(f"❌ [OpenAQ] 處理數據失敗: {e}")
        print("--- OpenAQ Traceback Start ---")
        traceback.print_exc()
        print("--- OpenAQ Traceback End ---")
        return pd.DataFrame()


# =================================================================
# Open-Meteo Weather Fetching Functions
# =================================================================

def fetch_weather_forecast(lat, lon, start_datetime):
    """
    從 Open-Meteo 獲取未來 24 小時的天氣預報 (從指定時間開始)。
    start_datetime 預期是從 OpenAQ 來的 timezone-naive Timestamp。
    """
    # 確保 start_datetime 是有效的 Timestamp 物件
    if start_datetime is None or pd.isna(start_datetime):
        print("⚠️ [OpenMeteo] 無效的開始時間戳記，無法獲取天氣預報。")
        return pd.DataFrame()
        
    try:
        # Open-Meteo API 參數
        params = {
            "latitude": lat,
            "longitude": lon,
            "hourly": ["temperature_2m", "relative_humidity_2m", "surface_pressure"],
            "timezone": "auto", # 讓 Open-Meteo 處理時區
            "forecast_hours": 48 # 獲取 48 小時預報
        }
        
        responses = openmeteo.weather_api(WEATHER_URL, params=params)
        
        # 僅使用第一個回應 (如果有多個經緯度)
        response = responses[0]
        
        # 獲取小時數據
        hourly = response.Hourly()
        
        hourly_data = {
            "datetime": pd.to_datetime(hourly.Time(), unit="s", utc=True), # 確保它是 UTC
            "temperature": hourly.Variables(0).ValuesAsNumpy(),
            "humidity": hourly.Variables(1).ValuesAsNumpy(),
            "pressure": hourly.Variables(2).ValuesAsNumpy()
        }
        
        weather_df = pd.DataFrame(hourly_data)
        
        # 1. 將 UTC 時間轉換為本地時區 (帶時區資訊)
        weather_df['datetime'] = weather_df['datetime'].dt.tz_convert(LOCAL_TZ)
        
        # 2. 移除時區資訊，變成 naive (匹配 OpenAQ 數據和模型訓練)
        weather_df['datetime'] = weather_df['datetime'].dt.tz_localize(None)
        
        # 3. 過濾出從開始時間之後的數據
        # start_datetime 已經是 naive Timestamp
        weather_df = weather_df[weather_df['datetime'] > start_datetime]
            
        # 僅保留未來 24 小時的預報
        weather_df = weather_df.sort_values(by='datetime').head(24).reset_index(drop=True)
        
        print(f"✅ [OpenMeteo] Fetched {len(weather_df)} hours of weather forecast.")
        
        return weather_df
        
    except Exception as e:
        print(f"❌ [OpenMeteo] 取得天氣預報失敗: {e}")
        print("--- Weather Traceback Start ---")
        traceback.print_exc()
        print("--- Weather Traceback End ---")
        return pd.DataFrame()


# =================================================================
# Model Initialization and Feature Engineering
# =================================================================

def load_models():
    """載入所有已儲存的 XGBoost 模型和模型元數據"""
    global TRAINED_MODELS, LAST_OBSERVATION, FEATURE_COLUMNS, INITIAL_AQI_INFO
    
    # 即使模型檔案不存在，也嘗試載入元數據 (Feature Columns/LAST_OBSERVATION)
    if os.path.exists(META_PATH):
        try:
            with open(META_PATH, 'r') as f:
                meta_data = json.load(f)
                FEATURE_COLUMNS = meta_data.get('feature_columns', [])
                
                # 載入 LAST_OBSERVATION
                last_obs_json = meta_data.get('last_observation')
                if last_obs_json:
                    last_obs_df = pd.DataFrame([last_obs_json])
                    # 確保 'datetime' 欄位被正確轉換
                    last_obs_df['datetime'] = pd.to_datetime(last_obs_df['datetime']).dt.tz_localize(None)
                    LAST_OBSERVATION = last_obs_df
                    print("✅ [Model] LAST_OBSERVATION 載入成功。")
                    
                INITIAL_AQI_INFO = meta_data.get('initial_aqi_info', {})
                
            # 載入每個 pollutant 的模型
            for param in POLLUTANT_TARGETS:
                model_path = os.path.join(MODELS_DIR, f'{param}_model.json')
                if os.path.exists(model_path):
                    xgb_model = xgb.XGBRegressor()
                    xgb_model.load_model(model_path)
                    TRAINED_MODELS[param] = xgb_model
                    print(f"✅ [Model] {param} 模型載入成功。")
                else:
                    print(f"⚠️ [Model] 找不到 {param} 模型 ({model_path})。")
                    
        except Exception as e:
            print(f"🚨 [Model] 載入元數據或模型時發生錯誤: {e}")
            
    if not TRAINED_MODELS:
        print("🚨 [Model] 未載入任何模型。預測功能將無法運作。")
        return False
    
    print(f"✅ [Model] 所有模型和元數據載入完成。總共 {len(TRAINED_MODELS)} 個模型。")
    return True

def create_datetime_features(df):
    """創建時間相關特徵：小時、星期幾、月份"""
    df['hour'] = df['datetime'].dt.hour
    df['dayofweek'] = df['datetime'].dt.dayofweek
    df['month'] = df['datetime'].dt.month
    return df

def generate_lag_features(df, param):
    """為單一污染物生成滯後特徵 (lag features)"""
    for lag in [1, 2, 3, 6, 12, 24]:
        df[f'{param}_lag_{lag}h'] = df[f'{param}_value'].shift(lag)
    return df

def generate_rolling_features(df, param):
    """為單一污染物生成滾動統計特徵 (rolling mean/std)"""
    for window in [6, 12, 24]:
        df[f'{param}_rolling_mean_{window}h'] = df[f'{param}_value'].rolling(window=window).mean()
        df[f'{param}_rolling_std_{window}h'] = df[f'{param}_value'].rolling(window=window).std()
    return df

def get_forecast_input_template(observation_for_prediction, weather_forecast_df):
    """
    建立未來 24 小時預測的輸入模板。
    它包含 t=0 的實際觀測值（已在 index() 中用最新數據覆蓋），以及 t+1 到 t+24 的時間和天氣預報。
    """
    
    # 1. 建立 t+1 到 t+24 的時間序列
    start_dt = observation_for_prediction['datetime'].iloc[0]
    future_datetimes = [start_dt + timedelta(hours=i) for i in range(1, 25)]
    future_df = pd.DataFrame({'datetime': future_datetimes})
    
    # 2. 合併天氣預報 (t+1 到 t+24)
    #    由於 weather_forecast_df 已經被過濾為 t+1 到 t+24，可以直接合併
    future_df = future_df.merge(weather_forecast_df, on='datetime', how='left')
    
    # 3. 建立完整的預測 DataFrame
    #    t=0 (實際觀測) + t+1 到 t+24 (未來預測)
    full_prediction_df = pd.concat([observation_for_prediction, future_df], ignore_index=True)
    
    # 4. 初始化所有污染物、AQI 欄位為 NaN (t+1 到 t+24 的值)
    for param in POLLUTANT_TARGETS:
        full_prediction_df[f'{param}_value'] = full_prediction_df.get(f'{param}_value', np.nan)
    full_prediction_df['aqi'] = full_prediction_df.get('aqi', np.nan)
    
    # 5. 創建所有必要的特徵欄位，並填入 NaN
    for col in FEATURE_COLUMNS:
        if col not in full_prediction_df.columns:
            full_prediction_df[col] = np.nan
    
    # 6. 創建時間特徵
    full_prediction_df = create_datetime_features(full_prediction_df)
    
    # 確保只有需要的特徵欄位
    return full_prediction_df


# =================================================================
# Main Prediction Logic
# =================================================================

def predict_future_multi(df, models, feature_cols):
    """
    執行遞歸多步預測。
    
    Args:
        df: 包含 t=0 實際觀測和 t+1 到 t+24 天氣預報的 DataFrame。
        models: 訓練好的模型字典。
        feature_cols: 模型需要的特徵列表。
        
    Returns:
        包含 t+1 到 t+24 預測結果的 DataFrame。
    """
    N_STEPS = 24 # 預測未來 24 小時
    
    # 將所有數值欄位轉換為 float
    for col in df.columns:
        if df[col].dtype == object and col != 'datetime':
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # 進行 24 步遞歸預測
    for t in range(1, N_STEPS + 1):
        
        # 1. 獲取當前要預測的時間點的行 (即 t 時刻)
        #    由於 df 是 t=0 到 t=24，t=1 是第二行 (index 1)
        current_idx = t
        
        if current_idx >= len(df):
            break

        # 2. 準備當前時間點 (t) 的特徵數據
        # 複製 t-1 的預測/觀測值到 t 時刻的滯後特徵
        prev_idx = current_idx - 1
        
        for param in POLLUTANT_TARGETS:
            # 填充 t 時刻的 1-hour lag (使用 t-1 時刻的 value)
            lag_1h_col = f'{param}_lag_{1}h'
            value_col = f'{param}_value'
            if lag_1h_col in df.columns and value_col in df.columns:
                 # 使用 .loc 進行精確賦值
                 df.loc[current_idx, lag_1h_col] = df.loc[prev_idx, value_col]
                 
        # 填充 t 時刻的 aqi lag 1h (使用 t-1 時刻的 aqi)
        if 'aqi_lag_1h' in df.columns and 'aqi' in df.columns:
             df.loc[current_idx, 'aqi_lag_1h'] = df.loc[prev_idx, 'aqi']

        # 獲取要傳入模型的特徵
        X_test = df.loc[current_idx, feature_cols].to_frame().T
        
        # 3. 執行預測
        current_predictions = {}
        for param, model in models.items():
            # 執行預測
            # 確保輸入 X_test 不含 NaN (XGBoost 不支援 NaN)
            X_test_filled = X_test.fillna(0) # ⚠️ 簡化處理: 僅用 0 填充缺失值，這可能影響準確性，但避免崩潰
            
            pred_value = model.predict(X_test_filled)[0]
            current_predictions[param] = max(0, pred_value) # 確保濃度不為負
            
            # 將預測值存回 DataFrame
            df.loc[current_idx, f'{param}_value'] = current_predictions[param]

        # 4. 計算並存儲 AQI 預測值
        #    首先計算 t 時刻的總體 AQI
        df.loc[current_idx, 'aqi'] = calculate_aqi(df.loc[current_idx])
        
        # 將最終的 AQI 預測值單獨儲存，以便在最後返回時使用
        df.loc[current_idx, 'aqi_pred'] = df.loc[current_idx, 'aqi']
        
        
    # 返回 t=1 到 t=24 的預測結果
    return df.iloc[1:].copy()


# =================================================================
# Flask Application Setup and Routes
# =================================================================

app = Flask(__name__)

# 應用程式啟動時載入模型和測站列表
if not TRAINED_MODELS:
    print("⏳ [App] 正在載入模型...")
    if load_models():
        print("✅ [App] 模型載入完成。")
        # 載入測站列表
        fetch_location_list()
    else:
        print("🚨 [App] 無法啟動應用程式，模型載入失敗。")
        # 即使模型載入失敗，仍嘗試載入測站列表以提供基本介面
        fetch_location_list()


@app.route('/', methods=['GET', 'POST'])
def index():
    """主頁面：顯示最新觀測和預測結果"""
    
    # ========== 1️⃣ 處理用戶輸入和狀態設定 ==========
    global TARGET_LAT, TARGET_LON, DEFAULT_LOCATION_ID, DEFAULT_LOCATION_NAME
    
    selected_location_id = request.form.get('location_id') or request.args.get('location_id')
    
    if selected_location_id:
        # 嘗試在 LOCATION_LIST 中找到對應的經緯度
        target_loc = next((loc for loc in LOCATION_LIST if str(loc['id']) == str(selected_location_id)), None)
        if target_loc:
            TARGET_LAT = target_loc['latitude']
            TARGET_LON = target_loc['longitude']
            DEFAULT_LOCATION_ID = target_loc['id']
            DEFAULT_LOCATION_NAME = target_loc['name']
        else:
            # 如果找不到，則回退到初始的預設值 (避免出錯)
            selected_location_id = DEFAULT_LOCATION_ID
            
    else:
         selected_location_id = DEFAULT_LOCATION_ID
         # 確保經緯度也是預設的
         target_loc = next((loc for loc in LOCATION_LIST if str(loc['id']) == str(selected_location_id)), None)
         if target_loc:
             TARGET_LAT = target_loc['latitude']
             TARGET_LON = target_loc['longitude']

    print(f"🌍 [Request] Selected Location: {DEFAULT_LOCATION_NAME} ({selected_location_id}) at ({TARGET_LAT}, {TARGET_LON})")
    
    
    # ========== 2️⃣ 獲取當前觀測數據 ==========
    current_observation_raw = fetch_latest_observation(selected_location_id)
    
    CURRENT_OBSERVATION_AQI = "N/A"
    CURRENT_OBSERVATION_TIME = "N/A"
    CURRENT_OBSERVATION_CATEGORY = "N/A"
    CURRENT_OBSERVATION_COLOR = "bg-gray-400"
    CURRENT_OBSERVATION_DT = None # 用於儲存 Timestamp 物件

    if not current_observation_raw.empty:
        latest_row = current_observation_raw.iloc[0]
        
        # --- Update AQI ---
        aqi_val = latest_row['aqi']
        CURRENT_OBSERVATION_AQI = int(aqi_val) if pd.notna(aqi_val) else "N/A"
        
        # --- Update Time (和儲存 Timestamp 物件) ---
        dt_val = latest_row['datetime']
        if pd.notna(dt_val):
            CURRENT_OBSERVATION_DT = dt_val 
            CURRENT_OBSERVATION_TIME = CURRENT_OBSERVATION_DT.strftime('%Y-%m-%d %H:%M')

        # --- Update Category and Color ---
        if CURRENT_OBSERVATION_AQI != "N/A":
             CURRENT_OBSERVATION_CATEGORY, CURRENT_OBSERVATION_COLOR = get_aqi_category(CURRENT_OBSERVATION_AQI)
        
        print(f"✅ [Observation] Latest AQI: {CURRENT_OBSERVATION_AQI} at {CURRENT_OBSERVATION_TIME}")
    else:
        print(f"⚠️ [Observation] OpenAQ returned empty data for location {selected_location_id}. Continuing in fallback mode.")


    # ========== 3️⃣ 獲取未來天氣預報 (使用 Timestamp 物件) ==========
    weather_forecast_df = pd.DataFrame()
    if CURRENT_OBSERVATION_DT is not None: 
        # 從當前觀測時間開始，獲取未來 24 小時的天氣預報 (用於 t+1 到 t+24)
        weather_forecast_df = fetch_weather_forecast(
            TARGET_LAT, 
            TARGET_LON, 
            CURRENT_OBSERVATION_DT # 直接傳遞 Timestamp 物件
        )
    else:
        print("⚠️ [Weather] Skipping weather fetch because CURRENT_OBSERVATION_DT is None.")
    
    
    # ========== 4️⃣ 檢查模型和數據完整性 ==========
    aqi_predictions = []
    
    # 模型必須存在、LAST_OBSERVATION 必須載入、天氣預報必須有 24 筆數據
    is_valid_for_prediction = bool(TRAINED_MODELS) and \
                             LAST_OBSERVATION is not None and \
                             not LAST_OBSERVATION.empty and \
                             weather_forecast_df.shape[0] == 24
    
    is_fallback_mode = True

    # ========== 5️⃣ 建立預測或回退顯示 ==========
    
    if is_valid_for_prediction and not current_observation_raw.empty:
        try:
            # 1. 以訓練時的 LAST_OBSERVATION 作為模板，保留其所有歷史/滯後特徵
            observation_for_prediction = LAST_OBSERVATION.iloc[:1].copy()
            
            latest_row = current_observation_raw.iloc[0]
            dt_val = CURRENT_OBSERVATION_DT # 使用我們已經驗證過的 Timestamp
                
            # 2. 核心修正: 將當前觀測的時間設置為起始時間 (t=0)
            observation_for_prediction['datetime'] = dt_val

            # 3. 核心修正: 用當前測站的觀測值覆蓋訓練時儲存的 "最新觀測值" (t=0)
            for col in POLLUTANT_TARGETS:
                col_to_match = f'{col}_value'
                if col_to_match in observation_for_prediction.columns:
                     observation_for_prediction[col_to_match] = latest_row.get(col_to_match, np.nan)
            
            if 'aqi' in observation_for_prediction.columns:
                observation_for_prediction['aqi'] = latest_row.get('aqi', np.nan)

            # 4. 進行額外檢查：用最新的觀測值來更新 t-1 的 LAG_1h 特徵
            for param in POLLUTANT_TARGETS:
                 value_col = f'{param}_value'
                 lag_1h_col = f'{param}_lag_1h'
                 if value_col in observation_for_prediction.columns and lag_1h_col in observation_for_prediction.columns:
                     # 使用當前最新觀測值作為 t-1 的輸入
                     observation_for_prediction[lag_1h_col] = observation_for_prediction[value_col].iloc[0]

            aqi_lag_1h_col = 'aqi_lag_1h'
            if 'aqi' in observation_for_prediction.columns and aqi_lag_1h_col in observation_for_prediction.columns:
                 # 使用當前最新 AQI 作為 t-1 的輸入
                 observation_for_prediction[aqi_lag_1h_col] = observation_for_prediction['aqi'].iloc[0]


            # 5. 執行預測
            # 建立 t=0 到 t=24 的完整輸入模板
            full_input_df = get_forecast_input_template(observation_for_prediction, weather_forecast_df)
            
            # 執行遞歸預測
            predictions_df = predict_future_multi(full_input_df, TRAINED_MODELS, FEATURE_COLUMNS)
            
            # 準備輸出格式
            predictions_df['datetime_local'] = predictions_df['datetime'].dt.tz_localize(LOCAL_TZ)
            predictions_df = predictions_df.loc[:, ['datetime_local', 'aqi_pred']].copy()
            
            # 計算最大預測 AQI
            max_aqi_val = predictions_df['aqi_pred'].max()
            max_aqi = int(max_aqi_val) if pd.notna(max_aqi_val) and max_aqi_val > 0 else CURRENT_OBSERVATION_AQI
            
            predictions_df['aqi'] = predictions_df['aqi_pred'].apply(
                lambda x: int(x) if pd.notna(x) else "N/A"
            ).astype(object)
            
            aqi_predictions = [
                {'time': item['datetime_local'].strftime('%Y-%m-%d %H:%M'), 'aqi': item['aqi']}
                for item in predictions_df.to_dict(orient='records')
            ]
            if aqi_predictions:
                is_fallback_mode = False
                print("✅ [Request] Prediction successful!")
        except Exception as e:
            print(f"❌ [Predict] Error during prediction logic: {e}")
            print("--- Prediction Traceback Start ---")
            traceback.print_exc()
            print("--- Prediction Traceback End ---")

    if is_fallback_mode:
        print("🚨 [Fallback Mode] Showing latest observed AQI only.")
        # 如果當前有觀測值，則只顯示觀測值
        if CURRENT_OBSERVATION_AQI != "N/A":
            aqi_predictions = [{
                'time': CURRENT_OBSERVATION_TIME,
                'aqi': CURRENT_OBSERVATION_AQI,
                'is_obs': True
            }]

    # ========== 6️⃣ 輸出頁面 =========
    return render_template(
        'index.html',
        max_aqi=max_aqi,
        current_aqi=CURRENT_OBSERVATION_AQI,
        current_time=CURRENT_OBSERVATION_TIME,
        current_category=CURRENT_OBSERVATION_CATEGORY,
        current_color=CURRENT_OBSERVATION_COLOR,
        selected_location_id=str(DEFAULT_LOCATION_ID), # 確保為字串
        selected_location_name=DEFAULT_LOCATION_NAME,
        location_list=LOCATION_LIST,
        aqi_predictions=aqi_predictions,
        is_fallback_mode=is_fallback_mode,
    )

if __name__ == '__main__':
    # Flask 應用程式會在伺服器中運行
    # app.run(debug=True)
    pass
