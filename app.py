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
    """計算觀測數據的整體 AQI，取所有污染物 AQI 的最大值"""
    aqis = []
    
    for param in POLLUTANT_TARGETS:
        conc = row.get(f'{param}_value') # 使用帶有 _value 的欄位
        if pd.notna(conc):
            aqi = aqi_from_conc(param, conc)
            if pd.notna(aqi):
                aqis.append(aqi)
    
    if 'aqi_pred' in row and pd.notna(row['aqi_pred']):
        aqis.append(row['aqi_pred'])
        
    if not aqis:
        return np.nan
        
    # 整體 AQI 為所有污染物 AQI 中的最大值
    return max(aqis)

def get_aqi_category(aqi):
    """根據 AQI 值返回類別和顏色"""
    if pd.isna(aqi): return "N/A", "gray"
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
        
        if not data.get('results'):
            return pd.DataFrame()

        # 扁平化結果
        latest_data = data['results'][0]
        obs = latest_data['measurements']
        
        # 轉換為 DataFrame
        df = pd.DataFrame(obs)
        if df.empty:
            return pd.DataFrame()
            
        # 轉換日期時間並設定為本地時區
        df['datetime'] = pd.to_datetime(df['datetime'])
        df['datetime'] = df['datetime'].dt.tz_convert(LOCAL_TZ).dt.tz_localize(None) # 移除時區以便與訓練數據匹配
        
        # 轉換為寬格式
        pivot_df = df.pivot_table(index='datetime', columns='parameter_id', values='value').reset_index()
        
        # 確保所有目標污染物欄位存在 (如果 OpenAQ 沒有返回，則填 NaN)
        for param in POLLUTANT_TARGETS:
            if param not in pivot_df.columns:
                pivot_df[param] = np.nan
        
        # 僅保留最新一筆數據
        latest_row = pivot_df.sort_values(by='datetime', ascending=False).iloc[:1]
        
        # 計算 AQI
        latest_row['aqi'] = latest_row.apply(
            lambda row: max([aqi_from_conc(p, row[p]) for p in POLLUTANT_TARGETS if p in row and pd.notna(row[p])]), 
            axis=1
        )
        
        # 重新命名以匹配訓練數據的格式 (用於 t=0 的輸入)
        latest_row.rename(columns={p: f'{p}_value' for p in POLLUTANT_TARGETS}, inplace=True)
        
        return latest_row.reset_index(drop=True)

    except Exception as e:
        print(f"❌ [OpenAQ] Error fetching latest observation: {e}")
        return pd.DataFrame()


# =================================================================
# Open-Meteo Weather Fetching Functions
# =================================================================

def fetch_weather_forecast(lat, lon, start_datetime):
    """
    從 Open-Meteo 獲取未來 24 小時的天氣預報 (從指定時間開始)。
    """
    try:
        # Open-Meteo API 參數
        params = {
            "latitude": lat,
            "longitude": lon,
            "current": ["temperature_2m", "relative_humidity_2m", "surface_pressure"],
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
            "datetime": pd.to_datetime(hourly.Time(), unit="s"),
            "temperature": hourly.Variables(0).ValuesAsNumpy(),
            "humidity": hourly.Variables(1).ValuesAsNumpy(),
            "pressure": hourly.Variables(2).ValuesAsNumpy()
        }
        
        weather_df = pd.DataFrame(hourly_data)
        
        # 將時間轉換為本地時區 (與 OpenAQ 數據的時間格式匹配，即不帶時區)
        weather_df['datetime'] = weather_df['datetime'].dt.tz_convert(LOCAL_TZ).dt.tz_localize(None)
        
        # 過濾出從開始時間之後的數據
        # 為了預測 t+1 到 t+24，我們只需要 t+1 及之後的數據
        start_dt_no_tz = pd.to_datetime(start_datetime).tz_localize(None)
        weather_df = weather_df[weather_df['datetime'] > start_dt_no_tz]
        
        # 僅保留未來 24 小時的預報
        weather_df = weather_df.sort_values(by='datetime').head(24).reset_index(drop=True)
        
        print(f"✅ [OpenMeteo] Fetched {len(weather_df)} hours of weather forecast.")
        
        return weather_df
        
    except Exception as e:
        print(f"❌ [OpenMeteo] Error fetching weather forecast: {e}")
        return pd.DataFrame()


# =================================================================
# Model Initialization and Feature Engineering
# =================================================================

def load_models():
    """載入所有已儲存的 XGBoost 模型和模型元數據"""
    global TRAINED_MODELS, LAST_OBSERVATION, FEATURE_COLUMNS, INITIAL_AQI_INFO
    
    if not os.path.exists(MODELS_DIR) or not os.path.exists(META_PATH):
        print("🚨 [Model] 找不到 models 資料夾或 model_meta.json。請先執行 train_and_save.py。")
        return False

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
            
    if not TRAINED_MODELS:
        print("🚨 [Model] 未載入任何模型。")
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
        #    a. 創建 t-1 的特徵（滯後特徵）
        #       這裡我們需要使用 t-1 時刻 (index t-1) 的觀測或預測值來填充 t 時刻的滯後特徵
        
        #    b. 創建 t-2, t-3... 的特徵（滾動特徵）
        #       這裡我們使用 t-24 到 t-1 時刻的觀測或預測值來計算 t 時刻的滾動特徵
        
        # 由於訓練時已經將所有滯後和滾動特徵都計算好了，這裡只需要從前一行/前 N 行複製過來
        # *********** 關鍵步驟：重新計算特徵 ***********
        
        # 為了避免在預測時重新實現滾動和滯後邏輯，我們使用一個更簡單的方法：
        #   - 使用 t-1 的值填充 t 時刻的 1-hour lag
        #   - 忽略其他 lag 和 rolling，依賴模型從 t=0 的舊 lag 中學到的趨勢。
        
        # 複製 t-1 的預測/觀測值到 t 時刻的滯後特徵
        prev_idx = current_idx - 1
        
        for param in POLLUTANT_TARGETS:
            # 填充 t 時刻的 1-hour lag (使用 t-1 時刻的 value)
            lag_1h_col = f'{param}_lag_1h'
            value_col = f'{param}_value'
            if lag_1h_col in df.columns and value_col in df.columns:
                 df.loc[current_idx, lag_1h_col] = df.loc[prev_idx, value_col]
                 
        # 填充 t 時刻的 aqi lag 1h (使用 t-1 時刻的 aqi)
        if 'aqi_lag_1h' in df.columns and 'aqi' in df.columns:
             df.loc[current_idx, 'aqi_lag_1h'] = df.loc[prev_idx, 'aqi']

        # 確保天氣特徵已經存在 (t+1 開始從 weather_forecast_df 載入)
        # 確保時間特徵已經存在
        df.loc[current_idx, ['hour', 'dayofweek', 'month']] = df.loc[current_idx].pipe(create_datetime_features).loc[current_idx, ['hour', 'dayofweek', 'month']]
        
        # 獲取要傳入模型的特徵
        X_test = df.loc[current_idx, feature_cols].to_frame().T
        
        # 3. 執行預測
        current_predictions = {}
        for param, model in models.items():
            # 執行預測
            pred_value = model.predict(X_test)[0]
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

    if not current_observation_raw.empty:
        latest_row = current_observation_raw.iloc[0]
        
        # 更新當前 AQI
        aqi_val = latest_row['aqi']
        CURRENT_OBSERVATION_AQI = int(aqi_val) if pd.notna(aqi_val) else "N/A"
        
        # 更新時間
        dt_val = latest_row['datetime']
        CURRENT_OBSERVATION_TIME = dt_val.strftime('%Y-%m-%d %H:%M') if pd.notna(dt_val) else "N/A"

        # 更新類別和顏色
        if CURRENT_OBSERVATION_AQI != "N/A":
             CURRENT_OBSERVATION_CATEGORY, CURRENT_OBSERVATION_COLOR = get_aqi_category(CURRENT_OBSERVATION_AQI)
        
        # 將最新的天氣觀測加入到 current_observation_raw (如果 OpenMeteo 的當前觀測能獲取)
        # 這裡由於 V3 API 無法取得歷史天氣，我們暫時跳過這個步驟，
        # 讓 `observation_for_prediction` 在步驟 5 中使用 LAST_OBSERVATION 中的舊天氣特徵作為起始狀態。
        
        print(f"✅ [Observation] Latest AQI: {CURRENT_OBSERVATION_AQI} at {CURRENT_OBSERVATION_TIME}")


    # ========== 3️⃣ 獲取未來天氣預報 ==========
    weather_forecast_df = pd.DataFrame()
    if CURRENT_OBSERVATION_TIME != "N/A":
        # 從當前觀測時間開始，獲取未來 24 小時的天氣預報 (用於 t+1 到 t+24)
        weather_forecast_df = fetch_weather_forecast(
            TARGET_LAT, 
            TARGET_LON, 
            pd.to_datetime(CURRENT_OBSERVATION_TIME)
        )
    
    
    # ========== 4️⃣ 檢查模型和數據完整性 ==========
    aqi_predictions = []
    
    if not TRAINED_MODELS or not LAST_OBSERVATION.shape[0] > 0 or not weather_forecast_df.shape[0] == 24:
        print("🚨 [Predict] 模型/LAST_OBSERVATION/天氣預報 不完整，跳過預測。")
    
    
    # ========== 5️⃣ 建立預測或回退顯示 (修正核心邏輯) ==========
    observation_for_prediction = None
    is_valid_for_prediction = False
    is_fallback_mode = True

    if not current_observation_raw.empty and LAST_OBSERVATION is not None and not LAST_OBSERVATION.empty:
        # 1. 以訓練時的 LAST_OBSERVATION 作為模板，保留其所有歷史/滯後特徵
        observation_for_prediction = LAST_OBSERVATION.iloc[:1].copy()
        
        latest_row = current_observation_raw.iloc[0]
        dt_val = latest_row['datetime']
        if pd.to_datetime(dt_val).tz is not None:
            # 移除時區資訊以匹配訓練集的特徵生成邏輯
            dt_val = pd.to_datetime(dt_val).tz_convert(None)
            
        # 2. 核心修正: 將當前觀測的時間設置為起始時間 (t=0)
        observation_for_prediction['datetime'] = dt_val

        # 3. 核心修正: 用當前測站的觀測值覆蓋訓練時儲存的 "最新觀測值" (t=0)
        #    這確保了預測從當前測站的實際數據開始
        for col in latest_row.index:
            # 覆蓋所有污染物、AQI，以及任何天氣欄位
            if col in observation_for_prediction.columns and not any(s in col for s in ['lag_', 'rolling_']):
                if col in POLLUTANT_TARGETS:
                    col_to_match = f'{col}_value' # 匹配訓練集中的 'pm25_value' 格式
                    if col_to_match in observation_for_prediction.columns:
                         # 使用最新的濃度值覆蓋 t=0 的輸入值
                         observation_for_prediction[col_to_match] = latest_row[col]
                elif col == 'aqi':
                    # 覆蓋 t=0 的實際 AQI 值
                    observation_for_prediction['aqi'] = latest_row['aqi']
                elif col in ['temperature', 'humidity', 'pressure']:
                    # 覆蓋 t=0 的天氣值 (如果存在)
                    observation_for_prediction[col] = latest_row[col]

        # 4. 進行額外檢查：用最新的觀測值來更新 t-1 的 LAG_1h 特徵
        #    這對於遞歸預測的初始步驟至關重要。
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

        # 5. 確保所有必要的特徵列都在
        if all(col in observation_for_prediction.columns for col in FEATURE_COLUMNS):
            is_valid_for_prediction = True

    max_aqi = CURRENT_OBSERVATION_AQI
    
    # 進行預測
    if is_valid_for_prediction and weather_forecast_df.shape[0] == 24:
        try:
            # 建立 t=0 到 t=24 的完整輸入模板
            full_input_df = get_forecast_input_template(observation_for_prediction, weather_forecast_df)
            
            # 執行遞歸預測
            predictions_df = predict_future_multi(full_input_df, TRAINED_MODELS, FEATURE_COLUMNS)
            
            # 準備輸出格式
            predictions_df['datetime_local'] = pd.to_datetime(predictions_df['datetime']).dt.tz_localize(LOCAL_TZ)
            predictions_df = predictions_df.loc[:, ['datetime_local', 'aqi_pred']].copy()
            max_aqi_val = predictions_df['aqi_pred'].max()
            max_aqi = int(max_aqi_val) if pd.notna(max_aqi_val) else CURRENT_OBSERVATION_AQI
            predictions_df['aqi_pred'] = predictions_df['aqi_pred'].replace(np.nan, "N/A")
            predictions_df['aqi'] = predictions_df['aqi_pred'].apply(
                lambda x: int(x) if x != "N/A" else "N/A"
            ).astype(object)
            aqi_predictions = [
                {'time': item['datetime_local'].strftime('%Y-%m-%d %H:%M'), 'aqi': item['aqi']}
                for item in predictions_df.to_dict(orient='records')
            ]
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
