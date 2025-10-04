# =================================================================
# Import all necessary libraries 
# =================================================================
import requests
import pandas as pd
# 修正：將 datetime 類別導入為 dt，以避免與 datetime 模組本身的命名衝突
from datetime import datetime as dt, timedelta, timezone 
import re
import os
import warnings
import numpy as np
import xgboost as xgb
import json
from flask import Flask, render_template

# 忽略警告 
warnings.filterwarnings('ignore')

# 模型和中繼資料路徑 
MODELS_DIR = 'models'
META_PATH = os.path.join(MODELS_DIR, 'model_meta.json')

# =================================================================
# OpenAQ API Constants
# =================================================================
# ⚠️ 請替換為您自己的 API Key 
API_KEY = "fb579916623e8483cd85344b14605c3109eea922202314c44b87a2df3b1fff77" 
HEADERS = {"X-API-Key": API_KEY}
# BASE V3
BASE = "https://api.openaq.org/v3"

# 目標地理坐標 (高雄市)
TARGET_LAT = 22.6324 
TARGET_LON = 120.2954

# 初始/預設地點 (啟動時會被 initialize_location 更新)
DEFAULT_LOCATION_ID = 2395624 # 預設：Kaohsiung-Qianjin (高雄-前金)
DEFAULT_LOCATION_NAME = "Kaohsiung-Qianjin" # 預設地點名稱

TARGET_PARAMS = ["co", "no2", "o3", "pm10", "pm25", "so2"]
PARAM_IDS = {"co": 8, "no2": 7, "o3": 10, "pm10": 1, "pm25": 2, "so2": 9}

# =================================================================
# Global Variables (可變動的全域變數)
# =================================================================
TRAINED_MODELS = {} 
LAST_OBSERVATION = None 
FEATURE_COLUMNS = []
POLLUTANT_PARAMS = [] 
HOURS_TO_PREDICT = 24

# 儲存最新的觀測數據 (用於回退/錯誤處理)
CURRENT_OBSERVATION_AQI = "N/A"
CURRENT_OBSERVATION_TIME = "N/A"

# 動態地點變數 (啟動時更新)
current_location_id = DEFAULT_LOCATION_ID
current_location_name = DEFAULT_LOCATION_NAME

# =================================================================
# Constants (常數)
# =================================================================
LOCAL_TZ = "Asia/Taipei"
LAG_HOURS = [1, 2, 3, 6, 12, 24]
ROLLING_WINDOWS = [6, 12, 24]
POLLUTANT_TARGETS = ["pm25", "pm10", "o3", "no2", "so2", "co"] 

# AQI 區間定義 (簡化版本)
AQI_BREAKPOINTS = {
    "pm25": [(0.0, 12.0, 0, 50), (12.1, 35.4, 51, 100), (35.5, 55.4, 101, 150), (55.5, 150.4, 151, 200)],
    "pm10": [(0, 54, 0, 50), (55, 154, 51, 100), (155, 254, 101, 150), (255, 354, 151, 200)],
    "o3": [(0, 54, 0, 50), (55, 70, 51, 100), (71, 85, 101, 150), (86, 105, 151, 200)],
    "co": [(0.0, 4.4, 0, 50), (4.5, 9.4, 51, 100), (9.5, 12.4, 101, 150), (12.5, 15.4, 151, 200)],
    "no2": [(0, 100, 0, 50), (101, 360, 51, 100), (361, 649, 101, 150), (650, 1249, 151, 200)],
    "so2": [(0, 35, 0, 50), (36, 75, 51, 100), (76, 185, 101, 150), (186, 304, 151, 200)],
}


# =================================================================
# OpenAQ Data Fetching Functions (OpenAQ 數據獲取函數)
# =================================================================

def get_location_meta(location_id: int):
    """Fetches location metadata including the last update time (Uses V3)."""
    try:
        r = requests.get(f"{BASE}/locations/{location_id}", headers=HEADERS, timeout=10)
        r.raise_for_status()
        row = r.json()["results"][0]
        last_utc = pd.to_datetime(row["datetimeLast"]["utc"], errors="coerce", utc=True)
        return {
            "id": int(row["id"]),
            "name": row["name"],
            "last_utc": last_utc,
        }
    except Exception as e:
        return None

def get_nearest_location(lat: float, lon: float, radius_km: int = 25): 
    """Searches for the closest monitoring station using V3 API with simplified parameters."""
    V3_LOCATIONS_URL = f"{BASE}/locations" 
    
    params = {
        "coordinates": f"{lat},{lon}",
        "radius": 25000, # 强制限制在 25km
        "limit": 5,
    }
    
    try:
        r = requests.get(V3_LOCATIONS_URL, headers=HEADERS, params=params, timeout=10)
        r.raise_for_status()
        results = r.json().get("results", [])
        
        if not results:
            return None, None
            
        for nearest_loc in results:
            # 檢查是否有 pm25 參數
            has_pm25 = any(p.get("id") == 2 or p.get("name").lower() == "pm25" for p in nearest_loc.get("parameters", []))
            
            if has_pm25:
                loc_id = int(nearest_loc["id"])
                loc_name = nearest_loc["name"]
                return loc_id, loc_name
            
        return None, None

    except Exception as e:
        return None, None
        
def get_location_latest_df(location_id: int) -> pd.DataFrame:
    """Fetches the 'latest' values for all parameters at a location (Uses V3)."""
    try:
        r = requests.get(f"{BASE}/locations/{location_id}/latest", headers=HEADERS, params={"limit": 1000}, timeout=10)
        if r.status_code == 404:
            return pd.DataFrame()
        r.raise_for_status()
        results = r.json().get("results", [])
        if not results:
            return pd.DataFrame()

        df = pd.json_normalize(results)

        df["parameter"] = df["parameter.name"].str.lower() if "parameter.name" in df.columns else df.get("parameter", df.get("name"))
        df["value"] = df["value"]

        df["ts_utc"] = pd.NaT
        # 嘗試從多個欄位中提取時間戳
        for col in ["datetime.utc", "period.datetimeTo.utc", "period.datetimeFrom.utc"]:
            if col in df.columns:
                ts = pd.to_datetime(df[col], errors="coerce", utc=True)
                df["ts_utc"] = df["ts_utc"].where(df["ts_utc"].notna(), ts)

        return df[["parameter", "value", "ts_utc"]]
    except Exception as e:
        return pd.DataFrame()

def get_parameters_latest_df(location_id: int, target_params) -> pd.DataFrame:
    """Fetches 'latest' value for specific parameters (Uses V3)."""
    rows = []
    try:
        for p in target_params:
            pid = PARAM_IDS.get(p)
            if not pid: continue
            r = requests.get(
                f"{BASE}/parameters/{pid}/latest",
                headers=HEADERS,
                params={"locationId": location_id, "limit": 50},
                timeout=10
            )
            if r.status_code == 404:
                continue
            r.raise_for_status()
            res = r.json().get("results", [])
            if not res:
                continue
            df = pd.json_normalize(res)

            df["parameter"] = p
            df["value"] = df["value"]

            df["ts_utc"] = pd.NaT
            # 嘗試從多個欄位中提取時間戳
            for col in ["datetime.utc", "period.datetimeTo.utc", "period.datetimeFrom.utc"]:
                if col in df.columns:
                    ts = pd.to_datetime(df[col], errors="coerce", utc=True)
                    df["ts_utc"] = df["ts_utc"].where(df["ts_utc"].notna(), ts)

            rows.append(df[["parameter", "value", "ts_utc"]])

    except Exception as e:
        pass

    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


# =================================================================
# Fetch Latest Weather Observation (獲取最新氣象觀測 - Open-Meteo)
# ================================================================= 

def fetch_latest_weather_observation(lat: float, lon: float) -> dict:
    """Fetches the latest (T=0) weather observation data (temperature, humidity, pressure)."""
    OM_CURRENT_URL = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "current": "temperature_2m,relative_humidity_2m,pressure_msl",
        "timezone": "UTC"
    }

    try:
        r = requests.get(OM_CURRENT_URL, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        current = data.get("current", {})
        
        if not current:
            return {}

        return {
            'datetime': pd.to_datetime(current.get('time'), utc=True),
            'temperature': current.get('temperature_2m'),
            'humidity': current.get('relative_humidity_2m'),
            'pressure': current.get('pressure_msl'),
        }
    
    except Exception as e:
        return {}


# =================================================================
# AQI Calculation and Data Wrangling (AQI 計算和數據整理)
# =================================================================

def calculate_aqi_sub_index(param: str, concentration: float) -> float:
    """Calculates the AQI sub-index (I) for a single pollutant concentration."""
    if pd.isna(concentration) or concentration < 0:
        return np.nan

    breakpoints = AQI_BREAKPOINTS.get(param)
    if not breakpoints:
        return np.nan

    for C_low, C_high, I_low, I_high in breakpoints:
        if C_low <= concentration <= C_high:
            if C_high == C_low:
                return I_high
            I = ((I_high - I_low) / (C_high - C_low)) * (concentration - C_low) + I_low
            return np.round(I)

        # 處理超出最高區間的情況
        if concentration > breakpoints[-1][1]:
            I_low, I_high = breakpoints[-1][2], breakpoints[-1][3]
            C_low, C_high = breakpoints[-1][0], breakpoints[-1][1]
            if C_high == C_low:
                return I_high
            # 使用最高區間的變化率進行外推
            I_rate = (I_high - I_low) / (C_high - C_low)
            I = I_high + I_rate * (concentration - C_high)
            return np.round(I)

    return np.nan

def calculate_aqi(row: pd.Series, params: list, is_pred=True) -> float:
    """Calculates the final AQI based on multiple pollutant concentrations (max sub-index)."""
    sub_indices = []
    for p in params:
        col_name = f'{p}_pred' if is_pred else p
        if col_name in row and pd.notna(row[col_name]):
            sub_index = calculate_aqi_sub_index(p, row[col_name])
            if pd.notna(sub_index):
                sub_indices.append(sub_index)

    if not sub_indices:
        return np.nan

    return np.max(sub_indices)

def fetch_latest_observation_data(location_id: int, target_params: list) -> pd.DataFrame:
    """
    Fetches the latest observation data from OpenAQ, prioritizing the freshest reading for each parameter.
    """
    
    df_loc_latest = get_location_latest_df(location_id)
    df_param_latest = get_parameters_latest_df(location_id, target_params)
    
    frames = [df for df in [df_loc_latest, df_param_latest] if not df.empty]
    if not frames:
        print("🚨 [Fetch] No pollutant data fetched from OpenAQ.")
        return pd.DataFrame()

    df_all = pd.concat(frames, ignore_index=True)
    df_all["parameter"] = df_all["parameter"].str.lower()
    df_all = df_all[df_all["parameter"].isin(target_params)]
    
    # 核心修正：排序並選擇每個參數的絕對最新讀數
    df_all["ts_utc"] = pd.to_datetime(df_all["ts_utc"], errors="coerce", utc=True)
    df_all = df_all.dropna(subset=['ts_utc'])
    
    df_all = df_all.sort_values(["parameter", "ts_utc"], ascending=[True, False])
    df_all = df_all.drop_duplicates(subset=["parameter"], keep="first")
    
    # 確保數據足夠新鮮 (只使用 3 小時內的數據)
    # 修正了錯誤：現在使用 dt.now() (類別方法)
    three_hours_ago = dt.now(timezone.utc) - timedelta(hours=3)
    df_all = df_all[df_all["ts_utc"] > three_hours_ago].copy()

    if df_all.empty:
        print("🚨 [Fetch] No valid and recent observations found within the last 3 hours.")
        return pd.DataFrame()
        
    latest_valid_ts = df_all["ts_utc"].max()
    df_all = df_all.drop(columns=["ts_local"] if "ts_local" in df_all.columns else [])
    
    # 轉換為模型輸入格式 (單行寬表)
    observation = df_all.pivot_table(
        index='parameter', values='value', aggfunc='first'
    ).T.reset_index(drop=True)
    
    # 設置統一的時間戳
    observation.insert(0, 'datetime', latest_valid_ts)
    
    # 計算 AQI 和最終時區處理
    if not observation.empty:
        observation['aqi'] = observation.apply(
            lambda row: calculate_aqi(row, target_params, is_pred=False), axis=1
        )
        # 確保 'datetime' 總是 UTC-aware
        if observation['datetime'].dt.tz is None:
             observation['datetime'] = observation['datetime'].dt.tz_localize('UTC')
        else:
             observation['datetime'] = observation['datetime'].dt.tz_convert('UTC')

    return observation


# =================================================================
# Prediction Function (預測函數 - 使用歷史平均趨勢替換隨機漫步)
# =================================================================
def predict_future_multi(models, last_data, feature_cols, pollutant_params, hours=24):
    """Predicts multiple target pollutants for N future hours (recursive prediction)."""
    predictions = []

    # 確保數據是 tz-aware (UTC)
    last_data['datetime'] = pd.to_datetime(last_data['datetime'])
    if last_data['datetime'].dt.tz is None:
        last_data['datetime'] = last_data['datetime'].dt.tz_localize('UTC')
    else:
        last_data['datetime'] = last_data['datetime'].dt.tz_convert('UTC')
        
    last_datetime_aware = last_data['datetime'].iloc[0]
    start_hour = last_datetime_aware.hour 
    
    current_data_dict = {col: last_data.get(col, np.nan).iloc[0] 
                              if col in last_data.columns and not last_data[col].empty 
                              else np.nan 
                              for col in feature_cols} 

    weather_feature_names_base = ['temperature', 'humidity', 'pressure']
    weather_feature_names = [col for col in weather_feature_names_base if col in feature_cols]
    has_weather = bool(weather_feature_names)
    
    # 擷取 T=0 的氣象觀測值作為基準
    start_weather_obs = {}
    for w_col in weather_feature_names:
        start_weather_obs[w_col] = current_data_dict.get(w_col) or np.nan

    for h in range(hours):
        future_time = last_datetime_aware + timedelta(hours=h + 1)
        pred_features = current_data_dict.copy()

        # 1. Update time-based features (更新時間特徵)
        pred_features['hour'] = future_time.hour
        pred_features['day_of_week'] = future_time.dayofweek
        pred_features['month'] = future_time.month
        pred_features['day_of_year'] = future_time.timetuple().tm_yday 
        pred_features['is_weekend'] = int(future_time.dayofweek in [5, 6])
        # 循環特徵
        pred_features['hour_sin'] = np.sin(2 * np.pi * future_time.hour / 24)
        pred_features['hour_cos'] = np.cos(2 * np.pi * future_time.hour / 24)
        pred_features['day_sin'] = np.sin(2 * np.pi * pred_features['day_of_year'] / 365)
        pred_features['day_cos'] = np.cos(2 * np.pi * pred_features['day_of_year'] / 365)

        # 2. Simulate future weather changes (模擬未來氣象變化 - 使用歷史平均趨勢)
        if has_weather:
            future_hour = future_time.hour
            
            # 趨勢計算：振幅已調高，模擬較大的日夜起伏
            temp_swing_factor = np.cos(2 * np.pi * (future_hour - 14) / 24)
            humid_swing_factor = np.cos(2 * np.pi * (future_hour - 6) / 24)
            # 使用固定種子確保每次運行結果一致，但隨時間/日期的不同而變化
            np.random.seed(future_time.hour + future_time.day + 42) 
            
            for w_col in weather_feature_names:
                start_value = start_weather_obs.get(w_col)
                
                if pd.isna(start_value):
                    new_weather_value = np.nan
                elif w_col == 'temperature':
                    start_factor = np.cos(2 * np.pi * (start_hour - 14) / 24)
                    temp_change = 8 * (temp_swing_factor - start_factor) # 振幅 8 度
                    new_weather_value = start_value + temp_change
                    
                elif w_col == 'humidity':
                    start_factor = np.cos(2 * np.pi * (start_hour - 6) / 24)
                    humid_change = 15 * (humid_swing_factor - start_factor) # 振幅 15%
                    new_weather_value = start_value + humid_change
                    
                elif w_col == 'pressure':
                    # 壓力變化較小，使用輕微的隨機漫步
                    new_weather_value = start_value + np.random.normal(0, 0.2)
                
                pred_features[w_col] = new_weather_value
                # 更新 current_data_dict 以在後續迭代中使用模擬天氣值
                current_data_dict[w_col] = new_weather_value


        current_prediction_row = {'datetime': future_time}
        new_pollutant_values = {}

        # 3. Predict all pollutants (預測所有污染物)
        for param in pollutant_params:
            model = models[param]
            # 準備模型的輸入特徵
            pred_input_list = [pred_features.get(col) for col in feature_cols]
            pred_input = np.array(pred_input_list, dtype=np.float64).reshape(1, -1)
            
            pred = model.predict(pred_input)[0]
            pred = max(0, pred) # 確保預測值不為負數

            current_prediction_row[f'{param}_pred'] = pred
            new_pollutant_values[param] = pred

        # 4. Calculate predicted AQI (計算預測 AQI)
        predicted_aqi = calculate_aqi(pd.Series(current_prediction_row), pollutant_params, is_pred=True)
        current_prediction_row['aqi_pred'] = predicted_aqi
        new_pollutant_values['aqi'] = predicted_aqi

        predictions.append(current_prediction_row)

        # 5. Update lag features for the next hour's prediction (更新滯後特徵 - 遞歸預測的關鍵)
        for param in pollutant_params + ['aqi']:
            for i in range(len(LAG_HOURS) - 1, 0, -1):
                lag_current = LAG_HOURS[i]
                lag_prev = LAG_HOURS[i-1]
                lag_current_col = f'{param}_lag_{lag_current}h'
                lag_prev_col = f'{param}_lag_{lag_prev}h'

                if lag_current_col in current_data_dict and lag_prev_col in current_data_dict:
                    # 將較小的滯後值推到較大的滯後位置 (例如 12h 變成 24h)
                    current_data_dict[lag_current_col] = current_data_dict[lag_prev_col]

            # 將新的預測值設置為 1 小時滯後特徵 (param_lag_1h)
            if f'{param}_lag_{1}h' in current_data_dict and param in new_pollutant_values:
                current_data_dict[f'{param}_lag_{1}h'] = new_pollutant_values[param]

    return pd.DataFrame(predictions)


# =================================================================
# Model Loading Logic (模型載入邏輯)
# =================================================================

def load_models_and_metadata():
    global TRAINED_MODELS, LAST_OBSERVATION, FEATURE_COLUMNS, POLLUTANT_PARAMS

    if not os.path.exists(MODELS_DIR) or not os.path.exists(META_PATH):
        print("🚨 [Load] 模型中繼資料檔案或目錄未找到。無法載入模型。")
        return

    try:
        with open(META_PATH, 'r', encoding='utf-8') as f:
            metadata = json.load(f)

        POLLUTANT_PARAMS = metadata.get('pollutant_params', [])
        FEATURE_COLUMNS = metadata.get('feature_columns', [])
        
        if 'last_observation_json' in metadata:
            LAST_OBSERVATION = pd.read_json(metadata['last_observation_json'], orient='records')

        TRAINED_MODELS = {}
        params_to_remove = []
        for param in POLLUTANT_PARAMS:
            model_path = os.path.join(MODELS_DIR, f'{param}_model.json')
            if os.path.exists(model_path):
                model = xgb.XGBRegressor()
                model.load_model(model_path)
                TRAINED_MODELS[param] = model
            else:
                # 如果模型檔案缺失，則從目標清單中移除
                params_to_remove.append(param)
        
        for param in params_to_remove:
             POLLUTANT_PARAMS.remove(param)

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
# Flask Application Setup and Initialization (Flask 應用程式設置與初始化)
# =================================================================

def initialize_location():
    """Finds the nearest location and updates the global variables."""
    global current_location_id, current_location_name, DEFAULT_LOCATION_ID, DEFAULT_LOCATION_NAME
    
    print(f"🗺️ [Init] 正在搜尋最近的空氣品質監測站 ({TARGET_LAT}, {TARGET_LON})...")
    loc_id, loc_name = get_nearest_location(TARGET_LAT, TARGET_LON)
    
    if loc_id is not None:
        current_location_id = loc_id
        current_location_name = loc_name
        print(f"✅ [Init] 找到監測站: ID={current_location_id}, Name='{current_location_name}'")
    else:
        current_location_id = DEFAULT_LOCATION_ID
        current_location_name = DEFAULT_LOCATION_NAME
        print(f"⚠️ [Init] 未找到帶有 PM2.5 數據的監測站，使用預設地點: ID={current_location_id}, Name='{current_location_name}'")

# Dynamically find the nearest location before app instantiation
initialize_location()


app = Flask(__name__)

# Load models when the application starts
with app.app_context():
    load_models_and_metadata() 

@app.route('/')
def index():
    global CURRENT_OBSERVATION_AQI, CURRENT_OBSERVATION_TIME, current_location_id, current_location_name
    station_name = current_location_name
    
    # 1. Attempt to fetch the latest observation data in real-time
    current_observation_raw = fetch_latest_observation_data(current_location_id, POLLUTANT_TARGETS)

    # NEW STEP: Fetch latest weather observation (T=0)
    latest_weather_obs = fetch_latest_weather_observation(TARGET_LAT, TARGET_LON)

    # 提取最新的觀測 AQI (用於回退)
    if not current_observation_raw.empty and 'aqi' in current_observation_raw.columns:
        obs_aqi_val = current_observation_raw['aqi'].iloc[0]
        obs_time_val = current_observation_raw['datetime'].iloc[0]
        
        CURRENT_OBSERVATION_AQI = int(obs_aqi_val) if pd.notna(obs_aqi_val) else "N/A"
        
        if pd.notna(obs_time_val):
            # 確保觀測時間是時區感知的 (UTC)，然後轉換到本地時區顯示
            if obs_time_val.tz is None:
                 obs_time_val = obs_time_val.tz_localize('UTC')
            
            CURRENT_OBSERVATION_TIME = obs_time_val.tz_convert(LOCAL_TZ).strftime('%Y-%m-%d %H:%M')
        else:
             CURRENT_OBSERVATION_TIME = "N/A"
    
    
    # 2. Prepare data for prediction (準備用於預測的數據)
    observation_for_prediction = None
    is_valid_for_prediction = False

    if not current_observation_raw.empty and LAST_OBSERVATION is not None and not LAST_OBSERVATION.empty:
        # 從上次儲存的數據中複製滯後和滾動特徵
        observation_for_prediction = LAST_OBSERVATION.iloc[:1].copy() 
        latest_row = current_observation_raw.iloc[0]
        
        # 提取最新時間戳
        dt_val = latest_row['datetime']
        if pd.to_datetime(dt_val).tz is not None:
             # 確保傳遞給預測函數的時間戳是時區感知的
             dt_val = pd.to_datetime(dt_val).tz_convert('UTC') 
             
        observation_for_prediction['datetime'] = dt_val
        
        # 用最新的污染物觀測值更新 T=0 數據
        for col in latest_row.index:
            if col in observation_for_prediction.columns and not any(s in col for s in ['lag_', 'rolling_']):
                 if col in POLLUTANT_TARGETS or col == 'aqi':
                      observation_for_prediction[col] = latest_row[col]

        # 用最新的氣象觀測值更新 T=0 數據
        for w_col, w_val in latest_weather_obs.items():
             if w_col in ['temperature', 'humidity', 'pressure'] and w_col in observation_for_prediction.columns:
                  observation_for_prediction[w_col] = w_val
        
        # 檢查是否所有必要的特徵都存在
        if all(col in observation_for_prediction.columns for col in FEATURE_COLUMNS):
             is_valid_for_prediction = True
        
    # =================================================================
    # T=0 數據診斷輸出 (請檢查您的終端機/Console)
    # =================================================================
    if observation_for_prediction is not None and 'aqi' in observation_for_prediction.columns:
        print("\n=============================================")
        print("--- DIAGNOSTIC: T=0 PREDICTION START DATA ---")
        
        start_time_utc = observation_for_prediction['datetime'].iloc[0]
        if pd.to_datetime(start_time_utc).tz is None:
             start_time_utc = pd.to_datetime(start_time_utc).tz_localize('UTC')
        
        print(f"START TIME (UTC): {start_time_utc.strftime('%Y-%m-%d %H:%M:%S')}")
        
        for p in POLLUTANT_TARGETS:
            if p in observation_for_prediction.columns:
                val = observation_for_prediction[p].iloc[0]
                print(f"  > Latest {p} value used: {val}")

        for w in ['temperature', 'humidity', 'pressure']:
             if w in observation_for_prediction.columns:
                 val = observation_for_prediction[w].iloc[0]
                 print(f"  > Latest {w} value used: {val}")
        
        calculated_aqi = observation_for_prediction['aqi'].iloc[0]
        print(f"  > Calculated T=0 AQI: {calculated_aqi}")
        print("=============================================\n")


    # 3. Perform prediction or fallback (執行預測或回退)
    max_aqi = CURRENT_OBSERVATION_AQI
    aqi_predictions = []
    is_fallback_mode = True

    if TRAINED_MODELS and POLLUTANT_PARAMS and is_valid_for_prediction and observation_for_prediction is not None:
        try:
            future_predictions = predict_future_multi(
                TRAINED_MODELS,
                observation_for_prediction,
                FEATURE_COLUMNS,
                POLLUTANT_PARAMS,
                hours=HOURS_TO_PREDICT
            )

            # 將 UTC 時間轉換為本地時區
            future_predictions['datetime_local'] = future_predictions['datetime'].dt.tz_convert(LOCAL_TZ)
            
            predictions_df = future_predictions[['datetime_local', 'aqi_pred']].copy()
            max_aqi_val = predictions_df['aqi_pred'].max()
            # 找到預測期間的最高 AQI
            max_aqi = int(max_aqi_val) if pd.notna(max_aqi_val) else CURRENT_OBSERVATION_AQI
            
            predictions_df['aqi_pred'] = predictions_df['aqi_pred'].replace(np.nan, "N/A")
            predictions_df['aqi'] = predictions_df['aqi_pred'].apply(
                lambda x: int(x) if x != "N/A" else "N/A"
            ).astype(object)

            # 準備用於網頁模板的預測列表
            aqi_predictions = [
                {
                    'time': item['datetime_local'].strftime('%Y-%m-%d %H:%M'), 
                    'aqi': item['aqi']
                }
                for item in predictions_df.to_dict(orient='records')
            ]
            
            if aqi_predictions:
                 is_fallback_mode = False
            else:
                 max_aqi = CURRENT_OBSERVATION_AQI
                 is_fallback_mode = True


        except Exception as e:
            print(f"❌ [Predict] 預測失敗: {e}")
            max_aqi = CURRENT_OBSERVATION_AQI
            aqi_predictions = []
            is_fallback_mode = True
            
    if is_fallback_mode:
             # 如果預測失敗，則使用最新的觀測值作為回退
             max_aqi = CURRENT_OBSERVATION_AQI
             
             if max_aqi != "N/A":
                 aqi_predictions = [{
                    'time': CURRENT_OBSERVATION_TIME,
                    'aqi': max_aqi,
                    'is_obs': True # 標記為觀測數據
                 }]

    # 4. Render template (渲染模板)
    return render_template('index.html', 
                            max_aqi=max_aqi, 
                            aqi_predictions=aqi_predictions, 
                            city_name=current_location_name,
                            current_obs_time=CURRENT_OBSERVATION_TIME,
                            is_fallback=is_fallback_mode)

if __name__ == '__main__':
    # 注意：在生產環境中，請移除 debug=True
    app.run(debug=True)
