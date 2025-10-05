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
DEFAULT_LOCATION_NAME = "Kaohsiung-Qianjin" # Default Location Name

TARGET_PARAMS = ["co", "no2", "o3", "pm10", "pm25", "so2"]
PARAM_IDS = {"co": 8, "no2": 7, "o3": 10, "pm10": 1, "pm25": 2, "so2": 9}

TOL_MINUTES_PRIMARY = 120
TOL_MINUTES_FALLBACK = 180

# =================================================================
# Global Variables (Mutable)
# =================================================================
TRAINED_MODELS = {} 
LAST_OBSERVATION = None 
FEATURE_COLUMNS = []
POLLUTANT_PARAMS = [] 
HOURS_TO_PREDICT = 24

# Store the latest observation data (for fallback)
CURRENT_OBSERVATION_AQI = "N/A"
CURRENT_OBSERVATION_TIME = "N/A"

# Dynamic Location Variables (Will be updated on startup)
current_location_id = DEFAULT_LOCATION_ID
current_location_name = DEFAULT_LOCATION_NAME

# =================================================================
# Constants
# =================================================================
LOCAL_TZ = "Asia/Taipei"
LAG_HOURS = [1, 2, 3, 6, 12, 24]
ROLLING_WINDOWS = [6, 12, 24]
POLLUTANT_TARGETS = ["pm25", "pm10", "o3", "no2", "so2", "co"] 

AQI_BREAKPOINTS = {
    "pm25": [(0.0, 12.0, 0, 50), (12.1, 35.4, 51, 100), (35.5, 55.4, 101, 150), (55.5, 150.4, 151, 200)],
    "pm10": [(0, 54, 0, 50), (55, 154, 51, 100), (155, 254, 101, 150), (255, 354, 151, 200)],
    "o3": [(0, 54, 0, 50), (55, 70, 51, 100), (71, 85, 101, 150), (86, 105, 151, 200)],
    "co": [(0.0, 4.4, 0, 50), (4.5, 9.4, 51, 100), (9.5, 12.4, 101, 150), (12.5, 15.4, 151, 200)],
    "no2": [(0, 100, 0, 50), (101, 360, 51, 100), (361, 649, 101, 150), (650, 1249, 151, 200)],
    "so2": [(0, 35, 0, 50), (36, 75, 51, 100), (76, 185, 101, 150), (186, 304, 151, 200)],
}


# =================================================================
# OpenAQ Data Fetching Functions
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

# =================================================================
# V3 API 穩健定位函式 (修正 422 錯誤)
# =================================================================
def get_nearest_location(lat: float, lon: float, radius_km: int = 25): 
    """
    Searches for the closest monitoring station using V3 API with simplified parameters.
    Now returns both ID, name, and coordinates.
    """
    V3_LOCATIONS_URL = f"{BASE}/locations" 
    params = {
        "coordinates": f"{lat},{lon}",
        "radius": 20000,
        "limit": 5,
    }
    try:
        r = requests.get(V3_LOCATIONS_URL, headers=HEADERS, params=params, timeout=10)
        r.raise_for_status()
        results = r.json().get("results", [])

        if not results:
            print("🚨 [Nearest] No stations found within 25km.")
            return None, None, None, None

        # 直接使用第一個（最近）站，無論有沒有 PM2.5
        nearest = results[0]
        loc_id = int(nearest["id"])
        loc_name = nearest["name"]
        coords = nearest.get("coordinates", {})
        lat_found = coords.get("latitude", "N/A")
        lon_found = coords.get("longitude", "N/A")

        print(f"✅ [Nearest] Found station: {loc_name} (ID: {loc_id})")
        print(f"📍 Coordinates: latitude={lat_found}, longitude={lon_found}")

        return loc_id, loc_name, lat_found, lon_found

    except Exception as e:
        print(f"❌ [Nearest] Failed to find station: {e}")
        return None, None, None, None

# -----------------------------------------------------------------
# Core Data Fetching Logic (All use V3 BASE)
# -----------------------------------------------------------------

def get_location_latest_df(location_id: int) -> pd.DataFrame:
    """Fetches the 'latest' values for all parameters at a location (Uses V3)."""
    try:
        r = requests.get(f"{BASE}/locations/{location_id}/latest", headers=HEADERS, params={"limit": 1000}, timeout=10)
        if r.status_code == 404:
            return pd.DataFrame()
        r.raise_for_status()
        results = r.json().get("results", [])
        print("\n🌍 [DEBUG] Raw stations returned by OpenAQ:")
        print(json.dumps(results, indent=2, ensure_ascii=False))

        if not results:
            return pd.DataFrame()

        df = pd.json_normalize(results)

        # Standardize column names
        df["parameter"] = df["parameter.name"].str.lower() if "parameter.name" in df.columns else df.get("parameter", df.get("name"))
        df["units"] = df["parameter.units"] if "parameter.units" in df.columns else df.get("units")
        df["value"] = df["value"]

        # Find the best UTC timestamp
        df["ts_utc"] = pd.NaT
        for col in ["datetime.utc", "period.datetimeTo.utc", "period.datetimeFrom.utc"]:
            if col in df.columns:
                ts = pd.to_datetime(df[col], errors="coerce", utc=True)
                df["ts_utc"] = df["ts_utc"].where(df["ts_utc"].notna(), ts)

        # Find local timestamp
        local_col = None
        for c in ["datetime.local", "period.datetimeTo.local", "period.datetimeFrom.local"]:
            if c in df.columns:
                local_col = c
                break
        df["ts_local"] = df[local_col] if local_col in df.columns else None

        return df[["parameter", "value", "units", "ts_utc", "ts_local"]]
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
            df["units"] = df["parameter.units"] if "parameter.units" in df.columns else df.get("units")
            df["value"] = df["value"]

            df["ts_utc"] = pd.NaT
            for col in ["datetime.utc", "period.datetimeTo.utc", "period.datetimeFrom.utc"]:
                if col in df.columns:
                    ts = pd.to_datetime(df[col], errors="coerce", utc=True)
                    df["ts_utc"] = df["ts_utc"].where(df["ts_utc"].notna(), ts)

            local_col = None
            for c in ["datetime.local", "period.datetimeTo.local", "period.datetimeFrom.local"]:
                if c in df.columns:
                    local_col = c
                    break
            df["ts_local"] = df[local_col] if local_col in df.columns else None

            rows.append(df[["parameter", "value", "units", "ts_utc", "ts_local"]])

    except Exception as e:
        pass

    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


# =================================================================
# Open-Meteo Weather Fetching (新增)
# =================================================================
# 設置快取和重試
cache_session = requests_cache.CachedSession('.cache', expire_after = 3600)
retry_session = openmeteo_requests.create_retry_session(session=cache_session)
openmeteo_client = openmeteo_requests.Client(session=retry_session)

def get_weather_forecast(lat: float, lon: float) -> pd.DataFrame:
    """
    Fetches 24-hour weather forecast for the given coordinates from Open-Meteo.
    Returns a DataFrame with 'datetime', 'temperature', 'humidity', 'pressure'.
    """
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ["temperature_2m", "relative_humidity_2m", "surface_pressure"],
        "timezone": "UTC",
        "forecast_days": 2, # 獲取足夠多的數據來覆蓋接下來 24 小時
    }
    
    try:
        responses = openmeteo_client.weather_api(url, params=params)
        
        if not responses or not responses[0].IsInitialized():
             print("❌ [Weather] Open-Meteo did not return initialized data.")
             return pd.DataFrame()
             
        response = responses[0]
        hourly = response.Hourly()
        
        # 轉換為 DataFrame
        hourly_data = {
            "datetime": pd.to_datetime([hourly.Time(i) for i in range(len(hourly.Time()))], unit="s", utc=True),
            "temperature": hourly.Variables(0).ValuesAsNumpy(),
            "humidity": hourly.Variables(1).ValuesAsNumpy(), # relative_humidity_2m
            "pressure": hourly.Variables(2).ValuesAsNumpy(), # surface_pressure
        }
        
        df = pd.DataFrame(hourly_data)
        
        # 確保列名與模型特徵匹配
        df = df.rename(columns={
            "temperature": "temperature",
            "humidity": "humidity", 
            "pressure": "pressure",
        })
        
        # 截取從下一個小時開始的 24 小時預報
        now_utc = pd.Timestamp.now(tz='UTC').floor('H')
        start_time = now_utc + timedelta(hours=1)
        
        df = df[df['datetime'] >= start_time].head(HOURS_TO_PREDICT).copy()
        
        print(f"✅ [Weather] Fetched {len(df)} hours of weather forecast.")
        
        return df
        
    except Exception as e:
        print(f"❌ [Weather] Failed to fetch weather forecast: {e}")
        return pd.DataFrame()


# =================================================================
# Helper Functions: AQI Calculation and Data Wrangling
# =================================================================

def pick_batch_near(df: pd.DataFrame, t_ref: pd.Timestamp, tol_minutes: int) -> pd.DataFrame:
    """Selects the batch of data closest to t_ref and within tol_minutes."""
    if df.empty or pd.isna(t_ref):
        return pd.DataFrame()

    df = df.copy()

    def _scalarize(v):
        if isinstance(v, (list, tuple, np.ndarray)):
            return v[0] if len(v) else None
        return v

    df["ts_utc"] = df["ts_utc"].map(_scalarize)
    df["ts_utc"] = pd.to_datetime(df["ts_utc"], errors="coerce", utc=True)

    df["dt_diff"] = (df["ts_utc"] - t_ref).abs()

    tol = pd.Timedelta(minutes=tol_minutes)
    df = df[df["dt_diff"] <= tol].copy()
    if df.empty:
        return df

    df = df.sort_values(["parameter", "dt_diff", "ts_utc"], ascending=[True, True, False])
    df = df.drop_duplicates(subset=["parameter"], keep="first")
    return df[["parameter", "value", "units", "ts_utc", "ts_local"]]


def fetch_latest_observation_data(location_id: int, target_params: list) -> pd.DataFrame:
    """
    Fetches the latest observation data from OpenAQ and converts it to a single-row wide format.
    Includes final timezone logic to ensure 'datetime' is consistently UTC-aware.
    """
    meta = get_location_meta(location_id)
    if not meta or pd.isna(meta["last_utc"]):
        return pd.DataFrame()

    df_loc_latest = get_location_latest_df(location_id)
    if df_loc_latest.empty:
        return pd.DataFrame()

    t_star_latest = df_loc_latest["ts_utc"].max()
    t_star_loc = meta["last_utc"]
    t_star = t_star_latest if pd.notna(t_star_latest) else t_star_loc

    if pd.isna(t_star):
        return pd.DataFrame()
    
    # 1. Try primary source / strict tolerance
    df_at_batch = pick_batch_near(df_loc_latest, t_star, TOL_MINUTES_PRIMARY)
    if df_at_batch.empty:
        # 2. Try primary source / fallback tolerance
        df_at_batch = pick_batch_near(df_loc_latest, t_star, TOL_MINUTES_FALLBACK)

    have = set(df_at_batch["parameter"].str.lower().tolist()) if not df_at_batch.empty else set()

    # 3. Try to fetch missing parameters using dedicated parameter endpoint
    missing = [p for p in target_params if p not in have]
    df_param_batch = pd.DataFrame()
    if missing:
        df_param_latest = get_parameters_latest_df(location_id, missing)
        df_param_batch = pick_batch_near(df_param_latest, t_star, TOL_MINUTES_PRIMARY)
        if df_param_batch.empty:
            df_param_batch = pick_batch_near(df_param_latest, t_star, TOL_MINUTES_FALLBACK)

    frames = [df for df in [df_at_batch, df_param_batch] if not df.empty]
    if not frames:
        return pd.DataFrame()

    df_all = pd.concat(frames, ignore_index=True)
    df_all["parameter"] = df_all["parameter"].str.lower()
    df_all = df_all[df_all["parameter"].isin(target_params)]

    # Final selection (ensure only one value per parameter)
    df_all["dt_diff"] = (df_all["ts_utc"] - t_star).abs()
    df_all = df_all.sort_values(["parameter", "dt_diff", "ts_utc"], ascending=[True, True, False])
    df_all = df_all.drop_duplicates(subset=["parameter"], keep="first")
    df_all = df_all.drop(columns=["dt_diff", "units", "ts_local"])

    # 4. Convert to model input format (single-row wide table)
    observation = df_all.pivot_table(
        index='ts_utc', columns='parameter', values='value', aggfunc='first'
    ).reset_index()
    observation = observation.rename(columns={'ts_utc': 'datetime'})
    
    # Calculate AQI
    if not observation.empty:
        observation['aqi'] = observation.apply(
            lambda row: calculate_aqi(row, target_params, is_pred=False), axis=1
        )
        
    # 核心修正：確保 'datetime' 總是 UTC-aware
    if not observation.empty:
        observation['datetime'] = pd.to_datetime(observation['datetime'])
        if observation['datetime'].dt.tz is None:
             # 如果沒有時區，本地化為 UTC
             observation['datetime'] = observation['datetime'].dt.tz_localize('UTC')
        else:
             # 如果已經有時區，轉換到 UTC (確保一致性)
             observation['datetime'] = observation['datetime'].dt.tz_convert('UTC')

    return observation


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

        # Handle concentrations above the highest defined range (simple linear extrapolation)
        if concentration > breakpoints[-1][1]:
            I_low, I_high = breakpoints[-1][2], breakpoints[-1][3]
            C_low, C_high = breakpoints[-1][0], breakpoints[-1][1]
            if C_high == C_low:
                return I_high
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


# =================================================================
# Prediction Function (使用 Open-Meteo 數據取代模擬)
# =================================================================
def predict_future_multi(models, last_data, feature_cols, pollutant_params, hours=24, weather_df=None):
    """
    Predicts multiple target pollutants for N future hours (recursive prediction) 
    and calculates AQI using real weather forecast data.
    """
    predictions = []

    # pandas 印出設定
    pd.set_option('display.max_columns', 10)
    pd.set_option('display.width', 140)

    # 確保 datetime 是 tz-aware (UTC)
    last_data['datetime'] = pd.to_datetime(last_data['datetime'])
    if last_data['datetime'].dt.tz is None:
        last_data['datetime'] = last_data['datetime'].dt.tz_localize('UTC')
    else:
        last_data['datetime'] = last_data['datetime'].dt.tz_convert('UTC')
        
    last_datetime_aware = last_data['datetime'].iloc[0]
    
    # 初始化特徵字典
    current_data_dict = {col: last_data.get(col, np.nan).iloc[0] 
                             if col in last_data.columns and not last_data[col].empty 
                             else np.nan 
                             for col in feature_cols} 

    weather_feature_names_base = ['temperature', 'humidity', 'pressure']
    weather_feature_names = [col for col in weather_feature_names_base if col in feature_cols]
    has_weather = bool(weather_feature_names)

    # 預處理天氣預報：設置 'datetime' 為索引並轉為字典
    weather_dict = {}
    if weather_df is not None and not weather_df.empty:
        # 確保天氣預報的 datetime 也是 UTC-aware
        weather_df['datetime'] = pd.to_datetime(weather_df['datetime']).dt.tz_convert('UTC')
        weather_df = weather_df.set_index('datetime')
        weather_dict = weather_df.to_dict(orient='index')
        print(f"✅ [Weather] Weather data loaded for {len(weather_dict)} hours.")


    total_predictions = 0

    try:
        for h in range(hours):
            future_time = last_datetime_aware + timedelta(hours=h + 1)
            pred_features = current_data_dict.copy()

            # 更新時間特徵
            pred_features['hour'] = future_time.hour
            pred_features['day_of_week'] = future_time.dayofweek
            pred_features['month'] = future_time.month
            pred_features['day_of_year'] = future_time.timetuple().tm_yday 
            pred_features['is_weekend'] = int(future_time.dayofweek in [5, 6])
            pred_features['hour_sin'] = np.sin(2 * np.pi * future_time.hour / 24)
            pred_features['hour_cos'] = np.cos(2 * np.pi * future_time.hour / 24)
            pred_features['day_sin'] = np.sin(2 * np.pi * pred_features['day_of_year'] / 365)
            pred_features['day_cos'] = np.cos(2 * np.pi * pred_features['day_of_year'] / 365)

            # ⭐️ 核心變動：使用 Open-Meteo 預報數據
            if has_weather:
                weather_key = future_time.replace(minute=0, second=0, microsecond=0) # 確保時間匹配整點
                
                if weather_key in weather_dict:
                    forecast = weather_dict[weather_key]
                    for w_col in weather_feature_names:
                        if w_col in forecast:
                            pred_features[w_col] = forecast[w_col]
                            # 為了下一輪預測的滯後特徵/最後已知值，更新 current_data_dict
                            current_data_dict[w_col] = forecast[w_col] 
                else:
                    print(f"⚠️ [Weather] Forecast missing for {future_time}. Using last known value.")
                    for w_col in weather_feature_names:
                         # 使用 current_data_dict 中最新的天氣值作為預測，以避免 NaN
                        pred_features[w_col] = current_data_dict.get(w_col, np.nan) 

            # -----------------------------------------------
            # 移除 np.random.seed() 和隨機模擬邏輯
            # -----------------------------------------------

            current_prediction_row = {'datetime': future_time}
            new_pollutant_values = {}

            # 預測每個污染物
            for param in pollutant_params:
                if param not in models:
                    print(f"⚠️ 模型 {param} 不存在，跳過。")
                    continue

                model = models[param]
                pred_input_list = [pred_features.get(col) for col in feature_cols]
                # 確保特徵數量一致
                if len(pred_input_list) != len(feature_cols):
                    print(f"❌ [Predict] 特徵數量不匹配，跳過 {param} 預測。")
                    continue

                pred_input = np.array(pred_input_list, dtype=np.float64).reshape(1, -1)

                # 印出資料內容（前 10 欄）
                print(f"\n📦 [Model Input for {param.upper()} — Hour +{h+1}] (feature count = {len(feature_cols)})")
                print(pd.DataFrame(pred_input, columns=feature_cols).iloc[:, :10])

                pred = model.predict(pred_input)[0]
                pred = max(0, pred)

                current_prediction_row[f'{param}_pred'] = pred
                new_pollutant_values[param] = pred
                total_predictions += 1

            # 計算 AQI
            predicted_aqi = calculate_aqi(pd.Series(current_prediction_row), pollutant_params, is_pred=True)
            current_prediction_row['aqi_pred'] = predicted_aqi
            new_pollutant_values['aqi'] = predicted_aqi
            predictions.append(current_prediction_row)

            # 更新滯後特徵
            for param in pollutant_params + ['aqi']:
                for i in range(len(LAG_HOURS) - 1, 0, -1):
                    lag_current = LAG_HOURS[i]
                    lag_prev = LAG_HOURS[i-1]
                    lag_current_col = f'{param}_lag_{lag_current}h'
                    lag_prev_col = f'{param}_lag_{lag_prev}h'

                    if lag_current_col in current_data_dict and lag_prev_col in current_data_dict:
                        current_data_dict[lag_current_col] = current_data_dict[lag_prev_col]

                if f'{param}_lag_1h' in current_data_dict and param in new_pollutant_values:
                    current_data_dict[f'{param}_lag_1h'] = new_pollutant_values[param]

        # 總結印出結果
        print(f"\n✅ [Summary] 模型共收到 {total_predictions} 筆輸入資料，"
              f"每筆包含 {len(feature_cols)} 個特徵。"
              f"→ 總特徵傳遞量 = {total_predictions * len(feature_cols):,} 數值")

    except Exception as e:
        print(f"❌ [Predict] 發生錯誤：{e}")

    return pd.DataFrame(predictions)



# =================================================================
# Model Loading Logic
# =================================================================

def load_models_and_metadata():
    global TRAINED_MODELS, LAST_OBSERVATION, FEATURE_COLUMNS, POLLUTANT_PARAMS

    if not os.path.exists(MODELS_DIR) or not os.path.exists(META_PATH):
        print("🚨 [Load] Model metadata file or directory not found. Cannot load models.")
        return

    try:
        with open(META_PATH, 'r', encoding='utf-8') as f:
            metadata = json.load(f)

        POLLUTANT_PARAMS = metadata.get('pollutant_params', [])
        FEATURE_COLUMNS = metadata.get('feature_columns', [])
        
        if 'last_observation_json' in metadata:
            # We rely on this to provide the initial lagged features
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
                print(f"❌ [Load] Model file for {param} not found: {model_path}")
                params_to_remove.append(param)
        
        for param in params_to_remove:
             POLLUTANT_PARAMS.remove(param)

        if TRAINED_MODELS:
            print(f"✅ [Load] Successfully loaded {len(TRAINED_MODELS)} models.")
        else:
            print("🚨 [Load] No models were loaded.")


    except Exception as e:
        print(f"❌ [Load] Model loading failed: {e}") 
        TRAINED_MODELS = {} 
        LAST_OBSERVATION = None
        FEATURE_COLUMNS = []
        POLLUTANT_PARAMS = []

# =================================================================
# Flask Application Setup and Initialization
# =================================================================

app = Flask(__name__)

# Load models when the application starts
with app.app_context():
    load_models_and_metadata() 


@app.route('/')
def index():
    global CURRENT_OBSERVATION_AQI, CURRENT_OBSERVATION_TIME
    global current_location_id, current_location_name
    global TARGET_LAT, TARGET_LON
    station_lat, station_lon = TARGET_LAT, TARGET_LON # 預設使用TARGET，如果找到測站則更新

    # ========== 1️⃣ 從網址參數抓座標 ==========
    lat_param = request.args.get('lat', type=float)
    lon_param = request.args.get('lon', type=float)

    if lat_param is not None and lon_param is not None:
        TARGET_LAT, TARGET_LON = lat_param, lon_param
        print(f"🌍 [Request] Using dynamic coordinates from URL → lat={TARGET_LAT}, lon={TARGET_LON}")
    else:
        print(f"⚙️ [Request] No coordinates provided, using default → lat={TARGET_LAT}, lon={TARGET_LON}")

    # ========== 2️⃣ 找最近測站 ==========
    loc_id, loc_name, lat_found, lon_found = get_nearest_location(TARGET_LAT, TARGET_LON)
    if loc_id:
        current_location_id = loc_id
        current_location_name = loc_name
        station_lat, station_lon = lat_found, lon_found # 使用測站的精確坐標來獲取天氣
        print(f"✅ [Nearest Station Found] {loc_name} (ID: {loc_id})")
        print(f"📍 Station Coordinates : {station_lat}, {station_lon}")
    else:
        print("⚠️ [Nearest] No valid station found, fallback to default Kaohsiung")
        current_location_id = DEFAULT_LOCATION_ID
        current_location_name = DEFAULT_LOCATION_NAME
        # 如果找不到測站，使用 TARGET 坐標來獲取天氣

    # ⭐️ 新增：獲取天氣預報
    weather_forecast_df = get_weather_forecast(station_lat, station_lon)

    # ========== 3️⃣ 取得觀測資料 ==========
    current_observation_raw = fetch_latest_observation_data(current_location_id, POLLUTANT_TARGETS)

    if not current_observation_raw.empty:
        print("\n📊 [OpenAQ Raw Observation DataFrame]")
        print(current_observation_raw.to_string(index=False))
    else:
        print("🚨 [OpenAQ] No data returned from API.")

    # ========== 4️⃣ 取得當前 AQI ==========
    if not current_observation_raw.empty and 'aqi' in current_observation_raw.columns:
        obs_aqi_val = current_observation_raw['aqi'].iloc[0]
        obs_time_val = current_observation_raw['datetime'].iloc[0]
        CURRENT_OBSERVATION_AQI = int(obs_aqi_val) if pd.notna(obs_aqi_val) else "N/A"
        if pd.notna(obs_time_val):
            if obs_time_val.tz is None:
                obs_time_val = obs_time_val.tz_localize('UTC')
            CURRENT_OBSERVATION_TIME = obs_time_val.tz_convert(LOCAL_TZ).strftime('%Y-%m-%d %H:%M')
    else:
        CURRENT_OBSERVATION_AQI = "N/A"
        CURRENT_OBSERVATION_TIME = "N/A"

    # ========== 5️⃣ 建立預測或回退顯示 ==========
    observation_for_prediction = None
    is_valid_for_prediction = False
    is_fallback_mode = True

    if not current_observation_raw.empty and LAST_OBSERVATION is not None and not LAST_OBSERVATION.empty:
        observation_for_prediction = LAST_OBSERVATION.iloc[:1].copy()
        latest_row = current_observation_raw.iloc[0]
        dt_val = latest_row['datetime']
        if pd.to_datetime(dt_val).tz is not None:
            dt_val = pd.to_datetime(dt_val).tz_convert(None)
        observation_for_prediction['datetime'] = dt_val

        for col in latest_row.index:
            if col in observation_for_prediction.columns and not any(s in col for s in ['lag_', 'rolling_']):
                if col in POLLUTANT_TARGETS or col == 'aqi' or col in ['temperature', 'humidity', 'pressure']:
                    observation_for_prediction[col] = latest_row[col]

        if all(col in observation_for_prediction.columns for col in FEATURE_COLUMNS):
            is_valid_for_prediction = True

    max_aqi = CURRENT_OBSERVATION_AQI
    aqi_predictions = []

    if TRAINED_MODELS and POLLUTANT_PARAMS and is_valid_for_prediction and observation_for_prediction is not None:
        try:
            # ⭐️ 傳遞天氣預報數據
            future_predictions = predict_future_multi(
                TRAINED_MODELS,
                observation_for_prediction,
                FEATURE_COLUMNS,
                POLLUTANT_PARAMS,
                hours=HOURS_TO_PREDICT,
                weather_df=weather_forecast_df # 傳遞 Open-Meteo 預報
            )
            
            future_predictions['datetime_local'] = future_predictions['datetime'].dt.tz_convert(LOCAL_TZ)
            predictions_df = future_predictions[['datetime_local', 'aqi_pred']].copy()
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

    # ========== 6️⃣ 輸出頁面 ==========
    return render_template(
        'index.html',
        max_aqi=max_aqi,
        aqi_predictions=aqi_predictions,
        city_name=current_location_name,
        current_obs_time=CURRENT_OBSERVATION_TIME,
        is_fallback=is_fallback_mode
    )


if __name__ == '__main__':
    app.run(debug=True)
