# app.py - FINAL PRODUCTION CODE (Global Auto-Selection, All Bug Fixes, and Neutral N/A Display)

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

# Target geographical coordinates (Default to Kaohsiung if no user coordinates are provided)
TARGET_LAT = 22.6324 
TARGET_LON = 120.2954

# Initial/Default Location (Hardcoded fallback if NO PM2.5 station is found worldwide)
DEFAULT_LOCATION_ID = 2395624 # Default: Kaohsiung-Qianjin (Used ONLY for model loading fallback)
DEFAULT_LOCATION_NAME = "Kaohsiung-Qianjin" 

TARGET_PARAMS = ["co", "no2", "o3", "pm10", "pm25", "so2"]
PARAM_IDS = {"co": 8, "no2": 7, "o3": 10, "pm10": 1, "pm25": 2, "so2": 9}

TOL_MINUTES_PRIMARY = 5
TOL_MINUTES_FALLBACK = 60

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

# Dynamic Location Variables (Will be updated on startup and on each user request)
current_location_id = None 
current_location_name = "System Initializing..."

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
# V3 API Global Auto-Selection Function - 已移除 PM2.5 篩選
# =================================================================
def get_nearest_location(lat: float, lon: float): 
    """
    Searches for the closest monitoring station in two phases.
    Filters PM2.5 availability only AFTER fetching the actual data (in index route).
    """
    V3_LOCATIONS_URL = f"{BASE}/locations" 
    
    # --- 搜尋階段設定 ---
    # 修正半徑限制為 OpenAQ V3 允許的最大值 25000 米
    search_phases = [
        {"radius_m": 25000, "limit": 5, "name": "Strict (25km/5)"}, 
        {"radius_m": 25000, "limit": 100, "name": "Fallback (25km/100)"}, 
    ]

    for phase in search_phases:
        radius_m = phase["radius_m"]
        limit = phase["limit"]
        
        params = {
            "coordinates": f"{lat},{lon}",
            "radius": radius_m, 
            "limit": limit,
        }
        
        try:
            r = requests.get(V3_LOCATIONS_URL, headers=HEADERS, params=params, timeout=10)
            r.raise_for_status()
            results = r.json().get("results", [])
            
            if not results:
                print(f"🚨 [Nearest] Phase {phase['name']}: No stations found.")
                continue 

            # **【關鍵修正】**：不再檢查 PM2.5 屬性，直接選中第一個站點 (距離最近)
            nearest_loc = results[0] 
            loc_id = int(nearest_loc["id"])
            loc_name = nearest_loc["name"]
            
            print(f"✅ [Nearest] Phase {phase['name']}: Found the closest station: {loc_name} (ID: {loc_id}).")
            return loc_id, loc_name
                 
        except Exception as e:
            status_code = r.status_code if 'r' in locals() else 'N/A'
            error_detail = r.text if 'r' in locals() else str(e)
            print(f"❌ [Nearest] Phase {phase['name']}: Failed to search. Status: {status_code}. Details: {error_detail}")
            continue 

    # 如果所有階段都失敗，則回傳 None
    return None, None
        
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
# Prediction Function (Timezone Aware)
# =================================================================
def predict_future_multi(models, last_data, feature_cols, pollutant_params, hours=24):
    """Predicts multiple target pollutants for N future hours (recursive prediction) and calculates AQI."""
    predictions = []

    # 確保數據是 tz-aware (UTC)
    last_data['datetime'] = pd.to_datetime(last_data['datetime'])
    if last_data['datetime'].dt.tz is None:
        # 如果沒有時區，賦予 UTC
        last_data['datetime'] = last_data['datetime'].dt.tz_localize('UTC')
    else:
        # 如果已經有時區，轉換為 UTC 
        last_data['datetime'] = last_data['datetime'].dt.tz_convert('UTC')
        
    last_datetime_aware = last_data['datetime'].iloc[0]
    
    # Initialize features dictionary from the last observation
    current_data_dict = {col: last_data.get(col, np.nan).iloc[0] 
                             if col in last_data.columns and not last_data[col].empty 
                             else np.nan 
                             for col in feature_cols} 

    weather_feature_names_base = ['temperature', 'humidity', 'pressure']
    weather_feature_names = [col for col in feature_cols if col in weather_feature_names_base]
    has_weather = bool(weather_feature_names)

    for h in range(hours):
        future_time = last_datetime_aware + timedelta(hours=h + 1)
        pred_features = current_data_dict.copy()

        # 1. Update time-based features
        pred_features['hour'] = future_time.hour
        pred_features['day_of_week'] = future_time.dayofweek
        pred_features['month'] = future_time.month
        pred_features['day_of_year'] = future_time.timetuple().tm_yday 
        pred_features['is_weekend'] = int(future_time.dayofweek in [5, 6])
        pred_features['hour_sin'] = np.sin(2 * np.pi * future_time.hour / 24)
        pred_features['hour_cos'] = np.cos(2 * np.pi * future_time.hour / 24)
        pred_features['day_sin'] = np.sin(2 * np.pi * pred_features['day_of_year'] / 365)
        pred_features['day_cos'] = np.cos(2 * np.pi * pred_features['day_of_year'] / 365)

        # 2. Simulate future weather changes (simple random walk for features without forecasts)
        if has_weather:
            # Seed for deterministic simulation across features for the same hour
            np.random.seed(future_time.hour + future_time.day + 42) 
            for w_col in weather_feature_names:
                base_value = current_data_dict.get(w_col)
                if base_value is not None and pd.notna(base_value):
                    new_weather_value = base_value + np.random.normal(0, 0.5) 
                    pred_features[w_col] = new_weather_value
                    current_data_dict[w_col] = new_weather_value 
                else:
                    pred_features[w_col] = np.nan
                    current_data_dict[w_col] = np.nan 


        current_prediction_row = {'datetime': future_time}
        new_pollutant_values = {}

        # 3. Predict all pollutants
        for param in pollutant_params:
            model = models[param]
            # Ensure input is in the expected feature order
            pred_input_list = [pred_features.get(col) for col in feature_cols]
            pred_input = np.array(pred_input_list, dtype=np.float64).reshape(1, -1)
            
            pred = model.predict(pred_input)[0]
            pred = max(0, pred) 

            current_prediction_row[f'{param}_pred'] = pred
            new_pollutant_values[param] = pred

        # 4. Calculate predicted AQI
        predicted_aqi = calculate_aqi(pd.Series(current_prediction_row), pollutant_params, is_pred=True)
        current_prediction_row['aqi_pred'] = predicted_aqi
        new_pollutant_values['aqi'] = predicted_aqi

        # 5. Update lag features for the next hour's prediction (recursive)
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

        predictions.append(current_prediction_row)

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
        
        # ⚠️ 這裡使用 DEFAULT_LOCATION_ID 的歷史數據作為模型的初始輸入
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

def initialize_location_on_startup():
    """
    Finds the nearest location using the default target coordinates. 
    If no PM2.5 station is found, current_location_id is set to None.
    """
    global current_location_id, current_location_name, DEFAULT_LOCATION_NAME
    
    print(f"🌐 [Startup] Initializing location using default coordinates: {TARGET_LAT}, {TARGET_LON}")
    loc_id, loc_name = get_nearest_location(TARGET_LAT, TARGET_LON)
    
    if loc_id is not None:
        # 找到站點，但我們還不能確定它有 PM2.5，先選中它
        current_location_id = loc_id
        current_location_name = loc_name
        print(f"✅ [Startup] Found a potential station: {current_location_name} (ID: {current_location_id})")
    else:
        # 找不到任何站點 (25km 半徑內)
        current_location_id = None
        current_location_name = "Default (No Station Found Near Target)"
        print(f"⚠️ [Startup] No station found near default location. Initializing to 'None' station ID.")

# Dynamically find the nearest location for the server's initial run
initialize_location_on_startup()


app = Flask(__name__)

# Load models when the application starts
with app.app_context():
    load_models_and_metadata() 

@app.route('/')
def index():
    global CURRENT_OBSERVATION_AQI, CURRENT_OBSERVATION_TIME, current_location_id, current_location_name, TARGET_LAT, TARGET_LON
    
    # === 獲取用戶座標或使用預設座標 ===
    user_lat = request.args.get('lat', type=float)
    user_lon = request.args.get('lon', type=float)
    
    # 用於 N/A 狀態顯示的座標
    display_lat = user_lat if user_lat is not None else TARGET_LAT
    display_lon = user_lon if user_lon is not None else TARGET_LON

    # 設置當前請求要使用的站點資訊 (預設使用全局變量)
    station_id = current_location_id
    station_name = current_location_name

    # 1. 處理有用戶座標的情況，嘗試尋找最近站點
    if user_lat is not None and user_lon is not None:
        target_lat = user_lat
        target_lon = user_lon
        
        print(f"✅ [Location] Using User Coordinates: LAT={target_lat}, LON={target_lon}")
        
        loc_id, loc_name = get_nearest_location(target_lat, target_lon)
        
        if loc_id is not None:
            # 找到新的最近站點，更新當前請求使用的站點資訊
            station_id = loc_id
            station_name = loc_name
        else:
            # 找不到任何站點 (25km 半徑內)
            station_id = None
            # 站名設置為一個中立的標籤，讓下面的 N/A 邏輯處理最終顯示名稱
            station_name = f"Location near {target_lat:.2f}, {target_lon:.2f}"
            print(f"⚠️ [Location] No station found near user. Entering 'No Data' mode.")
    
    else:
        # 2. 如果 URL 參數中沒有座標 (使用啟動時的結果)
        print(f"⚠️ [Location] No user coordinates found. Using current station: {station_name}")
        
    # =================================================================
    # 核心邏輯：站點有效性檢查與數據獲取
    # =================================================================

    # 3. 如果站點 ID 是 None (找不到任何站點)，則直接進入無法預測模式
    if station_id is None:
        max_aqi = "N/A"
        # ⚠️ 這裡使用座標來顯示中立名稱
        display_name = f"Location near {display_lat:.2f}, {display_lon:.2f} (No Data)"
        return render_template('index.html', 
                                max_aqi=max_aqi, 
                                aqi_predictions=[], 
                                city_name=display_name, 
                                current_obs_time="N/A",
                                is_fallback=True)
    
    # 4. 嘗試獲取最新的觀測數據
    current_observation_raw = fetch_latest_observation_data(station_id, POLLUTANT_TARGETS)

    # **【重要檢查】**：檢查獲取的實際數據中是否包含 PM2.5
    if current_observation_raw.empty or 'pm25' not in current_observation_raw.columns or pd.isna(current_observation_raw['pm25'].iloc[0]):
        print(f"🚨 [Data Check] Station {station_name} (ID: {station_id}) was selected but did NOT return valid PM2.5 data. Falling back to 'No Data' mode.")
        
        # ⚠️ 【最終修正】：觸發「無數據」回退，強制使用座標作為站名
        display_name = f"Location near {display_lat:.2f}, {display_lon:.2f} (No Data)"
        max_aqi = "N/A"
        return render_template('index.html', 
                                max_aqi=max_aqi, 
                                aqi_predictions=[], 
                                city_name=display_name, # 例如: "Location near 24.18, 120.68 (No Data)"
                                current_obs_time="N/A",
                                is_fallback=True)
        
    # =================================================================
    # 5. 數據有效：執行預測
    # =================================================================
    
    # Extract the latest observed AQI 
    obs_aqi_val = current_observation_raw['aqi'].iloc[0] if 'aqi' in current_observation_raw.columns else np.nan
    obs_time_val = current_observation_raw['datetime'].iloc[0]
        
    CURRENT_OBSERVATION_AQI = int(obs_aqi_val) if pd.notna(obs_aqi_val) else "N/A"
        
    if pd.notna(obs_time_val):
        if obs_time_val.tz is None:
             obs_time_val = obs_time_val.tz_localize('UTC')
            
        CURRENT_OBSERVATION_TIME = obs_time_val.tz_convert(LOCAL_TZ).strftime('%Y-%m-%d %H:%M')
    else:
        CURRENT_OBSERVATION_TIME = "N/A"
    
    # Prepare data for prediction
    observation_for_prediction = None
    is_valid_for_prediction = False

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
        else:
             print("⚠️ [Request] Missing required feature columns after integration, falling back.")
    else:
        print("🚨 [Request] Cannot get latest observation or lagged model data. Prediction is not possible.")


    # Perform prediction or fallback
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

            future_predictions['datetime_local'] = future_predictions['datetime'].dt.tz_convert(LOCAL_TZ)
            
            predictions_df = future_predictions[['datetime_local', 'aqi_pred']].copy()
            max_aqi_val = predictions_df['aqi_pred'].max()
            max_aqi = int(max_aqi_val) if pd.notna(max_aqi_val) else CURRENT_OBSERVATION_AQI
            
            predictions_df['aqi_pred'] = predictions_df['aqi_pred'].replace(np.nan, "N/A")
            predictions_df['aqi'] = predictions_df['aqi_pred'].apply(
                 lambda x: int(x) if x != "N/A" else "N/A"
            ).astype(object)

            aqi_predictions = [
                {
                    'time': item['datetime_local'].strftime('%Y-%m-%d %H:%M'), 
                    'aqi': item['aqi']
                }
                for item in predictions_df.to_dict(orient='records')
            ]
            
            if aqi_predictions:
                 is_fallback_mode = False
                 print("✅ [Request] Prediction successful!")
            else:
                 max_aqi = CURRENT_OBSERVATION_AQI
                 is_fallback_mode = True
                 print("⚠️ [Request] Prediction list is empty, falling back to latest observed AQI.")


        except Exception as e:
            max_aqi = CURRENT_OBSERVATION_AQI
            aqi_predictions = []
            is_fallback_mode = True
            print(f"❌ [Request] Prediction execution failed ({e}), falling back to latest observed AQI.") 
            
    if is_fallback_mode:
             print("🚨 [Request] Final result using fallback mode.")
             max_aqi = CURRENT_OBSERVATION_AQI
             
             if max_aqi != "N/A":
               aqi_predictions = [{
                 'time': CURRENT_OBSERVATION_TIME,
                 'aqi': max_aqi,
                 'is_obs': True 
               }]

    # 6. Render template - 使用動態站名或中立座標名稱
    display_city_name = station_name
    if is_fallback_mode and (station_id is None or current_observation_raw.empty or 'pm25' not in current_observation_raw.columns):
         display_city_name = f"Location near {display_lat:.2f}, {display_lon:.2f} (No Data)"

    return render_template('index.html', 
                            max_aqi=max_aqi, 
                            aqi_predictions=aqi_predictions, 
                            city_name=display_city_name,
                            current_obs_time=CURRENT_OBSERVATION_TIME,
                            is_fallback=is_fallback_mode)

if __name__ == '__main__':
    app.run(debug=True)
