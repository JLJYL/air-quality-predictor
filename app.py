# app.py - FINAL FINAL REVISION (Hardened Timezone and Location Check)

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
from flask import Flask, render_template

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

# Target geographical coordinates 
TARGET_LAT = 22.6324 
TARGET_LON = 120.2954

# Initial/Default Location (These will be updated by initialize_location)
DEFAULT_LOCATION_ID = 2395624 # Default: Kaohsiung-Qianjin
DEFAULT_LOCATION_NAME = "Kaohsiung-Qianjin" # Default Location Name

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
    硬性修正：強制使用 V3 API 要求的參數。
    """
    V3_LOCATIONS_URL = f"{BASE}/locations" 
    
    # 硬性修正: 確保 radius <= 25000，並只使用 V3 支援的參數
    params = {
        "coordinates": f"{lat},{lon}",
        "radius": 25000,  # 强制限制在 25km
        "limit": 5,
        # 移除 order_by 和 sort 參數，因為 V3 API 不允許
    }
    
    try:
        r = requests.get(V3_LOCATIONS_URL, headers=HEADERS, params=params, timeout=10)
        r.raise_for_status()
        results = r.json().get("results", [])
        
        if not results:
            print("🚨 [Nearest] V3: No stations found within the specified radius (25km).")
            return None, None
            
        # V3 回傳的結果預設應該是按照距離排序的
        # 篩選出第一個提供 PM2.5 數據的站點
        for nearest_loc in results:
            # 檢查該站點的 parameters 列表是否包含 PM2.5 (ID: 2 或 name: pm25)
            has_pm25 = any(p.get("id") == 2 or p.get("name").lower() == "pm25" for p in nearest_loc.get("parameters", []))
            
            if has_pm25:
                loc_id = int(nearest_loc["id"])
                loc_name = nearest_loc["name"]
                print(f"✅ [Nearest] V3: Successfully found nearest station: {loc_name} (ID: {loc_id})")
                return loc_id, loc_name
        
        # 如果前 5 個站點都沒有 PM2.5，則回傳 None
        print("🚨 [Nearest] V3: Found stations, but none of the nearest 5 offer PM2.5 data.")
        return None, None

    except Exception as e:
        status_code = r.status_code if 'r' in locals() else 'N/A'
        error_detail = r.text if 'r' in locals() else str(e)
        print(f"❌ [Nearest] V3: Failed to search for the nearest station. Status: {status_code}. Details: {error_detail}")
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
# Prediction Function (修正時區錯誤)
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
    weather_feature_names = [col for col in weather_feature_names_base if col in feature_cols]
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

        predictions.append(current_prediction_row)

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

def initialize_location():
    """Finds the nearest location and updates the global variables."""
    global current_location_id, current_location_name, DEFAULT_LOCATION_ID, DEFAULT_LOCATION_NAME
    
    loc_id, loc_name = get_nearest_location(TARGET_LAT, TARGET_LON)
    
    if loc_id is not None:
        current_location_id = loc_id
        current_location_name = loc_name
    else:
        # Fallback to the hardcoded default if API call fails
        current_location_id = DEFAULT_LOCATION_ID
        current_location_name = DEFAULT_LOCATION_NAME
        print(f"⚠️ Could not find the nearest station, using default station: {current_location_name} (ID: {current_location_id})")

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

    # Extract the latest observed AQI for fallback
    if not current_observation_raw.empty and 'aqi' in current_observation_raw.columns:
        obs_aqi_val = current_observation_raw['aqi'].iloc[0]
        obs_time_val = current_observation_raw['datetime'].iloc[0]
        
        CURRENT_OBSERVATION_AQI = int(obs_aqi_val) if pd.notna(obs_aqi_val) else "N/A"
        
        if pd.notna(obs_time_val):
            # 確保 time is UTC-aware for display, then convert to local
            if obs_time_val.tz is None:
                 obs_time_val = obs_time_val.tz_localize('UTC')
            
            CURRENT_OBSERVATION_TIME = obs_time_val.tz_convert(LOCAL_TZ).strftime('%Y-%m-%d %H:%M')
        else:
             CURRENT_OBSERVATION_TIME = "N/A"
    
    
    # 2. Prepare data for prediction
    observation_for_prediction = None
    is_valid_for_prediction = False

    if not current_observation_raw.empty and LAST_OBSERVATION is not None and not LAST_OBSERVATION.empty:
        # Integrate the latest observation into the lagged features
        observation_for_prediction = LAST_OBSERVATION.iloc[:1].copy() 
        latest_row = current_observation_raw.iloc[0]
        
        # 核心修正：安全地移除時區，為遞迴預測做準備
        dt_val = latest_row['datetime']
        
        # 雙重檢查：確保移除時區時不會觸發 'Already tz-aware' 錯誤
        if pd.to_datetime(dt_val).tz is not None:
            dt_val = pd.to_datetime(dt_val).tz_convert(None) 
            
        observation_for_prediction['datetime'] = dt_val
        
        # Update current values and features (non-lag/non-rolling)
        for col in latest_row.index:
            if col in observation_for_prediction.columns and not any(s in col for s in ['lag_', 'rolling_']):
                 if col in POLLUTANT_TARGETS or col == 'aqi' or col in ['temperature', 'humidity', 'pressure']:
                      observation_for_prediction[col] = latest_row[col]
            
        # Check if all required features are present
        if all(col in observation_for_prediction.columns for col in FEATURE_COLUMNS):
             is_valid_for_prediction = True
        else:
             print("⚠️ [Request] Missing required feature columns after integration, falling back.")
    else:
        print("🚨 [Request] Cannot get latest observation or lagged model data. Prediction is not possible.")


    # 3. Perform prediction or fallback
    max_aqi = CURRENT_OBSERVATION_AQI
    aqi_predictions = []
    
    is_fallback_mode = True

    if TRAINED_MODELS and POLLUTANT_PARAMS and is_valid_for_prediction and observation_for_prediction is not None:
        try:
            
            # The final time zone handling is done within predict_future_multi
            future_predictions = predict_future_multi(
                TRAINED_MODELS,
                observation_for_prediction,
                FEATURE_COLUMNS,
                POLLUTANT_PARAMS,
                hours=HOURS_TO_PREDICT
            )

            # Convert UTC time to local time for display
            # future_predictions['datetime'] is UTC-aware from predict_future_multi
            future_predictions['datetime_local'] = future_predictions['datetime'].dt.tz_convert(LOCAL_TZ)
            
            # Process NaN values and calculate Max AQI
            predictions_df = future_predictions[['datetime_local', 'aqi_pred']].copy()
            max_aqi_val = predictions_df['aqi_pred'].max()
            max_aqi = int(max_aqi_val) if pd.notna(max_aqi_val) else CURRENT_OBSERVATION_AQI
            
            # Replace NaN with "N/A" and convert valid numbers to integers
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
                 # Prediction list is empty, fallback to current observed AQI
                 max_aqi = CURRENT_OBSERVATION_AQI
                 is_fallback_mode = True
                 print("⚠️ [Request] Prediction list is empty, falling back to latest observed AQI.")


        except Exception as e:
            # Prediction failed, fallback
            max_aqi = CURRENT_OBSERVATION_AQI
            aqi_predictions = []
            is_fallback_mode = True
            print(f"❌ [Request] Prediction execution failed ({e}), falling back to latest observed AQI.") 
            
    if is_fallback_mode:
             # Models not loaded or data invalid, generate a single observation entry for fallback display
             print("🚨 [Request] Final result using fallback mode.")
             max_aqi = CURRENT_OBSERVATION_AQI
             
             # Create a list containing only the current observation, marked as observation
             if max_aqi != "N/A":
               aqi_predictions = [{
                 'time': CURRENT_OBSERVATION_TIME,
                 'aqi': max_aqi,
                 'is_obs': True # New marker for observation
               }]

    # 4. Render template
    return render_template('index.html', 
                            max_aqi=max_aqi, 
                            aqi_predictions=aqi_predictions, 
                            city_name=current_location_name, # Use the dynamically found location name
                            current_obs_time=CURRENT_OBSERVATION_TIME,
                            is_fallback=is_fallback_mode)

if __name__ == '__main__':
    app.run(debug=True)
