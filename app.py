# app.py - Open-Meteo Weather Integration Revision with Full Historical Feature Engineering

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
DEFAULT_LOCATION_ID = 2395624 # Default: Kaohsiung-Qianjin (模型訓練時常用的穩定 ID)
DEFAULT_LOCATION_NAME = "Kaohsiung-Qianjin" # Default Location Name

TARGET_PARAMS = ["co", "no2", "o3", "pm10", "pm25", "so2"]
PARAM_IDS = {"co": 8, "no2": 7, "o3": 10, "pm10": 1, "pm25": 2, "so2": 9}

TOL_MINUTES_PRIMARY = 120
TOL_MINUTES_FALLBACK = 180

# =================================================================
# Global Variables (Mutable)
# =================================================================
TRAINED_MODELS = {} 
LAST_OBSERVATION = None # NOTE: This will now only store the loaded features from meta, mainly for feature list reference.
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
def get_nearest_location(lat: float, lon: float, radius_km: int = 50): 
    """
    Searches for the closest monitoring station using V3 API with simplified parameters.
    Now returns both ID, name, and coordinates.
    """
    V3_LOCATIONS_URL = f"{BASE}/locations" 
    params = {
        "coordinates": f"{lat},{lon}",
        "radius": 50000, # 擴大搜索半徑到 50km
        "limit": 5,
    }
    try:
        r = requests.get(V3_LOCATIONS_URL, headers=HEADERS, params=params, timeout=10)
        r.raise_for_status()
        results = r.json().get("results", [])

        if not results:
            print("🚨 [Nearest] No stations found within 50km.")
            return None, None, None, None

        # 直接使用第一個（最近）站
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
        # print("\n🌍 [DEBUG] Raw stations returned by OpenAQ:") # Commented out for cleaner output

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
        print(f"❌ [Latest] Failed to fetch latest data: {e}")
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
# OpenAQ V3 穩健歷史數據獲取 (新增 - 核心修正)
# =================================================================
def get_location_history_df(location_id: int, hours_back: int = 26) -> pd.DataFrame:
    """
    Fetches historical hourly air quality data for the last 'hours_back' hours (Uses V3).
    Returns a DataFrame with 'datetime' (UTC) and pollutant columns.
    """
    # 設置時間範圍：從現在往前回溯
    end_time = datetime.datetime.now(timezone.utc)
    start_time = end_time - timedelta(hours=hours_back)
    
    # OpenAQ API V3 的 time_from 和 time_to 參數
    time_from_iso = start_time.isoformat().replace('+00:00', 'Z')
    time_to_iso = end_time.isoformat().replace('+00:00', 'Z')
    
    params = {
        "time_from": time_from_iso,
        "time_to": time_to_iso,
        "location_id": location_id,
        "parameter_id": [PARAM_IDS[p] for p in POLLUTANT_TARGETS if p in PARAM_IDS], # 限制為目標污染物
        "limit": 10000, # 獲取足夠多的點
        "sort": "desc"
    }

    try:
        r = requests.get(f"{BASE}/measurements", headers=HEADERS, params=params, timeout=20)
        r.raise_for_status()
        results = r.json().get("results", [])

        if not results:
            print(f"🚨 [History] No historical measurements found for location ID {location_id}.")
            return pd.DataFrame()

        df = pd.json_normalize(results)

        # 標準化時間和數值
        df["datetime"] = pd.to_datetime(df["datetime.utc"], errors="coerce", utc=True)
        df["parameter"] = df["parameter.name"].str.lower()
        df = df.rename(columns={"value": "value_raw"})
        
        # 僅保留目標污染物，並按時間和參數去重複
        df = df[df["parameter"].isin(POLLUTANT_TARGETS)].sort_values(
            ['datetime', 'parameter'], ascending=False
        ).drop_duplicates(subset=['datetime', 'parameter'], keep='first')

        # 轉為寬表格，每行一個時間點
        history_wide = df.pivot_table(
            index='datetime', columns='parameter', values='value_raw'
        ).reset_index()

        # 計算歷史 AQI
        if not history_wide.empty:
            history_wide['aqi'] = history_wide.apply(
                lambda row: calculate_aqi(row, POLLUTANT_TARGETS, is_pred=False), axis=1
            )

        print(f"✅ [History] Fetched {len(history_wide)} historical hours.")
        return history_wide.sort_values('datetime')

    except Exception as e:
        print(f"❌ [History] Failed to fetch historical data: {e}")
        return pd.DataFrame()


# =================================================================
# Open-Meteo Weather Fetching
# =================================================================
# 設置快取 (修正: 移除 create_retry_session，直接使用 CachedSession)
cache_session = requests_cache.CachedSession('.cache', expire_after = 3600)
openmeteo_client = openmeteo_requests.Client(session=cache_session)

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
        
        # 穩健性檢查 (修正: 避免使用 IsInitialized 方法的潛在版本問題)
        if not responses or len(responses) == 0:
             print("❌ [Weather] Open-Meteo did not return any responses.")
             return pd.DataFrame()

        response = responses[0]
        
        # 檢查 Hourly 資料是否真的存在且有長度
        # 這裡仍然使用 IsInitialized() 配合 Hourly() 的檢查來確保數據完整性
        if not response.IsInitialized() or response.Hourly().Time(0) is None:
             print("❌ [Weather] Open-Meteo response not initialized or missing hourly data.")
             return pd.DataFrame()
             
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

# -----------------------------------------------------------------
# 特徵工程生成器 (新增 - 核心修正)
# -----------------------------------------------------------------
def generate_features_for_prediction(history_df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    """
    Takes historical data (at least 24 hours) and generates the single-row 
    DataFrame with all required lag and rolling features for the prediction start point.
    """
    if history_df.empty:
        return pd.DataFrame()
    
    # 確保 history_df 的 datetime 是 UTC-aware
    history_df['datetime'] = pd.to_datetime(history_df['datetime']).dt.tz_localize('UTC')

    # 1. 處理滯後特徵 (Lagged Features)
    df_features = history_df.set_index('datetime').copy()
    pollutant_plus_aqi = POLLUTANT_TARGETS + ['aqi']
    
    for param in pollutant_plus_aqi:
        for lag in LAG_HOURS:
            lag_col = f'{param}_lag_{lag}h'
            if lag_col in feature_cols and param in df_features.columns:
                # 簡單地從過去 N 小時的觀測中取值
                df_features[lag_col] = df_features[param].shift(lag)


    # 2. 處理滾動特徵 (Rolling Features)
    for param in pollutant_plus_aqi:
        for window in ROLLING_WINDOWS:
            roll_col_mean = f'{param}_rolling_{window}h_mean'
            roll_col_std = f'{param}_rolling_{window}h_std'
            
            if param in df_features.columns:
                if roll_col_mean in feature_cols:
                    # 滾動視窗的平均值 (shift(1)確保我們只使用過去的數據)
                    df_features[roll_col_mean] = df_features[param].rolling(window=window, min_periods=1).mean().shift(1)
                
                if roll_col_std in feature_cols:
                    # 滾動視窗的標準差
                    df_features[roll_col_std] = df_features[param].rolling(window=window, min_periods=1).std().shift(1)

    # 3. 提取最後一個觀測點的特徵
    last_obs_row = df_features.iloc[-1].copy().to_frame().T.reset_index(names=['datetime'])
    
    # 處理時間特徵 (只對最後一筆觀測點生成)
    last_obs_row['hour'] = last_obs_row['datetime'].dt.hour
    last_obs_row['day_of_week'] = last_obs_row['datetime'].dt.dayofweek
    last_obs_row['month'] = last_obs_row['datetime'].dt.month
    last_obs_row['day_of_year'] = last_obs_row['datetime'].dt.dayofyear.astype(int)
    last_obs_row['is_weekend'] = last_obs_row['day_of_week'].apply(lambda x: int(x in [5, 6]))
    
    # 循環時間特徵
    last_obs_row['hour_sin'] = np.sin(2 * np.pi * last_obs_row['hour'] / 24)
    last_obs_row['hour_cos'] = np.cos(2 * np.pi * last_obs_row['hour'] / 24)
    last_obs_row['day_sin'] = np.sin(2 * np.pi * last_obs_row['day_of_year'] / 365)
    last_obs_row['day_cos'] = np.cos(2 * np.pi * last_obs_row['day_of_year'] / 365)
    
    # 選擇所有模型所需的特徵
    final_input_df = last_obs_row[feature_cols + ['datetime']].copy()
    
    # 簡易填充：用前一個有效值填充任何剩餘的 NaN (主要針對剛開始時的滾動/滯後特徵)
    # 並用 0 填充最前端的 NaN
    final_input_df = final_input_df.fillna(method='ffill', axis=1).fillna(0)
    
    print(f"✅ [Feature Eng] Generated 1 observation row with {len(final_input_df.columns) - 1} features.")
    return final_input_df


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
    last_data['datetime'] = pd.to_datetime(last_data['datetime']).dt.tz_convert('UTC')
        
    last_datetime_aware = last_data['datetime'].iloc[0]
    
    # 初始化特徵字典 (注意：這裡的 last_data 已經包含了正確的滯後/滾動特徵)
    current_data_dict = {col: last_data.get(col, np.nan).iloc[0] 
                              if col in last_data.columns and not last_data[col].empty 
                              else np.nan 
                              for col in feature_cols} 

    # 將當前觀測值 (非滯後) 也放入字典，以便在下一輪遞迴更新滯後特徵
    for p in pollutant_params + ['aqi', 'temperature', 'humidity', 'pressure']:
        if p in last_data.columns and not last_data[p].empty:
             current_data_dict[p] = last_data[p].iloc[0]
             
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
                    # ⚠️ 預報缺失：使用上一個時間點的預測值（即 current_data_dict 中最新的天氣值）
                    print(f"⚠️ [Weather] Forecast missing for {future_time}. Using last known value.")
                    for w_col in weather_feature_names:
                         pred_features[w_col] = current_data_dict.get(w_col, np.nan) 
                         
            # -----------------------------------------------
            current_prediction_row = {'datetime': future_time}
            new_pollutant_values = {}
            predicted_aqi = np.nan
            
            # 預測每個污染物
            for param in pollutant_params:
                if param not in models:
                    continue

                model = models[param]
                
                # 從 pred_features 中提取模型所需的特徵
                pred_input_list = [pred_features.get(col) for col in feature_cols]
                
                # 檢查是否有任何 NaN 或 None 的關鍵特徵
                if any(pd.isna(x) for x in pred_input_list):
                     print(f"❌ [Predict] Input feature missing or NaN for {param} at Hour +{h+1}. Skipping prediction.")
                     new_pollutant_values[param] = np.nan
                     continue

                pred_input = np.array(pred_input_list, dtype=np.float64).reshape(1, -1)

                pred = model.predict(pred_input)[0]
                pred = max(0, pred)

                current_prediction_row[f'{param}_pred'] = pred
                new_pollutant_values[param] = pred
                total_predictions += 1

            # 計算 AQI
            if new_pollutant_values:
                predicted_aqi = calculate_aqi(pd.Series(current_prediction_row), pollutant_params, is_pred=True)
            
            current_prediction_row['aqi_pred'] = predicted_aqi
            new_pollutant_values['aqi'] = predicted_aqi
            predictions.append(current_prediction_row)

            # 更新滯後特徵和滾動特徵 (遞迴核心)
            for param in pollutant_params + ['aqi']:
                if pd.isna(new_pollutant_values.get(param)): continue
                
                # 1. 更新滯後特徵：將 1h 的預測值設為當前 param 的最新值
                for i in range(len(LAG_HOURS) - 1, 0, -1):
                    lag_current = LAG_HOURS[i]
                    lag_prev = LAG_HOURS[i-1]
                    lag_current_col = f'{param}_lag_{lag_current}h'
                    lag_prev_col = f'{param}_lag_{lag_prev}h'

                    if lag_current_col in current_data_dict and lag_prev_col in current_data_dict:
                         current_data_dict[lag_current_col] = current_data_dict[lag_prev_col]

                if f'{param}_lag_1h' in current_data_dict:
                    current_data_dict[f'{param}_lag_1h'] = new_pollutant_values[param]
                    
                # 2. 更新滾動特徵
                for window in ROLLING_WINDOWS:
                    roll_mean_col = f'{param}_rolling_{window}h_mean'
                    roll_std_col = f'{param}_rolling_{window}h_std'
                    
                    if roll_mean_col in current_data_dict:
                        # 使用 EWMA 模擬滾動平均更新
                        alpha = 2 / (window + 1)
                        old_mean = current_data_dict[roll_mean_col]
                        new_value = new_pollutant_values[param]
                        
                        # 這是下一輪預測的滾動平均
                        current_data_dict[roll_mean_col] = (1 - alpha) * old_mean + alpha * new_value
                        
                    # 滾動標準差的遞迴更新非常複雜，這裡暫時保持不變 (使用前一時刻的值)

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
            # 這裡只是載入它來獲取欄位名稱和順序，實際數值將被新地點的數據取代
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

    # pandas 印出設定
    pd.set_option('display.max_columns', 15)
    pd.set_option('display.width', 180)

    # ========== 1️⃣ 從網址參數抓座標 ==========
    lat_param = request.args.get('lat', type=float)
    lon_param = request.args.get('lon', type=float)

    if lat_param is not None and lon_param is not None:
        TARGET_LAT, TARGET_LON = lat_param, lon_param
        print(f"🌍 [Request] Using dynamic coordinates from URL → lat={TARGET_LAT}, lon={TARGET_LON}")
    else:
        print(f"⚙️ [Request] No coordinates provided, using default → lat={TARGET_LAT}, lon={TARGET_LON}")

    # ========== 2️⃣ 找最近測站 (如果失敗，使用 DEFAULT 測站) ==========
    loc_id, loc_name, lat_found, lon_found = get_nearest_location(TARGET_LAT, TARGET_LON)
    
    if loc_id:
        current_location_id = loc_id
        current_location_name = loc_name
        station_lat, station_lon = lat_found, lon_found # 使用測站的精確坐標來獲取天氣
        print(f"✅ [Nearest Station Found] {loc_name} (ID: {loc_id})")
        print(f"📍 Station Coordinates : {station_lat}, {station_lon}")
    else:
        # 強制使用 DEFAULT_LOCATION_ID 來確保歷史數據獲取有更高的成功率
        print(f"⚠️ [Nearest] No valid station found, fallback to default model station: {DEFAULT_LOCATION_NAME}")
        current_location_id = DEFAULT_LOCATION_ID
        current_location_name = DEFAULT_LOCATION_NAME
        # 如果找不到測站，使用 TARGET 坐標來獲取天氣

    # ⭐️ 新增：獲取天氣預報 (使用測站或目標座標)
    weather_forecast_df = get_weather_forecast(station_lat, station_lon)

    # ========== 3️⃣ 取得觀測資料 (修正：新增歷史資料獲取) ==========
    # 獲取單一最新觀測點
    current_observation_raw = fetch_latest_observation_data(current_location_id, POLLUTANT_TARGETS)
    
    # 獲取歷史趨勢數據 (用於計算滯後特徵)
    # 注意：如果 current_location_id 的歷史數據 404，這裡可能會失敗
    historical_df = get_location_history_df(current_location_id, hours_back=26) 


    if not current_observation_raw.empty:
        print("\n📊 [OpenAQ Latest Observation DataFrame]")
        print(current_observation_raw.to_string(index=False))
    else:
        print("🚨 [OpenAQ] No LATEST data returned from API.")


    # ========== 4️⃣ 取得當前 AQI (保持不變) ==========
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

    # ========== 5️⃣ 建立預測起點 (替換為完整修正邏輯) ==========
    observation_for_prediction = None
    is_valid_for_prediction = False
    is_fallback_mode = True

    # 檢查是否有模型，有最新觀測，有歷史數據
    if TRAINED_MODELS and not current_observation_raw.empty and not historical_df.empty:
        try:
            # 1. 用歷史數據生成包含所有滯後特徵的單行 DataFrame
            obs_with_features = generate_features_for_prediction(historical_df, FEATURE_COLUMNS)
            
            # 2. 如果生成成功，則使用它作為預測起點
            if not obs_with_features.empty:
                
                # 確保最新的觀測值覆蓋歷史計算的最新點（尤其是在 API 延遲時）
                latest_row = current_observation_raw.iloc[0].to_dict()
                
                # 覆蓋污染物和 AQI
                for col in POLLUTANT_TARGETS + ['aqi']:
                    if col in latest_row and col in obs_with_features.columns:
                         obs_with_features[col] = latest_row[col]
                         
                # 確保將天氣數據的第一小時預報值也放入 (用於 t=0 的天氣特徵)
                if not weather_forecast_df.empty and 'temperature' in obs_with_features.columns:
                     latest_weather = weather_forecast_df.iloc[0].to_dict()
                     for w_col in ['temperature', 'humidity', 'pressure']:
                         if w_col in obs_with_features.columns and w_col in latest_weather:
                              # 使用預報的第一小時值（即 t+1 的預報，作為 t=0 的天氣觀測的替代）
                              obs_with_features[w_col] = latest_weather.get(w_col, obs_with_features[w_col].iloc[0]) 
                
                observation_for_prediction = obs_with_features
                
                # 最終檢查所有必要欄位是否存在
                missing_features = [col for col in FEATURE_COLUMNS if col not in observation_for_prediction.columns]
                if not missing_features:
                    is_valid_for_prediction = True
                    print("\n✅ [Feature Input] Prediction input row successfully generated:")
                    # 顯示污染物當前值和部分滯後特徵
                    cols_to_print = POLLUTANT_TARGETS + ['aqi', 'pm25_lag_1h', 'pm25_rolling_6h_mean', 'temperature', 'humidity']
                    print(observation_for_prediction.iloc[0][observation_for_prediction.columns.intersection(cols_to_print)].to_string())
                else:
                    print(f"❌ [Feature Input] Missing critical features for prediction: {missing_features}")

        except Exception as e:
            print(f"❌ [Feature Input] Error during feature generation: {e}")

    # --- 預測區塊 ---
    max_aqi = CURRENT_OBSERVATION_AQI
    aqi_predictions = []

    if TRAINED_MODELS and POLLUTANT_PARAMS and is_valid_for_prediction and observation_for_prediction is not None:
        try:
            # ⭐️ 傳遞天氣預報數據
            future_predictions = predict_future_multi(
                TRAINED_MODELS,
                observation_for_prediction, # 使用新的、有正確滯後特徵的數據
                FEATURE_COLUMNS,
                POLLUTANT_PARAMS,
                hours=HOURS_TO_PREDICT,
                weather_df=weather_forecast_df
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
            print(f"❌ [Predict] Error during main prediction process: {e}")

    if is_fallback_mode:
        print("🚨 [Fallback Mode] Prediction failed or data insufficient. Showing latest observed AQI only.")
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
