

import requests
import pandas as pd
import datetime
import os
import warnings
import numpy as np
import xgboost as xgb
import json
from datetime import timedelta, timezone
from flask import Flask, render_template, request
import openmeteo_requests
import requests_cache

warnings.filterwarnings('ignore')

MODELS_DIR = 'models'
META_PATH = os.path.join(MODELS_DIR, 'model_meta.json')

# OpenAQ API Constants
API_KEY = "fb579916623e8483cd85344b14605c3109eea922202314c44b87a2df3b1fff77" 
HEADERS = {"X-API-Key": API_KEY}
BASE = "https://api.openaq.org/v3"

TARGET_PARAMS = ["co", "no2", "o3", "pm10", "pm25", "so2"]
PARAM_IDS = {"co": 8, "no2": 7, "o3": 10, "pm10": 1, "pm25": 2, "so2": 9}

TOL_MINUTES_PRIMARY = 120
TOL_MINUTES_FALLBACK = 180

# ✅ 核心修改 1: 只用 1 小時 lag
LAG_HOURS = [1]  # 原本是 [1, 2, 3, 6, 12, 24]
ROLLING_WINDOWS = []  # 移除滾動窗口特徵

# ❌ 核心修改 2: 移除預設地點（設為 None）
DEFAULT_LOCATION_ID = None
DEFAULT_LOCATION_NAME = None
TARGET_LAT = None  # 不再有預設座標
TARGET_LON = None

# Global Variables
TRAINED_MODELS = {} 
LAST_OBSERVATION = None 
FEATURE_COLUMNS = []
POLLUTANT_PARAMS = [] 
HOURS_TO_PREDICT = 24

CURRENT_OBSERVATION_AQI = "N/A"
CURRENT_OBSERVATION_TIME = "N/A"

current_location_id = None
current_location_name = None

LOCAL_TZ = "Asia/Taipei"
POLLUTANT_TARGETS = ["pm25", "pm10", "o3", "no2", "so2", "co"] 

AQI_BREAKPOINTS = {
    "pm25": [(0.0, 12.0, 0, 50), (12.1, 35.4, 51, 100), (35.5, 55.4, 101, 150), (55.5, 150.4, 151, 200)],
    "pm10": [(0, 54, 0, 50), (55, 154, 51, 100), (155, 254, 101, 150), (255, 354, 151, 200)],
    "o3": [(0, 54, 0, 50), (55, 70, 51, 100), (71, 85, 101, 150), (86, 105, 151, 200)],
    "co": [(0.0, 4.4, 0, 50), (4.5, 9.4, 51, 100), (9.5, 12.4, 101, 150), (12.5, 15.4, 151, 200)],
    "no2": [(0, 100, 0, 50), (101, 360, 51, 100), (361, 649, 101, 150), (650, 1249, 151, 200)],
    "so2": [(0, 35, 0, 50), (36, 75, 51, 100), (76, 185, 101, 150), (186, 304, 151, 200)],
}

def get_location_meta(location_id: int):
    """Fetches location metadata including the last update time."""
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
    """搜尋最近且數據完整的監測站"""
    V3_LOCATIONS_URL = f"{BASE}/locations"
    radius_meters = radius_km * 1000
    
    params = {
        "coordinates": f"{lat},{lon}",
        "radius": radius_meters,
        "limit": 20,
    }
    
    try:
        r = requests.get(V3_LOCATIONS_URL, headers=HEADERS, params=params, timeout=10)
        r.raise_for_status()
        results = r.json().get("results", [])

        if not results:
            print(f"🚨 [Nearest] 在 {radius_km}km 內未找到測站")
            return None, None, None, None

        print(f"🔍 [Nearest] 找到 {len(results)} 個候選測站，正在評估...")

        best_station = None
        max_params = 0
        
        for idx, station in enumerate(results):
            station_name = station.get("name", "Unknown")
            distance = station.get("distance", 0)
            
            sensors = station.get("sensors", [])
            param_names = []
            
            for sensor in sensors:
                parameter = sensor.get("parameter", {})
                if isinstance(parameter, dict):
                    param_name = parameter.get("name", "").lower()
                    if param_name:
                        param_names.append(param_name)
            
            param_count = len([p for p in param_names if p in TARGET_PARAMS])
            
            last_update = station.get("datetimeLast", {}).get("utc")
            hours_since_update = 999
            if last_update:
                last_update_dt = pd.to_datetime(last_update, utc=True)
                hours_since_update = (pd.Timestamp.now(tz='UTC') - last_update_dt).total_seconds() / 3600
            
            print(f"   [{idx+1}] {station_name}: {param_count} 項目, "
                  f"{hours_since_update:.1f}h 前更新, 距離 {distance/1000:.1f}km")
            
            if param_names:
                unique_params = sorted(set([p for p in param_names if p in TARGET_PARAMS]))
                print(f"       → 監測項目: {', '.join(unique_params)}")
            
            if hours_since_update <= 24 and param_count > max_params:
                max_params = param_count
                best_station = station
            elif hours_since_update <= 24 and param_count == max_params and best_station:
                if distance < best_station.get("distance", 999999):
                    best_station = station
        
        if best_station is None:
            for station in sorted(results, key=lambda s: s.get("distance", 999999)):
                last_update = station.get("datetimeLast", {}).get("utc")
                if last_update:
                    last_update_dt = pd.to_datetime(last_update, utc=True)
                    days_since = (pd.Timestamp.now(tz='UTC') - last_update_dt).days
                    if days_since < 30:
                        best_station = station
                        print(f"⚠️ [Nearest] 選擇距離最近且 {days_since} 天內有更新的測站")
                        break
            
            if best_station is None:
                print("❌ [Nearest] 找不到任何有效測站")
                return None, None, None, None
        
        loc_id = int(best_station["id"])
        loc_name = best_station["name"]
        coords = best_station.get("coordinates", {})
        lat_found = coords.get("latitude", lat)
        lon_found = coords.get("longitude", lon)
        distance = best_station.get("distance", 0)
        
        sensors = best_station.get("sensors", [])
        final_param_names = []
        for sensor in sensors:
            parameter = sensor.get("parameter", {})
            if isinstance(parameter, dict):
                param_name = parameter.get("name", "").lower()
                if param_name in TARGET_PARAMS:
                    final_param_names.append(param_name)

        print(f"✅ [Nearest] 最終選擇: {loc_name} (ID: {loc_id})")
        print(f"   監測項目: {len(set(final_param_names))} 個 ({', '.join(sorted(set(final_param_names)))})")
        print(f"   距離: {distance/1000:.2f}km")
        print(f"   座標: ({lat_found}, {lon_found})")

        return loc_id, loc_name, lat_found, lon_found

    except Exception as e:
        print(f"❌ [Nearest] 搜尋失敗: {e}")
        return None, None, None, None

def get_location_latest_df(location_id: int) -> pd.DataFrame:
    """Fetches the 'latest' values for all parameters at a location."""
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

        return df[["parameter", "value", "units", "ts_utc", "ts_local"]]
    except Exception as e:
        return pd.DataFrame()

def get_parameters_latest_df(location_id: int, target_params) -> pd.DataFrame:
    """Fetches 'latest' value for specific parameters."""
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

cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
openmeteo_client = openmeteo_requests.Client(session=cache_session)

def get_weather_forecast(lat: float, lon: float) -> pd.DataFrame:
    """從 Open-Meteo 獲取天氣預報"""
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ["temperature_2m", "relative_humidity_2m", "surface_pressure"],
        "timezone": "UTC",
        "forecast_days": 2, 
    }
    
    try:
        responses = openmeteo_client.weather_api(url, params=params)
        
        if not responses:
            print("❌ [Weather] Open-Meteo 返回空響應")
            return pd.DataFrame()
             
        response = responses[0]

        if not response.Hourly():
            print("❌ [Weather] 缺少 Hourly 數據")
            return pd.DataFrame()
             
        hourly = response.Hourly()
        
        try:
            interval_seconds = response.Interval()
        except AttributeError:
            interval_seconds = 3600
            print("⚠️ [Weather] 使用預設間隔 3600 秒")
        
        temperature_data = hourly.Variables(0).ValuesAsNumpy()
        humidity_data = hourly.Variables(1).ValuesAsNumpy()
        pressure_data = hourly.Variables(2).ValuesAsNumpy()
        
        try:
            start_timestamp = response.Time()
        except:
            start_timestamp = pd.Timestamp.now(tz='UTC').timestamp()
            print("⚠️ [Weather] 使用當前時間作為起始時間")
        
        data_points = len(temperature_data)
        
        time_series = pd.date_range(
            start=pd.to_datetime(start_timestamp, unit="s", utc=True),
            periods=data_points,
            freq=f'{interval_seconds}s',
            tz='UTC'
        )
        
        df = pd.DataFrame({
            "datetime": time_series,
            "temperature": temperature_data,
            "humidity": humidity_data,
            "pressure": pressure_data,
        })
        
        now_utc = pd.Timestamp.now(tz='UTC').floor('H')
        start_time = now_utc + timedelta(hours=1)
        
        df = df[df['datetime'] >= start_time].head(HOURS_TO_PREDICT).copy()
        
        print(f"✅ [Weather] 成功獲取 {len(df)} 小時天氣預報")
        print(f"   溫度範圍: {df['temperature'].min():.1f}°C ~ {df['temperature'].max():.1f}°C")
        
        return df
        
    except Exception as e:
        print(f"❌ [Weather] 獲取失敗: {e}")
        return pd.DataFrame()

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
    """Fetches the latest observation data from OpenAQ."""
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
    
    df_at_batch = pick_batch_near(df_loc_latest, t_star, TOL_MINUTES_PRIMARY)
    if df_at_batch.empty:
        df_at_batch = pick_batch_near(df_loc_latest, t_star, TOL_MINUTES_FALLBACK)

    have = set(df_at_batch["parameter"].str.lower().tolist()) if not df_at_batch.empty else set()

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

    df_all["dt_diff"] = (df_all["ts_utc"] - t_star).abs()
    df_all = df_all.sort_values(["parameter", "dt_diff", "ts_utc"], ascending=[True, True, False])
    df_all = df_all.drop_duplicates(subset=["parameter"], keep="first")
    df_all = df_all.drop(columns=["dt_diff", "units", "ts_local"])

    observation = df_all.pivot_table(
        index='ts_utc', columns='parameter', values='value', aggfunc='first'
    ).reset_index()
    observation = observation.rename(columns={'ts_utc': 'datetime'})
    
    if not observation.empty:
        observation['aqi'] = observation.apply(
            lambda row: calculate_aqi(row, target_params, is_pred=False), axis=1
        )
        
    if not observation.empty:
        observation['datetime'] = pd.to_datetime(observation['datetime'])
        if observation['datetime'].dt.tz is None:
            observation['datetime'] = observation['datetime'].dt.tz_localize('UTC')
        else:
            observation['datetime'] = observation['datetime'].dt.tz_convert('UTC')

    return observation

def calculate_aqi_sub_index(param: str, concentration: float) -> float:
    """Calculates the AQI sub-index."""
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
    """Calculates the final AQI."""
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

def predict_future_multi(models, last_data, feature_cols, pollutant_params, hours=24, weather_df=None):
    """多污染物預測（修復版：處理缺失污染物）"""
    predictions = []

    last_data['datetime'] = pd.to_datetime(last_data['datetime'])
    if last_data['datetime'].dt.tz is None:
        last_data['datetime'] = last_data['datetime'].dt.tz_localize('UTC')
    else:
        last_data['datetime'] = last_data['datetime'].dt.tz_convert('UTC')
        
    last_datetime_aware = last_data['datetime'].iloc[0]
    
    # ✅ 修改：用 0 填充缺失值，而非 NaN
    current_data_dict = {}
    available_pollutants = []
    
    print("\n🔍 [Init] 檢查觀測數據特徵:")
    for col in feature_cols:
        if col in last_data.columns and not last_data[col].empty:
            val = last_data[col].iloc[0]
            if pd.notna(val):
                current_data_dict[col] = float(val)
                # 記錄哪些污染物有數據
                if col.endswith('_lag_1h') and not col.startswith('aqi'):
                    param = col.replace('_lag_1h', '')
                    if param in pollutant_params:
                        available_pollutants.append(param)
                print(f"   ✅ {col}: {val:.2f}")
            else:
                current_data_dict[col] = 0.0
                print(f"   ⚠️ {col}: NaN → 0")
        else:
            current_data_dict[col] = 0.0
            print(f"   ❌ {col}: 缺失 → 0")
    
    print(f"\n📊 [Init] 可用污染物: {available_pollutants}")

    weather_feature_names = ['temperature', 'humidity', 'pressure']
    weather_feature_names = [col for col in weather_feature_names if col in feature_cols]
    has_weather = bool(weather_feature_names)

    weather_dict = {}
    if weather_df is not None and not weather_df.empty:
        try:
            if weather_df['datetime'].dt.tz is None:
                weather_df['datetime'] = weather_df['datetime'].dt.tz_localize('UTC')
            else:
                weather_df['datetime'] = weather_df['datetime'].dt.tz_convert('UTC')
            
            weather_df = weather_df.set_index('datetime')
            weather_dict = weather_df.to_dict(orient='index')
            print(f"✅ [Weather] 載入 {len(weather_dict)} 小時天氣數據")
        except Exception as e:
            print(f"⚠️ [Weather] 天氣數據處理失敗: {e}")
            weather_dict = {}
    else:
        print("⚠️ [Weather] 無天氣數據，將使用默認值")

    total_predictions = 0
    skipped_reasons = {}

    try:
        for h in range(hours):
            future_time = last_datetime_aware + timedelta(hours=h + 1)
            pred_features = current_data_dict.copy()

            # 時間特徵
            pred_features['hour'] = future_time.hour
            pred_features['day_of_week'] = future_time.dayofweek
            pred_features['month'] = future_time.month
            pred_features['day_of_year'] = future_time.timetuple().tm_yday 
            pred_features['is_weekend'] = int(future_time.dayofweek in [5, 6])
            pred_features['hour_sin'] = np.sin(2 * np.pi * future_time.hour / 24)
            pred_features['hour_cos'] = np.cos(2 * np.pi * future_time.hour / 24)
            pred_features['day_sin'] = np.sin(2 * np.pi * pred_features['day_of_year'] / 365)
            pred_features['day_cos'] = np.cos(2 * np.pi * pred_features['day_of_year'] / 365)

            # 天氣特徵
            if has_weather and weather_dict:
                weather_key = future_time.replace(minute=0, second=0, microsecond=0)
                
                if weather_key in weather_dict:
                    forecast = weather_dict[weather_key]
                    for w_col in weather_feature_names:
                        if w_col in forecast and pd.notna(forecast[w_col]):
                            pred_features[w_col] = forecast[w_col]
                            current_data_dict[w_col] = forecast[w_col]
                else:
                    # 使用前一小時的天氣數據
                    for w_col in weather_feature_names:
                        pred_features[w_col] = current_data_dict.get(w_col, 0.0)

            current_prediction_row = {'datetime': future_time}
            new_pollutant_values = {}

            # ✅ 對每個污染物單獨預測
            for param in pollutant_params:
                if param not in models:
                    if param not in skipped_reasons:
                        skipped_reasons[param] = "模型不存在"
                    continue

                # ✅ 檢查該污染物是否有初始數據
                param_lag_col = f'{param}_lag_1h'
                if param_lag_col not in pred_features or pred_features[param_lag_col] == 0.0:
                    if param not in skipped_reasons:
                        skipped_reasons[param] = "缺少初始觀測值"
                    continue

                model = models[param]
                
                # 準備輸入，用 0 填充所有 NaN
                pred_input_list = []
                for col in feature_cols:
                    val = pred_features.get(col, 0.0)
                    pred_input_list.append(0.0 if pd.isna(val) else float(val))

                try:
                    pred_input = np.array(pred_input_list, dtype=np.float64).reshape(1, -1)
                    pred = model.predict(pred_input)[0]
                    pred = max(0, pred)

                    current_prediction_row[f'{param}_pred'] = pred
                    new_pollutant_values[param] = pred
                    total_predictions += 1
                    
                except Exception as e:
                    if param not in skipped_reasons:
                        skipped_reasons[param] = f"預測錯誤: {str(e)[:50]}"
                    continue

            if new_pollutant_values:
                predicted_aqi = calculate_aqi(pd.Series(current_prediction_row), pollutant_params, is_pred=True)
                current_prediction_row['aqi_pred'] = predicted_aqi
                new_pollutant_values['aqi'] = predicted_aqi
                predictions.append(current_prediction_row)

                # 更新 lag 特徵
                for param in list(new_pollutant_values.keys()):
                    if param == 'aqi':
                        lag_col = 'aqi_lag_1h'
                    else:
                        lag_col = f'{param}_lag_1h'
                    
                    if lag_col in current_data_dict:
                        current_data_dict[lag_col] = new_pollutant_values[param]

        print(f"\n✅ [Predict] 成功生成 {len(predictions)} 個預測時間點")
        print(f"   模型調用總次數: {total_predictions}")
        
        if skipped_reasons:
            print(f"\n⚠️ [Predict] 跳過的污染物:")
            for param, reason in skipped_reasons.items():
                print(f"   - {param}: {reason}")

    except Exception as e:
        print(f"❌ [Predict] 預測錯誤: {e}")
        import traceback
        traceback.print_exc()

    return pd.DataFrame(predictions)

def load_models_and_metadata():
    global TRAINED_MODELS, LAST_OBSERVATION, FEATURE_COLUMNS, POLLUTANT_PARAMS

    if not os.path.exists(MODELS_DIR) or not os.path.exists(META_PATH):
        print("🚨 [Load] 模型資料夾或 metadata 檔案不存在")
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
                print(f"❌ [Load] 找不到模型檔案: {model_path}")
                params_to_remove.append(param)
        
        for param in params_to_remove:
            POLLUTANT_PARAMS.remove(param)

        if TRAINED_MODELS:
            print(f"✅ [Load] 成功載入 {len(TRAINED_MODELS)} 個模型")
            print(f"   監測項目: {', '.join(POLLUTANT_PARAMS)}")
            print(f"   特徵數量: {len(FEATURE_COLUMNS)}")
        else:
            print("🚨 [Load] 沒有成功載入任何模型")

    except Exception as e:
        print(f"❌ [Load] 模型載入失敗: {e}") 
        import traceback
        traceback.print_exc()

app = Flask(__name__)

with app.app_context():
    load_models_and_metadata() 

@app.route('/')
def index():
    """主路由"""
    global CURRENT_OBSERVATION_AQI, CURRENT_OBSERVATION_TIME
    global current_location_id, current_location_name
    
    try:
        print("\n" + "="*60)
        print("🚀 [Request] 開始處理新請求")
        print("="*60)

        # ✅ 核心修改 3: 必須提供座標，否則報錯
        lat_param = request.args.get('lat', type=float)
        lon_param = request.args.get('lon', type=float)

        if lat_param is None or lon_param is None:
            print("❌ [Request] 缺少 lat/lon 參數")
            return render_template(
                'index.html',
                max_aqi="ERROR",
                aqi_predictions=[],
                city_name="錯誤：需要提供座標",
                current_obs_time="N/A",
                is_fallback=True,
                error_message="請允許瀏覽器定位或手動提供座標參數"
            )

        user_lat, user_lon = lat_param, lon_param
        print(f"📍 [Request] 使用座標 → lat={user_lat}, lon={user_lon}")

        # ✅ 核心修改 4: 找不到測站直接報錯，不回退
        loc_id, loc_name, lat_found, lon_found = get_nearest_location(user_lat, user_lon)
        
        if loc_id is None:
            print("❌ [Station] 找不到任何測站")
            return render_template(
                'index.html',
                max_aqi="N/A",
                aqi_predictions=[],
                city_name=f"({user_lat:.4f}, {user_lon:.4f})",
                current_obs_time="N/A",
                is_fallback=True,
                error_message="您所在區域附近 25km 內沒有可用的空氣品質監測站"
            )

        current_location_id = loc_id
        current_location_name = loc_name
        station_lat, station_lon = lat_found, lon_found

        print(f"\n🌤️  [Weather] 獲取天氣預報 ({station_lat}, {station_lon})")
        weather_forecast_df = get_weather_forecast(station_lat, station_lon)

        print(f"\n📊 [Observation] 獲取觀測數據 (測站 ID: {current_location_id})")
        current_observation_raw = fetch_latest_observation_data(current_location_id, POLLUTANT_TARGETS)

        if not current_observation_raw.empty:
            print(f"✅ [Observation] 獲得觀測數據")
            print(current_observation_raw.to_string(index=False))
        else:
            print("🚨 [Observation] 無觀測數據")

        if not current_observation_raw.empty and 'aqi' in current_observation_raw.columns:
            obs_aqi_val = current_observation_raw['aqi'].iloc[0]
            obs_time_val = current_observation_raw['datetime'].iloc[0]
            CURRENT_OBSERVATION_AQI = int(obs_aqi_val) if pd.notna(obs_aqi_val) else "N/A"
            if pd.notna(obs_time_val):
                if obs_time_val.tz is None:
                    obs_time_val = obs_time_val.tz_localize('UTC')
                CURRENT_OBSERVATION_TIME = obs_time_val.tz_convert(LOCAL_TZ).strftime('%Y-%m-%d %H:%M')
            print(f"📍 [Current AQI] {CURRENT_OBSERVATION_AQI} @ {CURRENT_OBSERVATION_TIME}")
        else:
            CURRENT_OBSERVATION_AQI = "N/A"
            CURRENT_OBSERVATION_TIME = "N/A"

        observation_for_prediction = None
        is_valid_for_prediction = False
        is_fallback_mode = True

        if not current_observation_raw.empty:
            observation_for_prediction = current_observation_raw.copy()
            observation_for_prediction['datetime'] = pd.to_datetime(observation_for_prediction['datetime'])
            if observation_for_prediction['datetime'].dt.tz is None:
                observation_for_prediction['datetime'] = observation_for_prediction['datetime'].dt.tz_localize('UTC')
            else:
                observation_for_prediction['datetime'] = observation_for_prediction['datetime'].dt.tz_convert('UTC')
            is_valid_for_prediction = True

        max_aqi = CURRENT_OBSERVATION_AQI
        aqi_predictions = []

        if TRAINED_MODELS and POLLUTANT_PARAMS and is_valid_for_prediction and observation_for_prediction is not None:
            print(f"\n🔮 [Prediction] 開始預測未來 {HOURS_TO_PREDICT} 小時")
            try:
                future_predictions = predict_future_multi(
                    TRAINED_MODELS,
                    observation_for_prediction,
                    FEATURE_COLUMNS,
                    POLLUTANT_PARAMS,
                    hours=HOURS_TO_PREDICT,
                    weather_df=weather_forecast_df
                )

                if not future_predictions.empty:
                    future_predictions['datetime_local'] = future_predictions['datetime'].dt.tz_convert(LOCAL_TZ)
                    predictions_df = future_predictions[['datetime_local', 'aqi_pred']].copy()
                    
                    if predictions_df['datetime_local'].duplicated().any():
                        print("⚠️ [Predict] 移除重複預測時間")
                        predictions_df = predictions_df.drop_duplicates(subset=['datetime_local'], keep='first')
                    
                    max_aqi_val = predictions_df['aqi_pred'].max()
                    max_aqi = int(max_aqi_val) if pd.notna(max_aqi_val) else CURRENT_OBSERVATION_AQI
                    
                    predictions_df['aqi'] = predictions_df['aqi_pred'].apply(
                        lambda x: int(x) if pd.notna(x) else "N/A"
                    )
                    
                    aqi_predictions = [
                        {'time': item['datetime_local'].strftime('%Y-%m-%d %H:%M'), 'aqi': item['aqi']}
                        for item in predictions_df.to_dict(orient='records')
                    ]
                    
                    if aqi_predictions:
                        is_fallback_mode = False
                        print(f"✅ [Predict] 預測成功")
                        
            except Exception as e:
                print(f"❌ [Predict] 預測失敗: {e}")
                import traceback
                traceback.print_exc()

        if is_fallback_mode:
            print("🚨 [Fallback] 僅顯示觀測值")
            if CURRENT_OBSERVATION_AQI != "N/A":
                aqi_predictions = [{
                    'time': CURRENT_OBSERVATION_TIME,
                    'aqi': CURRENT_OBSERVATION_AQI,
                    'is_obs': True
                }]

        print(f"\n📊 [Final] max_aqi={max_aqi}, predictions={len(aqi_predictions)}, fallback={is_fallback_mode}")
        print("="*60 + "\n")

        return render_template(
            'index.html',
            max_aqi=max_aqi,
            aqi_predictions=aqi_predictions,
            city_name=current_location_name,
            current_obs_time=CURRENT_OBSERVATION_TIME,
            is_fallback=is_fallback_mode
        )
        
    except Exception as e:
        print(f"❌ [Route] 嚴重錯誤: {e}")
        import traceback
        traceback.print_exc()
        
        return render_template(
            'index.html',
            max_aqi="ERROR",
            aqi_predictions=[],
            city_name="系統錯誤",
            current_obs_time="N/A",
            is_fallback=True
        )

@app.route('/health')
def health_check():
    """健康檢查端點"""
    import sys
    return {
        'status': 'ok',
        'models_loaded': len(TRAINED_MODELS),
        'pollutants': POLLUTANT_PARAMS,
        'features': len(FEATURE_COLUMNS),
        'python_version': sys.version,
        'simplified_mode': 'Only 1-hour lag features'
    }

if __name__ == '__main__':
    app.run(debug=True)
