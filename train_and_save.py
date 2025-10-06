# train_and_save.py - 單小時預測版本

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
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from meteostat import Point, Hourly, units

warnings.filterwarnings('ignore')

MODELS_DIR = 'models'
os.makedirs(MODELS_DIR, exist_ok=True)

# =================================================================
# 🔥 核心修改：簡化特徵配置（只需 1 小時數據）
# =================================================================
API_KEY = "fb579916623e8483cd85344b14605c3109eea922202314c44b87a2df3b1fff77"
API_BASE_URL = "https://api.openaq.org/v3/"
POLLUTANT_TARGETS = ["pm25", "pm10", "o3", "no2", "so2", "co"]
LOCAL_TZ = "Asia/Taipei"
MIN_DATA_THRESHOLD = 100

# ✅ 修改：只使用 1 小時 lag，移除長期 lag
LAG_HOURS = [1]  # 只保留 1 小時前的數據
ROLLING_WINDOWS = []  # 完全移除滾動窗口特徵

DAYS_TO_FETCH = 90 
N_ESTIMATORS = 150

# AQI 計算（保持不變）
AQI_BREAKPOINTS = {
    "pm25": [(0.0, 12.0, 0, 50), (12.1, 35.4, 51, 100), (35.5, 55.4, 101, 150), (55.5, 150.4, 151, 200)],
    "pm10": [(0, 54, 0, 50), (55, 154, 51, 100), (155, 254, 101, 150), (255, 354, 151, 200)],
    "o3": [(0, 54, 0, 50), (55, 70, 51, 100), (71, 85, 101, 150), (86, 105, 151, 200)],
    "co": [(0.0, 4.4, 0, 50), (4.5, 9.4, 51, 100), (9.5, 12.4, 101, 150), (12.5, 15.4, 151, 200)],
    "no2": [(0, 100, 0, 50), (101, 360, 51, 100), (361, 649, 101, 150), (650, 1249, 151, 200)],
    "so2": [(0, 35, 0, 50), (36, 75, 51, 100), (76, 185, 101, 150), (186, 304, 151, 200)],
}

GLOBAL_TRAINING_LOCATIONS = [
    (22.6273, 120.3014, "Kaohsiung, TW"),     
    (28.7041, 77.1025, "Delhi, IN"),         
    (40.7128, -74.0060, "New York, US"),       
    (43.6532, -79.3832, "Toronto, CA"),        
    (52.3676, 4.9041, "Amsterdam, NL"),       
    (48.8566, 2.3522, "Paris, FR"),            
    (39.9042, 116.4074, "Beijing, CN"),        
    (35.6895, 139.6917, "Tokyo, JP"),          
]

# AQI 計算函數（保持不變）
def calculate_aqi_sub_index(param: str, concentration: float) -> float:
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
    sub_indices = []
    for p in params:
        col_name = f'{p}_pred' if f'{p}_pred' in row else f'{p}_value'
        if col_name in row and not pd.isna(row[col_name]):
            sub_index = calculate_aqi_sub_index(p, row[col_name])
            sub_indices.append(sub_index)
    if not sub_indices:
        return np.nan
    return int(np.max(sub_indices))

# OpenAQ 函數（保持不變）
def sanitize_filename(name: str) -> str:
    return re.sub(r'[\\/:"*?<>|]+', '_', name)

def get_nearest_station(lat, lon, radius=20000, limit=50, days=DAYS_TO_FETCH):
    url = f"{API_BASE_URL}locations"
    headers = {"X-API-Key": API_KEY}
    params = {"coordinates": f"{lat},{lon}", "radius": radius, "limit": limit}
    try:
        resp = requests.get(url, headers=headers, params=params)
        resp.raise_for_status()
        j = resp.json()
    except Exception as e:
        return None
    if "results" not in j or not j["results"]:
        return None
    df = pd.json_normalize(j["results"])
    if "datetimeLast.utc" not in df.columns:
        return None
    df["datetimeLast.utc"] = pd.to_datetime(df["datetimeLast.utc"], errors="coerce", utc=True)
    now = pd.Timestamp.utcnow()
    cutoff = now - pd.Timedelta(days=days)
    df = df[(df["datetimeLast.utc"] >= cutoff) & (df["datetimeLast.utc"] <= now)]
    if df.empty:
        return None
    nearest = df.sort_values("distance").iloc[0]
    return nearest.to_dict()

def get_station_sensors(station_id):
    url = f"{API_BASE_URL}locations/{station_id}/sensors"
    headers = {"X-API-Key": API_KEY}
    try:
        resp = requests.get(url, headers=headers, params={"limit":1000})
        resp.raise_for_status()
        j = resp.json()
        return j.get("results", [])
    except Exception as e:
        return []

def _extract_datetime_from_measurement(item: dict):
    candidates = [("period", "datetimeFrom", "utc"), ("date", "utc"), ("datetime",)]
    for path in candidates:
        cur = item
        ok = True
        for k in path:
            if isinstance(cur, dict) and k in cur:
                cur = cur[k]
            else:
                ok = False
                break
        if ok and cur:
            return cur
    return None

def fetch_sensor_data(sensor_id, param_name, limit=500, days=DAYS_TO_FETCH):
    url = f"{API_BASE_URL}sensors/{sensor_id}/measurements"
    headers = {"X-API-Key": API_KEY}
    now = datetime.datetime.now(datetime.timezone.utc)
    date_from = (now - datetime.timedelta(days=days)).isoformat().replace("+00:00", "Z")
    params = {"limit": limit, "date_from": date_from}
    try:
        resp = requests.get(url, headers=headers, params=params)
        resp.raise_for_status()
        j = resp.json()
        results = j.get("results", [])
    except Exception as e:
        return pd.DataFrame()
    rows = []
    for r in results:
        dt_str = _extract_datetime_from_measurement(r)
        try:
            ts = pd.to_datetime(dt_str, utc=True)
        except Exception:
            ts = pd.NaT
        rows.append({"datetime": ts, param_name: r.get("value")})
    df = pd.DataFrame(rows).dropna(subset=["datetime"])
    if df.empty:
        return pd.DataFrame()
    df = df.sort_values("datetime", ascending=False).drop_duplicates(subset=["datetime"])
    return df

def get_all_target_data(station_id, target_params, days_to_fetch):
    sensors = get_station_sensors(station_id)
    sensor_map = {s.get("parameter", {}).get("name", "").lower(): s.get("id") for s in sensors}
    all_dfs = []
    found_params = []
    for param in target_params:
        sensor_id = sensor_map.get(param)
        if sensor_id:
            df_param = fetch_sensor_data(sensor_id, param, limit=500, days=days_to_fetch)
            if not df_param.empty:
                df_param.rename(columns={param: f'{param}_value'}, inplace=True)
                all_dfs.append(df_param)
                found_params.append(param)
    if not all_dfs:
        return pd.DataFrame(), []
    merged_df = all_dfs[0]
    for i in range(1, len(all_dfs)):
        merged_df = pd.merge(merged_df, all_dfs[i], on='datetime', how='outer')
    return merged_df, found_params

# WeatherCrawler（保持不變）
class WeatherCrawler:
    def __init__(self, lat, lon):
        self.point = Point(lat, lon)
        self.weather_cols = {
            'temp': 'temperature',
            'rhum': 'humidity',
            'pres': 'pressure',
        }

    def fetch_and_merge_weather(self, air_quality_df: pd.DataFrame):
        if air_quality_df.empty:
            return air_quality_df
        if air_quality_df['datetime'].dt.tz is None:
            air_quality_df['datetime'] = air_quality_df['datetime'].dt.tz_localize('UTC')
        start_time_utc_aware = air_quality_df['datetime'].min()
        end_time_utc_aware = air_quality_df['datetime'].max()
        start_dt = start_time_utc_aware.tz_convert(None).to_pydatetime()
        end_dt = end_time_utc_aware.tz_convert(None).to_pydatetime()
        try:
            data = Hourly(self.point, start_dt, end_dt)
            weather_data = data.fetch()
        except Exception as e:
            weather_data = pd.DataFrame()
        if weather_data.empty:
            empty_weather = pd.DataFrame({'datetime': air_quality_df['datetime'].unique()})
            for col in self.weather_cols.values():
                empty_weather[col] = np.nan
            return pd.merge(air_quality_df, empty_weather, on='datetime', how='left')
        weather_data = weather_data.reset_index()
        weather_data.rename(columns={'time': 'datetime'}, inplace=True)
        weather_data = weather_data.rename(columns=self.weather_cols)
        weather_data = weather_data[list(self.weather_cols.values()) + ['datetime']]
        weather_data['datetime'] = weather_data['datetime'].dt.tz_localize('UTC')
        merged_df = pd.merge(air_quality_df, weather_data, on='datetime', how='left')
        weather_cols_list = list(self.weather_cols.values())
        merged_df[weather_cols_list] = merged_df[weather_cols_list].fillna(method='ffill').fillna(method='bfill')
        return merged_df

    def get_weather_feature_names(self):
        return list(self.weather_cols.values())

# ✅ 核心修改：簡化特徵工程（只需要 1 小時的 lag）
def _preprocess_and_feature_engineer(df_input: pd.DataFrame, pollutant_params: list, weather_feature_names: list) -> pd.DataFrame:
    df = df_input.copy()
    value_cols = [f'{p}_value' for p in pollutant_params]
    all_data_cols = value_cols + weather_feature_names

    df.set_index('datetime', inplace=True)
    df = df[value_cols + weather_feature_names].resample('H').mean()
    df.reset_index(inplace=True)
    df = df.dropna(how='all', subset=all_data_cols)

    df['aqi_value'] = df.apply(lambda row: calculate_aqi(row, pollutant_params), axis=1)
    df = df.dropna(subset=all_data_cols + ['aqi_value']).reset_index(drop=True)
    
    if len(df) <= 1:  # 只需要 1 小時的歷史數據
        return pd.DataFrame()

    # 時間特徵
    df['hour'] = df['datetime'].dt.hour
    df['day_of_week'] = df['datetime'].dt.dayofweek
    df['month'] = df['datetime'].dt.month
    df['day_of_year'] = df.index
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)

    df = df.sort_values('datetime')
    
    # ✅ 只創建 1 小時 lag 特徵
    feature_base_cols = value_cols + ['aqi_value']
    for col_name in feature_base_cols:
        param = col_name.replace('_value', '')
        df[f'{param}_lag_1h'] = df[col_name].shift(1)  # 只有 1 小時 lag

    # ✅ 完全移除滾動窗口特徵（ROLLING_WINDOWS = []）
    
    df = df.dropna().reset_index(drop=True)
    return df

def train_and_save_models(locations: list, days_to_fetch: int):
    print(f"🔥 [單小時模型] 開始訓練（只需 1 小時歷史數據）...")

    all_df = []
    all_found_params = set()
    weather_feature_names = WeatherCrawler(0, 0).get_weather_feature_names()

    for lat, lon, name in locations:
        print(f"\n--- 🌍 處理地點: {name} ---")
        weather = WeatherCrawler(lat, lon)
        try:
            station = get_nearest_station(lat, lon, days=days_to_fetch)
            if not station:
                print(f"🚨 [{name}] 未找到測站，跳過")
                continue
            print(f"✅ [{name}] 找到測站: {station['name']}")
            df_raw, found_params = get_all_target_data(station["id"], POLLUTANT_TARGETS, days_to_fetch)
            print(f"   [{name}] 原始數據: {len(df_raw)} 點")
            if df_raw.empty or len(df_raw) < MIN_DATA_THRESHOLD:
                print(f"🚨 [{name}] 數據不足，跳過")
                continue
            df = weather.fetch_and_merge_weather(df_raw.copy())
            df_processed = _preprocess_and_feature_engineer(df, found_params, weather_feature_names)
            if not df_processed.empty:
                all_df.append(df_processed)
                all_found_params.update(found_params)
                print(f"📊 [{name}] 訓練數據: {len(df_processed)} 小時")
            else:
                print(f"🚨 [{name}] 處理後數據不足")
        except Exception as e:
            print(f"❌ [{name}] 失敗: {e}")
            continue

    if not all_df:
        raise ValueError("所有地點失敗")
    
    final_df = pd.concat(all_df, ignore_index=True)
    final_df = final_df.sort_values('datetime').reset_index(drop=True)
    
    print(f"\n📊 總訓練數據: {len(final_df)} 小時")
    print(f"🎯 訓練污染物: {POLLUTANT_TARGETS}")

    LAST_OBSERVATION = final_df.iloc[-1:].to_json(orient='records', date_format='iso')

    # ✅ 簡化後的特徵欄位（無長期 lag，無滾動窗口）
    base_time_features = ['hour', 'day_of_week', 'month', 'is_weekend', 'hour_sin', 'hour_cos', 'day_sin', 'day_cos']
    air_quality_features = []
    for param in POLLUTANT_TARGETS + ['aqi']:
        air_quality_features.append(f'{param}_lag_1h')  # 只有 1h lag

    FEATURE_COLUMNS = weather_feature_names + base_time_features + air_quality_features
    FEATURE_COLUMNS = [col for col in FEATURE_COLUMNS if col in final_df.columns]

    print(f"📋 特徵數量: {len(FEATURE_COLUMNS)}")

    split_idx = int(len(final_df) * 0.8)
    Y_cols = [f'{p}_value' for p in POLLUTANT_TARGETS]
    final_df.dropna(subset=Y_cols, inplace=True)
    
    if len(final_df) == 0:
        raise ValueError("清理後無數據")
    
    split_idx = int(len(final_df) * 0.8)
    X = final_df[FEATURE_COLUMNS]
    Y = {param: final_df[f'{param}_value'] for param in POLLUTANT_TARGETS if f'{param}_value' in final_df.columns}
    X_train = X[:split_idx]

    print(f"⏳ 開始訓練 {len(Y)} 個模型...")
    TRAINED_MODELS = {}
    
    for param, Y_series in Y.items():
        Y_train = Y_series[:split_idx]
        print(f"   訓練 {param}...")
        xgb_model = xgb.XGBRegressor(n_estimators=N_ESTIMATORS, max_depth=7, learning_rate=0.08, random_state=42, n_jobs=-1)
        xgb_model.fit(X_train, Y_train)
        TRAINED_MODELS[param] = xgb_model
        model_path = os.path.join(MODELS_DIR, f'{param}_model.json')
        xgb_model.save_model(model_path)
        print(f"   ✅ {param} 已儲存")

    metadata = {
        'pollutant_params': POLLUTANT_TARGETS,
        'feature_columns': FEATURE_COLUMNS,
        'last_observation_json': LAST_OBSERVATION
    }
    with open(os.path.join(MODELS_DIR, 'model_meta.json'), 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=4)

    print("✅ 所有模型儲存完成（單小時版本）")

if __name__ == '__main__':
    try:
        train_and_save_models(GLOBAL_TRAINING_LOCATIONS, DAYS_TO_FETCH)
    except Exception as e:
        print(f"❌ 訓練失敗: {e}")
