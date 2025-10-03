# train_and_save.py - 供本地訓練使用

# =================================================================
# 導入所有必要的庫
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
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from meteostat import Point, Hourly, units

# 忽略警告
warnings.filterwarnings('ignore')

# 創建一個 models 資料夾來儲存模型
MODELS_DIR = 'models'
os.makedirs(MODELS_DIR, exist_ok=True)

# =================================================================
# 複製 app.py 中的常數設定
# =================================================================
API_KEY = "fb579916623e8483cd85344b14605c3109eea922202314c44b87a2df3b1fff77"
API_BASE_URL = "https://api.openaq.org/v3/"
POLLUTANT_TARGETS = ["pm25", "pm10", "o3", "no2", "so2", "co"]
LOCAL_TZ = "Asia/Taipei"
MIN_DATA_THRESHOLD = 100
LAG_HOURS = [1, 2, 3, 6, 12, 24]
ROLLING_WINDOWS = [6, 12, 24]
# 【修改】設定為 90 天，提高尋找活躍測站的成功率
DAYS_TO_FETCH = 90 
N_ESTIMATORS = 150

# =================================================================
# 【核心修改】全球訓練地點列表 (8 個地點，替換柏林為阿姆斯特丹)
# =================================================================
# train_and_save.py 頂部常數部分

# ... (DAYS_TO_FETCH = 90 不變) ...

# 【核心修改】全球訓練地點列表
# train_and_save.py 頂部常數部分

GLOBAL_TRAINING_LOCATIONS = [
    # 亞洲 A：高濕度/南亞污染 (高數據量)
    (22.6273, 120.3014, "Kaohsiung, TW"),     
    (28.7041, 77.1025, "Delhi, IN"),         
    
    # 北美洲 (高監測標準)
    (40.7128, -74.0060, "New York, US"),       
    (43.6532, -79.3832, "Toronto, CA"),        
    
    # 歐洲 (數據穩定)
    (52.3676, 4.9041, "Amsterdam, NL"),       
    (48.8566, 2.3522, "Paris, FR"),            
    
    # 亞洲 B：**極致數據完整點** (替換失敗點，確保成功)
    (39.9042, 116.4074, "Beijing, CN"),        # 中國北京
    
    # 亞洲 C：東亞交通/工業 (數據穩定)
    (35.6895, 139.6917, "Tokyo, JP"),          # 日本東京
]


# 簡化的 AQI 分級表 (保持不變)
AQI_BREAKPOINTS = {
    "pm25": [(0.0, 12.0, 0, 50), (12.1, 35.4, 51, 100), (35.5, 55.4, 101, 150), (55.5, 150.4, 151, 200)],
    "pm10": [(0, 54, 0, 50), (55, 154, 51, 100), (155, 254, 101, 150), (255, 354, 151, 200)],
    "o3": [(0, 54, 0, 50), (55, 70, 51, 100), (71, 85, 101, 150), (86, 105, 151, 200)],
    "co": [(0.0, 4.4, 0, 50), (4.5, 9.4, 51, 100), (9.5, 12.4, 101, 150), (12.5, 15.4, 151, 200)],
    "no2": [(0, 100, 0, 50), (101, 360, 51, 100), (361, 649, 101, 150), (650, 1249, 151, 200)],
    "so2": [(0, 35, 0, 50), (36, 75, 51, 100), (76, 185, 101, 150), (186, 304, 151, 200)],
}

# =================================================================
# AQI 輔助函式 (保持不變)
# =================================================================
def calculate_aqi_sub_index(param: str, concentration: float) -> float:
    """計算單一污染物濃度對應的 AQI 子指數 (I)"""
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
    """根據多個污染物濃度計算最終 AQI (取最大子指數)"""
    sub_indices = []
    for p in params:
        col_name = f'{p}_pred' if f'{p}_pred' in row else f'{p}_value'
        if col_name in row and not pd.isna(row[col_name]):
            sub_index = calculate_aqi_sub_index(p, row[col_name])
            sub_indices.append(sub_index)

    if not sub_indices:
        return np.nan

    return int(np.max(sub_indices))


# =================================================================
# OpenAQ 輔助函式 (保持不變)
# =================================================================
def sanitize_filename(name: str) -> str:
    return re.sub(r'[\\/:"*?<>|]+', '_', name)

def get_nearest_station(lat, lon, radius=20000, limit=50, days=DAYS_TO_FETCH):
    """ 找離 (lat,lon) 最近且最近 days 內有更新的測站 """
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
    # 這裡使用 DAYS_TO_FETCH=90 天來篩選活躍測站
    df = df[(df["datetimeLast.utc"] >= cutoff) & (df["datetimeLast.utc"] <= now)] 
    if df.empty:
        return None

    nearest = df.sort_values("distance").iloc[0]
    return nearest.to_dict()

def get_station_sensors(station_id):
    """ 使用 /locations/{id}/sensors 取得 sensors 列表 """
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
    """ 嘗試從 measurement 物件抽出時間字串 """
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
    """ 擷取 sensor 的時間序列 """
    url = f"{API_BASE_URL}sensors/{sensor_id}/measurements"
    headers = {"X-API-Key": API_KEY}
    now = datetime.datetime.now(datetime.timezone.utc)
    date_from = (now - datetime.timedelta(days=days)).isoformat().replace("+00:00", "Z")
    # 這裡的 limit=500 是 OpenAQ API 的硬限制
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
    """獲取所有目標污染物數據並合併"""
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

# =================================================================
# WeatherCrawler 類 (保持不變)
# =================================================================
class WeatherCrawler:
    """Meteostat 小時級天氣數據爬蟲與整合"""

    def __init__(self, lat, lon):
        self.point = Point(lat, lon)
        self.weather_cols = {
            'temp': 'temperature',
            'rhum': 'humidity',
            'pres': 'pressure',
        }

    def fetch_and_merge_weather(self, air_quality_df: pd.DataFrame):
        """根據空氣品質數據的時間範圍，從 Meteostat 獲取小時級天氣數據並合併。"""
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

        merged_df = pd.merge(
            air_quality_df,
            weather_data,
            on='datetime',
            how='left'
        )

        weather_cols_list = list(self.weather_cols.values())
        merged_df[weather_cols_list] = merged_df[weather_cols_list].fillna(method='ffill').fillna(method='bfill')

        return merged_df

    def get_weather_feature_names(self):
        return list(self.weather_cols.values())


# =================================================================
# 特徵工程輔助函式 (保持不變)
# =================================================================
def _preprocess_and_feature_engineer(df_input: pd.DataFrame, pollutant_params: list, weather_feature_names: list) -> pd.DataFrame:
    """處理單一地點的數據、重採樣、計算 AQI 和所有特徵。"""
    
    df = df_input.copy()
    value_cols = [f'{p}_value' for p in pollutant_params]
    all_data_cols = value_cols + weather_feature_names

    # 重採樣到小時
    df.set_index('datetime', inplace=True)
    df = df[value_cols + weather_feature_names].resample('H').mean()
    df.reset_index(inplace=True)
    df = df.dropna(how='all', subset=all_data_cols)

    # 計算歷史 AQI
    df['aqi_value'] = df.apply(lambda row: calculate_aqi(row, pollutant_params), axis=1)

    # 移除任一污染物或天氣數據為 NaN 的行 (確保模型輸入完整)
    df = df.dropna(subset=all_data_cols + ['aqi_value']).reset_index(drop=True)
    
    if len(df) <= max(LAG_HOURS):
        return pd.DataFrame()

    # 特徵工程
    df['hour'] = df['datetime'].dt.hour
    df['day_of_week'] = df['datetime'].dt.dayofweek
    df['month'] = df['datetime'].dt.month
    df['day_of_year'] = df.index # 使用簡單的行索引作為年內天數的替代
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    # 由於我們沒有 365 天數據，這裡的 Day_sin/cos 只能基於相對索引，但仍保留
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365) 
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)

    df = df.sort_values('datetime')
    feature_base_cols = value_cols + ['aqi_value']

    for col_name in feature_base_cols:
        param = col_name.replace('_value', '')
        for lag in LAG_HOURS:
            df[f'{param}_lag_{lag}h'] = df[col_name].shift(lag)

        if 'aqi' not in param:
            for window in ROLLING_WINDOWS:
                df[f'{param}_rolling_mean_{window}h'] = df[col_name].rolling(window=window, min_periods=1).mean()
                df[f'{param}_rolling_std_{window}h'] = df[col_name].rolling(window=window, min_periods=1).std()

    # 移除因為 lag/rolling 創建的 NaN 行
    df = df.dropna().reset_index(drop=True)
    
    return df


# =================================================================
# 訓練與儲存模型的邏輯 (保持不變)
# =================================================================
def train_and_save_models(locations: list, days_to_fetch: int):
    print(f"🔥 [Local Init] 開始執行多地點 AQI 預測初始化流程 (尋找活躍測站範圍: {days_to_fetch} 天)...")

    # 1. 多地點數據收集、處理與合併
    all_df = []
    all_found_params = set()
    weather_feature_names = WeatherCrawler(0, 0).get_weather_feature_names() 

    for lat, lon, name in locations:
        print(f"\n--- 🌍 處理地點: {name} ({lat:.4f}, {lon:.4f}) ---")
        
        weather = WeatherCrawler(lat, lon)
        
        try:
            # 這裡使用 DAYS_TO_FETCH=90 尋找最近 90 天有數據的測站
            station = get_nearest_station(lat, lon, days=days_to_fetch) 
            
            if not station:
                print(f"🚨 [Init - {name}] 未找到活躍測站，跳過此地點。")
                continue
            
            print(f"✅ [Init - {name}] 找到測站: {station['name']} ({station['id']})")
            # 實際抓取數據（仍受 OpenAQ 的 500 筆限制）
            df_raw, found_params = get_all_target_data(station["id"], POLLUTANT_TARGETS, days_to_fetch)
            
            print(f"   [Init - {name}] 原始數據點數: {len(df_raw)}")
            if df_raw.empty or len(df_raw) < MIN_DATA_THRESHOLD:
                print(f"🚨 [Init - {name}] 原始數據量不足 ({len(df_raw)}), 跳過此地點。")
                continue
                
            # 合併 Meteostat 天氣數據
            df = weather.fetch_and_merge_weather(df_raw.copy())
            
            # 數據清理與特徵工程
            df_processed = _preprocess_and_feature_engineer(df, found_params, weather_feature_names)
            
            if not df_processed.empty:
                all_df.append(df_processed)
                all_found_params.update(found_params)
                print(f"📊 [Init - {name}] **最終訓練數據量**: {len(df_processed)} 小時")
            else:
                print(f"🚨 [Init - {name}] 特徵工程後數據量不足（小於 {max(LAG_HOURS)}），跳過此地點。")

        except Exception as e:
            print(f"❌ [Init - {name}] 處理失敗: {e}")
            continue

    # 2. 合併所有數據並準備訓練
    if not all_df:
        raise ValueError("所有地點數據收集失敗，訓練無法進行。")
        
    final_df = pd.concat(all_df, ignore_index=True)
    final_df = final_df.sort_values('datetime').reset_index(drop=True)
    
    POLLUTANT_PARAMS_TRAINED = list(POLLUTANT_TARGETS)
    
    print(f"\n=========================================================")
    print(f"📊 [Local Init] 最終用於訓練的**總數據量**: {len(final_df)} 小時")
    print(f"🎯 [Local Init] 訓練目標污染物: {POLLUTANT_PARAMS_TRAINED}")
    print(f"=========================================================")

    if len(final_df) == 0:
        raise ValueError("總數據量為零，訓練無法進行。")

    LAST_OBSERVATION = final_df.iloc[-1:].to_json(orient='records', date_format='iso')

    # 3. 確定特徵欄位
    base_time_features = ['hour', 'day_of_week', 'month', 'is_weekend', 'hour_sin', 'hour_cos', 'day_sin', 'day_cos']
    
    air_quality_features = []
    for param in POLLUTANT_TARGETS + ['aqi']: 
        for lag in LAG_HOURS:
            air_quality_features.append(f'{param}_lag_{lag}h')
        if param != 'aqi':
            for window in ROLLING_WINDOWS:
                air_quality_features.append(f'{param}_rolling_mean_{window}h')
                air_quality_features.append(f'{param}_rolling_std_{window}h')

    FEATURE_COLUMNS = weather_feature_names + base_time_features + air_quality_features
    FEATURE_COLUMNS = [col for col in FEATURE_COLUMNS if col in final_df.columns]

    # 4. 數據分割與模型訓練
    # train_and_save.py (修改 train_and_save_models 函式末段)

    # 4. 數據分割與模型訓練 (80% 訓練)
    split_idx = int(len(final_df) * 0.8)

    # 【核心修改點 A: 清理 NaN 目標】
    # 在提取 X 和 Y 之前，先移除所有訓練目標（Y）欄位中含有 NaN 的行。
    # 雖然之前已經做過一次 dropna，這裡的額外檢查是為了處理合併後的極端情況。
    Y_cols = [f'{p}_value' for p in POLLUTANT_TARGETS]
    final_df.dropna(subset=Y_cols, inplace=True) # 移除訓練目標（Y）欄位有 NaN 的行

    # 重新確保數據量仍足夠
    if len(final_df) == 0:
        raise ValueError("最終數據清理後總數據量為零，訓練無法進行。")
        
    # 重新計算分割索引
    split_idx = int(len(final_df) * 0.8) 

    X = final_df[FEATURE_COLUMNS]
    # 這裡 Y 的字典創建應該只包含 final_df 中存在的欄位
    Y = {param: final_df[f'{param}_value'] 
        for param in POLLUTANT_TARGETS 
        if f'{param}_value' in final_df.columns} 

    X_train = X[:split_idx]
    # ... (其餘訓練代碼不變)
    
    print(f"⏳ [Local Init] 開始訓練 {len(Y)} 個 XGBoost 模型 (N={N_ESTIMATORS})...")
    TRAINED_MODELS = {}
    
    for param, Y_series in Y.items():
        Y_train = Y_series[:split_idx]
        print(f"       訓練 {param} 模型...")
        
        xgb_model = xgb.XGBRegressor(
            n_estimators=N_ESTIMATORS, max_depth=7, learning_rate=0.08, random_state=42, n_jobs=-1
        )
        xgb_model.fit(X_train, Y_train)
        TRAINED_MODELS[param] = xgb_model
        
        model_path = os.path.join(MODELS_DIR, f'{param}_model.json')
        xgb_model.save_model(model_path)
        print(f"       ✅ {param} 模型已儲存至 {model_path}")

    # 5. 儲存模型元數據 (Metadata)
    metadata = {
        'pollutant_params': POLLUTANT_TARGETS, 
        'feature_columns': FEATURE_COLUMNS,
        'last_observation_json': LAST_OBSERVATION
    }
    with open(os.path.join(MODELS_DIR, 'model_meta.json'), 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=4)

    print("✅ [Local Init] 所有模型和元數據儲存完成。")

if __name__ == '__main__':
    try:
        train_and_save_models(GLOBAL_TRAINING_LOCATIONS, DAYS_TO_FETCH)
    except Exception as e:
        print(f"❌ [Local Init] 訓練執行失敗: {e}")