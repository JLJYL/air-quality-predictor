# app.py: Air Quality Prediction Flask Application

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
from datetime import timedelta, timezone
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from meteostat import Point, Hourly, units
from flask import Flask, render_template

# 忽略警告
warnings.filterwarnings('ignore')

# =================================================================
# 常數設定 (與您的原始腳本相同)
# =================================================================
API_KEY = "68af34aea77a19aa1137ee5fd9b287229ccf23a686309b4521924a04963ac663"
API_BASE_URL = "https://api.openaq.org/v3/"
POLLUTANT_TARGETS = ["pm25", "pm10", "o3", "no2", "so2", "co"]
LOCAL_TZ = "Asia/Taipei"
MIN_DATA_THRESHOLD = 100
LAG_HOURS = [1, 2, 3, 6, 12, 24]
ROLLING_WINDOWS = [6, 12, 24]
DAYS_TO_FETCH = 14 # 減少抓取天數以加速 Web 應用啟動時間

# 簡化的 AQI 分級表 (基於小時值和 US EPA 標準的常用數值)
AQI_BREAKPOINTS = {
    "pm25": [(0.0, 12.0, 0, 50), (12.1, 35.4, 51, 100), (35.5, 55.4, 101, 150), (55.5, 150.4, 151, 200)],
    "pm10": [(0, 54, 0, 50), (55, 154, 51, 100), (155, 254, 101, 150), (255, 354, 151, 200)],
    "o3": [(0, 54, 0, 50), (55, 70, 51, 100), (71, 85, 101, 150), (86, 105, 151, 200)],
    "co": [(0.0, 4.4, 0, 50), (4.5, 9.4, 51, 100), (9.5, 12.4, 101, 150), (12.5, 15.4, 151, 200)],
    "no2": [(0, 100, 0, 50), (101, 360, 51, 100), (361, 649, 101, 150), (650, 1249, 151, 200)],
    "so2": [(0, 35, 0, 50), (36, 75, 51, 100), (76, 185, 101, 150), (186, 304, 151, 200)],
}

# =================================================================
# 輔助函式: AQI 計算 (從您的原始腳本複製)
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
# OpenAQ V3 數據爬取/輔助函式 (從您的原始腳本複製)
# =================================================================
def sanitize_filename(name: str) -> str:
    return re.sub(r'[\\/:"*?<>|]+', '_', name)

def get_nearest_station(lat, lon, radius=20000, limit=50, days=7):
    """ 找離 (lat,lon) 最近且最近 days 內有更新的測站 """
    url = f"{API_BASE_URL}locations"
    headers = {"X-API-Key": API_KEY}
    params = {"coordinates": f"{lat},{lon}", "radius": radius, "limit": limit}
    try:
        resp = requests.get(url, headers=headers, params=params)
        resp.raise_for_status()
        j = resp.json()
    except Exception as e:
        print(f"Error fetching nearest station: {e}")
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
    """ 使用 /locations/{id}/sensors 取得 sensors 列表 """
    url = f"{API_BASE_URL}locations/{station_id}/sensors"
    headers = {"X-API-Key": API_KEY}
    try:
        resp = requests.get(url, headers=headers, params={"limit":1000})
        resp.raise_for_status()
        j = resp.json()
        return j.get("results", [])
    except Exception as e:
        print(f"Error fetching sensors: {e}")
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

def fetch_sensor_data(sensor_id, param_name, limit=500, days=7):
    """ 擷取 sensor 的時間序列 """
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
        print(f"❌ 抓取 {param_name} 數據失敗: {e}")
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

def generate_fake_data(limit=10, params=POLLUTANT_TARGETS):
    """生成所有目標污染物 (含 AQI) 的模擬數據"""
    now = datetime.datetime.now(datetime.timezone.utc)
    base_rows = []
    for i in range(limit):
        dt = now - datetime.timedelta(minutes=i*60)
        row = {'datetime': dt}

        for param in params:
            if param in ["pm25", "pm10"]: value = round(random.uniform(10, 60), 1)
            elif param == "o3": value = round(random.uniform(20, 100), 1)
            elif param in ["no2", "so2"]: value = round(random.uniform(1, 40), 1)
            elif param == 'co': value = round(random.uniform(0.1, 5), 1)
            row[f'{param}_value'] = value

        row['temperature'] = round(random.uniform(15, 30), 1)
        row['humidity'] = round(random.uniform(50, 95), 1)
        row['pressure'] = round(random.uniform(1000, 1020), 1)

        aqi_val = calculate_aqi(pd.Series(row), params)
        row['aqi_value'] = aqi_val
        base_rows.append(row)

    df = pd.DataFrame(base_rows)
    df['datetime'] = df['datetime'].dt.tz_localize('UTC')
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
            df_param = fetch_sensor_data(sensor_id, param, days=days_to_fetch)
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
# Meteostat 天氣爬蟲類 (從您的原始腳本複製)
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
        except Exception:
            weather_data = pd.DataFrame()

        if weather_data.empty:
            # 如果抓取失敗，則填充 NaN
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
        # 使用 ffill/bfill 處理缺失天氣數據
        merged_df[weather_cols_list] = merged_df[weather_cols_list].fillna(method='ffill').fillna(method='bfill')

        return merged_df

    def get_weather_feature_names(self):
        return list(self.weather_cols.values())


# =================================================================
# 預測函式 (從您的原始腳本複製)
# =================================================================

def predict_future_multi(models, last_data, feature_cols, pollutant_params, hours=24):
    """預測未來 N 小時的多個目標污染物 (遞迴預測) 並計算 AQI"""
    predictions = []

    last_datetime_aware = last_data['datetime'].iloc[0]
    current_data_dict = last_data[feature_cols].iloc[0].to_dict()

    weather_feature_names_base = ['temperature', 'humidity', 'pressure']
    weather_feature_names = [col for col in weather_feature_names_base if col in feature_cols]
    has_weather = bool(weather_feature_names)

    for h in range(hours):
        future_time = last_datetime_aware + timedelta(hours=h + 1)
        pred_features = current_data_dict.copy()

        # 1. 更新時間特徵
        pred_features['hour'] = future_time.hour
        pred_features['day_of_week'] = future_time.dayofweek
        pred_features['month'] = future_time.month
        pred_features['day_of_year'] = future_time.dayofyear
        pred_features['is_weekend'] = int(future_time.dayofweek in [5, 6])
        pred_features['hour_sin'] = np.sin(2 * np.pi * future_time.hour / 24)
        pred_features['hour_cos'] = np.cos(2 * np.pi * future_time.hour / 24)
        pred_features['day_sin'] = np.sin(2 * np.pi * future_time.dayofyear / 365)
        pred_features['day_cos'] = np.cos(2 * np.pi * future_time.dayofyear / 365)

        # 2. 模擬未來天氣變化
        if has_weather:
            np.random.seed(future_time.hour + future_time.day + 42)
            for w_col in weather_feature_names:
                base_value = current_data_dict[w_col]
                pred_features[w_col] = base_value + np.random.normal(0, 0.5)
                current_data_dict[w_col] = pred_features[w_col]

        current_prediction_row = {'datetime': future_time}
        new_pollutant_values = {}

        # 3. 預測所有污染物
        for param in pollutant_params:
            model = models[param]
            pred_input = np.array([pred_features[col] for col in feature_cols]).reshape(1, -1)
            pred = model.predict(pred_input)[0]
            pred = max(0, pred)

            current_prediction_row[f'{param}_pred'] = pred
            new_pollutant_values[param] = pred

        # 4. 計算預測的 AQI
        predicted_aqi = calculate_aqi(pd.Series(current_prediction_row), pollutant_params)
        current_prediction_row['aqi_pred'] = predicted_aqi
        new_pollutant_values['aqi'] = predicted_aqi

        predictions.append(current_prediction_row)

        # 5. 更新滯後特徵
        for param in pollutant_params + ['aqi']:
            for i in range(len(LAG_HOURS) - 1, 0, -1):
                lag_current = LAG_HOURS[i]
                lag_prev = LAG_HOURS[i-1]
                lag_current_col = f'{param}_lag_{lag_current}h'
                lag_prev_col = f'{param}_lag_{lag_prev}h'

                if lag_current_col in current_data_dict and lag_prev_col in current_data_dict:
                    current_data_dict[lag_current_col] = current_data_dict[lag_prev_col]

            if f'{param}_lag_1h' in current_data_dict:
                current_data_dict[f'{param}_lag_1h'] = new_pollutant_values[param]

    return pd.DataFrame(predictions)


# =================================================================
# 核心執行邏輯 (封裝您的步驟 3-6)
# =================================================================

def run_aqi_prediction(lat: float, lon: float, days_to_fetch: int, hours_to_predict: int):
    """執行空氣品質預測的整個流程，並返回預測結果。"""

    weather = WeatherCrawler(lat, lon)
    
    # 1. 數據收集
    station = get_nearest_station(lat, lon, days=days_to_fetch)

    if not station:
        # 如果找不到測站，則使用模擬數據
        df = generate_fake_data(limit=days_to_fetch * 24, params=POLLUTANT_TARGETS)
        found_target_params = POLLUTANT_TARGETS
    else:
        df_raw, found_target_params = get_all_target_data(station["id"], POLLUTANT_TARGETS, days_to_fetch)

        if df_raw.empty or len(df_raw) < MIN_DATA_THRESHOLD:
            df = generate_fake_data(limit=days_to_fetch * 24, params=POLLUTANT_TARGETS)
            found_target_params = POLLUTANT_TARGETS
        else:
            df = df_raw.copy()
        
        # 合併 Meteostat 天氣數據
        df = weather.fetch_and_merge_weather(df)

    weather_feature_names = weather.get_weather_feature_names()
    value_cols = [f'{p}_value' for p in found_target_params]
    all_data_cols = value_cols + weather_feature_names

    # 重採樣到小時
    df.set_index('datetime', inplace=True)
    df = df[value_cols + weather_feature_names].resample('H').mean()
    df.reset_index(inplace=True)
    df = df.dropna(how='all', subset=all_data_cols)
    
    # 計算歷史 AQI
    df['aqi_value'] = df.apply(lambda row: calculate_aqi(row, found_target_params), axis=1)

    # 移除任一污染物或天氣數據為 NaN 的行 (確保模型輸入完整)
    df = df.dropna(subset=all_data_cols + ['aqi_value']).reset_index(drop=True)

    if len(df) <= max(LAG_HOURS):
        raise ValueError("最終數據量不足，無法進行滯後特徵工程和訓練。")


    # 2. 特徵工程
    df['hour'] = df['datetime'].dt.hour
    df['day_of_week'] = df['datetime'].dt.dayofweek
    df['month'] = df['datetime'].dt.month
    df['day_of_year'] = df['datetime'].dt.dayofyear
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
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
    
    df = df.dropna().reset_index(drop=True)

    base_time_features = ['hour', 'day_of_week', 'month', 'is_weekend', 'hour_sin', 'hour_cos', 'day_sin', 'day_cos']
    
    air_quality_features = []
    for param in found_target_params + ['aqi']:
        for lag in LAG_HOURS:
            air_quality_features.append(f'{param}_lag_{lag}h')
        if param != 'aqi':
            for window in ROLLING_WINDOWS:
                air_quality_features.append(f'{param}_rolling_mean_{window}h')
                air_quality_features.append(f'{param}_rolling_std_{window}h')

    feature_cols = weather_feature_names + base_time_features + air_quality_features
    feature_cols = [col for col in feature_cols if col in df.columns]

    # 3. 數據分割與模型訓練 (80% 訓練)
    split_idx = int(len(df) * 0.8)
    X = df[feature_cols]
    Y = {param: df[f'{param}_value'] for param in found_target_params}
    
    X_train, X_test = X[:split_idx], X[split_idx:]
    Y_train = {param: Y[param][:split_idx] for param in found_target_params}

    multi_models = {}
    for param in found_target_params:
        xgb_model = xgb.XGBRegressor(
            n_estimators=150, max_depth=7, learning_rate=0.08, random_state=42, n_jobs=-1
        )
        xgb_model.fit(X_train, Y_train[param])
        multi_models[param] = xgb_model

    # 4. 預測未來
    last_observation = df.iloc[-1:].copy()
    future_predictions = predict_future_multi(
        multi_models,
        last_observation,
        feature_cols,
        found_target_params,
        hours=hours_to_predict
    )

    # 5. 格式化結果
    future_predictions['datetime_local'] = future_predictions['datetime'].dt.tz_convert(LOCAL_TZ)

    # 提取最高 AQI (必須轉為 int)
    max_aqi = int(future_predictions['aqi_pred'].max())

    # 提取未來 24 小時的 AQI 列表
    aqi_list = [
        {'time': item['datetime_local'].strftime('%Y-%m-%d %H:%M'), 'aqi': int(item['aqi_pred'])}
        for item in future_predictions.to_dict(orient='records')
    ]

    return max_aqi, aqi_list

# =================================================================
# Flask 應用程式設定
# =================================================================
app = Flask(__name__)

@app.route('/')
def index():
    # 高雄市中心經緯度
    lat, lon = 22.6273, 120.3014
    city_name = "高雄"
    
    try:
        # 執行核心預測邏輯 (預測未來 24 小時)
        max_aqi, aqi_predictions = run_aqi_prediction(lat, lon, DAYS_TO_FETCH, 24)
    except ValueError as ve:
        # 處理數據不足的錯誤
        max_aqi = "N/A"
        aqi_predictions = []
        print(f"數據錯誤: {ve}")
    except Exception as e:
        # 處理其他網路或計算錯誤
        max_aqi = "N/A"
        aqi_predictions = []
        print(f"預測執行失敗: {e}") 
        
    # 渲染 index.html 模板並傳遞數據
    return render_template('index.html', max_aqi=max_aqi, aqi_predictions=aqi_predictions, city_name=city_name)

if __name__ == '__main__':
    # 在本地環境運行時使用
    app.run(debug=True)
