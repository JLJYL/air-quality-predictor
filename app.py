# app.py - 供 Render 部署使用

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
from flask import Flask, render_template

# 忽略警告
warnings.filterwarnings('ignore')

# 模型與元數據路徑
MODELS_DIR = 'models'
META_PATH = os.path.join(MODELS_DIR, 'model_meta.json')

# =================================================================
# OpenAQ API 相關常數
# =================================================================
# ⚠️ 請替換成您自己的 API Key
API_KEY = "98765df2082f04dc9449e305bc736e93624b66e250fa9dfabcca53b31fc11647" 
HEADERS = {"X-API-Key": API_KEY}
BASE = "https://api.openaq.org/v3"

LOCATION_ID = 2395624  # 高雄市-前金 (請替換為您需要的站點 ID)
TARGET_PARAMS = ["co", "no2", "o3", "pm10", "pm25", "so2"]
PARAM_IDS = {"co": 8, "no2": 7, "o3": 10, "pm10": 1, "pm25": 2, "so2": 9}

TOL_MINUTES_PRIMARY = 5
TOL_MINUTES_FALLBACK = 60

# =================================================================
# 全域變數
# =================================================================
TRAINED_MODELS = {} 
LAST_OBSERVATION = None # 從 model_meta.json 載入，用於提供滯後特徵的基礎
FEATURE_COLUMNS = []
POLLUTANT_PARAMS = [] # 實際找到並訓練的模型參數
HOURS_TO_PREDICT = 24

# =================================================================
# 常數設定
# =================================================================
LOCAL_TZ = "Asia/Taipei"
LAG_HOURS = [1, 2, 3, 6, 12, 24]
ROLLING_WINDOWS = [6, 12, 24]
POLLUTANT_TARGETS = ["pm25", "pm10", "o3", "no2", "so2", "co"] 

# 簡化的 AQI 分級表
AQI_BREAKPOINTS = {
    "pm25": [(0.0, 12.0, 0, 50), (12.1, 35.4, 51, 100), (35.5, 55.4, 101, 150), (55.5, 150.4, 151, 200)],
    "pm10": [(0, 54, 0, 50), (55, 154, 51, 100), (155, 254, 101, 150), (255, 354, 151, 200)],
    "o3": [(0, 54, 0, 50), (55, 70, 51, 100), (71, 85, 101, 150), (86, 105, 151, 200)],
    "co": [(0.0, 4.4, 0, 50), (4.5, 9.4, 51, 100), (9.5, 12.4, 101, 150), (12.5, 15.4, 151, 200)],
    "no2": [(0, 100, 0, 50), (101, 360, 51, 100), (361, 649, 101, 150), (650, 1249, 151, 200)],
    "so2": [(0, 35, 0, 50), (36, 75, 51, 100), (76, 185, 101, 150), (186, 304, 151, 200)],
}


# =================================================================
# OpenAQ 資料抓取函式
# =================================================================

def get_location_meta(location_id: int):
    """取得站點最後更新時間"""
    try:
        r = requests.get(f"{BASE}/locations/{location_id}", headers=HEADERS, timeout=10)
        r.raise_for_status()
        row = r.json()["results"][0]
        last_utc = pd.to_datetime(row["datetimeLast"]["utc"], errors="coerce", utc=True)
        last_local = row["datetimeLast"]["local"]
        return {
            "id": int(row["id"]),
            "name": row["name"],
            "last_utc": last_utc,
            "last_local": last_local,
        }
    except Exception as e:
        return None


def get_location_latest_df(location_id: int) -> pd.DataFrame:
    """站點各參數的『最新值清單』→ 正規化時間成 ts_utc / ts_local"""
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
    """用 /parameters/{pid}/latest?locationId= 拿各參數『最新值』並合併"""
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


def pick_batch_near(df: pd.DataFrame, t_ref: pd.Timestamp, tol_minutes: int) -> pd.DataFrame:
    """從 DataFrame 中挑選最接近 t_ref 且時間差異在 tol_minutes 內的資料批次"""
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
    """從 OpenAQ 抓取並轉換成單行寬表（只含當前原始值）"""
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

    # 4. 轉換成模型輸入格式 (單行寬表)
    observation = df_all.pivot_table(
        index='ts_utc', columns='parameter', values='value', aggfunc='first'
    ).reset_index()
    observation = observation.rename(columns={'ts_utc': 'datetime'})
    
    # 計算 AQI
    if not observation.empty:
        observation['aqi'] = observation.apply(
            lambda row: calculate_aqi(row, target_params, is_pred=False), axis=1
        )
        
    if not observation.empty:
        observation['datetime'] = pd.to_datetime(observation['datetime'])
        if observation['datetime'].dt.tz is None:
             observation['datetime'] = observation['datetime'].dt.tz_localize('UTC')

    return observation


# =================================================================
# 輔助函式: AQI 計算
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

def calculate_aqi(row: pd.Series, params: list, is_pred=True) -> int:
    """根據多個污染物濃度計算最終 AQI (取最大子指數)"""
    sub_indices = []
    for p in params:
        col_name = f'{p}_pred' if is_pred else p
        if col_name in row and not pd.isna(row[col_name]):
            sub_index = calculate_aqi_sub_index(p, row[col_name])
            sub_indices.append(sub_index)

    if not sub_indices:
        return np.nan

    return int(np.max(sub_indices))


# =================================================================
# 預測函式
# =================================================================

def predict_future_multi(models, last_data, feature_cols, pollutant_params, hours=24):
    """預測未來 N 小時的多個目標污染物 (遞迴預測) 並計算 AQI"""
    predictions = []

    # 🚨 修正：現在我們信任 index() 傳入的時間是 Naive 的，因此可以直接本地化為 UTC
    last_data['datetime'] = pd.to_datetime(last_data['datetime']).dt.tz_localize('UTC')
         
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
        pred_features['day_of_year'] = future_time.timetuple().tm_yday 
        pred_features['is_weekend'] = int(future_time.dayofweek in [5, 6])
        pred_features['hour_sin'] = np.sin(2 * np.pi * future_time.hour / 24)
        pred_features['hour_cos'] = np.cos(2 * np.pi * future_time.hour / 24)
        pred_features['day_sin'] = np.sin(2 * np.pi * pred_features['day_of_year'] / 365)
        pred_features['day_cos'] = np.cos(2 * np.pi * pred_features['day_of_year'] / 365)

        # 2. 模擬未來天氣變化 
        if has_weather:
            np.random.seed(future_time.hour + future_time.day + 42)
            for w_col in weather_feature_names:
                base_value = current_data_dict.get(w_col)
                if base_value is not None and not np.isnan(base_value):
                    new_weather_value = base_value + np.random.normal(0, 0.5) 
                    pred_features[w_col] = new_weather_value
                    current_data_dict[w_col] = new_weather_value 

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
        predicted_aqi = calculate_aqi(pd.Series(current_prediction_row), pollutant_params, is_pred=True)
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

            if f'{param}_lag_1h' in current_data_dict and param in new_pollutant_values:
                current_data_dict[f'{param}_lag_1h'] = new_pollutant_values[param]

    return pd.DataFrame(predictions)


# =================================================================
# 模型載入邏輯
# =================================================================

def load_models_and_metadata():
    global TRAINED_MODELS, LAST_OBSERVATION, FEATURE_COLUMNS, POLLUTANT_PARAMS

    if not os.path.exists(META_PATH):
        print("🚨 [Load] 找不到模型元數據檔案 (model_meta.json)，無法載入模型。")
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
                print(f"❌ [Load] 找不到 {param} 的模型檔案: {model_path}")
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
# Flask 應用程式設定與啟動
# =================================================================
app = Flask(__name__)

# 應用程式啟動時載入模型
with app.app_context():
    load_models_and_metadata() 

@app.route('/')
def index():
    city_name = "高雄"
    
    if not TRAINED_MODELS or not POLLUTANT_PARAMS:
        print("🚨 [Request] 模型或參數尚未初始化，無法進行預測。")
        return render_template('index.html', max_aqi="N/A", aqi_predictions=[], city_name=city_name)
    
    # 1. 嘗試即時抓取最新觀測數據
    current_observation_raw = fetch_latest_observation_data(LOCATION_ID, POLLUTANT_TARGETS)

    observation_for_prediction = None
    
    # 2. 數據整合邏輯
    if current_observation_raw.empty or LAST_OBSERVATION is None or LAST_OBSERVATION.empty:
        print("🚨 [Request] 無法取得最新觀測數據或模型滯後數據，退回使用模型載入時的數據。")
        observation_for_prediction = LAST_OBSERVATION
    else:
        observation_for_prediction = LAST_OBSERVATION.iloc[:1].copy() 

        latest_row = current_observation_raw.iloc[0]

        observation_for_prediction['datetime'] = latest_row['datetime']
        
        for col in latest_row.index:
            if col in observation_for_prediction.columns and not any(s in col for s in ['lag_', 'rolling_']):
                 if col in POLLUTANT_TARGETS or col == 'aqi' or col in ['temperature', 'humidity', 'pressure']:
                    observation_for_prediction[col] = latest_row[col]
                 
        print(f"✅ [Request] 成功整合最新觀測數據 (UTC: {observation_for_prediction['datetime'].iloc[0]})")
        
    # 3. 檢查最終預測數據來源
    if observation_for_prediction is None or observation_for_prediction.empty:
        print("🚨 [Request] 最終預測數據來源為空，無法進行預測。")
        return render_template('index.html', max_aqi="N/A", aqi_predictions=[], city_name=city_name)
    
    # 🚨 最終修正：在傳入預測函式前，強制移除所有時區資訊，避免重複本地化錯誤
    try:
        # 確保是 datetime 類型
        observation_for_prediction['datetime'] = pd.to_datetime(observation_for_prediction['datetime'])
        # 移除時區資訊
        if observation_for_prediction['datetime'].dt.tz is not None:
             observation_for_prediction['datetime'] = observation_for_prediction['datetime'].dt.tz_localize(None)
        print("✅ [Fix] 已安全清除預測數據中的時區資訊。")
    except Exception as e:
        # 如果失敗，記錄錯誤但仍然嘗試繼續預測
        print(f"⚠️ [Fix] 清除時區資訊失敗: {e}")
    
    # 4. 進行預測
    # app.py (在 index 函式的 try 區塊內)

    # ... (前略：步驟 3. 時區修正完成後) ...

    # 4. 進行預測
    try:
        # 執行未來小時的預測
        future_predictions_df = predict_future_multi(
            TRAINED_MODELS,
            observation_for_prediction, # 傳入 Naive 時間的數據
            FEATURE_COLUMNS,
            POLLUTANT_PARAMS,
            hours=HOURS_TO_PREDICT
        )
        
        # 🚨 修正核心：創建當前觀測值 (t+0) 的數據行
        current_data = observation_for_prediction.iloc[0].copy()
        
        # 獲取當前時間和實時 AQI
        current_time_aware = pd.to_datetime(current_data['datetime']).tz_localize('UTC').tz_convert(LOCAL_TZ)
        current_aqi = int(current_data.get('aqi', 0)) # 從實時抓取的數據中獲取 AQI
        
        # 格式化當前數據行
        current_prediction_row = pd.DataFrame([{
            'datetime': current_time_aware,
            'aqi_pred': current_aqi,
            'is_current': True # 添加標記以便在前端顯示 "現在"
        }])
        current_prediction_row = current_prediction_row.rename(
            columns={'datetime': 'datetime_local', 'aqi_pred': 'aqi'}
        )
        
        # 合併當前數據和未來預測數據
        # 將未來預測 DataFrame 的欄位改名以匹配
        future_predictions_df = future_predictions_df.rename(columns={'aqi_pred': 'aqi'})
        
        # 由於 future_predictions['datetime'] 是 UTC-aware，我們需要將 current_time_aware 轉回 UTC
        current_prediction_row['datetime'] = current_prediction_row['datetime'].dt.tz_convert('UTC')
        current_prediction_row = current_prediction_row.drop(columns=['is_current'])
        
        # 重計算 future_predictions 的 datetime_local (保持原有邏輯)
        future_predictions_df['datetime_local'] = future_predictions_df['datetime'].dt.tz_convert(LOCAL_TZ)

        # 確保當前數據行和預測數據行只有 'datetime', 'aqi', 'datetime_local' 這些欄位
        current_data_row = {'datetime_local': current_time_aware, 'aqi': current_aqi}
        
        # 建立最終的預測列表
        final_predictions = future_predictions_df[['datetime_local', 'aqi']].copy()
        
        # 確保 current_data_row 是一個 Series 或 DataFrame
        current_df = pd.DataFrame([current_data_row])
        
        # 將當前數據添加到列表的開頭
        combined_predictions_df = pd.concat([current_df, final_predictions], ignore_index=True)


        # 格式化最終結果
        max_aqi = int(future_predictions_df['aqi'].max()) # 最大AQI仍只計算未來小時
        
        aqi_predictions = [
            {
                'time': item['datetime_local'].strftime('%Y-%m-%d %H:%M'), 
                'aqi': int(item['aqi']),
                # 如果是第一行，顯示 '現在' 標籤
                'is_current': idx == 0 
            }
            for idx, item in combined_predictions_df.to_dict(orient='records')
        ]
        
    # ... (後略：except 區塊保持不變) ...
        
    except Exception as e:
        max_aqi = "N/A"
        aqi_predictions = []
        print(f"❌ [Request] 預測執行失敗: {e}") 

    return render_template('index.html', max_aqi=max_aqi, aqi_predictions=aqi_predictions, city_name=city_name)

if __name__ == '__main__':
    app.run(debug=True)
