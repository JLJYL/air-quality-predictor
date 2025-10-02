# app.py - 供 Render 部署使用 (已移除耗時的訓練邏輯)

# =================================================================
# 導入所有必要的庫 (新增 requests, numpy, pandas, json, xgboost, datetime 的導入)
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
# OpenAQ API 相關常數 (從您的第一個腳本複製)
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
# 全域變數 - 改為從檔案載入
# =================================================================
TRAINED_MODELS = {} 
# ⚠️ LAST_OBSERVATION 不再從檔案載入，改為即時抓取，但在載入時仍讀取以備模型需要
LAST_OBSERVATION = None 
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
# ... (AQI_BREAKPOINTS 保持不變) ...
AQI_BREAKPOINTS = {
    "pm25": [(0.0, 12.0, 0, 50), (12.1, 35.4, 51, 100), (35.5, 55.4, 101, 150), (55.5, 150.4, 151, 200)],
    "pm10": [(0, 54, 0, 50), (55, 154, 51, 100), (155, 254, 101, 150), (255, 354, 151, 200)],
    "o3": [(0, 54, 0, 50), (55, 70, 51, 100), (71, 85, 101, 150), (86, 105, 151, 200)],
    "co": [(0.0, 4.4, 0, 50), (4.5, 9.4, 51, 100), (9.5, 12.4, 101, 150), (12.5, 15.4, 151, 200)],
    "no2": [(0, 100, 0, 50), (101, 360, 51, 100), (361, 649, 101, 150), (650, 1249, 151, 200)],
    "so2": [(0, 35, 0, 50), (36, 75, 51, 100), (76, 185, 101, 150), (186, 304, 151, 200)],
}


# =================================================================
# OpenAQ 資料抓取函式 (從您的第一個腳本複製過來)
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
        print(f"❌ [Fetch] get_location_meta 失敗: {e}")
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

        # 參數名與單位
        df["parameter"] = df["parameter.name"].str.lower() if "parameter.name" in df.columns else df.get("parameter", df.get("name"))
        df["units"] = df["parameter.units"] if "parameter.units" in df.columns else df.get("units")
        df["value"] = df["value"]

        # 取代表該筆的UTC時間
        df["ts_utc"] = pd.NaT
        for col in ["datetime.utc", "period.datetimeTo.utc", "period.datetimeFrom.utc"]:
            if col in df.columns:
                ts = pd.to_datetime(df[col], errors="coerce", utc=True)
                df["ts_utc"] = df["ts_utc"].where(df["ts_utc"].notna(), ts)

        # 地方時間
        local_col = None
        for c in ["datetime.local", "period.datetimeTo.local", "period.datetimeFrom.local"]:
            if c in df.columns:
                local_col = c
                break
        df["ts_local"] = df[local_col] if local_col in df.columns else None

        return df[["parameter", "value", "units", "ts_utc", "ts_local"]]
    except Exception as e:
        print(f"❌ [Fetch] get_location_latest_df 失敗: {e}")
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

            # 參數名與單位
            df["parameter"] = p
            df["units"] = df["parameter.units"] if "parameter.units" in df.columns else df.get("units")
            df["value"] = df["value"]

            # 時間欄位
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
        print(f"❌ [Fetch] get_parameters_latest_df 失敗: {e}")

    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def pick_batch_near(df: pd.DataFrame, t_ref: pd.Timestamp, tol_minutes: int) -> pd.DataFrame:
    """從 DataFrame 中挑選最接近 t_ref 且時間差異在 tol_minutes 內的資料批次"""
    if df.empty or pd.isna(t_ref):
        return pd.DataFrame()

    df = df.copy()

    # ★ 確保 ts_utc 是單一值且為 NaT-aware
    def _scalarize(v):
        if isinstance(v, (list, tuple, np.ndarray)):
            return v[0] if len(v) else None
        return v

    df["ts_utc"] = df["ts_utc"].map(_scalarize)
    df["ts_utc"] = pd.to_datetime(df["ts_utc"], errors="coerce", utc=True)

    # 接著就能安全做時間距離比較
    df["dt_diff"] = (df["ts_utc"] - t_ref).abs()

    tol = pd.Timedelta(minutes=tol_minutes)
    df = df[df["dt_diff"] <= tol].copy()
    if df.empty:
        return df

    # 排序：參數、時間距離最小、最新時間 (確保同一參數只留最接近 t_ref 的那筆)
    df = df.sort_values(["parameter", "dt_diff", "ts_utc"], ascending=[True, True, False])
    df = df.drop_duplicates(subset=["parameter"], keep="first")
    return df[["parameter", "value", "units", "ts_utc", "ts_local"]]


def fetch_latest_observation_data(location_id: int, target_params: list) -> pd.DataFrame:
    """
    主要數據獲取流程：
    1. 取得站點最後更新時間 t_star。
    2. 從 /locations/{id}/latest 取得 df_loc_latest。
    3. 以 t_star 為基準，在 df_loc_latest 中尋找最接近且時間對齊的一批數據。
    4. 針對缺少的參數，從 /parameters/{pid}/latest 補齊，並也對齊 t_star。
    5. 合併結果，返回單行、時間對齊的 DataFrame。
    """
    meta = get_location_meta(location_id)
    if not meta or pd.isna(meta["last_utc"]):
        print("🚨 [Fetch] 無法取得站點元數據或最後更新時間。")
        return pd.DataFrame()

    df_loc_latest = get_location_latest_df(location_id)
    if df_loc_latest.empty:
        print("⚠️ [Fetch] /locations/{id}/latest 沒有任何資料。")
        return pd.DataFrame()

    # 決定對齊時間 t_star (使用站點 meta 或 latest 中的最大時間)
    t_star_latest = df_loc_latest["ts_utc"].max()
    t_star_loc = meta["last_utc"]
    t_star = t_star_latest if pd.notna(t_star_latest) else t_star_loc

    if pd.isna(t_star):
        print("🚨 [Fetch] 無法決定有效的批次對齊時間。")
        return pd.DataFrame()
    
    # 1. 在 /locations/{id}/latest 中找「接近 t_star」的一批
    df_at_batch = pick_batch_near(df_loc_latest, t_star, TOL_MINUTES_PRIMARY)
    if df_at_batch.empty:
        df_at_batch = pick_batch_near(df_loc_latest, t_star, TOL_MINUTES_FALLBACK)

    have = set(df_at_batch["parameter"].str.lower().tolist()) if not df_at_batch.empty else set()

    # 2. 還缺的參數，用 /parameters/{pid}/latest?locationId= 補
    missing = [p for p in target_params if p not in have]
    df_param_batch = pd.DataFrame()
    if missing:
        df_param_latest = get_parameters_latest_df(location_id, missing)
        df_param_batch = pick_batch_near(df_param_latest, t_star, TOL_MINUTES_PRIMARY)
        if df_param_batch.empty:
            df_param_batch = pick_batch_near(df_param_latest, t_star, TOL_MINUTES_FALLBACK)

    # 3. 合併、只留目標參數、去重
    frames = [df for df in [df_at_batch, df_param_batch] if not df.empty]
    if not frames:
        print("⚠️ [Fetch] 在最後一批時間附近，目標污染物都沒有資料。")
        return pd.DataFrame()

    df_all = pd.concat(frames, ignore_index=True)
    df_all["parameter"] = df_all["parameter"].str.lower()
    df_all = df_all[df_all["parameter"].isin(target_params)]

    # 最終去重 (取最接近 t_star 的那筆)
    df_all["dt_diff"] = (df_all["ts_utc"] - t_star).abs()
    df_all = df_all.sort_values(["parameter", "dt_diff", "ts_utc"], ascending=[True, True, False])
    df_all = df_all.drop_duplicates(subset=["parameter"], keep="first")
    df_all = df_all.drop(columns=["dt_diff", "units", "ts_local"]) # 移除不必要的欄位

    # 4. 轉換成模型輸入格式 (單行寬表)
    observation = df_all.pivot_table(
        index='ts_utc', columns='parameter', values='value', aggfunc='first'
    ).reset_index()
    observation = observation.rename(columns={'ts_utc': 'datetime'})
    
    # 計算 AQI，並確保 column name 為 aqi
    observation['aqi'] = observation.apply(
        lambda row: calculate_aqi(row, target_params), axis=1
    )
    
    # 確保只有一筆數據，並且時間是 UTC-aware
    if not observation.empty:
        observation['datetime'] = observation['datetime'].dt.tz_localize(None).dt.tz_localize('UTC')

    return observation


# =================================================================
# 輔助函式: AQI 計算 (保持不變)
# =================================================================
# ... (calculate_aqi_sub_index 保持不變) ...
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

# ... (calculate_aqi 保持不變) ...
def calculate_aqi(row: pd.Series, params: list) -> int:
    """根據多個污染物濃度計算最終 AQI (取最大子指數)"""
    sub_indices = []
    for p in params:
        col_name = f'{p}_pred' if f'{p}_pred' in row else p # 注意這裡 p 即可，因為是預測前的 raw value
        if col_name in row and not pd.isna(row[col_name]):
            sub_index = calculate_aqi_sub_index(p, row[col_name])
            sub_indices.append(sub_index)

    if not sub_indices:
        return np.nan

    return int(np.max(sub_indices))

# =================================================================
# 預測函式 (保持不變)
# =================================================================

def predict_future_multi(models, last_data, feature_cols, pollutant_params, hours=24):
    """預測未來 N 小時的多個目標污染物 (遞迴預測) 並計算 AQI"""
    predictions = []

    # last_data 現在是單行 DataFrame，需要先轉換時間格式
    last_data['datetime'] = pd.to_datetime(last_data['datetime']).dt.tz_localize('UTC')
    last_datetime_aware = last_data['datetime'].iloc[0]
    # 注意：這裡使用 to_dict() 創建一個可變字典副本作為迭代的基礎
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
                    current_data_dict[w_col] = new_weather_value # 更新以便下一小時使用

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

            if f'{param}_lag_1h' in current_data_dict and param in new_pollutant_values:
                current_data_dict[f'{param}_lag_1h'] = new_pollutant_values[param]

    return pd.DataFrame(predictions)


# =================================================================
# 模型載入邏輯 (保持不變)
# =================================================================
# ... (load_models_and_metadata 保持不變) ...

def load_models_and_metadata():
    global TRAINED_MODELS, LAST_OBSERVATION, FEATURE_COLUMNS, POLLUTANT_PARAMS

    if not os.path.exists(META_PATH):
        print("🚨 [Load] 找不到模型元數據檔案 (model_meta.json)，無法載入模型。")
        return

    try:
        # 1. 載入元數據
        with open(META_PATH, 'r', encoding='utf-8') as f:
            metadata = json.load(f)

        POLLUTANT_PARAMS = metadata.get('pollutant_params', [])
        FEATURE_COLUMNS = metadata.get('feature_columns', [])
        
        # 將最後一筆數據的 JSON 轉換回 DataFrame
        if 'last_observation_json' in metadata:
            LAST_OBSERVATION = pd.read_json(metadata['last_observation_json'], orient='records')

        # 2. 載入 XGBoost 模型
        TRAINED_MODELS = {}
        for param in POLLUTANT_PARAMS:
            model_path = os.path.join(MODELS_DIR, f'{param}_model.json')
            if os.path.exists(model_path):
                model = xgb.XGBRegressor()
                model.load_model(model_path)
                TRAINED_MODELS[param] = model
            else:
                print(f"❌ [Load] 找不到 {param} 的模型檔案: {model_path}")
                del POLLUTANT_PARAMS[POLLUTANT_PARAMS.index(param)]
        
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

with app.app_context():
    load_models_and_metadata() 

@app.route('/')
def index():
    city_name = "高雄"
    
    # 檢查模型是否成功載入
    if not TRAINED_MODELS or not POLLUTANT_PARAMS:
        print("🚨 [Request] 模型或參數尚未初始化，無法進行預測。")
        return render_template('index.html', max_aqi="N/A", aqi_predictions=[], city_name=city_name)
    
    # ⭐⭐⭐ 新增：即時抓取最新觀測數據 ⭐⭐⭐
    current_observation_df = fetch_latest_observation_data(LOCATION_ID, POLLUTANT_TARGETS)

    if current_observation_df.empty or len(current_observation_df) == 0:
        print("🚨 [Request] 無法取得最新的空氣品質觀測數據。")
        # ⚠️ 可選：如果抓取失敗，退回到使用 LAST_OBSERVATION 進行預測
        observation_for_prediction = LAST_OBSERVATION
    else:
        observation_for_prediction = current_observation_df
        print(f"✅ [Request] 成功取得最新觀測數據 (UTC: {observation_for_prediction['datetime'].iloc[0]})")


    if observation_for_prediction is None or observation_for_prediction.empty:
        print("🚨 [Request] 預測數據來源為空，無法進行預測。")
        return render_template('index.html', max_aqi="N/A", aqi_predictions=[], city_name=city_name)

    # 必須確保 observation_for_prediction 包含所有 FEATURE_COLUMNS
    # 這裡我們信任模型訓練時的邏輯，假設缺失的數據會在模型訓練時被處理成 Nan 或其他預設值
    
    # ⭐⭐⭐ 核心預測邏輯 (使用 observation_for_prediction) ⭐⭐⭐
    try:
        future_predictions = predict_future_multi(
            TRAINED_MODELS,
            observation_for_prediction, # 使用最新或備用數據
            FEATURE_COLUMNS,
            POLLUTANT_PARAMS,
            hours=HOURS_TO_PREDICT
        )

        # 格式化結果
        future_predictions['datetime_local'] = future_predictions['datetime'].dt.tz_convert(LOCAL_TZ)
        max_aqi = int(future_predictions['aqi_pred'].max())

        aqi_predictions = [
            {'time': item['datetime_local'].strftime('%Y-%m-%d %H:%M'), 'aqi': int(item['aqi_pred'])}
            for item in future_predictions.to_dict(orient='records')
        ]
        
    except Exception as e:
        max_aqi = "N/A"
        aqi_predictions = []
        print(f"❌ [Request] 預測執行失敗: {e}") 

    return render_template('index.html', max_aqi=max_aqi, aqi_predictions=aqi_predictions, city_name=city_name)

if __name__ == '__main__':
    # 在本地環境運行時使用
    app.run(debug=True)
