# app.py - 最終修正版本 (請替換掉您所有的 app.py 內容)

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

LOCATION_ID = 2395624 # 高雄市-前金 (請替換為您需要的站點 ID)
TARGET_PARAMS = ["co", "no2", "o3", "pm10", "pm25", "so2"]
PARAM_IDS = {"co": 8, "no2": 7, "o3": 10, "pm10": 1, "pm25": 2, "so2": 9}

TOL_MINUTES_PRIMARY = 5
TOL_MINUTES_FALLBACK = 60

# =================================================================
# 全域變數
# =================================================================
TRAINED_MODELS = {} 
LAST_OBSERVATION = None 
FEATURE_COLUMNS = []
POLLUTANT_PARAMS = [] 
HOURS_TO_PREDICT = 24

# 儲存最新的觀測數據 (用於回退)
CURRENT_OBSERVATION_AQI = "N/A"
CURRENT_OBSERVATION_TIME = "N/A"

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
    """根據多個污染物濃度計算最終 AQI (取最大子指數)"""
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
# 預測函式
# =================================================================

def predict_future_multi(models, last_data, feature_cols, pollutant_params, hours=24):
    """預測未來 N 小時的多個目標污染物 (遞迴預測) 並計算 AQI"""
    predictions = []

    last_data['datetime'] = pd.to_datetime(last_data['datetime']).dt.tz_localize('UTC')
        
    last_datetime_aware = last_data['datetime'].iloc[0]
    
    # 檢查並補齊特徵列，缺失則以 np.nan 填充 (讓 XGBoost 處理)
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
                if base_value is not None and pd.notna(base_value):
                    new_weather_value = base_value + np.random.normal(0, 0.5) 
                    pred_features[w_col] = new_weather_value
                    current_data_dict[w_col] = new_weather_value 
                else:
                    pred_features[w_col] = np.nan
                    current_data_dict[w_col] = np.nan 


        current_prediction_row = {'datetime': future_time}
        new_pollutant_values = {}

        # 3. 預測所有污染物
        for param in pollutant_params:
            model = models[param]
            # 確保輸入是模型期望的特徵順序
            pred_input_list = [pred_features.get(col) for col in feature_cols]
            pred_input = np.array(pred_input_list, dtype=np.float64).reshape(1, -1)
            
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
    global CURRENT_OBSERVATION_AQI, CURRENT_OBSERVATION_TIME
    city_name = "高雄"
    
    # 1. 嘗試即時抓取最新觀測數據
    current_observation_raw = fetch_latest_observation_data(LOCATION_ID, POLLUTANT_TARGETS)

    # 提取最新觀測 AQI 以供回退 (FALLBACK)
    if not current_observation_raw.empty and 'aqi' in current_observation_raw.columns:
        obs_aqi_val = current_observation_raw['aqi'].iloc[0]
        obs_time_val = current_observation_raw['datetime'].iloc[0]
        
        CURRENT_OBSERVATION_AQI = int(obs_aqi_val) if pd.notna(obs_aqi_val) else "N/A"
        
        if pd.notna(obs_time_val):
            if obs_time_val.tz is None:
                 obs_time_val = obs_time_val.tz_localize('UTC')
            
            CURRENT_OBSERVATION_TIME = obs_time_val.tz_convert(LOCAL_TZ).strftime('%Y-%m-%d %H:%M')
        else:
             CURRENT_OBSERVATION_TIME = "N/A"
    
    
    # 2. 準備預測數據
    observation_for_prediction = None
    is_valid_for_prediction = False

    if not current_observation_raw.empty and LAST_OBSERVATION is not None and not LAST_OBSERVATION.empty:
        # 整合最新觀測數據到滯後數據中
        observation_for_prediction = LAST_OBSERVATION.iloc[:1].copy() 
        latest_row = current_observation_raw.iloc[0]
        observation_for_prediction['datetime'] = latest_row['datetime']
        
        for col in latest_row.index:
            if col in observation_for_prediction.columns and not any(s in col for s in ['lag_', 'rolling_']):
                 if col in POLLUTANT_TARGETS or col == 'aqi' or col in ['temperature', 'humidity', 'pressure']:
                     observation_for_prediction[col] = latest_row[col]
            
        # 檢查是否有足夠的特徵列
        if all(col in observation_for_prediction.columns for col in FEATURE_COLUMNS):
             is_valid_for_prediction = True
             print("✅ [Request] 數據準備完成，準備進行預測。")
        else:
             print("⚠️ [Request] 整合數據後缺少模型所需的特徵列，將使用回退數據。")
    else:
        print("🚨 [Request] 無法取得最新觀測數據或模型滯後數據，無法進行預測。")


    # 3. 進行預測或回退
    max_aqi = CURRENT_OBSERVATION_AQI
    aqi_predictions = []
    
    is_fallback_mode = True

    if TRAINED_MODELS and POLLUTANT_PARAMS and is_valid_for_prediction and observation_for_prediction is not None:
        try:
            # 最終檢查時間區
            observation_for_prediction['datetime'] = pd.to_datetime(observation_for_prediction['datetime'])
            if observation_for_prediction['datetime'].dt.tz is not None:
                 observation_for_prediction['datetime'] = observation_for_prediction['datetime'].dt.tz_localize(None)

            future_predictions = predict_future_multi(
                TRAINED_MODELS,
                observation_for_prediction,
                FEATURE_COLUMNS,
                POLLUTANT_PARAMS,
                hours=HOURS_TO_PREDICT
            )

            future_predictions['datetime_local'] = future_predictions['datetime'].dt.tz_convert(LOCAL_TZ)
            
            # 處理 NaN 值並計算 Max AQI
            predictions_df = future_predictions[['datetime_local', 'aqi_pred']].copy()
            max_aqi_val = predictions_df['aqi_pred'].max()
            max_aqi = int(max_aqi_val) if pd.notna(max_aqi_val) else CURRENT_OBSERVATION_AQI
            
            # 將 NaN 替換為字串 "N/A"，並將有效數值轉換為整數
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
                 print("✅ [Request] 預測成功！")
            else:
                 # 預測列表為空，退回到顯示當前 AQI
                 max_aqi = CURRENT_OBSERVATION_AQI
                 is_fallback_mode = True
                 print("⚠️ [Request] 預測列表為空，退回顯示最新觀測 AQI。")


        except Exception as e:
            # 預測流程失敗，回退
            max_aqi = CURRENT_OBSERVATION_AQI
            aqi_predictions = []
            is_fallback_mode = True
            print(f"❌ [Request] 預測執行失敗 ({e})，退回顯示最新觀測 AQI。") 
            
    if is_fallback_mode:
         # 模型未載入或數據無效，產生一個簡單的當前觀測數據列表作為回退
         print("🚨 [Request] 最終使用回退模式。")
         max_aqi = CURRENT_OBSERVATION_AQI
         
         # 創建一個只包含當前觀測值的列表，告訴前端這是觀測值
         if max_aqi != "N/A":
             aqi_predictions = [{
                'time': CURRENT_OBSERVATION_TIME,
                'aqi': max_aqi,
                'is_obs': True # 新增標記
             }]

    # 4. 渲染模板
    return render_template('index.html', 
                           max_aqi=max_aqi, 
                           aqi_predictions=aqi_predictions, 
                           city_name=city_name,
                           current_obs_time=CURRENT_OBSERVATION_TIME,
                           is_fallback=is_fallback_mode)

if __name__ == '__main__':
    app.run(debug=True)
