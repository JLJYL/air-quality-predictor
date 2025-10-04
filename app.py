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

# 忽略所有警告 (例如 Pandas 的 SettingWithCopyWarning)
warnings.filterwarnings('ignore')

# =================================================================
# 模型與環境設置
# =================================================================
MODELS_DIR = 'models'
META_PATH = os.path.join(MODELS_DIR, 'model_meta.json')

# OpenAQ API Key (請使用您的金鑰)
API_KEY = "fb579916623e8483cd85344b14605c3109eea922202314c44b87a2df3b1fff77" 
HEADERS = {"X-API-Key": API_KEY}
BASE = "https://api.openaq.org/v3"

# 預設座標（高雄-前金）
TARGET_LAT = 22.6324 
TARGET_LON = 120.2954
DEFAULT_LOCATION_ID = 2395624 
DEFAULT_LOCATION_NAME = "Kaohsiung-Qianjin" 

# 監測參數與 ID
TARGET_PARAMS = ["co", "no2", "o3", "pm10", "pm25", "so2"]
PARAM_IDS = {"co": 8, "no2": 7, "o3": 10, "pm10": 1, "pm25": 2, "so2": 9}

# 數據容忍時間
TOL_MINUTES_PRIMARY = 5
TOL_MINUTES_FALLBACK = 60

# =================================================================
# 全域變數
# =================================================================
TRAINED_MODELS = {} 
LAST_OBSERVATION = None # 存儲模型訓練時的 Lag Template
FEATURE_COLUMNS = []
POLLUTANT_PARAMS = [] 
HOURS_TO_PREDICT = 24

CURRENT_OBSERVATION_AQI = "N/A"
CURRENT_OBSERVATION_TIME = "N/A"

current_location_id = None 
current_location_name = "System Initializing..."

LOCAL_TZ = "Asia/Taipei"
LAG_HOURS = [1, 2, 3, 6, 12, 24]
ROLLING_WINDOWS = [6, 12, 24]
POLLUTANT_TARGETS = ["pm25", "pm10", "o3", "no2", "so2", "co"] 

# AQI 分界點
AQI_BREAKPOINTS = {
    "pm25": [(0.0, 12.0, 0, 50), (12.1, 35.4, 51, 100), (35.5, 55.4, 101, 150), (55.5, 150.4, 151, 200)],
    "pm10": [(0, 54, 0, 50), (55, 154, 51, 100), (155, 254, 101, 150), (255, 354, 151, 200)],
    "o3": [(0, 54, 0, 50), (55, 70, 51, 100), (71, 85, 101, 150), (86, 105, 151, 200)],
    "co": [(0.0, 4.4, 0, 50), (4.5, 9.4, 51, 100), (9.5, 12.4, 101, 150), (12.5, 15.4, 151, 200)],
    "no2": [(0, 100, 0, 50), (101, 360, 51, 100), (361, 649, 101, 150), (650, 1249, 151, 200)],
    "so2": [(0, 35, 0, 50), (36, 75, 51, 100), (76, 185, 101, 150), (186, 304, 151, 200)],
}

# =================================================================
# OpenAQ 數據獲取與處理函數
# =================================================================

def get_location_meta(location_id: int):
    """獲取站點元數據，包括最後更新時間 (V3)"""
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

def get_nearest_location(lat: float, lon: float): 
    """強制搜尋地理上最近的單一監測站，忽略數據有效性 (V3)"""
    V3_LOCATIONS_URL = f"{BASE}/locations" 
    params = {"coordinates": f"{lat},{lon}"}
    
    try:
        r = requests.get(V3_LOCATIONS_URL, headers=HEADERS, params=params, timeout=10)
        r.raise_for_status()
        results = r.json().get("results", [])
        
        if not results:
            print(f"🚨 [Nearest] No station found globally for coordinates.")
            return None, None

        nearest_loc = results[0] 
        loc_id = int(nearest_loc["id"])
        loc_name = nearest_loc["name"]
        
        print(f"✅ [Nearest] Found the ABSOLUTELY closest station: {loc_name} (ID: {loc_id}).")
        return loc_id, loc_name
             
    except Exception as e:
        status_code = r.status_code if 'r' in locals() else 'N/A'
        error_detail = r.text if 'r' in locals() else str(e)
        print(f"❌ [Nearest] Failed to search for closest station. Status: {status_code}. Details: {error_detail}")
        return None, None
        
def get_location_latest_df(location_id: int) -> pd.DataFrame:
    """獲取站點所有參數的最新值 (V3)"""
    try:
        r = requests.get(f"{BASE}/locations/{location_id}/latest", headers=HEADERS, params={"limit": 1000}, timeout=10)
        if r.status_code == 404:
            return pd.DataFrame()
        r.raise_for_status()
        results = r.json().get("results", [])
        if not results:
            return pd.DataFrame()

        df = pd.json_normalize(results)
        df["parameter"] = df["parameter.name"].str.lower()
        df["value"] = df["value"]
        
        # 尋找最佳 UTC 時間戳
        df["ts_utc"] = pd.NaT
        for col in ["datetime.utc", "period.datetimeTo.utc"]:
            if col in df.columns:
                ts = pd.to_datetime(df[col], errors="coerce", utc=True)
                df["ts_utc"] = df["ts_utc"].where(df["ts_utc"].notna(), ts)

        return df[["parameter", "value", "ts_utc"]]
    except Exception as e:
        return pd.DataFrame()

def get_parameters_latest_df(location_id: int, target_params) -> pd.DataFrame:
    """獲取特定參數的最新值 (V3)"""
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
            df["value"] = df["value"]
            
            # 尋找最佳 UTC 時間戳
            df["ts_utc"] = pd.NaT
            for col in ["datetime.utc", "period.datetimeTo.utc"]:
                if col in df.columns:
                    ts = pd.to_datetime(df[col], errors="coerce", utc=True)
                    df["ts_utc"] = df["ts_utc"].where(df["ts_utc"].notna(), ts)

            rows.append(df[["parameter", "value", "ts_utc"]])

    except Exception as e:
        pass

    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)

def pick_batch_near(df: pd.DataFrame, t_ref: pd.Timestamp, tol_minutes: int) -> pd.DataFrame:
    """選取最接近 t_ref 且在容忍時間內的單批數據"""
    if df.empty or pd.isna(t_ref):
        return pd.DataFrame()

    df = df.copy()
    df["ts_utc"] = pd.to_datetime(df["ts_utc"], errors="coerce", utc=True)
    df["dt_diff"] = (df["ts_utc"] - t_ref).abs()

    tol = pd.Timedelta(minutes=tol_minutes)
    df = df[df["dt_diff"] <= tol].copy()
    if df.empty:
        return df

    df = df.sort_values(["parameter", "dt_diff", "ts_utc"], ascending=[True, True, False])
    df = df.drop_duplicates(subset=["parameter"], keep="first")
    return df[["parameter", "value", "ts_utc"]]

def fetch_latest_observation_data(location_id: int, target_params: list) -> pd.DataFrame:
    """獲取最新的觀測數據並轉換為單行寬格式"""
    meta = get_location_meta(location_id)
    if not meta or pd.isna(meta["last_utc"]):
        return pd.DataFrame()

    df_loc_latest = get_location_latest_df(location_id)
    if df_loc_latest.empty:
        return pd.DataFrame()

    t_star = meta["last_utc"] # 以站點元數據中的最新時間為準

    if pd.isna(t_star):
        return pd.DataFrame()
    
    # 嘗試在兩種容忍度下獲取數據
    df_at_batch = pick_batch_near(df_loc_latest, t_star, TOL_MINUTES_PRIMARY)
    if df_at_batch.empty:
        df_at_batch = pick_batch_near(df_loc_latest, t_star, TOL_MINUTES_FALLBACK)

    # 嘗試獲取缺失參數
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
    df_all = df_all.drop_duplicates(subset=["parameter"], keep="first")
    
    # 轉換為模型輸入格式
    observation = df_all.pivot_table(
        index='ts_utc', columns='parameter', values='value', aggfunc='first'
    ).reset_index().rename(columns={'ts_utc': 'datetime'})
    
    # 計算 AQI (缺失值會被視為 0)
    if not observation.empty:
        observation['aqi'] = observation.apply(
            lambda row: calculate_aqi(row, target_params, is_pred=False), axis=1
        )
        
    # 確保 'datetime' 總是 UTC-aware
    if not observation.empty:
        observation['datetime'] = pd.to_datetime(observation['datetime']).dt.tz_localize('UTC')

    return observation

def calculate_aqi_sub_index(param: str, concentration: float) -> float:
    """計算單一污染物濃度對應的 AQI 分指數"""
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
            # 簡單線性外推
            I_low, I_high = breakpoints[-1][2], breakpoints[-1][3]
            C_low, C_high = breakpoints[-1][0], breakpoints[-1][1]
            if C_high == C_low:
                return I_high
            I_rate = (I_high - I_low) / (C_high - C_low)
            I = I_high + I_rate * (concentration - C_high)
            return np.round(I)

    return np.nan

def calculate_aqi(row: pd.Series, params: list, is_pred=True) -> float:
    """計算最終 AQI (取最大分指數)"""
    sub_indices = []
    for p in params:
        col_name = f'{p}_pred' if is_pred else p
        if col_name in row and pd.notna(row[col_name]):
            sub_index = calculate_aqi_sub_index(p, row[col_name])
            if pd.notna(sub_index):
                sub_indices.append(sub_index)

    if not sub_indices:
        # 強制返回 0 避免 N/A
        return 0.0

    return np.max(sub_indices)

# =================================================================
# 預測與模型載入函數
# =================================================================

def predict_future_multi(models, last_data, feature_cols, pollutant_params, hours=24):
    """預測多個污染物 N 小時 (遞迴預測) 並計算 AQI"""
    predictions = []
    last_data['datetime'] = pd.to_datetime(last_data['datetime']).dt.tz_localize('UTC')
    last_datetime_aware = last_data['datetime'].iloc[0]
    
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
            # 強制將所有 NaN 輸入設為 0.0
            pred_input_list = [pred_features.get(col) if pd.notna(pred_features.get(col)) else 0.0 
                               for col in feature_cols]
            pred_input = np.array(pred_input_list, dtype=np.float64).reshape(1, -1)
            
            pred = model.predict(pred_input)[0]
            pred = max(0, pred) 

            current_prediction_row[f'{param}_pred'] = pred
            new_pollutant_values[param] = pred

        # 4. 計算預測 AQI
        predicted_aqi = calculate_aqi(pd.Series(current_prediction_row), pollutant_params, is_pred=True)
        current_prediction_row['aqi_pred'] = predicted_aqi
        new_pollutant_values['aqi'] = predicted_aqi

        # 5. 遞迴更新 Lag Features
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

def load_models_and_metadata():
    """載入模型和元數據"""
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
            LAST_OBSERVATION = pd.read_json(metadata['last_observation_json'], orient='records')

        TRAINED_MODELS = {}
        # ... (模型載入邏輯保持不變)
        params_to_remove = []
        for param in POLLUTANT_PARAMS:
            model_path = os.path.join(MODELS_DIR, f'{param}_model.json')
            if os.path.exists(model_path):
                model = xgb.XGBRegressor()
                model.load_model(model_path)
                TRAINED_MODELS[param] = model
            else:
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

def initialize_location_on_startup():
    """服務器啟動時，尋找預設座標的最近站點"""
    global current_location_id, current_location_name, DEFAULT_LOCATION_NAME
    
    print(f"🌐 [Startup] Initializing location using default coordinates: {TARGET_LAT}, {TARGET_LON}")
    loc_id, loc_name = get_nearest_location(TARGET_LAT, TARGET_LON)
    
    if loc_id is not None:
        current_location_id = loc_id
        current_location_name = loc_name
        print(f"✅ [Startup] Found a potential station: {current_location_name} (ID: {current_location_id})")
    else:
        current_location_id = None
        current_location_name = "Default (No Station Found Globally)"
        print(f"⚠️ [Startup] No station found globally. Initializing to 'None' station ID.")

# 啟動時執行初始化
initialize_location_on_startup()


app = Flask(__name__)

# 載入模型
with app.app_context():
    load_models_and_metadata() 

# =================================================================
# Flask 路由
# =================================================================
@app.route('/')
def index():
    global CURRENT_OBSERVATION_AQI, CURRENT_OBSERVATION_TIME, current_location_id, current_location_name, TARGET_LAT, TARGET_LON, LAST_OBSERVATION
    
    user_lat = request.args.get('lat', type=float)
    user_lon = request.args.get('lon', type=float)
    
    station_id = current_location_id
    station_name = current_location_name

    # 1. 處理用戶座標，尋找最近站點
    if user_lat is not None and user_lon is not None:
        print(f"✅ [Location] Using User Coordinates: LAT={user_lat}, LON={user_lon}")
        loc_id, loc_name = get_nearest_location(user_lat, user_lon)
        
        if loc_id is not None:
            station_id = loc_id
            station_name = loc_name
        else:
            station_id = None
            station_name = f"Location near {user_lat:.2f}, {user_lon:.2f}"
            print(f"⚠️ [Location] No station found near user. Cannot proceed to prediction.")
    
    else:
        print(f"⚠️ [Location] No user coordinates found. Using current station: {station_name}")
        
    # 2. 致命錯誤檢查：無站點或模型未載入
    if station_id is None or not TRAINED_MODELS or LAST_OBSERVATION is None:
        max_aqi = 0
        display_name = station_name if station_id is None else "System Error (No Model)"
        print("🚨 [Fatal] Cannot proceed. No station found or models/lag data missing.")
        return render_template('index.html', 
                                max_aqi=max_aqi, 
                                aqi_predictions=[], 
                                city_name=display_name, 
                                current_obs_time="N/A",
                                is_fallback=True)
    
    # 3. 嘗試獲取最新的觀測數據
    current_observation_raw = fetch_latest_observation_data(station_id, POLLUTANT_TARGETS)

    # --- 處理觀測數據 (確保不為 N/A) ---
    obs_aqi_val = 0
    obs_time_val = None
    
    if not current_observation_raw.empty:
         obs_aqi_val = current_observation_raw['aqi'].iloc[0] if 'aqi' in current_observation_raw.columns else 0
         obs_time_val = current_observation_raw['datetime'].iloc[0]

    CURRENT_OBSERVATION_AQI = int(obs_aqi_val) if pd.notna(obs_aqi_val) and obs_aqi_val >= 0 else 0
        
    if pd.notna(obs_time_val):
        CURRENT_OBSERVATION_TIME = obs_time_val.tz_convert(LOCAL_TZ).strftime('%Y-%m-%d %H:%M')
    else:
        CURRENT_OBSERVATION_TIME = "N/A"
        
    # 4. 準備預測數據：從模型 Lag Template 開始
    observation_for_prediction = LAST_OBSERVATION.iloc[:1].copy()
    is_valid_for_prediction = False
    
    # 使用當前時間覆蓋歷史時間
    if obs_time_val is not None:
        dt_val = obs_time_val
        if pd.to_datetime(dt_val).tz is not None:
             dt_val = pd.to_datetime(dt_val).tz_convert(None) 
        observation_for_prediction['datetime'] = dt_val
        
    # --- 5. 核心：強制填充所有 Lag Features 為當前觀測值 ---
    if not current_observation_raw.empty:
        latest_row = current_observation_raw.iloc[0]
        
        # 5a. 填充當前觀測值 (T0) 和基礎氣象數據
        for col in latest_row.index:
            if col in observation_for_prediction.columns and not any(s in col for s in ['lag_', 'rolling_']):
                 if col in POLLUTANT_TARGETS or col == 'aqi' or col in ['temperature', 'humidity', 'pressure']:
                      # 覆蓋當前欄位 T0
                      observation_for_prediction[col] = latest_row[col]

        print("🚨 [Prediction Input] Filling ALL lag features with current observed values.")
        
        # 5b. 填充所有 Lag features 和 Rolling Features
        for param in POLLUTANT_TARGETS + ['aqi']:
            if param in latest_row.index and pd.notna(latest_row[param]):
                current_value = latest_row[param]
                
                # 遍歷所有 Lag_*h 欄位
                for lag_hour in LAG_HOURS:
                    lag_col = f'{param}_lag_{lag_hour}h'
                    if lag_col in observation_for_prediction.columns:
                        observation_for_prediction[lag_col] = current_value
                
                # 遍歷所有 Rolling_*h 欄位
                for rolling_window in ROLLING_WINDOWS:
                    roll_col = f'{param}_rolling_{rolling_window}h'
                    if roll_col in observation_for_prediction.columns:
                        observation_for_prediction[roll_col] = current_value
        
        is_valid_for_prediction = True

    # 6. 執行預測
    max_aqi = CURRENT_OBSERVATION_AQI
    aqi_predictions = []
    is_fallback_mode = True 

    if is_valid_for_prediction and observation_for_prediction is not None:
        try:
            print(f"🚀 [Prediction] Forcing prediction using a STUB data frame.")
            future_predictions = predict_future_multi(
                TRAINED_MODELS,
                observation_for_prediction,
                FEATURE_COLUMNS,
                POLLUTANT_PARAMS,
                hours=HOURS_TO_PREDICT
            )

            # 處理預測結果
            future_predictions['datetime_local'] = future_predictions['datetime'].dt.tz_convert(LOCAL_TZ)
            predictions_df = future_predictions[['datetime_local', 'aqi_pred']].copy()
            max_aqi_pred = int(predictions_df['aqi_pred'].max())
            max_aqi = max(CURRENT_OBSERVATION_AQI, max_aqi_pred)
            
            predictions_df['aqi'] = predictions_df['aqi_pred'].apply(
                 lambda x: int(x) if pd.notna(x) and x >= 0 else 0
            ).astype(object)

            aqi_predictions = [
                {
                    'time': item['datetime_local'].strftime('%Y-%m-%d %H:%M'), 
                    'aqi': item['aqi']
                }
                for item in predictions_df.to_dict(orient='records')
            ]
            
            # 將當前觀測值加入到圖表開頭
            aqi_predictions.insert(0, {
                 'time': CURRENT_OBSERVATION_TIME,
                 'aqi': CURRENT_OBSERVATION_AQI,
                 'is_obs': True
            })
            
            is_fallback_mode = False
            print("✅ [Request] Prediction executed and displayed.")


        except Exception as e:
            max_aqi = CURRENT_OBSERVATION_AQI
            is_fallback_mode = True
            print(f"❌ [Request] Prediction execution failed ({e}), falling back to latest observed AQI.") 
            
    # 7. 預測失敗的回退邏輯 (僅顯示觀測值)
    if is_fallback_mode and max_aqi >= 0:
             print("🚨 [Request] Final result using fallback mode (only observed data).")
             aqi_predictions = [{
                 'time': CURRENT_OBSERVATION_TIME,
                 'aqi': max_aqi,
                 'is_obs': True 
               }]
             
    # 8. 渲染頁面
    return render_template('index.html', 
                            max_aqi=max_aqi, 
                            aqi_predictions=aqi_predictions, 
                            city_name=station_name, 
                            current_obs_time=CURRENT_OBSERVATION_TIME,
                            is_fallback=is_fallback_mode)

if __name__ == '__main__':
    app.run(debug=True)
