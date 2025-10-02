# app.py - 供 Render 部署使用 (已移除耗時的訓練邏輯)

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
import io 
from datetime import timedelta, timezone
from flask import Flask, render_template

# 忽略警告
warnings.filterwarnings('ignore')

# 模型與元數據路徑
MODELS_DIR = 'models'
META_PATH = os.path.join(MODELS_DIR, 'model_meta.json')

# =================================================================
# OpenAQ V3 常數設定 (新增)
# =================================================================
# 請務必使用您的實際 API Key，這裡使用範例 Key
API_KEY = "98765df2082f04dc9449e305bc736e93624b66e250fa9dfabcca53b31fc11647"
headers = {"X-API-Key": API_KEY}
BASE = "https://api.openaq.org/v3"
LOCATION_ID = 2395624  # 高雄市-前金

TARGET_PARAMS = ["co", "no2", "o3", "pm10", "pm25", "so2"]
PARAM_IDS = {"co": 8, "no2": 7, "o3": 10, "pm10": 1, "pm25": 2, "so2": 9}

# 對齊時間允許容忍（先用 ±5 分找批次，找不到再放寬到 ±60 分）
TOL_MINUTES_PRIMARY = 5
TOL_MINUTES_FALLBACK = 60


# =================================================================
# 全域變數 - 改為從檔案載入
# =================================================================
TRAINED_MODELS = {}
LAST_OBSERVATION = None # 載入訓練時的最後一筆數據，用於提供 Lag/Weather 特徵
FEATURE_COLUMNS = []
POLLUTANT_PARAMS = [] # 實際找到並訓練的模型參數
HOURS_TO_PREDICT = 24

# =================================================================
# 常數設定 (僅保留與預測相關的常數)
# =================================================================
LOCAL_TZ = "Asia/Taipei"
LAG_HOURS = [1, 2, 3, 6, 12, 24] # 預測遞迴需要這些參數
ROLLING_WINDOWS = [6, 12, 24] # 預測遞迴需要這些參數
POLLUTANT_TARGETS = ["pm25", "pm10", "o3", "no2", "so2", "co"] # 用於 AQI 計算

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
# 輔助函式: OpenAQ V3 數據抓取 (整合自第一個腳本)
# =================================================================

def get_location_meta(location_id: int):
    """獲取站點的元數據，包含最後更新時間。"""
    r = requests.get(f"{BASE}/locations/{location_id}", headers=headers)
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


def get_location_latest_df(location_id: int) -> pd.DataFrame:
    """站點各參數的『最新值清單』→ 正規化時間成 ts_utc / ts_local"""
    r = requests.get(f"{BASE}/locations/{location_id}/latest", headers=headers, params={"limit": 1000})
    if r.status_code == 404:
        return pd.DataFrame()
    r.raise_for_status()
    results = r.json().get("results", [])
    if not results:
        return pd.DataFrame()

    df = pd.json_normalize(results)

    # 參數名與單位
    if "parameter.name" in df.columns:
        df["parameter"] = df["parameter.name"].str.lower()
    elif "parameter" in df.columns:
        df["parameter"] = df["parameter"].str.lower()
    else:
        df["parameter"] = None
    # 處理單位欄位，以適應不同 API 返回格式
    df["units"] = df.get("parameter.units") if "parameter.units" in df.columns else df.get("units")
    df["value"] = df["value"]

    # 取代表該筆的UTC時間（依優先序）
    df["ts_utc"] = pd.NaT
    for col in ["datetime.utc", "period.datetimeTo.utc", "period.datetimeFrom.utc"]:
        if col in df.columns:
            ts = pd.to_datetime(df[col], errors="coerce", utc=True)
            df["ts_utc"] = df["ts_utc"].where(df["ts_utc"].notna(), ts)

    # 地方時間（若有）
    local_col = None
    for c in ["datetime.local", "period.datetimeTo.local", "period.datetimeFrom.local"]:
        if c in df.columns:
            local_col = c
            break
    df["ts_local"] = df[local_col] if local_col else None

    return df[["parameter", "value", "units", "ts_utc", "ts_local"]]


def get_parameters_latest_df(location_id: int, target_params) -> pd.DataFrame:
    """用 /parameters/{pid}/latest?locationId= 拿各參數『最新值』並合併"""
    rows = []
    for p in target_params:
        pid = PARAM_IDS[p]
        r = requests.get(
            f"{BASE}/parameters/{pid}/latest",
            headers=headers,
            params={"locationId": location_id, "limit": 50},
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
        df["units"] = df.get("parameter.units") if "parameter.units" in df.columns else df.get("units")
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
        df["ts_local"] = df[local_col] if local_col else None

        rows.append(df[["parameter", "value", "units", "ts_utc", "ts_local"]])

    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def pick_batch_near(df: pd.DataFrame, t_ref: pd.Timestamp, tol_minutes: int) -> pd.DataFrame:
    """從 DataFrame 中選出時間最接近 t_ref 且在容忍度內的每種參數的單一觀測值。"""
    if df.empty or pd.isna(t_ref):
        return pd.DataFrame()

    df = df.copy()

    # 確保 ts_utc 欄位中的每個值都是單一時間戳（處理 list/ndarray 的情況）
    def _scalarize(v):
        if isinstance(v, (list, tuple, np.ndarray)):
            return v[0] if len(v) else None
        return v

    df["ts_utc"] = df["ts_utc"].map(_scalarize)
    df["ts_utc"] = pd.to_datetime(df["ts_utc"], errors="coerce", utc=True)

    # 接著就能安全做時間距離比較
    df["dt_diff"] = (df["ts_utc"] - t_ref).abs()

    tol = pd.Timedelta(minutes=tol_minutes)
    df = df[df["dt_diff"] <= tol]
    if df.empty:
        return df

    # 排序：優先參數、時間差最小、時間戳最新
    df = df.sort_values(["parameter", "dt_diff", "ts_utc"], ascending=[True, True, False])
    # 每個參數只留一筆
    df = df.drop_duplicates(subset=["parameter"], keep="first")
    return df[["parameter", "value", "units", "ts_utc", "ts_local"]]


def fetch_latest_data_for_prediction(location_id: int, target_params: list, historical_data: pd.DataFrame) -> pd.DataFrame:
    """
    從 OpenAQ 抓取最新一批的目標污染物資料，並與歷史數據合併以形成預測輸入。
    """
    if historical_data.empty:
        print("⚠️ [Merge] 缺少訓練時的歷史數據 (LAST_OBSERVATION)，無法提供 Lag/Weather 特徵。")
        return pd.DataFrame()
    
    try:
        meta = get_location_meta(location_id)
        t_star_loc = meta["last_utc"]
        
        # 1. 站點『最新值清單』
        df_loc_latest = get_location_latest_df(location_id)
        if df_loc_latest.empty:
            print("⚠️ [Fetch] /locations/{id}/latest 沒有任何資料。")
            return pd.DataFrame()

        # 2. 確定對齊時間
        t_star_latest = df_loc_latest["ts_utc"].max()
        t_star = t_star_latest if pd.notna(t_star_latest) else t_star_loc
        if pd.isna(t_star):
             print("⚠️ [Fetch] 無法確定最新批次時間。")
             return pd.DataFrame()

        # 3. 找接近 t_star 的一批數據 (df_all)
        df_at_batch = pick_batch_near(df_loc_latest, t_star, TOL_MINUTES_PRIMARY)
        if df_at_batch.empty:
            df_at_batch = pick_batch_near(df_loc_latest, t_star, TOL_MINUTES_FALLBACK)

        have = set(df_at_batch["parameter"].str.lower().tolist()) if not df_at_batch.empty else set()
        missing = [p for p in TARGET_PARAMS if p not in have]
        df_param_batch = pd.DataFrame()

        if missing:
            df_param_latest = get_parameters_latest_df(location_id, missing)
            df_param_batch = pick_batch_near(df_param_latest, t_star, TOL_MINUTES_PRIMARY)
            if df_param_batch.empty:
                df_param_batch = pick_batch_near(df_param_latest, t_star, TOL_MINUTES_FALLBACK)
        
        frames = []
        if not df_at_batch.empty:
            frames.append(df_at_batch)
        if not df_param_batch.empty:
            frames.append(df_param_batch)
        
        if not frames:
            print(f"⚠️ [Fetch] 在 {t_star} 附近，六項污染物都沒有資料。")
            return pd.DataFrame()

        # 最終合併所有最新數據，確保每個參數只有一筆
        df_all = pd.concat(frames, ignore_index=True)
        df_all["dt_diff"] = (df_all["ts_utc"] - t_star).abs()
        df_all = df_all.sort_values(["parameter", "dt_diff", "ts_utc"], ascending=[True, True, False])
        df_all = df_all.drop_duplicates(subset=["parameter"], keep="first")
        
        # 4. 建立單行最新觀測數據 (只包含時間和污染物濃度)
        final_row = {'datetime': t_star}
        for _, row in df_all.iterrows():
            # 使用 _value 結尾來匹配模型訓練時的命名規範
            final_row[f'{row["parameter"]}_value'] = row["value"] 

        current_obs_df = pd.DataFrame([final_row])

        # 5. 合併最新觀測與歷史特徵 (重點步驟)
        
        # 複製歷史特徵 (包含所有 lag/weather/seasonal features)
        # to_frame().T 確保它是一個單行 DataFrame
        input_df = historical_data.copy().iloc[0].to_frame().T
        
        # 從歷史數據中移除舊的 datetime 和舊的污染物值
        pollutant_value_cols = [f'{p}_value' for p in TARGET_PARAMS if f'{p}_value' in input_df.columns]
        input_df = input_df.drop(columns=['datetime'] + pollutant_value_cols, errors='ignore')
        
        # 將最新抓到的數據（時間和污染物值）合併到 input_df 中
        # concat 將會把最新的數據作為新的一行
        final_input_df = pd.concat([input_df, current_obs_df], axis=1).iloc[0].to_frame().T
        
        # 確保 final_input_df 只有需要的欄位 (feature_cols + 'datetime')
        required_cols = list(FEATURE_COLUMNS) + ['datetime']
        final_input_df = final_input_df.reindex(columns=required_cols, fill_value=np.nan)

        print(f"✅ [Fetch] 成功抓取最新批次資料，時間: {t_star.strftime('%Y-%m-%d %H:%M:%S%Z')}")
        return final_input_df

    except requests.exceptions.RequestException as e:
        print(f"❌ [Fetch] OpenAQ API 請求失敗: {e}")
        return pd.DataFrame()
    except Exception as e:
        print(f"❌ [Fetch] 抓取最新資料或合併失敗: {e}")
        return pd.DataFrame()


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

def calculate_aqi(row: pd.Series, params: list) -> int:
    """根據多個污染物濃度計算最終 AQI (取最大子指數)"""
    sub_indices = []
    for p in params:
        # 檢查預測值和觀測值欄位
        col_pred = f'{p}_pred'
        col_obs = f'{p}_value'
        
        # 預測值優先
        if col_pred in row and not pd.isna(row[col_pred]):
            concentration = row[col_pred]
        elif col_obs in row and not pd.isna(row[col_obs]):
            concentration = row[col_obs]
        else:
            continue

        sub_index = calculate_aqi_sub_index(p, concentration)
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

    # last_data 現在是單行 DataFrame，需要確保其時間格式和時區設定正確
    last_data['datetime'] = pd.to_datetime(last_data['datetime'])

    # 修正時區問題: 確保日期時間為 UTC-aware。
    if last_data['datetime'].dt.tz is not None:
        last_data['datetime'] = last_data['datetime'].dt.tz_convert('UTC')
    else:
        try:
            last_data['datetime'] = last_data['datetime'].dt.tz_localize('UTC')
        except Exception as e:
            if "tz-aware" in str(e):
                print("⚠️ [TZ Fix] .dt.tz 檢查失效，實際為 tz-aware，使用 tz_convert 修正。")
                last_data['datetime'] = last_data['datetime'].dt.tz_convert('UTC')
            else:
                raise e


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
        pred_features['day_of_year'] = future_time.timetuple().tm_yday # 使用 day_of_year
        pred_features['is_weekend'] = int(future_time.dayofweek in [5, 6])
        pred_features['hour_sin'] = np.sin(2 * np.pi * future_time.hour / 24)
        pred_features['hour_cos'] = np.cos(2 * np.pi * future_time.hour / 24)
        pred_features['day_sin'] = np.sin(2 * np.pi * pred_features['day_of_year'] / 365)
        pred_features['day_cos'] = np.cos(2 * np.pi * pred_features['day_of_year'] / 365)

        # 2. 模擬未來天氣變化 (使用前一小時的天氣值進行隨機擾動)
        if has_weather:
            # 確保種子與時間相關，讓每次運行結果一致，但每個小時不同
            np.random.seed(future_time.year + future_time.month + future_time.day + future_time.hour + 42)
            for w_col in weather_feature_names:
                base_value = current_data_dict.get(w_col)
                if base_value is not None and not np.isnan(base_value):
                    # 模擬輕微隨機變化 (使用前一小時的天氣值進行遞迴)
                    new_weather_value = base_value + np.random.normal(0, 0.5)
                    pred_features[w_col] = new_weather_value
                    # 將新的天氣值更新到 current_data_dict，以便下一小時使用
                    current_data_dict[w_col] = new_weather_value

        current_prediction_row = {'datetime': future_time}
        new_pollutant_values = {}

        # 3. 預測所有污染物
        for param in pollutant_params:
            model = models[param]
            # 確保輸入特徵的順序與模型訓練時一致
            pred_input_data = {col: pred_features.get(col, 0) for col in feature_cols}
            pred_input = np.array([pred_input_data[col] for col in feature_cols]).reshape(1, -1)
            pred = model.predict(pred_input)[0]
            pred = max(0, pred) # 濃度不能小於 0

            current_prediction_row[f'{param}_pred'] = pred
            new_pollutant_values[param] = pred

        # 4. 計算預測的 AQI
        predicted_aqi = calculate_aqi(pd.Series(current_prediction_row), pollutant_params)
        current_prediction_row['aqi_pred'] = predicted_aqi
        new_pollutant_values['aqi'] = predicted_aqi

        predictions.append(current_prediction_row)

        # 5. 更新滯後特徵 (遞迴更新)
        for param in pollutant_params + ['aqi']:
            # 從最大的 Lag 開始更新，避免覆蓋
            for i in range(len(LAG_HOURS) - 1, 0, -1):
                lag_current = LAG_HOURS[i]
                lag_prev = LAG_HOURS[i-1]
                lag_current_col = f'{param}_lag_{lag_current}h'
                lag_prev_col = f'{param}_lag_{lag_prev}h'

                if lag_current_col in current_data_dict and lag_prev_col in current_data_dict:
                    current_data_dict[lag_current_col] = current_data_dict[lag_prev_col]

            # 更新 1 小時滯後特徵為當前預測值
            if f'{param}_lag_1h' in current_data_dict and param in new_pollutant_values:
                current_data_dict[f'{param}_lag_1h'] = new_pollutant_values[param]

        # 6. 滾動平均/標準差特徵無法在遞迴中準確更新，這裡保持省略


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
        # 1. 載入元數據
        with open(META_PATH, 'r', encoding='utf-8') as f:
            metadata = json.load(f)

        POLLUTANT_PARAMS = metadata.get('pollutant_params', [])
        FEATURE_COLUMNS = metadata.get('feature_columns', [])

        # 將最後一筆數據的 JSON 轉換回 DataFrame
        if 'last_observation_json' in metadata:
            # 使用 StringIO 模擬檔案讀取，確保格式正確
            LAST_OBSERVATION = pd.read_json(io.StringIO(metadata['last_observation_json']), orient='records')

        # 2. 載入 XGBoost 模型
        TRAINED_MODELS = {}
        params_to_check = list(POLLUTANT_PARAMS)

        for param in params_to_check:
            model_path = os.path.join(MODELS_DIR, f'{param}_model.json')
            if os.path.exists(model_path):
                model = xgb.XGBRegressor()
                model.load_model(model_path)
                TRAINED_MODELS[param] = model
            else:
                print(f"❌ [Load] 找不到 {param} 的模型檔案: {model_path}")

        # 最終更新 POLLUTANT_PARAMS，只保留成功載入模型的
        POLLUTANT_PARAMS = list(TRAINED_MODELS.keys())

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

# 應用程式啟動時，立即執行模型載入 (快速)
with app.app_context():
    load_models_and_metadata()

@app.route('/')
def index():
    global LAST_OBSERVATION # 允許讀取全域變數
    city_name = "高雄"
    
    # 檢查模型是否成功載入
    if TRAINED_MODELS and LAST_OBSERVATION is not None and not LAST_OBSERVATION.empty:
        try:
            # 1. 🚨 即時抓取最新觀測數據並與歷史數據合併 🚨
            # 獲取單行且更新了最新污染物值和時間戳的 DataFrame
            final_input_data = fetch_latest_data_for_prediction(
                LOCATION_ID,
                TARGET_PARAMS,
                LAST_OBSERVATION.copy() # 傳遞副本
            )

            if final_input_data.empty or 'datetime' not in final_input_data.columns:
                max_aqi = "N/A"
                aqi_predictions = []
                print("🚨 [Request] 無法取得最新數據，或合併數據格式錯誤。")
                
            else:
                # 2. 執行預測
                future_predictions = predict_future_multi(
                    TRAINED_MODELS,
                    final_input_data, # 使用即時且合併後的數據
                    FEATURE_COLUMNS,
                    POLLUTANT_PARAMS,
                    hours=HOURS_TO_PREDICT
                )

                # 格式化結果
                future_predictions['datetime_local'] = future_predictions['datetime'].dt.tz_convert(LOCAL_TZ)

                # 確保 aqi_pred 是數字再取 max
                if not future_predictions.empty and future_predictions['aqi_pred'].dtype in [np.int64, np.float64]:
                    max_aqi = int(future_predictions['aqi_pred'].max())
                else:
                    max_aqi = "N/A"

                aqi_predictions = [
                    {'time': item['datetime_local'].strftime('%Y-%m-%d %H:%M'), 'aqi': int(item['aqi_pred'])}
                    for item in future_predictions.to_dict(orient='records')
                ]

        except Exception as e:
            max_aqi = "N/A"
            aqi_predictions = []
            print(f"❌ [Request] 預測執行失敗: {e}")

    else:
        max_aqi = "N/A"
        aqi_predictions = []
        print("🚨 [Request] 模型或數據尚未初始化，無法進行預測。")

    return render_template('index.html', max_aqi=max_aqi, aqi_predictions=aqi_predictions, city_name=city_name)

if __name__ == '__main__':
    # 在本地環境運行時使用
    app.run(debug=True)
