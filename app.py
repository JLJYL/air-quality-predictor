import requests
import pandas as pd
import numpy as np
import os
import json
import warnings
from datetime import timedelta, timezone, datetime
from flask import Flask, render_template, request
from pytz import timezone as pytz_timezone # 導入 pytz 的 timezone 模組

# 忽略所有警告
warnings.filterwarnings('ignore')

# --- 常數定義 ---

# OpenAQ API Key (請注意：此 key 應保密，這裡僅供範例)
API_KEY = "fb579916623e8483cd85344b14605c3109eea922202314c44b87a2df3b1fff77" 
HEADERS = {"X-API-Key": API_KEY}
BASE = "https://api.openaq.org/v3"

# 目標污染物參數
TARGET_PARAMS = ["co", "no2", "o3", "pm10", "pm25", "so2"]
POLLUTANT_IDS = {
    "co": 8, "no2": 7, "o3": 10, "pm10": 1, "pm25": 2, "so2": 9
}

# 預設位置：台中市（經緯度）
DEFAULT_LAT = 24.1477
DEFAULT_LON = 120.6736
DEFAULT_LOCATION_NAME = "台中市"
LOCAL_TZ = pytz_timezone('Asia/Taipei') # 台灣時區

# AQI Breakpoints (基於 EPA 2024 標準，單位：μg/m³ for PM, ppb for gases except CO)
# PM2.5, PM10: µg/m³
# CO: ppm (mg/m³ to ppm conversion handled in calculation if necessary)
# SO2, NO2, O3: ppb
AQI_BREAKPOINTS = {
    'pm25': [0, 15.4, 35.4, 55.4, 150.4, 250.4, 350.4, 500.4],
    'pm10': [0, 54, 154, 254, 354, 424, 504, 604],
    'o3': [0, 59, 75, 95, 105, 200, 404, 604], # 8-hour average (ppb)
    'co': [0, 4.4, 9.4, 12.4, 15.4, 30.4, 40.4, 50.4], # 8-hour average (ppm)
    'so2': [0, 35, 75, 185, 304, 604, 804, 1004], # 1-hour average (ppb)
    'no2': [0, 53, 100, 360, 649, 1249, 1649, 2049], # 1-hour average (ppb)
    'aqi': [0, 50, 100, 150, 200, 300, 400, 500],
}

# --- AQI 計算邏輯 (從原有程式碼保留) ---

def get_concentration_unit(param):
    """根據參數返回其在 OpenAQ 中的單位，用於數據清洗。"""
    # OpenAQ often reports PM in µg/m³ and gases in ppb/ppm.
    if param in ['pm25', 'pm10', 'so2', 'no2', 'o3']:
        # For simplicity, assume OpenAQ provides these in the units required by EPA (µg/m³ or ppb)
        return 'µg/m³' if param in ['pm25', 'pm10'] else 'ppb'
    elif param == 'co':
        return 'ppm' # CO typically reported in ppm

def calculate_aqi_sub_index(param, concentration):
    """
    計算單一污染物的 AQI 子指標。
    :param param: 污染物名稱 (e.g., 'pm25')
    :param concentration: 污染物濃度值
    :return: 該污染物對應的 AQI 子指標值
    """
    if pd.isna(concentration) or concentration < 0:
        return 0

    try:
        C_i = concentration
        Bp = AQI_BREAKPOINTS[param]
        Ip = AQI_BREAKPOINTS['aqi']
        
        # 尋找濃度 C_i 所在的範圍 [Bp_low, Bp_high]
        i = 0
        while i < len(Bp) - 1 and C_i > Bp[i+1]:
            i += 1
        
        # 處理超過最高級別的情況
        if i >= len(Bp) - 1:
            i = len(Bp) - 2 # 使用最高級別的範圍
            
        Bp_low = Bp[i]
        Bp_high = Bp[i+1]
        Ip_low = Ip[i]
        Ip_high = Ip[i+1]

        # 應用線性轉換公式：Ip = [(I_high - I_low) / (B_high - B_low)] * (C_i - B_low) + I_low
        if Bp_high == Bp_low:
             # 避免除以零，通常只在最高範圍發生，此時 Ip_high
            return Ip_high

        sub_index = ((Ip_high - Ip_low) / (Bp_high - Bp_low)) * (C_i - Bp_low) + Ip_low
        return round(sub_index)
        
    except Exception as e:
        print(f"AQI Sub-index calculation failed for {param}: {e}")
        return 0

def calculate_aqi(row, params, is_pred=False):
    """
    計算最終的 AQI 值（取所有污染物子指標的最大值）。
    :param row: 包含污染物濃度的 pandas Series
    :param params: 要計算的污染物列表
    :param is_pred: 是否為預測數據 (影響欄位名稱)
    :return: 最終 AQI 值
    """
    sub_indices = []
    
    for param in params:
        col = f'{param}_pred' if is_pred else param
        if col in row and pd.notna(row[col]):
            concentration = row[col]
            aqi_val = calculate_aqi_sub_index(param, concentration)
            sub_indices.append(aqi_val)

    # 最終 AQI 是所有子指標的最大值
    return max(sub_indices) if sub_indices else 0

# --- 歷史數據獲取與處理 ---

def fetch_recent_aqi_history(lat, lon, hours=24):
    """
    從 OpenAQ 獲取指定經緯度周圍的最近歷史觀測數據。
    :param lat: 緯度
    :param lon: 經度
    :param hours: 要獲取的小時數 (e.g., 24)
    :return: 過去 N 小時的 AQI 趨勢數據列表 (for Chart.js)
    """
    print("--- 開始獲取歷史觀測數據 ---")
    
    # 計算時間範圍 (UTC)
    time_now_utc = datetime.now(timezone.utc)
    time_from_utc = time_now_utc - timedelta(hours=hours)
    
    # OpenAQ v3 measurements API endpoint
    url = f"{BASE}/measurements"
    
    # 獲取所有目標污染物的 ID 列表
    param_ids = ','.join(map(str, POLLUTANT_IDS.values()))

    params = {
        'coordinates': f'{lat},{lon}',
        'date_from': time_from_utc.isoformat().replace('+00:00', 'Z'),
        'date_to': time_now_utc.isoformat().replace('+00:00', 'Z'),
        'parameter_id': param_ids,
        'limit': 1000, # 假設 1000 筆足夠涵蓋多個站點 24 小時的數據
        'sort': 'desc' # 確保最新的數據先來，但後續會按時間排序
    }

    try:
        response = requests.get(url, headers=HEADERS, params=params, timeout=30)
        response.raise_for_status()
        data = response.json().get('results', [])
    except Exception as e:
        print(f"❌ OpenAQ 歷史數據獲取失敗: {e}")
        return []

    if not data:
        print("⚠️ 未找到任何歷史觀測數據。")
        return []

    # 1. 轉換為 DataFrame
    df = pd.json_normalize(data)
    
    # 2. 數據清洗和準備
    df['datetime'] = pd.to_datetime(df['date.utc'])
    df['parameter'] = df['parameter.name'].str.lower()
    df['unit'] = df['parameter.unit']
    df = df[df['parameter'].isin(TARGET_PARAMS)]
    
    # 3. 按時間 (小時) 和參數進行聚合
    # 將時間四捨五入到最近的小時
    df['hour'] = df['datetime'].dt.round('H')
    
    # 計算每小時每種污染物的平均濃度
    hourly_mean_df = df.groupby(['hour', 'parameter'])['value'].mean().reset_index()
    
    # 將參數從行轉換為列 (pivot)
    pivoted_df = hourly_mean_df.pivot(index='hour', columns='parameter', values='value')
    pivoted_df = pivoted_df.reset_index().rename(columns={'hour': 'datetime'})
    
    # 確保所有小時都有記錄 (如果中間有缺小時，則填充)
    time_range = pd.date_range(end=time_now_utc.floor('H'), periods=hours + 1, freq='H', tz=timezone.utc)
    pivoted_df = pivoted_df.set_index('datetime').reindex(time_range).reset_index().rename(columns={'index': 'datetime'})
    
    # 4. 對每個聚合小時計算 AQI
    processed_history = []
    
    # 只計算完整小時的數據 (忽略當前小時，因為數據通常不完整)
    pivoted_df = pivoted_df[pivoted_df['datetime'] < time_now_utc.floor('H')].sort_values(by='datetime', ascending=True)

    for index, row in pivoted_df.iterrows():
        # 如果該小時有污染物數據，則計算 AQI
        if row[TARGET_PARAMS].notna().any():
            final_aqi = calculate_aqi(row, TARGET_PARAMS, is_pred=False)
        else:
            final_aqi = np.nan # 該小時無數據

        # 轉換時間為當地時區並格式化
        local_time = row['datetime'].astimezone(LOCAL_TZ).strftime('%m/%d %H:%M')

        processed_history.append({
            'time': local_time,
            'aqi': int(final_aqi) if pd.notna(final_aqi) and final_aqi > 0 else None # None 會被前端處理為 NA
        })

    print(f"✅ 成功處理 {len(processed_history)} 筆歷史 AQI 數據。")
    return [d for d in processed_history if d['aqi'] is not None]


# --- Flask 應用程式設定 ---

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    """
    主頁面路由：獲取最近 24 小時的 AQI 觀測趨勢數據並渲染圖表。
    """
    # 獲取使用者輸入的經緯度 (如果沒有，則使用預設值)
    try:
        lat = float(request.args.get('lat', DEFAULT_LAT))
        lon = float(request.args.get('lon', DEFAULT_LON))
        location_name = request.args.get('location', DEFAULT_LOCATION_NAME)
    except (TypeError, ValueError):
        lat = DEFAULT_LAT
        lon = DEFAULT_LON
        location_name = DEFAULT_LOCATION_NAME

    # 獲取最近 24 小時的 AQI 歷史觀測數據
    history_data = fetch_recent_aqi_history(lat, lon, hours=24)
    
    # 檢查是否有任何有效的歷史數據
    has_history_data = bool(history_data)

    # 渲染模板，傳遞歷史數據給前端
    return render_template(
        'index.html',
        location_name=location_name,
        current_aqi=history_data[-1]['aqi'] if has_history_data else 'N/A',
        aqi_predictions=history_data, # 將歷史數據視為要繪製的趨勢線
        is_history_mode=True,
        has_data=has_history_data
    )

if __name__ == '__main__':
    # 這是本地測試用的，Render 部署通常會使用 Gunicorn 或其他 WSGI 伺服器
    app.run(debug=True, host='0.0.0.0', port=os.environ.get('PORT', 5000))
