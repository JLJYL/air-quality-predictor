import requests
import pandas as pd
import datetime
import os
import warnings
import numpy as np
import json
from datetime import timedelta, timezone
from flask import Flask, render_template, request
import requests_cache

warnings.filterwarnings('ignore')

# 啟用 requests 快取，防止對 OpenAQ API 頻繁呼叫
# 緩存時間設為 5 分鐘
cache = requests_cache.install_cache('openaq_cache', expire_after=300)

# OpenAQ API Constants
# ⚠️ 注意: 您的 API Key 仍在此處，請確保在正式環境中安全處理
API_KEY = "fb579916623e8483cd85344b14605c3109eea922202314c44b87a2df3b1fff77" 
HEADERS = {"X-API-Key": API_KEY}
BASE = "https://api.openaq.org/v3"

# 預設定位點 (台中)
DEFAULT_LAT = 24.1477
DEFAULT_LON = 120.6736
DEFAULT_LOCATION_NAME = "台中市"
LOCAL_TZ = timezone(timedelta(hours=8), name='Asia/Taipei')

# U.S. EPA AQI 指標污染物
POLLUTANT_TARGETS = ["co", "no2", "o3", "pm10", "pm25", "so2"]

# U.S. EPA AQI 分級表 (2024 年 5 月更新)
# 格式: {污染物: [[C_low, C_high, I_low, I_high], ...]}
# 濃度單位: PM2.5/PM10: μg/m3, O3/SO2/NO2: ppb, CO: ppm
AQI_BREAKPOINTS = {
    # PM2.5 (24hr avg, μg/m3) - **U.S. EPA 2024/5 更新標準**
    "pm25_24h": [
        [0.0, 9.0, 0, 50], [9.1, 35.4, 51, 100], [35.5, 55.4, 101, 150],
        [55.5, 150.4, 151, 200], [150.5, 250.4, 201, 300], [250.5, 350.4, 301, 400],
        [350.5, 500.4, 401, 500]
    ],
    # PM10 (24hr avg, μg/m3) - U.S. EPA 標準
    "pm10_24h": [
        [0.0, 54.0, 0, 50], [55.0, 154.0, 51, 100], [155.0, 254.0, 101, 150],
        [255.0, 354.0, 151, 200], [355.0, 424.0, 201, 300], [425.0, 504.0, 301, 400],
        [505.0, 604.0, 401, 500]
    ],
    # O3 (8hr avg, ppb) - U.S. EPA 標準
    "o3_8h": [
        [0.0, 54.0, 0, 50], [55.0, 70.0, 51, 100], [71.0, 85.0, 101, 150],
        [86.0, 105.0, 151, 200], [106.0, 200.0, 201, 300] # 8hr O3 不用於 AQI > 300
    ],
    # O3 (1hr avg, ppb) - U.S. EPA 標準 (用於 AQI > 300 或 O3 1hr 濃度更高時)
    "o3_1h": [
        [125.0, 164.0, 101, 150], [165.0, 204.0, 151, 200], [205.0, 404.0, 201, 300],
        [405.0, 504.0, 301, 400], [505.0, 604.0, 401, 500]
    ],
    # CO (8hr avg, ppm) - U.S. EPA 標準
    "co_8h": [
        [0.0, 4.4, 0, 50], [4.5, 9.4, 51, 100], [9.5, 12.4, 101, 150],
        [12.5, 15.4, 151, 200], [15.5, 30.4, 201, 300], [30.5, 40.4, 301, 400],
        [40.5, 50.4, 401, 500]
    ],
    # SO2 (1hr avg, ppb) - U.S. EPA 標準 (用於 AQI < 200)
    "so2_1h": [
        [0.0, 35.0, 0, 50], [36.0, 75.0, 51, 100], [76.0, 185.0, 101, 150],
        [186.0, 304.0, 151, 200]
    ],
    # NO2 (1hr avg, ppb) - U.S. EPA 標準
    "no2_1h": [
        [0.0, 53.0, 0, 50], [54.0, 100.0, 51, 100], [101.0, 360.0, 101, 150],
        [361.0, 649.0, 151, 200], [650.0, 1249.0, 201, 300], [1250.0, 1649.0, 301, 400],
        [1650.0, 2049.0, 401, 500]
    ]
}


app = Flask(__name__)

# --- 核心數據抓取與計算函式 ---

def find_best_location_v3(lat, lon):
    """從 OpenAQ 尋找最近且數據最完整的台灣監測站"""
    V3_LOCATIONS_URL = f"{BASE}/locations"
    
    params = {
        "coordinates": f"{lat},{lon}",
        "radius": 50000, # 50 公里半徑
        "limit": 100,
        "country_id": "TW", 
        "parameter_id": [2, 1, 10], # 優先找有 PM25, PM10, O3 的站
    }

    try:
        r = requests.get(V3_LOCATIONS_URL, headers=HEADERS, params=params, timeout=15)
        r.raise_for_status()
        results = r.json().get("results", [])
        
        if not results:
            print("🚨 [Location] 未找到任何測站。")
            return None, DEFAULT_LOCATION_NAME, DEFAULT_LAT, DEFAULT_LON

        # 這裡簡化：取第一個結果作為「最佳」測站
        best_loc = results[0]
        location_id = best_loc.get("id")
        location_name = best_loc.get("name")
        loc_lat = best_loc.get("coordinates", {}).get("latitude")
        loc_lon = best_loc.get("coordinates", {}).get("longitude")
        
        print(f"✅ [Location] 找到最佳測站: {location_name} (ID: {location_id})")
        return location_id, location_name, loc_lat, loc_lon

    except Exception as e:
        print(f"❌ [Location] 尋找測站失敗: {e}")
        return None, DEFAULT_LOCATION_NAME, DEFAULT_LAT, DEFAULT_LON


def get_historical_measurements(location_id: int, hours: int):
    """
    從 OpenAQ V3 抓取指定測站過去 N 小時的污染物觀測數據。
    """
    V3_MEASUREMENTS_URL = f"{BASE}/measurements"
    end_time = datetime.datetime.now(timezone.utc)
    start_time = end_time - timedelta(hours=hours)

    params = {
        "location_id": location_id,
        "date_from": start_time.isoformat().replace('+00:00', 'Z'),
        "date_to": end_time.isoformat().replace('+00:00', 'Z'),
        "limit": 1000, # 抓多一點確保數據完整
    }
    
    try:
        r = requests.get(V3_MEASUREMENTS_URL, headers=HEADERS, params=params, timeout=15)
        r.raise_for_status()
        
        # 將結果轉換為 DataFrame
        data = r.json().get("results", [])
        if not data:
            return pd.DataFrame()
            
        df_list = []
        for item in data:
            value = item.get("value")
            param_name = item["parameter"]["name"].lower()
            
            if value is not None and value >= 0 and param_name in POLLUTANT_TARGETS:
                df_list.append({
                    'datetime': pd.to_datetime(item["datetime"]["utc"], utc=True),
                    'param': param_name,
                    'value': value
                })
        
        if not df_list:
            return pd.DataFrame()

        df = pd.DataFrame(df_list)
        # 轉換時區到本地 (台灣時間)
        df['datetime'] = df['datetime'].dt.tz_convert(LOCAL_TZ)
        
        # 將 'param' 轉為欄位，並用 'datetime' 作為索引
        df_pivot = df.pivot_table(index='datetime', columns='param', values='value', aggfunc='mean')
        
        # 確保 'datetime' 索引是連續且完整的 (每小時一筆)
        full_index = pd.date_range(start=start_time.tz_convert(LOCAL_TZ).floor('H'), 
                                   end=end_time.tz_convert(LOCAL_TZ).floor('H'), 
                                   freq='H', 
                                   name='datetime',
                                   inclusive='left') # 不包含當前正在發生的時間
        df_reindexed = df_pivot.reindex(full_index)

        # 只保留我們需要的目標污染物欄位，並將缺失值填補為 NaN
        df_final = df_reindexed[POLLUTANT_TARGETS].copy()
        
        # 依照 U.S. EPA 標準，計算平均值
        
        # O3 (8hr and 1hr)
        df_final['o3_8h'] = df_final['o3'].rolling(window=8, min_periods=5).mean().round(3)
        df_final['o3_1h'] = df_final['o3'] # 1hr avg
        
        # CO (8hr)
        df_final['co_8h'] = df_final['co'].rolling(window=8, min_periods=5).mean().round(3)
        
        # PM2.5/PM10 (24hr)
        # U.S. EPA 24hr PM 需要 75% 有效數據 (18/24)
        df_final['pm25_24h'] = df_final['pm25'].rolling(window=24, min_periods=18).mean().round(3) 
        df_final['pm10_24h'] = df_final['pm10'].rolling(window=24, min_periods=18).mean().round(3) 

        # SO2 (1hr) and NO2 (1hr)
        df_final['so2_1h'] = df_final['so2'] # 1hr avg
        df_final['no2_1h'] = df_final['no2'] # 1hr avg
        
        # 移除原始污染物欄位
        df_final = df_final.drop(columns=['o3', 'co', 'pm25', 'pm10', 'so2', 'no2'])

        return df_final

    except Exception as e:
        print(f"🚨 [Historical] 抓取歷史數據失敗: {e}")
        return pd.DataFrame()


def calculate_iaqi(conc, param_key):
    """
    使用 AQI 線性轉換公式計算單一污染物濃度對應的 IAQI (基於 U.S. EPA 標準)。
    IAQI = [(I_high - I_low) / (C_high - C_low)] * (Conc - C_low) + I_low
    """
    breakpoints = AQI_BREAKPOINTS.get(param_key)
    if not breakpoints or pd.isna(conc):
        return 0
    
    iaqi = 0
    
    for C_low, C_high, I_low, I_high in breakpoints:
        # 使用容錯的範圍檢查 (處理浮點數邊界問題)
        if C_low - 0.001 <= conc <= C_high + 0.001: 
            if C_high == C_low:
                iaqi = I_high
            else:
                iaqi = ((I_high - I_low) / (C_high - C_low)) * (conc - C_low) + I_low
            break
            
    return int(round(iaqi))


# --- Flask 路由 ---

@app.route('/')
def index():
    # 1. 取得用戶坐標，如果沒有則使用預設值
    try:
        lat = request.args.get('lat', type=float)
        lon = request.args.get('lon', type=float)
        if lat is None or lon is None:
            lat = DEFAULT_LAT
            lon = DEFAULT_LON
    except:
        lat = DEFAULT_LAT
        lon = DEFAULT_LON

    # 2. 尋找最佳監測站
    location_id, location_name, _, _ = find_best_location_v3(lat, lon)
    
    if location_id is None:
        return render_template('index.html', 
                               error_message="未找到可用測站數據", 
                               location_name=DEFAULT_LOCATION_NAME)

    # 3. 抓取過去 48 小時的歷史數據
    # 抓 48 小時才能計算出 PM2.5/PM10 的 24 小時平均
    history_df = get_historical_measurements(location_id, hours=48)
    
    if history_df.empty or len(history_df) < 24:
        return render_template('index.html', 
                               error_message=f"在 {location_name} 過去 48 小時內未抓到足夠的觀測數據。", 
                               location_name=location_name)

    # 4. 針對每筆觀測值計算 AQI
    observed_data = []
    
    # 只取最新的 24 筆數據 (確保有足夠的 Nhr 平均數據來計算)
    plot_df = history_df.iloc[-24:].copy() 
    
    for index, row in plot_df.iterrows():
        
        iaqi_pm25 = calculate_iaqi(row.get('pm25_24h'), 'pm25_24h')
        iaqi_pm10 = calculate_iaqi(row.get('pm10_24h'), 'pm10_24h')
        iaqi_co = calculate_iaqi(row.get('co_8h'), 'co_8h')
        iaqi_so2 = calculate_iaqi(row.get('so2_1h'), 'so2_1h')
        iaqi_no2 = calculate_iaqi(row.get('no2_1h'), 'no2_1h')

        # O3 邏輯 (U.S. EPA): 
        # a. 8hr O3: 用於 AQI 0-300
        # b. 1hr O3: 用於 AQI > 300
        iaqi_o3_8h = calculate_iaqi(row.get('o3_8h'), 'o3_8h')
        iaqi_o3_1h = calculate_iaqi(row.get('o3_1h'), 'o3_1h')
        
        # 8hr O3 IAQI 只計算到 300，因此我們取 max(8hr O3 <= 300, 1hr O3)
        if iaqi_o3_8h > 0 and iaqi_o3_8h <= 300:
            iaqi_o3 = iaqi_o3_8h
        else:
            iaqi_o3 = iaqi_o3_1h
        
        # 找出最大的 IAQI 作為該小時的 AQI
        iaqis = {
            'PM2.5': iaqi_pm25, 'PM10': iaqi_pm10, 'O3': iaqi_o3, 
            'CO': iaqi_co, 'SO2': iaqi_so2, 'NO2': iaqi_no2
        }
        
        # 排除 0 (代表無數據或濃度過低)
        valid_iaqis = {poll: aqi for poll, aqi in iaqis.items() if aqi > 0}
        
        if not valid_iaqis:
            continue # 無有效數據，跳過此小時

        max_iaqi = max(valid_iaqis.values())
        main_pollutant = max(valid_iaqis, key=valid_iaqis.get)
        
        # 記錄結果
        observed_data.append({
            # 格式化為: 06/01 10:00
            'datetime': index.strftime('%m/%d %H:%M'),
            'aqi': max_iaqi,
            'main_pollutant': main_pollutant
        })
            
    # 5. 準備傳給前端的數據
    final_data = [d for d in observed_data if d['aqi'] > 0]
    
    if not final_data:
         return render_template('index.html', 
                               error_message=f"在 {location_name} 過去 24 小時內未計算出有效的 AQI 觀測數據。", 
                               location_name=location_name)

    # 取得最新一筆數據作為顯示
    latest_data = final_data[-1]

    # 將數據傳到前端
    return render_template(
        'index.html',
        location_name=location_name,
        latest_aqi=latest_data['aqi'],
        latest_time=latest_data['datetime'],
        chart_data=final_data # 只傳觀測的歷史數據
    )
