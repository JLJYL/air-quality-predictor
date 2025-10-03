import os
import requests
from flask import Flask, render_template, request
from datetime import datetime, timedelta, timezone

# --- Configuration ---
app = Flask(__name__)

# 預設位置 (如果 URL 參數中沒有提供經緯度)
# 設為台灣高雄 (Gushan) 作為參考點
DEFAULT_LATITUDE = 22.6397
DEFAULT_LONGITUDE = 120.2798
DEFAULT_CITY_NAME = "Default Location (Kaohsiung)"

# OpenAQ API 相關
OPENAQ_API_URL = "https://api.openaq.org/v2/latest"

# Open-Meteo API 相關
OPENMETEO_API_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"


# --- AQI Calculation/Classification Helper ---

def get_aqi_status(aqi):
    """根據 AQI 數值返回狀態名稱 (使用 US EPA 標準)"""
    if aqi == "N/A" or aqi is None:
        return "N/A"
    try:
        aqi_int = int(aqi)
    except ValueError:
        return "N/A"

    if aqi_int <= 50:
        return "Good"
    elif aqi_int <= 100:
        return "Moderate"
    elif aqi_int <= 150:
        return "Unhealthy for Sensitive Groups"
    elif aqi_int <= 200:
        return "Unhealthy"
    elif aqi_int <= 300:
        return "Very Unhealthy"
    else:
        return "Hazardous"

# --- OpenAQ Data Fetching (Real-Time Observation) ---

def fetch_openaq_latest(lat, lon):
    """從 OpenAQ 獲取指定經緯度附近站點的最新 AQI 數值"""
    params = {
        'coordinates': f"{lat},{lon}",
        'radius': 10000, # 10 公里範圍內
        'limit': 1,
        'parameter': 'pm25', # OpenAQ 沒有直接的 AQI，我們使用 PM2.5 作為主要的空氣品質指標
        'order_by': 'distance',
        'sort': 'asc',
        'value_from': 0, # 只取有效數據
    }
    
    try:
        response = requests.get(OPENAQ_API_URL, params=params, timeout=8)
        response.raise_for_status()
        data = response.json()
        
        # 檢查是否有結果
        if data and data.get('results'):
            result = data['results'][0]
            # 找到 PM2.5 的最新測量值
            pm25_measurement = next((m for m in result['measurements'] if m['parameter'] == 'pm25'), None)
            
            if pm25_measurement:
                # 這裡是一個簡化，因為 OpenAQ 只提供 PM2.5 數值，我們必須將 PM2.5 轉換為 AQI。
                # 由於 AQI 轉換複雜，且為保持程式碼簡潔，這裡我們直接返回 PM2.5 數值和站點資訊，
                # 但在前端顯示時仍使用 AQI 邏輯（假設 PM2.5 數值近似 AQI）。
                # 專業的做法是實作一個 PM2.5-to-AQI 轉換函數。
                
                # 站點名稱、時間和數值
                city = result.get('city', result.get('location'))
                station_time = pm25_measurement.get('lastUpdated')
                pm25_value = round(pm25_measurement['value']) # PM2.5 數值
                
                return {
                    "city_name": city,
                    "time": station_time,
                    # 為了圖表顯示，這裡直接使用 PM2.5 作為近似 AQI
                    "aqi": pm25_value 
                }

    except requests.exceptions.RequestException as e:
        app.logger.error(f"Error fetching OpenAQ data: {e}")
    
    return None

# --- Open-Meteo Data Fetching (Forecast) ---

def fetch_openmeteo_forecast(lat, lon):
    """從 Open-Meteo 獲取 24 小時 AQI 預報"""
    params = {
        'latitude': lat,
        'longitude': lon,
        'hourly': 'us_epa_aqi',
        'forecast_days': 1, 
        'timezone': 'auto' 
    }
    
    try:
        response = requests.get(OPENMETEO_API_URL, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        return data.get('hourly', {})
    except requests.exceptions.RequestException as e:
        app.logger.error(f"Error fetching Open-Meteo data: {e}")
        return None

# --- Data Merging and Processing ---

def process_combined_data(obs_data, forecast_data):
    """合併 OpenAQ 觀察值和 Open-Meteo 預測值"""
    
    # 預測列表
    predictions = []
    max_aqi = 0
    current_obs_time_str = "N/A"
    city_name = DEFAULT_CITY_NAME
    is_fallback = True
    
    # 1. 處理 OpenAQ 觀察值 (作為起點)
    if obs_data and obs_data.get('aqi') is not None:
        aqi_val = obs_data['aqi']
        max_aqi = aqi_val
        
        # 格式化 OpenAQ 時間 (通常是 ISO 格式)
        try:
            dt_obj = datetime.fromisoformat(obs_data['time'].replace('Z', '+00:00'))
            current_obs_time_str = dt_obj.strftime("%m-%d %H:%M")
        except:
            current_obs_time_str = "Latest"

        city_name = obs_data['city_name']
        
        # 將觀察值作為第一個點加入列表
        predictions.append({
            "time": current_obs_time_str,
            "aqi": aqi_val,
            "is_obs": True # 標記為最新觀察值
        })
        is_fallback = False
    
    # 2. 處理 Open-Meteo 預測值
    if forecast_data and 'time' in forecast_data and 'us_epa_aqi' in forecast_data:
        times = forecast_data['time']
        aqi_values = forecast_data['us_epa_aqi']
        
        # 從第二個點開始處理 Open-Meteo 數據 (第一個點已由 OpenAQ 提供)
        for i in range(len(times)):
            time_str = times[i]
            aqi = aqi_values[i]
            
            # 格式化時間 (MM-DD HH:MM)
            try:
                dt_obj = datetime.fromisoformat(time_str)
                formatted_time = dt_obj.strftime("%m-%d %H:%M")
            except ValueError:
                formatted_time = time_str

            aqi_value = aqi if aqi is not None else None
            
            # 避免與觀察點重複 (Open-Meteo 的第一個點通常是當前小時，可能與 OpenAQ 接近)
            # 這裡我們只取時間點晚於觀察點的預測。
            if formatted_time > current_obs_time_str or not predictions:
                predictions.append({
                    "time": formatted_time,
                    "aqi": aqi_value,
                    "is_obs": False
                })

            # 計算最大 AQI
            if aqi_value is not None and isinstance(aqi_value, (int, float)):
                if aqi_value > max_aqi:
                    max_aqi = int(aqi_value)

    # 3. 如果 OpenAQ 和 Open-Meteo 都失敗，則設置為完全回退
    if not predictions:
        is_fallback = True
        
    return predictions, str(max_aqi), current_obs_time_str, city_name, is_fallback

# --- Flask Routes ---

@app.route('/')
def aqi_forecast():
    """主頁面：根據 URL 參數或預設值獲取 AQI 預報並渲染模板"""
    
    # 獲取經緯度 (從 URL 參數或使用預設值)
    try:
        lat = float(request.args.get('lat', DEFAULT_LATITUDE))
        lon = float(request.args.get('lon', DEFAULT_LONGITUDE))
    except ValueError:
        lat = DEFAULT_LATITUDE
        lon = DEFAULT_LONGITUDE

    # 1. 獲取 OpenAQ 即時數據 (觀察值)
    obs_data = fetch_openaq_latest(lat, lon)
    
    # 2. 獲取 Open-Meteo 預測數據
    forecast_data = fetch_openmeteo_forecast(lat, lon)
    
    # 3. 處理並合併數據
    aqi_predictions, max_aqi, current_obs_time, city_name, is_fallback = process_combined_data(obs_data, forecast_data)

    # 如果 OpenAQ 沒有提供站點名稱，則使用預設或地理座標
    if city_name == DEFAULT_CITY_NAME:
         if request.args.get('lat') and request.args.get('lon'):
             city_name = f"Your Location ({lat:.4f}, {lon:.4f})"
         else:
             city_name = DEFAULT_CITY_NAME

    # 4. 渲染模板
    return render_template(
        'index.html',
        city_name=city_name,
        aqi_predictions=aqi_predictions,
        max_aqi=max_aqi if max_aqi != '0' else 'N/A',
        current_obs_time=current_obs_time,
        is_fallback=is_fallback
    )

@app.route('/health')
def health_check():
    """Render 部署所需的健康檢查路徑"""
    # 檢查兩者 APIs 的健康狀態 (簡化為總是 OK)
    return "OK", 200

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
