from flask import Flask, render_template
from datetime import datetime, timedelta
import random
import json
import os

# 為了滿足 requirements.txt 中的依賴，即使我們在這裡使用模擬數據，
# 也建議將這些庫導入。
import pandas as pd
# import requests # 通常用於實際 API 調用
# import xgboost as xgb
# from meteostat import Point, Daily
# 註解掉的行表示在實際的生產環境中，您會在這裡實現真正的數據獲取和模型預測邏輯。

# 初始化 Flask 應用程式
# 假設您的 HTML 檔案位於名為 'templates' 的目錄中
app = Flask(__name__)

# --- 數據模擬和處理函式 ---

def get_aqi_status(aqi: int) -> str:
    """根據 AQI 數值返回對應的空氣品質狀態描述"""
    if aqi <= 50:
        return "Good (良好)"
    elif aqi <= 100:
        return "Moderate (普通)"
    elif aqi <= 150:
        return "Unhealthy for Sensitive Groups (對敏感族群不健康)"
    elif aqi <= 200:
        return "Unhealthy (不健康)"
    elif aqi <= 300:
        return "Very Unhealthy (非常不健康)"
    else:
        return "Hazardous (危害)"

def generate_mock_aqi_forecast(start_aqi: int = 55) -> tuple[list, int]:
    """
    生成一個模擬的 24 小時 AQI 預測列表。
    第一個點是當前觀察值 (is_obs=True)，後續 24 個點是預測值。
    """
    now = datetime.now()
    forecast = []
    current_aqi = start_aqi
    max_aqi = start_aqi
    
    # 0. 當前觀察 (Current Observation - 作為預測的起點)
    forecast.append({
        'time': now.strftime('%Y-%m-%d %H:%M'),
        'aqi': str(current_aqi),
        'is_obs': True
    })

    # 1-24. 預測小時
    for i in range(1, 25):
        forecast_time = now + timedelta(hours=i)
        
        # 簡單的模擬：AQI 隨機浮動
        change = random.randint(-8, 12)
        current_aqi = max(10, min(350, current_aqi + change)) # 限制範圍 10-350
        
        # 在下午/晚上模擬輕微污染高峰
        hour_of_day = forecast_time.hour
        if 16 <= hour_of_day <= 20:
             current_aqi = min(350, current_aqi + random.randint(0, 15))
        
        if current_aqi > max_aqi:
            max_aqi = current_aqi
            
        forecast.append({
            'time': forecast_time.strftime('%Y-%m-%d %H:%M'),
            'aqi': str(current_aqi),
            'is_obs': False
        })
        
    return forecast, max_aqi

# --- Flask 路由定義 ---

@app.route('/')
def index():
    """主頁路由，用於渲染 AQI 預測儀表板"""
    
    # --- 實際應用中，此處應為從數據庫或 API 獲取數據的邏輯 ---
    # 這裡我們使用模擬數據
    
    CITY_NAME = "高雄市 (Kaohsiung, Taiwan)"
    
    # 隨機模擬模型偶爾失敗，只顯示當前觀察值
    # IS_FALLBACK = random.choice([True, False, False, False]) # 約 25% 機率觸發 fallback
    IS_FALLBACK = False # 為了演示，我們通常保持預測模式

    if IS_FALLBACK:
        # 僅顯示當前觀察值的備用模式
        current_obs_time = datetime.now().strftime('%Y-%m-%d %H:%M')
        # 模擬一個當前 AQI 值
        initial_aqi = random.randint(50, 100)
        aqi_predictions = [{'time': current_obs_time, 'aqi': str(initial_aqi), 'is_obs': True}]
        MAX_AQI = str(initial_aqi)
        
    else:
        # 生成完整的 24 小時預測數據
        start_aqi = random.randint(40, 110)
        aqi_predictions, max_aqi_int = generate_mock_aqi_forecast(start_aqi)
        current_obs_time = aqi_predictions[0]['time']
        MAX_AQI = str(max_aqi_int)
    
    # 將數據傳遞給 Jinja2 範本
    return render_template(
        'index.html',
        city_name=CITY_NAME,
        max_aqi=MAX_AQI,
        # 將 Python 列表轉換為 JSON 字符串，以安全地嵌入到 JS 中
        aqi_predictions=json.dumps(aqi_predictions), 
        is_fallback=IS_FALLBACK,
        current_obs_time=current_obs_time
    )

if __name__ == '__main__':
    # 在本地開發環境運行 Flask
    app.run(debug=True)
