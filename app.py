from flask import Flask, jsonify
import pandas as pd
import numpy as np
from datetime import timedelta
import xgboost as xgb

# 匯入你原本寫好的函數（數據抓取、處理、模型訓練）
from 空氣品質預測系統_ import (
    calculate_aqi, generate_fake_data, predict_future_multi,
    POLLUTANT_TARGETS, LOCAL_TZ
)

app = Flask(__name__)

# 初始化：這裡可以預先跑一次訓練
print("🚀 正在初始化模型...")
df = generate_fake_data(limit=24*7, params=POLLUTANT_TARGETS)   # 先用假數據示範
df['aqi_value'] = df.apply(lambda row: calculate_aqi(row, POLLUTANT_TARGETS), axis=1)

# 簡化：只訓練一個隨機模型（正式版可以照你原本的流程）
X = df[['temperature','humidity','pressure']]
Y = df['pm25_value']
model = xgb.XGBRegressor().fit(X, Y)

@app.route("/status")
def status():
    return jsonify({"status": "ok", "message": "API 正常運作 🚀"})

@app.route("/predict")
def predict():
    """回傳 24 小時預測結果 (簡化版本)"""
    last_row = df.iloc[-1:].copy()
    future = predict_future_multi(
        models={"pm25": model},    # 這裡可以放多個污染物模型
        last_data=last_row,
        feature_cols=['temperature','humidity','pressure'],
        pollutant_params=["pm25"], # 測試先只用 PM2.5
        hours=24
    )

    # 整理成 JSON
    future['datetime_local'] = future['datetime'].dt.tz_convert(LOCAL_TZ).astype(str)
    result = {
        "timestamps": future['datetime_local'].tolist(),
        "pm25_pred": future['pm25_pred'].round(2).tolist(),
        "aqi_pred": future['aqi_pred'].astype(int).tolist()
    }
    return jsonify(result)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
