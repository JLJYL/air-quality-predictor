def fetch_latest_observation_data(location_id: int, target_params: list) -> pd.DataFrame:
    """
    Fetches the latest observation data from OpenAQ.
    修正: 移除剛性時間匹配，直接選擇每個參數的絕對最新讀數。
    """
    
    # 1. Fetch all 'latest' data from two main sources
    df_loc_latest = get_location_latest_df(location_id)
    df_param_latest = get_parameters_latest_df(location_id, target_params)
    
    # Combine all fetched data
    frames = [df for df in [df_loc_latest, df_param_latest] if not df.empty]
    if not frames:
        print("🚨 [Fetch] No pollutant data fetched from OpenAQ.")
        return pd.DataFrame()

    df_all = pd.concat(frames, ignore_index=True)
    df_all["parameter"] = df_all["parameter"].str.lower()
    df_all = df_all[df_all["parameter"].isin(target_params)]
    
    # 2. 核心修正：排序並選擇每個參數的絕對最新讀數
    
    # 確保 ts_utc 是 tz-aware
    df_all["ts_utc"] = pd.to_datetime(df_all["ts_utc"], errors="coerce", utc=True)
    df_all = df_all.dropna(subset=['ts_utc'])
    
    # 依參數排序，再依時間降序排序 (最新在前)
    df_all = df_all.sort_values(["parameter", "ts_utc"], ascending=[True, False])
    
    # 選擇每個參數的第一筆 (即最新) 讀數
    df_all = df_all.drop_duplicates(subset=["parameter"], keep="first")
    
    # 3. 確保數據足夠新鮮 (防止模型從幾天前的數據開始預測)
    three_hours_ago = datetime.now(timezone.utc) - timedelta(hours=3)
    df_all = df_all[df_all["ts_utc"] > three_hours_ago].copy()

    if df_all.empty:
        print("🚨 [Fetch] No valid and recent observations found within the last 3 hours.")
        return pd.DataFrame()
        
    # 4. 為最終的單行 DataFrame 確定一個統一的時間戳 (使用所有最新讀數中最新的那個)
    latest_valid_ts = df_all["ts_utc"].max()
    
    # 移除不需要的輔助欄位
    df_all = df_all.drop(columns=["units", "ts_local"])

    # 5. 轉換為模型輸入格式 (單行寬表)
    observation = df_all.pivot_table(
        index='parameter', columns='ts_utc', values='value', aggfunc='first'
    ).T.reset_index()
    
    # 統一欄位名稱
    observation = observation.rename(columns={'ts_utc': 'datetime'})
    
    # 6. 計算 AQI 和最終時區處理
    if not observation.empty:
        observation['aqi'] = observation.apply(
            lambda row: calculate_aqi(row, target_params, is_pred=False), axis=1
        )
        # 確保 'datetime' 總是 UTC-aware (使用統一的最新時間)
        observation['datetime'] = latest_valid_ts
        if observation['datetime'].dt.tz is None:
             observation['datetime'] = observation['datetime'].dt.tz_localize('UTC')
        else:
             observation['datetime'] = observation['datetime'].dt.tz_convert('UTC')

    return observation
