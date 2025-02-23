import pandas as pd
import numpy as np
from datetime import timedelta
from xgboost import XGBRegressor
from prophet import Prophet

def read_csv(file_input):
    df = pd.read_csv(file_input)
    df["date"] = pd.to_datetime(df["date"])
    return df

def fill_missing_dates(df):
    start_date = df["date"].min()
    end_date = df["date"].max()
    date_range = pd.date_range(start_date, end_date)
    full_data = pd.DataFrame({"date": date_range})
    full_data = full_data.merge(df, on="date", how="left")
    full_data["orders_count"].fillna(0, inplace=True)
    full_data["is_holiday"].fillna(False, inplace=True)
    full_data["day_of_week"] = full_data["date"].dt.dayofweek
    full_data["month"] = full_data["date"].dt.month
    full_data["is_holiday"] = full_data["is_holiday"].astype(int)
    return full_data

def run_forecast_table(file_input, forecast_period=28):
    # 1) Зчитуємо дані
    df_raw = read_csv(file_input)
    df_full = fill_missing_dates(df_raw)
    df_full.set_index("date", inplace=True)
    
    # 2) Формуємо ознаки та ціль
    X = df_full[["is_holiday", "day_of_week", "month"]]
    y = df_full["orders_count"]
    
    # 3) Тренувальна вибірка: 80% (можна змінити, якщо треба фіксовані дати)
    train_size = int(len(X)*0.8)
    X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
    y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]
    
    # 4) XGBoost-модель
    xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
    xgb_model.fit(X_train, y_train)
    
    # 5) Prophet-модель
    #   Перетворюємо у формат Prophet (ds, y) + дод. регресори
    prophet_data = df_full.reset_index().rename(columns={"date": "ds", "orders_count": "y"})
    prophet_data["ds"] = pd.to_datetime(prophet_data["ds"])
    prophet_data["is_holiday"] = prophet_data["is_holiday"].astype(int)
    # Prophet вимагає, щоб регресори були додані перед fit
    prophet_model = Prophet(yearly_seasonality=True, weekly_seasonality=True)
    prophet_model.add_regressor("is_holiday")
    prophet_model.add_regressor("day_of_week")
    prophet_model.add_regressor("month")
    prophet_model.fit(prophet_data)
    
    # 6) Прогноз XGBoost
    last_date = df_full.index[-1]
    future_dates = [last_date + timedelta(days=i) for i in range(1, forecast_period+1)]
    future_exog = pd.DataFrame({
        "is_holiday": [0]*forecast_period,
        "day_of_week": [d.dayofweek for d in future_dates],
        "month": [d.month for d in future_dates]
    }, index=future_dates)
    xgb_pred = xgb_model.predict(future_exog)
    
    # 7) Прогноз Prophet
    #   Згенеруємо data frame з майбутніми датами
    future_df = prophet_model.make_future_dataframe(periods=forecast_period)
    #   Додамо регресори
    future_df["is_holiday"] = 0
    future_df["day_of_week"] = future_df["ds"].dt.dayofweek
    future_df["month"] = future_df["ds"].dt.month
    prophet_pred_df = prophet_model.predict(future_df)
    #   Беремо лише останні forecast_period рядків
    prophet_pred_future = prophet_pred_df[["ds","yhat"]].tail(forecast_period)
    
    # 8) Комбінуємо
    #   Робимо DataFrame з XGBoost і Prophet
    df_xgb = pd.DataFrame({"date": future_dates, "xgb_forecast": xgb_pred})
    df_prophet = prophet_pred_future.rename(columns={"ds":"date","yhat":"prophet_forecast"})
    df_merged = pd.merge(df_xgb, df_prophet, on="date")
    df_merged["combined_forecast"] = (df_merged["xgb_forecast"] + df_merged["prophet_forecast"]) / 2
    
    # 9) Готуємо історію (history), XGBoost-прогноз, Prophet-прогноз, combined
    #   Історія – вся наявна (або остання частина, якщо треба)
    history_series = {"type": "actual", "data": []}
    for d, val in df_full["orders_count"].items():
        history_series["data"].append({"x": str(d.date()), "y": float(val)})
    
    xgb_series = {"type": "xgb_forecast", "data": []}
    for i, row in df_merged.iterrows():
        xgb_series["data"].append({"x": str(row["date"].date()), "y": float(row["xgb_forecast"])})
    
    prophet_series = {"type": "prophet_forecast", "data": []}
    for i, row in df_merged.iterrows():
        prophet_series["data"].append({"x": str(row["date"].date()), "y": float(row["prophet_forecast"])})
    
    combined_series = {"type": "combined_forecast", "data": []}
    for i, row in df_merged.iterrows():
        combined_series["data"].append({"x": str(row["date"].date()), "y": float(row["combined_forecast"])})
    
    # Повертаємо список із 4-ма серіями
    # (history, xgb, prophet, combined)
    # MAPE в даному випадку можна обчислити на тестовому відрізку, якщо треба
    mape_val = 0.0  # тут не обчислюємо, залишимо 0
    
    return [history_series, xgb_series, prophet_series, combined_series]
