import pandas as pd
import numpy as np
from scipy.stats import boxcox
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.preprocessing import MinMaxScaler

def read_csv(file_input):
    df = pd.read_csv(file_input)
    df['date'] = pd.to_datetime(df['date'])
    df = df.rename(columns={'date': 'ds', 'orders_count': 'y'})
    df.sort_values(by='ds', inplace=True)
    return df

def invboxcox(y, lmbda):
    return np.exp(y) if lmbda == 0 else np.exp(np.log(lmbda * y + 1) / lmbda)

def run_forecast_table(file_input, forecast_start, forecast_end):
    df = read_csv(file_input)
    df.set_index('ds', inplace=True)
    df.sort_index(inplace=True)
    df = df.asfreq('D', method='bfill')
    df.ffill(inplace=True)
    df['y_box'], lmbda = boxcox(df['y'] + 1)
    best_model = sm.tsa.statespace.SARIMAX(df['y_box'], order=(4,1,2), seasonal_order=(4,1,0,7)).fit(disp=-1)
    df['model'] = invboxcox(best_model.fittedvalues, lmbda)
    forecast_steps = (pd.to_datetime(forecast_end) - pd.to_datetime(forecast_start)).days + 1
    forecast_box = best_model.predict(start=df.shape[0], end=df.shape[0] + forecast_steps - 1)
    forecast_values = invboxcox(forecast_box, lmbda)
    forecast_dates = pd.date_range(start=forecast_start, end=forecast_end)
    forecast_series = {"type": "forecast", "data": []}
    for dte, v in zip(forecast_dates, forecast_values):
        forecast_series["data"].append({"x": str(dte.date()), "y": float(v)})
    actual_series = {"type": "actual", "data": []}
    for dte, v in zip(df.index, df['y'].values):
        actual_series["data"].append({"x": str(dte.date()), "y": float(v)})
    mape = mean_absolute_percentage_error(df['y'].values, df['model'].values) * 100
    # Normalize output data
    all_points = forecast_series["data"] + actual_series["data"]
    all_vals = np.array([pt["y"] for pt in all_points]).reshape(-1, 1)
    scaler_out = MinMaxScaler((0, 1))
    scaled_vals = scaler_out.fit_transform(all_vals)
    idx = 0
    for pt in forecast_series["data"]:
        pt["y"] = float(scaled_vals[idx, 0])
        idx += 1
    for pt in actual_series["data"]:
        pt["y"] = float(scaled_vals[idx, 0])
        idx += 1
    return [forecast_series, actual_series], mape
