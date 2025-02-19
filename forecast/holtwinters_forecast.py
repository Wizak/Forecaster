import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_percentage_error

def read_csv(file_stream):
    """Read CSV and preprocess for Holt-Winters/SARIMA forecasting."""
    data = pd.read_csv(file_stream)
    data['ds'] = pd.to_datetime(data['date'])
    data = data.rename(columns={'orders_count': 'y'})
    data.set_index('ds', inplace=True)
    data.sort_index(inplace=True)
    data = data.asfreq('D', method='bfill')
    data.ffill(inplace=True)
    return data

def run_forecast(data):
    """
    Fit a SARIMAX model (as a proxy for Holt-Winters with Brutlag)
    and produce a forecast plot for a 28-day horizon.
    """
    train_size = int(len(data)*0.8)
    train = data.iloc[:train_size]
    test = data.iloc[train_size:]
    try:
        model = SARIMAX(train['y'], order=(4,1,2), seasonal_order=(4,1,1,7))
        results = model.fit(disp=False)
    except Exception as e:
        raise Exception(f"Error fitting SARIMAX: {e}")
    test_forecast = results.get_forecast(steps=len(test)).predicted_mean
    mape = mean_absolute_percentage_error(test['y'], test_forecast)*100
    future_steps = 28
    future_forecast = results.get_forecast(steps=future_steps).predicted_mean
    future_index = pd.date_range(start=data.index[-1], periods=future_steps+1, closed='right')
    
    plt.figure(figsize=(10,6))
    plt.plot(data.index, data['y'], label='Historical')
    plt.plot(test.index, test_forecast, label='Test Forecast', linestyle='--')
    plt.plot(future_index, future_forecast, label='Future Forecast', linestyle='--', color='red')
    plt.xlabel('Date')
    plt.ylabel('Orders Count')
    plt.title('Holt-Winters/SARIMAX Forecast')
    plt.legend()
    plt.grid()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    return img_base64, mape
