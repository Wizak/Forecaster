import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
import numpy as np
import io
import base64

def read_csv(file_stream):
    """Read the CSV file and preprocess the data."""
    data = pd.read_csv(file_stream)
    data['date'] = pd.to_datetime(data['date'])
    data = data.rename(columns={'date': 'ds', 'orders_count': 'y'})
    data['is_holiday'] = data['is_holiday'].astype(bool)
    data['dow'] = data['ds'].dt.dayofweek
    data['moy'] = data['ds'].dt.month
    return data

def filter_last_six_months(data):
    """Filter the data to include only the last six months."""
    last_date = data['ds'].max()
    six_months_ago = last_date - pd.DateOffset(months=6)
    return data[data['ds'] >= six_months_ago]

def train_forecast_model(data, future_holidays):
    """Train the Prophet model and make a forecast."""
    model = Prophet()
    model.add_regressor('is_holiday')
    model.add_regressor('dow')
    model.add_regressor('moy')
    model.fit(data)
    future = model.make_future_dataframe(periods=30)  # Forecast for 30 days ahead
    future['is_holiday'] = future['ds'].isin(future_holidays)
    future['dow'] = future['ds'].dt.dayofweek
    future['moy'] = future['ds'].dt.month
    forecast = model.predict(future)
    return model, forecast

def calculate_mape(actual, predicted):
    """Calculate Mean Absolute Percentage Error (MAPE)."""
    actual, predicted = np.array(actual), np.array(predicted)
    return np.mean(np.abs((actual - predicted) / actual)) * 100

def run_forecast(data, forecast):
    """Single function that trains and plots without needing a second arg."""
    # Define future holidays or pass them in as needed
    future_holidays = [
        pd.Timestamp('2025-01-20'),
        pd.Timestamp('2025-02-14'),
        pd.Timestamp('2025-03-17')
    ]

    # Train the model and get the forecast
    model, forecast = train_forecast_model(data, future_holidays)

    # The rest of your existing code:
    last_six_months_data = filter_last_six_months(data)
    forecast_for_actual = forecast[forecast['ds'].isin(last_six_months_data['ds'])]
    mape = calculate_mape(last_six_months_data['y'], forecast_for_actual['yhat'])
    last_six_months_data = filter_last_six_months(data)
    forecast_for_actual = forecast[forecast['ds'].isin(last_six_months_data['ds'])]
    mape = calculate_mape(last_six_months_data['y'], forecast_for_actual['yhat'])

    plt.figure(figsize=(12, 6))
    plt.plot(last_six_months_data['ds'], last_six_months_data['y'], label='Actual', marker='o')
    plt.plot(forecast['ds'], forecast['yhat'], label='Forecast', linestyle='--')
    plt.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'],
                     color='gray', alpha=0.2, label='Uncertainty Interval')
    plt.xlabel('Date')
    plt.ylabel('Order Count')
    plt.title(f'Time Series Forecasting\nMAPE (Last 6 Months): {mape:.2f}%')
    plt.legend()
    plt.grid()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    return img_base64, mape
