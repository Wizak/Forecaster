import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import io
import base64

FORECAST_START = '2024-11-01'
FORECAST_END = '2024-11-30'
SEQ_LENGTH = 30

def read_csv(file_stream):
    """Read CSV, add day-of-week and day-of-year features, and return raw data."""
    data = pd.read_csv(file_stream)
    data['ds'] = pd.to_datetime(data['date'])
    data = data.rename(columns={'orders_count': 'y'})
    data['dow'] = data['ds'].dt.dayofweek
    data['doy'] = data['ds'].dt.dayofyear
    # Expect a 'service' column for aggregation
    data.sort_values(by=['ds', 'service'], inplace=True)
    return data

def preprocess_data(service_data):
    """Scale features for one service."""
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(service_data[['y', 'dow', 'doy', 'service']])
    return data_scaled, scaler

def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length, :])
        y.append(data[i+seq_length, 0])
    return np.array(X), np.array(y)

def run_forecast(data):
    """
    Aggregate data by day and service, train a separate LSTM per service, and produce forecasts.
    Returns a Base64 plot (for the first service) and a dummy MAPE.
    (In a full implementation you might loop over all services.)
    """
    daily_data = data.groupby(['ds', 'service']).sum().reset_index()
    # For demonstration, use the first unique service
    unique_services = daily_data['service'].unique()
    service = unique_services[0]
    service_data = daily_data[daily_data['service'] == service].copy()
    service_data.set_index('ds', inplace=True)
    
    scaler, data_scaled = None, None
    data_scaled, scaler = preprocess_data(service_data)
    X, y = create_sequences(data_scaled, SEQ_LENGTH)
    
    model = Sequential([
        LSTM(50, activation='relu', input_shape=(SEQ_LENGTH, X.shape[2])),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=50, batch_size=32, verbose=0)
    
    # Forecast future period
    forecast_steps = (pd.to_datetime(FORECAST_END) - pd.to_datetime(FORECAST_START)).days + 1
    forecast_input = data_scaled[-SEQ_LENGTH:].copy()
    forecast_values = []
    for _ in range(forecast_steps):
        pred = model.predict(forecast_input[np.newaxis, :, :])[0, 0]
        forecast_values.append(pred)
        next_input = np.hstack([pred, forecast_input[-1, 1:]]).reshape(1, -1)
        forecast_input = np.append(forecast_input[1:], next_input, axis=0)
    forecast_values = scaler.inverse_transform(np.hstack([np.array(forecast_values).reshape(-1, 1), np.zeros((forecast_steps, 3))]))[:, 0]
    forecast_dates = pd.date_range(start=FORECAST_START, end=FORECAST_END)
    
    plt.figure(figsize=(14,8))
    plt.plot(service_data.index, service_data['y'], label='Actual')
    plt.plot(forecast_dates, forecast_values, label='Forecast', linestyle='--')
    plt.xlabel('Date')
    plt.ylabel('Orders Count')
    plt.title(f'LSTM Aggregate Forecast for Service {service}')
    plt.legend()
    plt.grid()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    dummy_mape = 0.0  # For demonstration
    return img_base64, dummy_mape
