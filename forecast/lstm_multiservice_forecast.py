import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import io
import base64

# Define forecast period constants (you can adjust as needed)
TRAIN_START = '2022-01-01'
TRAIN_END = '2024-08-31'
TEST_START = '2024-09-01'
TEST_END = '2024-10-31'
FORECAST_START = '2024-11-01'
FORECAST_END = '2024-11-30'
SEQ_LENGTH = 30

def read_csv(file_stream):
    """Read CSV and preprocess for multiservice LSTM forecasting."""
    data = pd.read_csv(file_stream)
    data['date'] = pd.to_datetime(data['date'])
    data = data.rename(columns={'date': 'ds', 'orders_count': 'y'})
    # Add day-of-week and convert service to numeric codes
    data['dow'] = data['ds'].dt.dayofweek
    data['service'] = data['service'].astype('category').cat.codes
    data.set_index('ds', inplace=True)
    data.sort_index(inplace=True)
    data = data.asfreq('D', method='bfill')
    data.ffill(inplace=True)
    return data

def preprocess_data(data):
    """Scale columns used in forecasting."""
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data[['y', 'dow', 'service']])
    return data_scaled, scaler

def calculate_mape(actual, predicted):
    actual, predicted = np.array(actual), np.array(predicted)
    return np.mean(np.abs((actual - predicted)/actual)) * 100

def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length, :])
        y.append(data[i+seq_length, 0])
    return np.array(X), np.array(y)

def run_forecast(data):
    """Train a multiservice LSTM forecast model and produce forecast plot and MAPE."""
    # Split data by date ranges
    train_data = data[TRAIN_START:TRAIN_END]
    test_data = data[TEST_START:TEST_END]
    
    train_scaled, train_scaler = preprocess_data(train_data)
    test_scaled, test_scaler = preprocess_data(test_data)
    
    X_train, y_train = create_sequences(train_scaled, SEQ_LENGTH)
    X_test, y_test = create_sequences(test_scaled, SEQ_LENGTH)
    
    model = Sequential([
        LSTM(50, activation='relu', input_shape=(SEQ_LENGTH, X_train.shape[2])),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)
    
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    # Inverse transform predictions (padding zeros for extra columns)
    train_pred = train_scaler.inverse_transform(np.hstack([train_pred, np.zeros((train_pred.shape[0], 2))]))[:, 0]
    test_pred = test_scaler.inverse_transform(np.hstack([test_pred, np.zeros((test_pred.shape[0], 2))]))[:, 0]
    
    # Forecast future period
    forecast_steps = (pd.to_datetime(FORECAST_END) - pd.to_datetime(FORECAST_START)).days + 1
    forecast_input = test_scaled[-SEQ_LENGTH:].copy()
    forecast_values = []
    for _ in range(forecast_steps):
        pred = model.predict(forecast_input[np.newaxis, :, :])[0, 0]
        forecast_values.append(pred)
        # Update sequence (keeping other features unchanged)
        next_input = np.hstack([pred, forecast_input[-1, 1:]]).reshape(1, -1)
        forecast_input = np.append(forecast_input[1:], next_input, axis=0)
    forecast_values = test_scaler.inverse_transform(np.hstack([np.array(forecast_values).reshape(-1, 1), np.zeros((forecast_steps, 2))]))[:, 0]
    forecast_dates = pd.date_range(start=FORECAST_START, end=FORECAST_END)
    
    # Create plot
    plt.figure(figsize=(14,8))
    plt.plot(train_data.index, train_data['y'], label='Training Data')
    plt.plot(test_data.index, test_data['y'], label='Testing Data')
    plt.plot(forecast_dates, forecast_values, label='Forecast', linestyle='--')
    plt.xlabel('Date')
    plt.ylabel('Orders Count')
    plt.title('LSTM Multiservice Forecast')
    plt.legend()
    plt.grid()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    mape = calculate_mape(test_data['y'][SEQ_LENGTH:], test_pred[:len(test_data)-SEQ_LENGTH])
    return img_base64, mape
