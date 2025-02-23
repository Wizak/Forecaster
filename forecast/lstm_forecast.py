import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

def read_csv(file_input):
    data = pd.read_csv(file_input)
    data['date'] = pd.to_datetime(data['date'])
    data = data.rename(columns={'date': 'ds', 'orders_count': 'y'})
    data.set_index('ds', inplace=True)
    data.sort_index(inplace=True)
    data = data.asfreq('D', method='bfill')
    data.ffill(inplace=True)
    return data

def preprocess_data(data):
    scaler = MinMaxScaler(feature_range=(0,1))
    data_scaled = scaler.fit_transform(data)
    return data_scaled, scaler

def calculate_mape(actual, predicted):
    actual = np.array(actual)
    predicted = np.array(predicted)
    return np.mean(np.abs((actual - predicted) / actual)) * 100

def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

def run_forecast_table(
    file_input,
    train_start,
    train_end,
    test_start,
    test_end,
    forecast_start,
    forecast_end,
):
    seq_length=30
    data = read_csv(file_input)
    train_data = data[train_start:train_end][['y']]
    test_data = data[test_start:test_end][['y']]
    train_data_scaled, train_scaler = preprocess_data(train_data)
    test_data_scaled, test_scaler = preprocess_data(test_data)
    X_train, y_train = create_sequences(train_data_scaled, seq_length)
    X_test, y_test = create_sequences(test_data_scaled, seq_length)
    model = Sequential([
        LSTM(50, activation='relu', input_shape=(seq_length, 1)),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)
    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)
    train_predictions = train_scaler.inverse_transform(train_predictions)
    test_predictions = test_scaler.inverse_transform(test_predictions)
    y_test_inv = test_scaler.inverse_transform(y_test.reshape(-1,1))
    mape_val = calculate_mape(y_test_inv, test_predictions)
    forecast_period = (pd.to_datetime(forecast_end) - pd.to_datetime(forecast_start)).days + 1
    forecast_vals = []
    last_sequence = X_test[-1]
    for _ in range(forecast_period):
        current_sequence = last_sequence.reshape(1, seq_length, 1)
        next_pred = model.predict(current_sequence)[0][0]
        forecast_vals.append(next_pred)
        last_sequence = np.append(last_sequence[1:], next_pred)
    forecast_vals = train_scaler.inverse_transform(np.array(forecast_vals).reshape(-1,1)).flatten()
    test_index = test_data.index[seq_length:]
    historical_table = []
    for i, date_i in enumerate(test_index):
        historical_table.append({
            'ds': str(date_i.date()),
            'y': float(y_test_inv[i])
        })
    forecast_dates = pd.date_range(start=forecast_start, end=forecast_end)
    forecast_table = []
    for d, val in zip(forecast_dates, forecast_vals):
        forecast_table.append({
            'ds': str(d.date()),
            'yhat': float(val)
        })
    df_h = pd.DataFrame(historical_table)
    df_h.rename(columns={'y': 'val'}, inplace=True)
    df_h['is_forecast'] = False
    df_f = pd.DataFrame(forecast_table)
    df_f.rename(columns={'yhat': 'val'}, inplace=True)
    df_f['is_forecast'] = True
    merged = pd.concat([df_h, df_f], ignore_index=True)
    merged['val'] = pd.to_numeric(merged['val'], errors='coerce').fillna(0.0)
    scaler_out = MinMaxScaler(feature_range=(0,1))
    arr = merged[['val']].values
    arr_scaled = scaler_out.fit_transform(arr)
    merged['val'] = arr_scaled
    merged_h = merged[merged['is_forecast']==False].copy()
    merged_f = merged[merged['is_forecast']==True].copy()
    historical_table_norm = []
    for _, row in merged_h.iterrows():
        historical_table_norm.append({'x': row['ds'], 'y': float(row['val'])})
    forecast_table_norm = []
    for _, row in merged_f.iterrows():
        forecast_table_norm.append({'x': row['ds'], 'y': float(row['val'])})

    data = [
        {
            "type": "actual",
            "data": historical_table_norm,
        },
                {
            "type": "forecast",
            "data": forecast_table_norm,
        },
    ]
    return data, mape_val
