import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

def read_csv(file_input):
    df = pd.read_csv(file_input)
    df['date'] = pd.to_datetime(df['date'])
    df = df.rename(columns={'date': 'ds', 'orders_count': 'y'})
    df['dow'] = df['ds'].dt.dayofweek
    df['doy'] = df['ds'].dt.dayofyear
    df.sort_values(by=['ds', 'service'], inplace=True)
    return df

def preprocess_data(df):
    sc = MinMaxScaler((0, 1))
    arr = sc.fit_transform(df[['y','dow','doy','service']])
    return arr, sc

def create_sequences(arr, seq_length):
    X, y = [], []
    for i in range(len(arr) - seq_length):
        X.append(arr[i:i+seq_length, :])
        y.append(arr[i+seq_length, 0])
    return np.array(X), np.array(y)

def run_forecast_table(file_input, forecast_start, forecast_end):
    seq_length = 30
    epochs = 50
    batch_size = 32
    df = read_csv(file_input)
    daily = df.groupby(['ds','service']).sum(numeric_only=True).reset_index()
    services = daily['service'].unique()
    result = []
    for s in services:
        sdf = daily[daily['service'] == s].copy()
        arr, sc = preprocess_data(sdf)
        X, y_seq = create_sequences(arr, seq_length)
        if len(X) == 0:
            continue
        model = Sequential([
            LSTM(50, activation='relu', input_shape=(seq_length, X.shape[2])),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        model.fit(X, y_seq, epochs=epochs, batch_size=batch_size, verbose=1)
        steps = (pd.to_datetime(forecast_end) - pd.to_datetime(forecast_start)).days + 1
        fi = arr[-seq_length:].copy()
        preds = []
        for _ in range(steps):
            p = model.predict(fi[np.newaxis, :, :])[0, 0]
            preds.append(p)
            ni = np.hstack([np.array([[p]]), fi[-1, 1:].reshape(1, -1)])
            fi = np.append(fi[1:], ni, axis=0)
        preds = sc.inverse_transform(np.hstack([np.array(preds).reshape(-1, 1), np.zeros((steps, 3))]))[:, 0]
        fdates = pd.date_range(start=forecast_start, end=forecast_end)
        forecast_series = {"type": f"forecast_service_{str(s)}", "data": []}
        for dte, v in zip(fdates, preds):
            forecast_series["data"].append({"x": str(dte.date()), "y": float(v)})
        actual_series = {"type": f"actual_service_{str(s)}", "data": []}
        for _, row in sdf.iterrows():
            dte = pd.to_datetime(row['ds'])
            actual_series["data"].append({"x": str(dte.date()), "y": float(row['y'])})
        result.append(forecast_series)
        result.append(actual_series)
    # Normalize output data across all series
    all_vals = []
    for series in result:
        for point in series["data"]:
            all_vals.append(point["y"])
    all_vals = np.array(all_vals).reshape(-1, 1)
    out_scaler = MinMaxScaler((0, 1))
    scaled_vals = out_scaler.fit_transform(all_vals)
    idx = 0
    for series in result:
        for point in series["data"]:
            point["y"] = float(scaled_vals[idx, 0])
            idx += 1
    return result
