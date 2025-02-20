import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def read_csv(file_input):
    df = pd.read_csv(file_input)
    df['date'] = pd.to_datetime(df['date'])
    df = df.rename(columns={'date': 'ds', 'orders_count': 'y'})
    df.set_index('ds', inplace=True)
    df.sort_index(inplace=True)
    df = df.asfreq('D', method='bfill')
    df.ffill(inplace=True)
    return df

def univariate_data(dataset, start_index, end_index, history_size, target_size):
    data, labels = [], []
    start_index = start_index + history_size
    if end_index is None:
        end_index = len(dataset) - target_size
    for i in range(start_index, end_index):
        indices = range(i-history_size, i)
        data.append(np.reshape(dataset[indices], (history_size, 1)))
        labels.append(dataset[i+target_size])
    return np.array(data), np.array(labels)

def run_forecast_table(file_input, forecast_start, forecast_end):
    history_size = 28
    epochs = 10
    batch_size = 28
    df = read_csv(file_input)
    train_data = df[df.index < forecast_start]['y'].astype('float32')
    train_mean = train_data.mean()
    train_std = train_data.std()
    norm_data = (train_data - train_mean) / train_std
    TRAIN_SPLIT = len(norm_data)
    x_train, y_train = univariate_data(norm_data.values, 0, TRAIN_SPLIT, history_size, 0)
    dataset_train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    dataset_train = dataset_train.cache().shuffle(1000).batch(batch_size, drop_remainder=True).repeat()
    model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(8, input_shape=x_train.shape[-2:]),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mae')
    model.fit(dataset_train, epochs=epochs, steps_per_epoch=7, verbose=0)
    forecast_steps = (pd.to_datetime(forecast_end) - pd.to_datetime(forecast_start)).days + 1
    window = norm_data.values[-history_size:]
    preds = []
    for _ in range(forecast_steps):
        inp = np.reshape(window, (1, history_size, 1))
        p = model.predict(inp)[0, 0]
        preds.append(p)
        window = np.append(window[1:], p)
    preds_denorm = np.array(preds) * train_std + train_mean
    forecast_dates = pd.date_range(start=forecast_start, end=forecast_end)
    forecast_series = {"type": "forecast", "data": []}
    for dte, v in zip(forecast_dates, preds_denorm):
        forecast_series["data"].append({"x": str(dte.date()), "y": float(v)})
    actual_series = {"type": "actual", "data": []}
    for dte, v in zip(train_data.index, train_data.values):
        actual_series["data"].append({"x": str(dte.date()), "y": float(v)})
    # Normalize output data across forecast and actual series
    all_points = forecast_series["data"] + actual_series["data"]
    all_y = np.array([pt["y"] for pt in all_points]).reshape(-1, 1)
    scaler_out = MinMaxScaler((0, 1))
    scaled_y = scaler_out.fit_transform(all_y)
    idx = 0
    for pt in forecast_series["data"]:
        pt["y"] = float(scaled_y[idx, 0])
        idx += 1
    for pt in actual_series["data"]:
        pt["y"] = float(scaled_y[idx, 0])
        idx += 1
    return [forecast_series, actual_series]
