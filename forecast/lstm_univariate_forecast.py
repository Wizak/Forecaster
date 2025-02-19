import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import io
import base64

def read_csv(file_stream):
    """Read CSV and preprocess for univariate LSTM forecasting."""
    data = pd.read_csv(file_stream)
    data['ds'] = pd.to_datetime(data['date'])
    data = data.rename(columns={'orders_count': 'y'})
    data.set_index('ds', inplace=True)
    data.sort_index(inplace=True)
    data = data.asfreq('D', method='bfill')
    data.ffill(inplace=True)
    return data

def univariate_data(series, history_size, target_size):
    X, y = [], []
    for i in range(history_size, len(series)-target_size):
        X.append(series[i-history_size:i].values.reshape(history_size,1))
        y.append(series.iloc[i+target_size])
    return np.array(X), np.array(y)

def run_forecast(data):
    """Train a univariate LSTM and forecast 28 days ahead."""
    series = data['y']
    TRAIN_SPLIT = int(len(series)*0.8)
    uni_train = series.iloc[:TRAIN_SPLIT]
    uni_test = series.iloc[TRAIN_SPLIT:]
    mean = uni_train.mean()
    std = uni_train.std()
    norm_series = (series-mean)/std
    history_size = 28
    target_size = 0
    X, y = univariate_data(norm_series, history_size, target_size)
    # Split into training and validation
    X_train, y_train = X[:TRAIN_SPLIT - history_size], y[:TRAIN_SPLIT - history_size]
    X_val, y_val = X[TRAIN_SPLIT - history_size:], y[TRAIN_SPLIT - history_size:]
    
    model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(8, input_shape=(history_size, 1)),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mae')
    model.fit(X_train, y_train, epochs=10, batch_size=28, verbose=0)
    
    # Forecast future 28 days using last available history
    last_history = norm_series.iloc[-history_size:].values.reshape(1, history_size, 1)
    future_preds = []
    for _ in range(28):
        pred = model.predict(last_history)[0,0]
        future_preds.append(pred)
        last_history = np.append(last_history[:,1:,:], [[pred]], axis=1)
    future_preds = np.array(future_preds)*std+mean
    future_index = pd.date_range(start=data.index[-1], periods=29, closed='right')
    
    plt.figure(figsize=(10,6))
    plt.plot(data.index, data['y'], label='Actual')
    plt.plot(future_index, future_preds, label='Forecast', linestyle='--')
    plt.xlabel('Date')
    plt.ylabel('Orders Count')
    plt.title('Univariate LSTM Forecast')
    plt.legend()
    plt.grid()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    dummy_mape = 0.0  # You can compute error on validation if needed
    return img_base64, dummy_mape
