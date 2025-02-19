import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import io
import base64

def read_csv(file_stream):
    """Read CSV and preprocess for multivariate interval LSTM forecasting."""
    data = pd.read_csv(file_stream)
    data['ds'] = pd.to_datetime(data['date'])
    data = data.rename(columns={'orders_count': 'y'})
    data['is_holiday'] = data['is_holiday'].astype(int)
    data['dow'] = data['ds'].dt.dayofweek
    data['moy'] = data['ds'].dt.month
    data.set_index('ds', inplace=True)
    data.sort_index(inplace=True)
    data = data.asfreq('D', method='bfill')
    data.ffill(inplace=True)
    return data

def multivariate_data(dataset, target, history_size, target_size, step=1, single_step=False):
    X, y = [], []
    start = history_size
    end = len(dataset) - target_size
    for i in range(start, end):
        indices = range(i-history_size, i, step)
        X.append(dataset[indices])
        y.append(target[i:i+target_size])
    return np.array(X), np.array(y)

def run_forecast(data):
    """
    Train a multivariate LSTM for interval forecasting.
    In addition to the point forecast, you might compute prediction intervals
    (here we simply mimic an interval by adding/subtracting a fixed factor).
    """
    features = data[['dow', 'y', 'moy']]
    dataset = features.values
    TRAIN_SPLIT = int(len(dataset)*0.8)
    data_mean = dataset[-TRAIN_SPLIT:].mean(axis=0)
    data_std = dataset[-TRAIN_SPLIT:].std(axis=0)
    norm_data = (dataset-data_mean)/data_std
    history_size = 28
    future_target = 7
    X, y = multivariate_data(norm_data, norm_data[:,1], history_size, future_target, single_step=False)
    batch_size = 28
    train_data = tf.data.Dataset.from_tensor_slices((X[:TRAIN_SPLIT-history_size], y[:TRAIN_SPLIT-history_size])).batch(batch_size)
    
    model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(32, return_sequences=True, input_shape=(history_size, X.shape[2])),
        tf.keras.layers.LSTM(16, activation='relu'),
        tf.keras.layers.Dense(future_target)
    ])
    model.compile(optimizer=tf.keras.optimizers.RMSprop(clipvalue=1.0), loss='mae')
    model.fit(train_data, epochs=10, verbose=0)
    
    last_history = norm_data[-history_size:]
    last_history = np.expand_dims(last_history, axis=0)
    point_forecast = model.predict(last_history)[0]
    point_forecast = point_forecast * data_std[1] + data_mean[1]
    # Mimic interval forecasts by a fixed margin (e.g., ±10%)
    lower_forecast = point_forecast * 0.9
    upper_forecast = point_forecast * 1.1
    future_index = pd.date_range(start=data.index[-1], periods=future_target+1, closed='right')
    
    plt.figure(figsize=(10,6))
    plt.plot(data.index, data['y'], label='Actual')
    plt.plot(future_index, point_forecast, 'ro-', label='Forecast')
    plt.fill_between(future_index, lower_forecast, upper_forecast, color='gray', alpha=0.3, label='Prediction Interval')
    plt.xlabel('Date')
    plt.ylabel('Orders Count')
    plt.title('Multivariate LSTM Interval Forecast')
    plt.legend()
    plt.grid()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    dummy_mape = 0.0
    return img_base64, dummy_mape
