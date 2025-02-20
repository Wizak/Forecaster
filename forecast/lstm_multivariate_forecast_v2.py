import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from datetime import timedelta

def read_csv(file_path):
    data = pd.read_csv(file_path)
    data['date'] = pd.to_datetime(data['date'])
    data = data.rename(columns={'date':'ds','orders_count':'y'})
    data['is_holiday'] = data['is_holiday'].astype(int)
    data['dow'] = data['ds'].dt.dayofweek
    data['moy'] = data['ds'].dt.month
    data.set_index('ds', inplace=True)
    data.sort_index(inplace=True)
    data = data.asfreq('D')
    data.fillna(method='ffill', inplace=True)
    return data

def run_forecast_table(data):
    features = data[['dow','y','moy']].values
    TRAIN_SPLIT = int(len(features)*0.8)
    mean = features[:TRAIN_SPLIT].mean(axis=0)
    std = features[:TRAIN_SPLIT].std(axis=0)
    norm_data = (features-mean)/std
    history_size = 28
    future_target = 28
    X,y = [],[]
    for i in range(history_size, len(norm_data)-future_target):
        X.append(norm_data[i-history_size:i])
        y.append(norm_data[i+future_target-1,1])
    X = np.array(X)
    y = np.array(y)
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(history_size, norm_data.shape[1])),
        LSTM(32, activation='relu'),
        Dense(future_target)
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mae')
    model.fit(X, y, epochs=20, batch_size=32, verbose=0)
    mape_val = 0.0
    last_history = norm_data[-history_size:]
    last_history = np.expand_dims(last_history, axis=0)
    future_preds = model.predict(last_history)[0]
    future_preds = future_preds*std[1]+mean[1]
    last_date = data.index[-1]
    dates = pd.date_range(start=last_date, periods=future_target+1, closed='right')
    forecast_table = []
    for d, val in zip(dates, future_preds):
        forecast_table.append({
            'ds': str(d.date()),
            'yhat': float(val),
            'yhat_lower': float(val*0.9),
            'yhat_upper': float(val*1.1)
        })
    historical_table = []
    for idx, row in data.iterrows():
        historical_table.append({
            'ds': str(idx.date()),
            'y': float(row['y'])
        })
    return historical_table, forecast_table, mape_val
