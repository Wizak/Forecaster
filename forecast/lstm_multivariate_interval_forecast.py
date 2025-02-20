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
    data = data.asfreq('D', method='bfill')
    data.ffill(inplace=True)
    return data

def run_forecast_table(data):
    features = data[['dow','y','moy']].values
    TRAIN_SPLIT = int(len(features)*0.8)
    mean = features[:TRAIN_SPLIT].mean(axis=0)
    std = features[:TRAIN_SPLIT].std(axis=0)
    norm_data = (features-mean)/std
    history_size = 28
    future_target = 7
    X,y = [],[]
    for i in range(history_size, len(norm_data)-future_target):
        X.append(norm_data[i-history_size:i])
        y.append(norm_data[i+future_target-1,1])
    X = np.array(X)
    y = np.array(y)
    model = Sequential([
        LSTM(32, return_sequences=True, input_shape=(history_size, norm_data.shape[1])),
        LSTM(16, activation='relu'),
        Dense(future_target)
    ])
    model.compile(optimizer=tf.keras.optimizers.RMSprop(clipvalue=1.0), loss='mae')
    model.fit(X, y, epochs=10, batch_size=28, verbose=0)
    mape_val = 0.0
    last_history = norm_data[-history_size:]
    last_history = np.expand_dims(last_history, axis=0)
    point_forecast = model.predict(last_history)[0]
    point_forecast = point_forecast*std[1]+mean[1]
    lower_forecast = point_forecast*0.9
    upper_forecast = point_forecast*1.1
    last_date = data.index[-1]
    dates = pd.date_range(start=last_date, periods=future_target+1, closed='right')
    forecast_table = []
    for d, pf, lf, uf in zip(dates, point_forecast, lower_forecast, upper_forecast):
        forecast_table.append({
            'ds': str(d.date()),
            'yhat': float(pf),
            'yhat_lower': float(lf),
            'yhat_upper': float(uf)
        })
    historical_table = []
    for idx, row in data.iterrows():
        historical_table.append({
            'ds': str(idx.date()),
            'y': float(row['y'])
        })
    return historical_table, forecast_table, mape_val
