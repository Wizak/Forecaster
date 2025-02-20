import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

def read_csv(file_path):
    data = pd.read_csv(file_path)
    data['date'] = pd.to_datetime(data['date'])
    data = data.rename(columns={'date': 'ds', 'orders_count': 'y'})
    data['dow'] = data['ds'].dt.dayofweek
    data['service'] = data['service'].astype('category').cat.codes
    data.drop_duplicates(subset='ds', inplace=True)
    data.set_index('ds', inplace=True)
    data.sort_index(inplace=True)
    data = data.asfreq('D', method='bfill')
    data.ffill(inplace=True)
    return data

def preprocess_data(data):
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled = scaler.fit_transform(data[['y','dow','service']])
    return scaled, scaler

def calculate_mape(actual, predicted):
    actual = np.array(actual)
    predicted = np.array(predicted)
    mask = actual != 0
    return np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100

def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length, :])
        y.append(data[i+seq_length, 0])
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
    seq_length = 30
    data = read_csv(file_input)
    train_data = data[train_start:train_end]
    test_data = data[test_start:test_end]
    train_arr, train_scaler = preprocess_data(train_data[['y','dow','service']])
    test_arr, test_scaler   = preprocess_data(test_data[['y','dow','service']])
    X_train, y_train = create_sequences(train_arr, seq_length)
    X_test,  y_test  = create_sequences(test_arr, seq_length)

    model = Sequential([
        LSTM(50, activation='relu', input_shape=(seq_length, X_train.shape[2])),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)

    train_pred = model.predict(X_train)
    test_pred  = model.predict(X_test)

    train_pred_inv = train_scaler.inverse_transform(
        np.hstack([train_pred, np.zeros((len(train_pred),2))])
    )[:,0]
    test_pred_inv  = test_scaler.inverse_transform(
        np.hstack([test_pred, np.zeros((len(test_pred),2))])
    )[:,0]
    y_test_inv     = test_scaler.inverse_transform(
        np.hstack([y_test.reshape(-1,1), np.zeros((len(y_test),2))])
    )[:,0]

    train_index = train_data.index
    test_index  = test_data.index[seq_length:]
    mape_val = calculate_mape(y_test_inv, test_pred_inv)

    forecast_period = (pd.to_datetime(forecast_end) - pd.to_datetime(forecast_start)).days + 1
    forecast_vals = []
    last_seq = X_test[-1]
    for _ in range(forecast_period):
        pred = model.predict(last_seq.reshape(1, seq_length, X_test.shape[2]))[0][0]
        forecast_vals.append(pred)
        next_input = np.hstack([pred, last_seq[-1,1:]])
        last_seq = np.append(last_seq[1:], next_input).reshape(seq_length, X_test.shape[2])
    forecast_vals_inv = train_scaler.inverse_transform(
        np.hstack([np.array(forecast_vals).reshape(-1,1), np.zeros((forecast_period,2))])
    )[:,0]

    tr_idx = train_data.index[seq_length:] if len(train_data) > seq_length else train_data.index
    training_table = []
    for i, date_i in enumerate(tr_idx):
        if i < len(train_pred_inv):
            training_table.append({
                'ds': str(date_i.date()),
                'actual': float(train_data.iloc[seq_length + i]['y']) if len(train_data) > seq_length else float(train_data.iloc[i]['y']),
                'pred': float(train_pred_inv[i])
            })

    testing_table = []
    for i, date_i in enumerate(test_index):
        testing_table.append({
            'ds': str(date_i.date()),
            'actual': float(y_test_inv[i]),
            'pred': float(test_pred_inv[i])
        })

    forecast_dates = pd.date_range(start=forecast_start, end=forecast_end)
    forecast_table = []
    for d, val in zip(forecast_dates, forecast_vals_inv):
        forecast_table.append({'ds': str(d.date()), 'pred': float(val)})

    df_tr = pd.DataFrame(training_table)
    df_ts = pd.DataFrame(testing_table)
    df_fc = pd.DataFrame(forecast_table)

    df_hist = []
    for _, row in df_tr.iterrows():
        df_hist.append({
            'ds': row['ds'],
            'val': row['actual'],
            'type': 'train_actual'
        })
        df_hist.append({
            'ds': row['ds'],
            'val': row['pred'],
            'type': 'train_pred'
        })
    for _, row in df_ts.iterrows():
        df_hist.append({
            'ds': row['ds'],
            'val': row['actual'],
            'type': 'test_actual'
        })
        df_hist.append({
            'ds': row['ds'],
            'val': row['pred'],
            'type': 'test_pred'
        })
    df_hist = pd.DataFrame(df_hist)

    df_fc2 = []
    for _, row in df_fc.iterrows():
        df_fc2.append({
            'ds': row['ds'],
            'val': row['pred'],
            'type': 'forecast'
        })
    df_fc2 = pd.DataFrame(df_fc2)

    merged = pd.concat([df_hist, df_fc2], ignore_index=True)
    merged['val'] = pd.to_numeric(merged['val'], errors='coerce').fillna(0.0)
    merged['ds'] = merged['ds'].astype(str)

    out_scaler = MinMaxScaler(feature_range=(0,1))
    arr = merged[['val']].values
    arr_scaled = out_scaler.fit_transform(arr)
    merged['val'] = arr_scaled

    data = [
        {
            "type": "train_actual",
            "data": [],
        },
        {
            "type": "train_pred",
            "data": [],
        },
        {
            "type": "test_actual",
            "data": [],
        },
        {
            "type": "test_pred",
            "data": [],
        },
        {
            "type": "forecast",
            "data": [],
        },
    ]

    for _, row in merged.iterrows():
        idx = None
        if row['type'] == "train_actual":
            idx = 0
        elif row['type'] == "train_pred":
            idx = 1
        elif row['type'] == "test_actual":
            idx = 2
        elif row['type'] == "test_pred":
            idx = 3
        else:
            idx = 4

        data[idx]["data"].append({
            'x': row['ds'],
            'y': float(row['val']),
        })

    return data, mape_val
