import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.preprocessing import MinMaxScaler

def read_csv(file_input):
    data = pd.read_csv(file_input)
    data['date'] = pd.to_datetime(data['date'])
    data = data.rename(columns={'date': 'ds', 'orders_count': 'y'})
    data['is_holiday'] = data['is_holiday'].astype(bool)
    data['dow'] = data['ds'].dt.dayofweek
    data['moy'] = data['ds'].dt.month
    return data

def filter_last_six_months(data):
    last_date = data['ds'].max()
    six_months_ago = last_date - pd.DateOffset(months=6)
    return data[data['ds'] >= six_months_ago]

def train_forecast_model(data, future_holidays, forecast_period=0):
    model = Prophet()
    model.add_regressor('is_holiday')
    model.add_regressor('dow')
    model.add_regressor('moy')
    model.fit(data)
    future = model.make_future_dataframe(periods=forecast_period)
    future['is_holiday'] = future['ds'].isin(future_holidays)
    future['dow'] = future['ds'].dt.dayofweek
    future['moy'] = future['ds'].dt.month
    forecast = model.predict(future)
    if 'yhat_lower' not in forecast.columns:
        forecast['yhat_lower'] = forecast['yhat']
    if 'yhat_upper' not in forecast.columns:
        forecast['yhat_upper'] = forecast['yhat']
    return forecast

def calculate_mape(actual, predicted):
    actual = np.array(actual)
    predicted = np.array(predicted)
    mask = actual != 0
    return np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100

def run_forecast_table(file_input, training_cutoff_date=None, forecast_days_ahead=0, future_holidays=None):
    if future_holidays is None:
        future_holidays = []
    data = read_csv(file_input)
    full_end = data['ds'].max()
    if training_cutoff_date:
        cutoff = pd.to_datetime(training_cutoff_date)
        train_data = data[data['ds'] <= cutoff]
        forecast_period = (full_end - cutoff).days + forecast_days_ahead
    else:
        train_data = data.copy()
        forecast_period = forecast_days_ahead
    forecast = train_forecast_model(train_data, future_holidays, forecast_period)
    last_six = filter_last_six_months(data)
    overlap = forecast[forecast['ds'].isin(last_six['ds'])]
    mape_val = calculate_mape(last_six['y'], overlap['yhat'])

    historical_table = []
    for _, row in last_six.iterrows():
        historical_table.append({
            'ds': str(row['ds'].date()),
            'y': float(row['y'])
        })

    forecast_table = []
    for _, rowf in forecast.iterrows():
        forecast_table.append({
            'ds': str(rowf['ds'].date()),
            'yhat': float(rowf['yhat']),
            'yhat_lower': float(rowf['yhat_lower']),
            'yhat_upper': float(rowf['yhat_upper'])
        })

    df_h = pd.DataFrame(historical_table)
    df_h.rename(columns={'y': 'val'}, inplace=True)
    df_h['val_lower'] = 0.0
    df_h['val_upper'] = 0.0
    df_h['is_forecast'] = False

    df_f = pd.DataFrame(forecast_table)
    df_f.rename(columns={
        'yhat': 'val',
        'yhat_lower': 'val_lower',
        'yhat_upper': 'val_upper'
    }, inplace=True)
    df_f['is_forecast'] = True

    merged = pd.concat([df_h, df_f], ignore_index=True)
    merged[['val', 'val_lower', 'val_upper']] = merged[['val', 'val_lower', 'val_upper']].apply(pd.to_numeric, errors='coerce').fillna(0.0)

    scaler_out = MinMaxScaler(feature_range=(0,1))
    arr_3cols = merged[['val', 'val_lower', 'val_upper']].to_numpy()
    arr_3cols_scaled = scaler_out.fit_transform(arr_3cols)
    merged[['val', 'val_lower', 'val_upper']] = arr_3cols_scaled

    merged_h = merged[merged['is_forecast']==False].copy()
    merged_f = merged[merged['is_forecast']==True].copy()

    historical_table_norm = []
    for _, row in merged_h.iterrows():
        historical_table_norm.append({
            'x': row['ds'],
            'y': float(row['val'])
        })

    forecast_table_norm = []
    for _, row in merged_f.iterrows():
        forecast_table_norm.append({
            'x': row['ds'],
            'y': float(row['val']),
        })

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
