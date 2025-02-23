import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def read_csv(file_input):
    df = pd.read_csv(file_input)
    df['date'] = pd.to_datetime(df['date'])
    df = df.rename(columns={'date': 'ds', 'orders_count': 'y'})
    df['is_holiday'] = df['is_holiday'].astype(int)
    df['dow'] = df['ds'].dt.dayofweek
    df['moy'] = df['ds'].dt.month
    df.set_index('ds', inplace=True)
    df.sort_index(inplace=True)
    df = df.asfreq('D', method='bfill')
    df.ffill(inplace=True)
    return df

def multivariate_data(dataset, target, start_index, end_index, history_size, target_size, step, single_step=False):
    data, labels = [], []
    start_index = start_index + history_size
    if end_index is None:
        end_index = len(dataset) - target_size
    for i in range(start_index, end_index):
        indices = range(i - history_size, i, step)
        data.append(dataset[indices])
        labels.append(target[i:i+target_size])
    return np.array(data), np.array(labels)

def create_time_steps(length):
    return list(range(-length, 0))

def run_forecast_table(file_input, forecast_start, forecast_end):
    # Parameters
    past_history = 28      # history window length
    future_target = 7      # forecast horizon (vector length)
    STEP = 1
    EPOCHS = 10
    BATCH_SIZE = 28
    BUFFER_SIZE = 28 * 5
    TRAIN_SPLIT = 28 * 20

    # Read data (use data up to forecast_end)
    df = read_csv(file_input)
    df = df[:forecast_end]
    features_considered = ['dow','y','moy']
    dataset = df[features_considered].values.astype('float32')
    
    # Compute normalization stats from training portion
    train_data = dataset[-TRAIN_SPLIT:]
    data_mean = train_data.mean(axis=0)
    data_std = train_data.std(axis=0)
    dataset_norm = (dataset - data_mean) / data_std

    # Create training and validation sets; target is column index 1 ('y')
    x_train, y_train = multivariate_data(dataset_norm, dataset_norm[:,1], 0, TRAIN_SPLIT, past_history, future_target, STEP, single_step=False)
    x_val, y_val = multivariate_data(dataset_norm, dataset_norm[:,1], TRAIN_SPLIT, None, past_history, future_target, STEP, single_step=False)

    # Build multi-step LSTM model
    multi_step_model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(32, return_sequences=True, input_shape=x_train.shape[-2:]),
        tf.keras.layers.LSTM(16, activation='relu'),
        tf.keras.layers.Dense(future_target)
    ])
    multi_step_model.compile(optimizer=tf.keras.optimizers.RMSprop(clipvalue=1.0), loss='mae')
    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_ds = train_ds.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True).repeat()
    multi_step_model.fit(train_ds, epochs=EPOCHS, steps_per_epoch=7, verbose=0)
    
    # Select the first validation sample for plotting
    x_sample = x_val[0:1]        # shape: (1, past_history, num_features)
    y_true_norm = y_val[0]       # shape: (future_target,)
    # Model prediction: vector of length future_target
    pred_norm = multi_step_model.predict(x_sample)[0]
    
    # Denormalize: target 'y' is at index 1
    history_denorm = x_sample[0][:,1] * data_std[1] + data_mean[1]
    y_true_denorm = y_true_norm * data_std[1] + data_mean[1]
    pred_denorm = pred_norm * data_std[1] + data_mean[1]
    
    # Build one combined graph with three series:
    # Series 1: History (x-axis: negative time steps)
    history_series = {"type": "actual", "data": []}
    time_steps = create_time_steps(past_history)
    for t, val in zip(time_steps, history_denorm):
        history_series["data"].append({"x": str(t), "y": float(val)})
    # Series 2: True Future (x-axis: 0,1,...,future_target-1)
    true_future_series = {"type": "true_future", "data": []}
    for i, val in enumerate(y_true_denorm):
        true_future_series["data"].append({"x": str(i), "y": float(val)})
    # Series 3: Model Prediction (x-axis: same as true future)
    forecast_series = {"type": "forecast", "data": []}
    for i, val in enumerate(pred_denorm):
        forecast_series["data"].append({"x": str(i), "y": float(val)})
    
    # Combine all y-values for final normalization
    combined = history_series["data"] + true_future_series["data"] + forecast_series["data"]
    all_y = np.array([pt["y"] for pt in combined]).reshape(-1, 1)
    scaler_out = MinMaxScaler((0,1))
    scaled_y = scaler_out.fit_transform(all_y)
    idx = 0
    for series in [history_series, true_future_series, forecast_series]:
        for pt in series["data"]:
            pt["y"] = float(scaled_y[idx, 0])
            idx += 1

    return [history_series, true_future_series, forecast_series]
