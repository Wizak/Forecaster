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

def multivariate_data(dataset, target, start_index, end_index, history_size, target_size, step, single_step=True):
    data, labels = [], []
    start_index = start_index + history_size
    if end_index is None:
        end_index = len(dataset) - target_size
    for i in range(start_index, end_index):
        indices = range(i - history_size, i, step)
        data.append(dataset[indices])
        if single_step:
            labels.append(target[i+target_size])
        else:
            labels.append(target[i:i+target_size])
    return np.array(data), np.array(labels)

def create_time_steps(length):
    return list(range(-length, 0))

def run_forecast_table(file_input, forecast_start, forecast_end):
    # Parameters
    past_history = 28   # history window length (days)
    future_target = 0   # we predict a single future point
    STEP = 1
    EPOCHS = 10
    BATCH_SIZE = 28
    BUFFER_SIZE = 28 * 5
    TRAIN_SPLIT = 28 * 20

    # Read data (using data up to forecast_end)
    df = read_csv(file_input)
    df = df[:forecast_end]
    features_considered = ['dow', 'y', 'moy']
    dataset = df[features_considered].values.astype('float32')

    # Compute normalization stats from the training portion
    train_data = dataset[-TRAIN_SPLIT:]
    data_mean = train_data.mean(axis=0)
    data_std = train_data.std(axis=0)
    dataset_norm = (dataset - data_mean) / data_std

    # Prepare training and validation sets (target = column 1: 'y')
    x_train, y_train = multivariate_data(dataset_norm, dataset_norm[:, 1], 0, TRAIN_SPLIT, past_history, future_target, STEP, single_step=True)
    x_val, y_val = multivariate_data(dataset_norm, dataset_norm[:, 1], TRAIN_SPLIT, None, past_history, future_target, STEP, single_step=True)

    # Build a single-step LSTM model
    model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(32, input_shape=x_train.shape[-2:]),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer=tf.keras.optimizers.RMSprop(), loss='mae')
    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_ds = train_ds.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True).repeat()
    model.fit(train_ds, epochs=EPOCHS, steps_per_epoch=7, verbose=0)

    # Use the first validation sample for visualization
    x_sample = x_val[0:1]       # shape: (1, past_history, num_features)
    true_future_norm = y_val[0] # scalar in normalized scale
    pred_norm = model.predict(x_sample)[0, 0]

    # Denormalize: target 'y' is at index 1
    history_norm = x_sample[0][:, 1]
    history_denorm = history_norm * data_std[1] + data_mean[1]
    true_future_denorm = true_future_norm * data_std[1] + data_mean[1]
    pred_denorm = pred_norm * data_std[1] + data_mean[1]

    # Build one combined chart with three series:
    # Series 1: "Actual" history (x: negative time steps)
    actual_series = {"type": "actual", "data": []}
    time_steps = create_time_steps(past_history)
    for t, val in zip(time_steps, history_denorm):
        actual_series["data"].append({"x": str(t), "y": float(val)})

    # Series 2: "True Future" (a single point at x=0)
    true_future_series = {"type": "true_future", "data": [{"x": "0", "y": float(true_future_denorm)}]}

    # Series 3: "Forecast" (a single point at x=0)
    forecast_series = {"type": "forecast", "data": [{"x": "0", "y": float(pred_denorm)}]}

    # Final normalization over all output values for unified display
    combined = actual_series["data"] + true_future_series["data"] + forecast_series["data"]
    all_y = np.array([pt["y"] for pt in combined]).reshape(-1, 1)
    scaler_out = MinMaxScaler((0, 1))
    scaled_y = scaler_out.fit_transform(all_y)
    idx = 0
    for series in [actual_series, true_future_series, forecast_series]:
        for pt in series["data"]:
            pt["y"] = float(scaled_y[idx, 0])
            idx += 1

    return [actual_series, true_future_series, forecast_series]
