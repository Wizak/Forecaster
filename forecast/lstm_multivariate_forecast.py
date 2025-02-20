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
        indices = range(i-history_size, i, step)
        data.append(dataset[indices])
        if single_step:
            labels.append(target[i+target_size])
        else:
            labels.append(target[i:i+target_size])
    return np.array(data), np.array(labels)

def run_forecast_table(file_input, forecast_start, forecast_end):
    # Parameters
    past_history = 28  # history window length
    future_target = 7  # target offset for single-step prediction
    STEP = 1
    EPOCHS = 10
    BATCH_SIZE = 28
    BUFFER_SIZE = 28 * 5

    # Read and slice data (use data up to forecast_end)
    df = read_csv(file_input)
    df = df[:forecast_end]
    features_considered = ['dow', 'y', 'moy']
    dataset = df[features_considered].values.astype('float32')

    # Use last TRAIN_SPLIT points from training portion to compute normalization stats
    TRAIN_SPLIT = 28 * 20
    train_data = dataset[-TRAIN_SPLIT:]
    data_mean = train_data.mean(axis=0)
    data_std = train_data.std(axis=0)
    dataset_norm = (dataset - data_mean) / data_std

    # Create training and validation sets (target is column index 1: 'y')
    x_train, y_train = multivariate_data(dataset_norm, dataset_norm[:, 1], 0, TRAIN_SPLIT, past_history, future_target, STEP, single_step=True)
    x_val, y_val = multivariate_data(dataset_norm, dataset_norm[:, 1], TRAIN_SPLIT, None, past_history, future_target, STEP, single_step=True)

    # Build and train LSTM model
    model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(32, input_shape=x_train.shape[-2:]),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer=tf.keras.optimizers.RMSprop(), loss='mae')
    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_ds = train_ds.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True).repeat()
    model.fit(train_ds, epochs=EPOCHS, steps_per_epoch=7, verbose=0)

    # Select a few samples from the validation set to mimic the reference visualization.
    # For each sample, we output three series:
    #  - "actual_i": the history window (denormalized 'y' values from column index 1)
    #  - "true_future_i": the actual future value (denormalized)
    #  - "forecast_i": the model's prediction (denormalized)
    samples = 3
    output_series = []
    all_output_values = []  # to gather all y-values for global normalization

    for i in range(samples):
        # History from the validation sample (denormalize 'y' values)
        hist = x_val[i][:, 1] * data_std[1] + data_mean[1]
        true_future = y_val[i] * data_std[1] + data_mean[1]
        pred = model.predict(x_val[i:i+1])[0, 0] * data_std[1] + data_mean[1]

        series_actual = {"type": f"actual_{i+1}", "data": []}
        # Use time steps from -past_history to -1
        time_steps = list(range(-past_history, 0))
        for t, val in zip(time_steps, hist):
            series_actual["data"].append({"x": str(t), "y": float(val)})
            all_output_values.append(val)
        series_true_future = {"type": f"true_future_{i+1}", "data": [{"x": "0", "y": float(true_future)}]}
        series_forecast = {"type": f"forecast_{i+1}", "data": [{"x": "0", "y": float(pred)}]}
        all_output_values.append(true_future)
        all_output_values.append(pred)
        output_series.extend([series_actual, series_true_future, series_forecast])

    # Normalize all output y-values (across all series) to [0,1] for unified display
    all_y = np.array(all_output_values).reshape(-1, 1)
    scaler_out = MinMaxScaler((0, 1))
    scaled_y = scaler_out.fit_transform(all_y)
    idx = 0
    for series in output_series:
        for pt in series["data"]:
            pt["y"] = float(scaled_y[idx, 0])
            idx += 1

    return output_series
