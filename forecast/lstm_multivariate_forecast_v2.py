import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def read_csv(file_input):
    try:
        data = pd.read_csv(file_input)
        data['date'] = pd.to_datetime(data['date'])
        data = data.rename(columns={'date': 'ds', 'orders_count': 'y'})
        data['is_holiday'] = data['is_holiday'].astype(int)
        data['dow'] = data['ds'].dt.dayofweek
        data['moy'] = data['ds'].dt.month

        data.set_index('ds', inplace=True)
        data.sort_index(inplace=True)
        data = data.asfreq('D')
        data.fillna(method='ffill', inplace=True)
        return data
    except Exception as e:
        raise ValueError(f"Error processing CSV file: {e}")

def multivariate_data(dataset, target, start_index, end_index, history_size, target_size, step):
    data, labels = [], []
    start_index += history_size
    if end_index is None:
        end_index = len(dataset) - target_size

    for i in range(start_index, end_index):
        indices = range(i - history_size, i, step)
        data.append(dataset[indices])
        labels.append(target[i:i + target_size])
    return np.array(data), np.array(labels)

def run_forecast_table(file_input):
    """
    Адаптація оригінального коду.
    Використовує “штучну” вісь X: [-past_history..-1] для історії,
    [0..(future_target-1)] для майбутнього.
    В кінці робимо фінальну нормалізацію [0..1] для графіка.
    """
    tf.random.set_seed(13)

    # 1. Завантаження даних
    df = read_csv(file_input)
    features_considered = ['dow', 'y', 'moy']
    features = df[features_considered]

    # 2. Розділення та нормалізація
    TRAIN_SPLIT = int(len(features) * 0.8)
    dataset = features.values
    data_mean = dataset[:TRAIN_SPLIT].mean(axis=0)
    data_std = dataset[:TRAIN_SPLIT].std(axis=0)
    dataset = (dataset - data_mean) / data_std

    # Параметри
    past_history = 28
    future_target = 28
    step = 1
    batch_size = 32
    buffer_size = 10000
    epochs = 20

    # 3. Формуємо train/val
    x_train, y_train = multivariate_data(
        dataset, dataset[:, 1], 0, TRAIN_SPLIT,
        past_history, future_target, step
    )
    x_val, y_val = multivariate_data(
        dataset, dataset[:, 1], TRAIN_SPLIT, None,
        past_history, future_target, step
    )

    # 4. Датасети
    train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_data = train_data.cache().shuffle(buffer_size).batch(batch_size).repeat()

    val_data = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    val_data = val_data.batch(batch_size).repeat()

    # 5. Модель
    model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(64, return_sequences=True, input_shape=x_train.shape[-2:]),
        tf.keras.layers.LSTM(32, activation='relu'),
        tf.keras.layers.Dense(future_target)
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='mae')

    steps_per_epoch = len(x_train) // batch_size
    model.fit(
        train_data,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_data,
        validation_steps=50,
        verbose=0
    )

    # 6. Беремо 1 batch з val_data
    x_batch, y_batch = next(iter(val_data))
    # Беремо перший елемент у batch
    x_sample, y_sample = x_batch[0], y_batch[0]
    predictions = model.predict(tf.expand_dims(x_sample, axis=0))[0]

    # 7. Денормалізація
    # history => x_sample[:,1]
    history = x_sample[:, 1] * data_std[1] + data_mean[1]  # (28,)
    true_future = y_sample * data_std[1] + data_mean[1]    # (28,)
    pred_future = predictions * data_std[1] + data_mean[1] # (28,)

    # 8. Формуємо 3 серії:
    # History (x=-28..-1)
    history_series = {"type": "actual", "data": []}
    for i, val in enumerate(history):
        x_val_ = i - len(history)  # i=0 => x=-28, ..., i=27 => x=-1
        history_series["data"].append({"x": str(x_val_), "y": float(val)})

    # True Future (x=0..27)
    true_series = {"type": "true_future", "data": []}
    for i, val in enumerate(true_future):
        true_series["data"].append({"x": str(i), "y": float(val)})

    # Predicted Future (x=0..27)
    forecast_series = {"type": "forecast", "data": []}
    for i, val in enumerate(pred_future):
        forecast_series["data"].append({"x": str(i), "y": float(val)})

    # 9. (Опційно) Фінальна нормалізація [0..1] для всіх точок
    combined = history_series["data"] + true_series["data"] + forecast_series["data"]
    all_vals = np.array([float(pt["y"]) for pt in combined]).reshape(-1, 1)
    sc_final = MinMaxScaler(feature_range=(0, 1))
    scaled = sc_final.fit_transform(all_vals)
    idx = 0
    for series in [history_series, true_series, forecast_series]:
        for pt in series["data"]:
            pt["y"] = float(scaled[idx, 0])
            idx += 1

    return [history_series, true_series, forecast_series]
