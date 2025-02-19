import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler
import io
import base64

# Define constants for splitting the data
TRAIN_START = '2022-01-01'
TRAIN_END = '2024-08-31'
TEST_START = '2024-09-01'
TEST_END = '2024-10-31'
FORECAST_PERIOD = 30  # 30-day forecast
SEQ_LENGTH = 30       # Number of timesteps to look back

def read_csv(file_stream):
    """
    Read the CSV file from an in-memory stream and preprocess the data.
    Expects the CSV to have columns: 'date', 'orders_count'.
    """
    data = pd.read_csv(file_stream)
    data['date'] = pd.to_datetime(data['date'])
    data = data.rename(columns={'date': 'ds', 'orders_count': 'y'})
    data.set_index('ds', inplace=True)
    data.sort_index(inplace=True)
    # Set frequency to daily (business fill for missing days)
    data = data.asfreq('D', method='bfill')
    data.ffill(inplace=True)
    return data

def preprocess_data(data):
    """
    Normalize the data using MinMaxScaler.
    Returns the scaled data and the scaler.
    """
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data)
    return data_scaled, scaler

def create_sequences(data, seq_length):
    """
    Create input sequences and corresponding targets for training/testing.
    """
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

def calculate_mape(actual, predicted):
    """
    Calculate Mean Absolute Percentage Error (MAPE).
    """
    actual, predicted = np.array(actual), np.array(predicted)
    return np.mean(np.abs((actual - predicted) / actual)) * 100

def run_forecast(data):
    """
    Trains an LSTM model on the given data and produces a 30-day forecast.
    Returns a tuple: (forecast_plot_base64, mape)
    """
    try:
        # Split data into training and testing segments
        train_data = data[TRAIN_START:TRAIN_END]
        test_data = data[TEST_START:TEST_END]
        
        # Preprocess data (scale)
        train_data_scaled, train_scaler = preprocess_data(train_data)
        test_data_scaled, test_scaler = preprocess_data(test_data)
        
        # Create sequences for training and testing
        X_train, y_train = create_sequences(train_data_scaled, SEQ_LENGTH)
        X_test, y_test = create_sequences(test_data_scaled, SEQ_LENGTH)
        
        # Build and train the LSTM model
        model = Sequential([
            LSTM(50, activation='relu', input_shape=(SEQ_LENGTH, 1)),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)
        
        # Make predictions on training and testing data
        train_predictions = model.predict(X_train)
        test_predictions = model.predict(X_test)
        train_predictions_inv = train_scaler.inverse_transform(train_predictions)
        test_predictions_inv = test_scaler.inverse_transform(test_predictions)
        
        # Forecast the next FORECAST_PERIOD days using the last test sequence
        forecast = []
        last_sequence = X_test[-1]  # shape: (SEQ_LENGTH, 1)
        for _ in range(FORECAST_PERIOD):
            current_sequence = last_sequence.reshape(1, SEQ_LENGTH, 1)
            next_pred = model.predict(current_sequence)[0][0]
            forecast.append(next_pred)
            last_sequence = np.append(last_sequence[1:], next_pred).reshape(SEQ_LENGTH, 1)
        forecast = train_scaler.inverse_transform(np.array(forecast).reshape(-1, 1))
        
        # Create the plot
        plt.figure(figsize=(10, 6))
        # Plot actual values from the test period (using original unscaled data)
        plt.plot(test_data.index, test_data['y'], label='Actual', color='blue')
        # Create forecast dates starting from the last date in the data
        forecast_dates = pd.date_range(start=data.index[-1], periods=FORECAST_PERIOD+1, freq='D')[1:]
        plt.plot(forecast_dates, forecast, label='Forecast', color='red', linestyle='--')
        plt.title('Orders Time Series Forecasting (30-day Forecast)')
        plt.xlabel('Date')
        plt.ylabel('Number of Orders')
        plt.legend()
        plt.grid()
        
        # Save the plot to a bytes buffer and encode as Base64
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close()
        img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        
        # Calculate MAPE on the test set (excluding the first SEQ_LENGTH samples)
        mape = calculate_mape(test_data['y'].values[SEQ_LENGTH:], test_predictions_inv.flatten())
        
        return img_base64, mape
    except Exception as e:
        raise e
