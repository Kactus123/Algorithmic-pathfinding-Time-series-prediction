import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from meteostat import Point, Daily
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from sklearn.model_selection import train_test_split
import sys

# Set system encoding to utf-8
sys.stdout.reconfigure(encoding='utf-8')

# Constants
LOCATION = Point(49.2497, -123.1193, 70)
START_DATE = datetime(2023, 1, 1)
END_DATE = datetime(2024, 5, 31)
PREDICT_MONTH_START = datetime(2024, 6, 1)
PREDICT_MONTH_END = datetime(2024, 6, 30)
FEATURES = ['tavg', 'tmin', 'tmax']
SEQUENCE_LENGTHS = [5, 10, 15, 20]

# Fetch the temperature data from START_DATE to END_DATE
data = Daily(LOCATION, START_DATE, END_DATE)
data = data.fetch()
data = data[FEATURES]

# Plot the original data
plt.figure(figsize=(12, 18))
plt.subplot(5, 2, 1)
plt.plot(data.index, data['tavg'], label='tavg', color='red')
plt.plot(data.index, data['tmin'], label='tmin', color='green')
plt.plot(data.index, data['tmax'], label='tmax', color='blue')
plt.title('Original Data')
plt.xlabel('Date')
plt.ylabel('Temperature (°C)')
plt.legend()

# Normalize the data
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data[FEATURES])
data_scaled = pd.DataFrame(data_scaled, columns=FEATURES, index=data.index)

# Plot the normalized data
plt.subplot(5, 2, 2)
plt.plot(data_scaled.index, data_scaled['tavg'], label='tavg', color='red')
plt.plot(data_scaled.index, data_scaled['tmin'], label='tmin', color='green')
plt.plot(data_scaled.index, data_scaled['tmax'], label='tmax', color='blue')
plt.title('Data After Normalization')
plt.xlabel('Date')
plt.ylabel('Normalized Temperature')
plt.legend()

# Handle missing data using forward fill
data_scaled.ffill(inplace=True)

# Plot after handling missing data
plt.subplot(5, 2, 3)
plt.plot(data_scaled.index, data_scaled['tavg'], label='tavg', color='red')
plt.plot(data_scaled.index, data_scaled['tmin'], label='tmin', color='green')
plt.plot(data_scaled.index, data_scaled['tmax'], label='tmax', color='blue')
plt.title('After Handling Missing Data')
plt.xlabel('Date')
plt.ylabel('Normalized Temperature')
plt.legend()

# Create sequences for the LSTM model
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

# Define LSTM model architecture
def create_lstm_model(seq_length, features):
    model = Sequential()
    model.add(Input(shape=(seq_length, features)))
    model.add(LSTM(units=64, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=64))
    model.add(Dropout(0.2))
    model.add(Dense(units=features))
    model.compile(optimizer='adam', loss='mse')
    return model

# Prepare training and testing data
train_data, test_data = train_test_split(data_scaled.values, test_size=0.2, shuffle=False)

# Loop over different sequence lengths and plot training/validation losses
for i, seq_length in enumerate(SEQUENCE_LENGTHS):
    X_train, y_train = create_sequences(train_data, seq_length)
    X_test, y_test = create_sequences(test_data, seq_length)

    lstm_model = create_lstm_model(seq_length, len(FEATURES))

    history = lstm_model.fit(X_train, y_train, validation_split=0.2, epochs=50, batch_size=32, verbose=0)

    # Plot training and validation loss
    plt.subplot(5, 2, 4 + i)
    plt.plot(history.history['loss'], label='Training Loss', color='red')
    plt.plot(history.history['val_loss'], label='Validation Loss', color='blue')
    plt.title(f'Training and Validation Loss (Seq Length {seq_length})')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    test_loss = lstm_model.evaluate(X_test, y_test, verbose=0)
    print(f"Test MSE for sequence length {seq_length}: {test_loss}")

    # Predict temperatures for June 2024
    if seq_length == 20:
        last_sequence = data_scaled.values[-seq_length:]
        num_days_to_predict = (PREDICT_MONTH_END - PREDICT_MONTH_START).days + 1
        predictions = []
        current_sequence = last_sequence.reshape(1, seq_length, len(FEATURES))

        for _ in range(num_days_to_predict):
            next_step = lstm_model.predict(current_sequence)
            predictions.append(next_step[0])
            current_sequence = np.append(current_sequence[:, 1:, :], next_step.reshape(1, 1, len(FEATURES)), axis=1)

        predictions = np.array(predictions)
        predicted_actual = scaler.inverse_transform(predictions)

        future_dates = pd.date_range(start=PREDICT_MONTH_START, periods=num_days_to_predict)
        past_dates = pd.date_range(end=END_DATE, periods=seq_length)

        plt.subplot(5, 2, 10)
        plt.plot(past_dates, scaler.inverse_transform(last_sequence), label='Actual', color='black', marker='o', linestyle='-')
        plt.scatter(future_dates, predicted_actual[:, 0], color='red', label='Predicted tavg', marker='o', zorder=5)
        plt.scatter(future_dates, predicted_actual[:, 1], color='green', label='Predicted tmin', marker='o', zorder=5)
        plt.scatter(future_dates, predicted_actual[:, 2], color='blue', label='Predicted tmax', marker='o', zorder=5)
        plt.title('Predicted Temperatures for June 2024')
        plt.xlabel('Date')
        plt.ylabel('Temperature (°C)')
        plt.legend()

plt.tight_layout()
plt.show()