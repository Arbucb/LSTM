import numpy as np
import pandas as pd
import yfinance as yf  # API to fetch stock market data
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Step 1: Fetch historical stock data
def fetch_stock_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    return data

# Step 2: Data Preprocessing
def preprocess_data(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))
    
    # Create sequences
    sequence_length = 60
    x_train, y_train = [], []
    
    for i in range(sequence_length, len(scaled_data)):
        x_train.append(scaled_data[i-sequence_length:i, 0])
        y_train.append(scaled_data[i, 0])
        
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    
    return x_train, y_train, scaler

# Step 3: Build LSTM Model
def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=25))
    model.add(Dense(units=1))
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Step 4: Train the model
def train_model(model, x_train, y_train, epochs=5, batch_size=32):
    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)
    return model

# Step 5: Make Predictions
def make_predictions(model, data, scaler):
    last_60_days = data[-60:].values
    last_60_days_scaled = scaler.transform(last_60_days.reshape(-1, 1))
    
    x_test = []
    x_test.append(last_60_days_scaled)
    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    
    predicted_price = model.predict(x_test)
    predicted_price = scaler.inverse_transform(predicted_price)
    
    return predicted_price

# Example usage
ticker = 'AAPL'
start_date = '2015-01-01'
end_date = '2024-08-13'

# Fetch data
data = fetch_stock_data(ticker, start_date, end_date)

# Preprocess data
x_train, y_train, scaler = preprocess_data(data)

# Build and train the model
model = build_lstm_model((x_train.shape[1], 1))
model = train_model(model, x_train, y_train, epochs=10, batch_size=32)

# Predict future price
predicted_price = make_predictions(model, data['Close'], scaler)
print(f"Predicted next day price: {predicted_price[0][0]}")