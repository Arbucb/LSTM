import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

def fetch_stock_data(ticker, start_date, end_date):
    stock = yf.Ticker(ticker)
    data = stock.history(start=start_date, end=end_date)
    return data['Close'].values.reshape(-1, 1)

def prepare_data(data, look_back=60):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    
    X, y = [], []
    for i in range(look_back, len(scaled_data)):
        X.append(scaled_data[i-look_back:i, 0])
        y.append(scaled_data[i, 0])
    
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    
    return X, y, scaler

def create_model(look_back):
    model = Sequential([
        LSTM(units=50, return_sequences=True, input_shape=(look_back, 1)),
        Dropout(0.2),
        LSTM(units=50, return_sequences=True),
        Dropout(0.2),
        LSTM(units=50),
        Dropout(0.2),
        Dense(units=1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    return model

def train_model(X, y, epochs=50, batch_size=32):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = create_model(X.shape[1])
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test), verbose=1)
    return model, history

def predict_next_day(model, data, scaler, look_back):
    last_sequence = data[-look_back:]
    last_sequence = scaler.transform(last_sequence)
    last_sequence = np.reshape(last_sequence, (1, look_back, 1))
    prediction = model.predict(last_sequence)
    return scaler.inverse_transform(prediction)[0, 0]

# Main execution
if __name__ == "__main__":
    ticker = "AAPL"  # Example: Apple Inc.
    start_date = "2020-01-01"
    end_date = "2024-08-12"
    look_back = 60  # Number of previous days to use for prediction
   
    # Fetch data
    data = fetch_stock_data(ticker, start_date, end_date)

    # Prepare data
    X, y, scaler = prepare_data(data, look_back)

    # Train model
    model, history = train_model(X, y, epochs=10)

    # Predict next day's price
    next_day_price = predict_next_day(model, data, scaler, look_back)
    print(f"Predicted price for next day: ${next_day_price:.2f}")

    # Plot training history
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # Plot actual vs predicted prices
    actual_prices = scaler.inverse_transform(y.reshape(-1, 1))
    predicted_prices = model.predict(X)
    predicted_prices = scaler.inverse_transform(predicted_prices)

    plt.figure(figsize=(12, 6))
    plt.plot(actual_prices, label='Actual Prices')
    plt.plot(predicted_prices, label='Predicted Prices')
    plt.title(f'{ticker} Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.show()