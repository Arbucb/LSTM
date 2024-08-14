import numpy as np
import pandas as pd
import yfinance as yf  # For accessing stock data via API
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error, r2_score

# 1. Data Collection
def fetch_stock_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    return data

# Example usage
ticker_symbol = 'AAPL'  # Adjust as needed
start_date = '2015-01-01'
end_date = '2023-08-14'

data = fetch_stock_data(ticker_symbol, start_date, end_date)

# 2. Data Preprocessing
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))


# 3. Create Sequences for LSTM
def create_sequences(dataset, look_back):
    X, y = [], []
    for i in range(len(dataset) - look_back - 1):
        X.append(dataset[i:(i + look_back), 0])
        y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(y)

look_back = 60  # Number of past days to consider
X, y = create_sequences(scaled_data, look_back)

# 4. Split into Train and Test
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# 5. Build the LSTM Model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')


# 6. Train the Model
model.fit(X_train, y_train, epochs=50, batch_size=32)  # Adjust epochs as needed

# 7. Make Predictions
predicted_stock_price = model.predict(X_test)
predicted_stock_price = scaler.inverse_transform(predicted_stock_price)


# Evaluate and Visualize (Optional)
# ... (Add code to evaluate model performance and visualize predictions)

# ... (Previous code remains the same)

# 8. Evaluate Model Performance
mse = mean_squared_error(y_test, predicted_stock_price)
r2 = r2_score(y_test, predicted_stock_price)
print(f"Mean Squared Error (MSE): {mse}")
print(f"R-squared (R2): {r2}")

# 9. Visualize Predictions
plt.figure(figsize=(12, 6))
plt.plot(scaler.inverse_transform(y_test.reshape(-1, 1)), label='Actual Stock Price')
plt.plot(predicted_stock_price, label='Predicted Stock Price')
plt.title(f"{ticker_symbol} Stock Price Prediction")
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()