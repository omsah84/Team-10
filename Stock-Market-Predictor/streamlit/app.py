import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# ===========================================
# Load Model with Error Handling
# ===========================================
try:
    model = load_model('Stock_Predictions_Model.keras')
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# ===========================================
# Streamlit UI
# ===========================================
st.title("📈 Stock Market Predictor")

# User input for stock symbol
stock = st.text_input('Enter Stock Symbol (e.g., AAPL, TSLA, GOOG)', 'GOOG').upper()

# Validate stock input
if not stock.isalpha():
    st.error("Invalid stock symbol. Please enter a valid ticker symbol (e.g., AAPL, TSLA, GOOG).")
    st.stop()

start = '2012-01-01'
end = '2022-12-31'

# ===========================================
# Fetch Stock Data with Error Handling
# ===========================================
try:
    data = yf.download(stock, start, end)
    if data.empty:
        st.error(f"No data found for {stock}. Please check the stock symbol and try again.")
        st.stop()
except Exception as e:
    st.error(f"Error fetching stock data: {e}")
    st.stop()

# Display stock data
st.subheader(f"Stock Data for {stock}")
st.write(data.head())

# Check if 'Close' column exists
if 'Close' not in data.columns:
    st.error("Missing 'Close' price data. The stock may not have sufficient historical data.")
    st.stop()

# ===========================================
# Train-Test Split
# ===========================================
data_train = pd.DataFrame(data.Close[:int(len(data) * 0.80)])
data_test = pd.DataFrame(data.Close[int(len(data) * 0.80):])

# ===========================================
# Scaling Data
# ===========================================
scaler = MinMaxScaler(feature_range=(0, 1))

# Handle missing values (if any)
data_train.dropna(inplace=True)
data_test.dropna(inplace=True)

# Check if there is enough data
if len(data_train) < 100:
    st.error("Not enough historical data for training. Try another stock with more historical data.")
    st.stop()

# Prepare data for testing
past_100_days = data_train.tail(100)
data_test = pd.concat([past_100_days, data_test], ignore_index=True)
data_test_scaled = scaler.fit_transform(data_test)

# ===========================================
# Moving Averages
# ===========================================
st.subheader(f"📊 Moving Averages for {stock}")

# MA50
ma_50_days = data.Close.rolling(50).mean()
fig1 = plt.figure(figsize=(8, 6))
plt.plot(ma_50_days, 'r', label="MA50")
plt.plot(data.Close, 'g', label="Closing Price")
plt.legend()
st.pyplot(fig1)

# MA100 vs MA50
ma_100_days = data.Close.rolling(100).mean()
fig2 = plt.figure(figsize=(8, 6))
plt.plot(ma_50_days, 'r', label="MA50")
plt.plot(ma_100_days, 'b', label="MA100")
plt.plot(data.Close, 'g', label="Closing Price")
plt.legend()
st.pyplot(fig2)

# MA100 vs MA200
ma_200_days = data.Close.rolling(200).mean()
fig3 = plt.figure(figsize=(8, 6))
plt.plot(ma_100_days, 'r', label="MA100")
plt.plot(ma_200_days, 'b', label="MA200")
plt.plot(data.Close, 'g', label="Closing Price")
plt.legend()
st.pyplot(fig3)

# ===========================================
# Prepare Data for Prediction
# ===========================================
x, y = [], []

for i in range(100, data_test_scaled.shape[0]):
    x.append(data_test_scaled[i - 100:i])
    y.append(data_test_scaled[i, 0])

x, y = np.array(x), np.array(y)

# Predict test data
predict = model.predict(x)

# Inverse scaling
scale = 1 / scaler.scale_
predict = predict * scale
y = y * scale

# ===========================================
# Display Original vs Predicted Prices
# ===========================================
st.subheader(f"📉 Original vs Predicted Prices for {stock}")
fig4 = plt.figure(figsize=(8, 6))
plt.plot(y, 'g', label='Original Price')
plt.plot(predict, 'r', label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig4)

# ===========================================
# Future Prediction
# ===========================================
st.subheader("🔮 Future Price Prediction")

# Get last 100 days of stock prices for future prediction
future_input = data_test_scaled[-100:]

# Predict for the next N days
n_days = st.number_input("Enter number of days to predict:", min_value=1, max_value=200, value=7)

future_predictions = []
for _ in range(n_days):
    future_input = future_input.reshape(1, 100, 1)  # Reshape to match LSTM input shape
    future_pred = model.predict(future_input)[0][0]  # Predict next day
    future_predictions.append(future_pred)

    # Update input data by adding the predicted value
    future_input = np.append(future_input[0][1:], future_pred).reshape(100, 1)

# Inverse transform to get real prices
future_predictions = np.array(future_predictions) * scale

# Display future predictions
st.write(f"Predicted Prices for the next {n_days} days:")
future_df = pd.DataFrame({'Day': list(range(1, n_days + 1)), 'Predicted Price': future_predictions})
st.dataframe(future_df)

# Plot future predictions
fig5 = plt.figure(figsize=(8, 6))
plt.plot(range(1, n_days + 1), future_predictions, 'r', marker='o', label="Predicted Future Prices")
plt.xlabel('Days Ahead')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig5)

# ===========================================
# Handling Model & Data Errors
# ===========================================
if len(future_predictions) == 0:
    st.error("Prediction failed. Ensure the stock data is complete and try again.")
