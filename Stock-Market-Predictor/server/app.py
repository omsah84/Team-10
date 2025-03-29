import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
from flask import Flask, request, jsonify
from sklearn.preprocessing import MinMaxScaler
from flask_cors import CORS

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Allow requests from any frontend

# Load your trained model
try:
    model = load_model('../model/Stock_Predictions_Model.keras')
except Exception as e:
    print(f"Error loading model: {e}")
    raise e

@app.route('/')
def home():
    return jsonify({'message': '📈 Stock Prediction API is running!'})

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    stock = data.get('symbol', '').upper()
    n_days = int(data.get('days', 7))

    if not stock or not stock.isalpha():
        return jsonify({'error': 'Invalid stock symbol.'}), 400

    try:
        df = yf.download(stock, start='2012-01-01', end='2022-12-31')
        if df.empty or 'Close' not in df.columns:
            return jsonify({'error': f"No valid data found for {stock}."}), 400

        # Prepare training and testing data
        data_train = pd.DataFrame(df.Close[:int(len(df) * 0.80)])
        data_test = pd.DataFrame(df.Close[int(len(df) * 0.80):])

        if len(data_train) < 100:
            return jsonify({'error': 'Not enough historical data for prediction.'}), 400

        scaler = MinMaxScaler(feature_range=(0, 1))
        data_train.dropna(inplace=True)
        data_test.dropna(inplace=True)

        past_100_days = data_train.tail(100)
        full_test_data = pd.concat([past_100_days, data_test], ignore_index=True)
        data_scaled = scaler.fit_transform(full_test_data)

        future_input = data_scaled[-100:]
        scale = 1 / scaler.scale_

        # Predict future prices
        future_predictions = []
        for _ in range(n_days):
            future_input = future_input.reshape(1, 100, 1)
            future_pred = model.predict(future_input, verbose=0)[0][0]
            future_predictions.append(future_pred)
            future_input = np.append(future_input[0][1:], future_pred).reshape(100, 1)

        future_predictions = np.array(future_predictions) * scale

        result = {
            'symbol': stock,
            'days': n_days,
            'predicted_prices': [round(float(p), 2) for p in future_predictions]
        }

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
