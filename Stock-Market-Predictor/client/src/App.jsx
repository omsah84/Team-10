import React, { useState } from 'react';
import './App.css';

function App() {
  const [symbol, setSymbol] = useState('');
  const [days, setDays] = useState(7);
  const [predictions, setPredictions] = useState([]);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  const handlePredict = async () => {
    setError('');
    setPredictions([]);
    setLoading(true);

    try {
      const response = await fetch('http://localhost:5000/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ symbol, days }),
      });

      const data = await response.json();

      if (response.ok) {
        setPredictions(data.predicted_prices);
      } else {
        setError(data.error || 'Prediction failed. Try again.');
      }
    } catch (err) {
      setError('Server error. Make sure Flask API is running.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="App">
      <h1>📈 Stock Predictor</h1>
      <div className="form">
        <input
          type="text"
          placeholder="Enter Stock Symbol (e.g. AAPL)"
          value={symbol}
          onChange={(e) => setSymbol(e.target.value.toUpperCase())}
        />
        <input
          type="number"
          min="1"
          max="30"
          value={days}
          onChange={(e) => setDays(e.target.value)}
        />
        <button onClick={handlePredict}>Predict</button>
      </div>

      {loading && <p>Loading predictions...</p>}
      {error && <p className="error">{error}</p>}

      {predictions.length > 0 && (
        <div className="results">
          <h2>📅 Predicted Prices for {symbol}</h2>
          <ul>
            {predictions.map((price, index) => (
              <li key={index}>Day {index + 1}: ${price.toFixed(2)}</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

export default App;
