# AI Trading Signals – Regime Detection & Anomaly Analysis

This is a quantitative research project focused on identifying market regimes and anomalous behavior, using lightweight AI models designed to work locally on a standard laptop.

## Main objective

Instead of predicting price or giving direct trading advice, this system identifies the current **market state**:

- Uptrend
- Downtrend
- Sideways / Range
- Volatility / Anomaly regimes

This information helps reduce emotional and chaotic decision-making of retail traders and provides a research environment for studying market microstructure.

## Core methods

### Regime Detection
- Hidden Markov Models (HMM)
- Regime-aware LSTM/GRU
- Multi-timeframe engineered features (1m/5m/1H)

### Anomaly Detection
- Isolation Forest / LOF
- Autoencoder reconstruction error
- Volatility and volume anomaly tracking

### Baseline & Support Models
- XGBoost / LightGBM
- Statistical benchmarks (ARIMA / Prophet)

## Data sources

- MT4 / historical CSV
- WebSocket feeds
- Crypto / Forex / Index futures

## Output

The system does not output trading signals.

It outputs probabilities such as:

- P(uptrend)
- P(downtrend)
- P(range)
- P(anomaly)

This allows the user to make their own decisions based on market context, not emotional impulses.

## Philosophy

This is a research and educational project.
No financial advice. No performance guarantees.

Goal: clarity, structure, discipline, learning & recovery through data.
