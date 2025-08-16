# Solana Token ML Dashboard (Simulated)

This Streamlit app simulates on-chain–style Solana token data, generates trading signals with a machine learning model, backtests a simple strategy, and clusters tokens into Emerging / Stable / Declining categories based on recent trends.

## Features
- **Synthetic Data Simulation** – 30 tokens, hourly data, random spikes in activity/price/liquidity.
- **Feature Engineering** – returns, volatility, EMA ratios, z-scores for activity metrics, liquidity changes.
- **ML Model** – Gradient Boosting Classifier to predict next-hour up moves.
- **Backtesting** – TP/SL/timed exits with fee assumptions.
- **Trend Detection** – 14-day clustering into Emerging, Stable, Declining groups.
- **Interactive UI** – inspect token price charts with signal markers, equity curves, and activity trends.

## How to Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py

