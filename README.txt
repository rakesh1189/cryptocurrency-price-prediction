# Cryptocurrency Price Prediction Using Machine Learning

## Setup Instructions

### Step 1: Install dependencies
Open Command Prompt inside this folder and run:
pip install -r requirements.txt
### Step 2: Run files IN ORDER
python 1_data_collection.py
python 2_preprocessing.py
python 3_linear_regression.py
python 4_lstm_model.py
python 5_visualize.py
## File Description

| File | Description |
|------|-------------|
| 1_data_collection.py  | Downloads BTC historical data using yfinance |
| 2_preprocessing.py    | Cleans and normalizes the data |
| 3_linear_regression.py| Trains and evaluates Linear Regression model |
| 4_lstm_model.py       | Trains LSTM deep learning model |
| 5_visualize.py        | Creates analysis dashboard with charts |
| requirements.txt      | All Python libraries needed |

## Output Files Generated

- data/BTC_USD.csv           → Raw downloaded data
- data/BTC_processed.csv     → Cleaned data
- models/lstm_crypto.h5      → Saved LSTM model
- plots/linear_regression.png
- plots/lstm_prediction.png
- plots/lstm_loss.png
- plots/dashboard.png

## Tips
- LSTM training takes 5–15 minutes depending on your machine
- You can change "BTC-USD" to "ETH-USD" or "BNB-USD" for other coins
