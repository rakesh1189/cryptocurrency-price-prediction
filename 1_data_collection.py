import yfinance as yf
import pandas as pd
import os

os.makedirs("data", exist_ok=True)

# Download Bitcoin historical data
ticker = "BTC-USD"
print(f"Downloading {ticker} data...")
df = yf.download(ticker, start="2018-01-01", end="2024-12-31")

df.to_csv("data/BTC_USD.csv")
print(df.head())
print(f"Shape: {df.shape}")
print("Data saved to data/BTC_USD.csv")
