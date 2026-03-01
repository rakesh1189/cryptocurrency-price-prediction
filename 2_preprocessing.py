import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Read CSV - skip first 2 rows (Price, Ticker), 3rd row is Date header
df = pd.read_csv("data/BTC_USD.csv", skiprows=[0, 1], index_col=0, parse_dates=True)
df.index.name = "Date"

# Rename columns cleanly
df.columns = ['Close', 'High', 'Low', 'Open', 'Volume']

# Remove the empty 'Date,,,,,' row if present
df = df[df.index.notna()]
df = df[df['Close'].notna()]
df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
df = df.dropna()

print("Shape:", df.shape)
print("Missing values:", df.isnull().sum())
print(df.head())
print(df.describe())

# Normalize
scaler = MinMaxScaler(feature_range=(0, 1))
df['Scaled_Close'] = scaler.fit_transform(df[['Close']])

df.to_csv("data/BTC_processed.csv")
print("\nPreprocessing done! Saved to data/BTC_processed.csv")