import pandas as pd
import matplotlib.pyplot as plt
import os

os.makedirs("plots", exist_ok=True)

# Fixed CSV reading for new yfinance format
df = pd.read_csv("data/BTC_USD.csv", skiprows=[0, 1], index_col=0, parse_dates=True)
df.index.name = "Date"
df.columns = ['Close', 'High', 'Low', 'Open', 'Volume']
df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce')
df = df.dropna()

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle("Cryptocurrency Analysis Dashboard", fontsize=16, fontweight='bold')

# 1. Close Price
axes[0, 0].plot(df['Close'], color='blue')
axes[0, 0].set_title("BTC Closing Price")
axes[0, 0].set_ylabel("USD")

# 2. Volume
axes[0, 1].bar(df.index, df['Volume'], color='green', alpha=0.6)
axes[0, 1].set_title("Trading Volume")

# 3. Daily Returns
df['Daily_Return'] = df['Close'].pct_change()
axes[1, 0].plot(df['Daily_Return'], color='red', alpha=0.7)
axes[1, 0].set_title("Daily Returns")
axes[1, 0].set_ylabel("% Change")

# 4. Moving Averages
df['MA30'] = df['Close'].rolling(30).mean()
df['MA90'] = df['Close'].rolling(90).mean()
axes[1, 1].plot(df['Close'], label='Close', alpha=0.5)
axes[1, 1].plot(df['MA30'], label='30-day MA', color='orange')
axes[1, 1].plot(df['MA90'], label='90-day MA', color='red')
axes[1, 1].set_title("Moving Averages")
axes[1, 1].legend()

plt.tight_layout()
plt.savefig("plots/dashboard.png", dpi=150)
plt.show()
print("Dashboard saved to plots/dashboard.png")