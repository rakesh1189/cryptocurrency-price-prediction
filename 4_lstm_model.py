import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import os

os.makedirs("models", exist_ok=True)
os.makedirs("plots", exist_ok=True)

# Fixed CSV reading for new yfinance format
df = pd.read_csv("data/BTC_USD.csv", skiprows=[0, 1], index_col=0, parse_dates=True)
df.index.name = "Date"
df.columns = ['Close', 'High', 'Low', 'Open', 'Volume']
df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
data = df[['Close']].dropna().values

# Scale data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Create sequences of 60 days
def create_sequences(data, seq_len=60):
    X, y = [], []
    for i in range(seq_len, len(data)):
        X.append(data[i-seq_len:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

SEQ_LEN = 60
X, y = create_sequences(scaled_data, SEQ_LEN)

# Train/test split (80/20)
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Reshape for LSTM [samples, timesteps, features]
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test  = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# Build LSTM Model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(SEQ_LEN, 1)),
    Dropout(0.2),
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(25),
    Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')
model.summary()

# Train the model
print("\nTraining LSTM model (this may take a few minutes)...")
history = model.fit(
    X_train, y_train,
    batch_size=32,
    epochs=20,
    validation_data=(X_test, y_test)
)

# Save model
model.save("models/lstm_crypto.h5")
print("Model saved to models/lstm_crypto.h5")

# Predict
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)
actual = scaler.inverse_transform(y_test.reshape(-1, 1))

# Plot predictions
plt.figure(figsize=(14, 6))
plt.plot(actual, label='Actual Price', color='blue')
plt.plot(predictions, label='LSTM Predicted', color='orange')
plt.title("LSTM: BTC Price Prediction")
plt.xlabel("Days")
plt.ylabel("Price (USD)")
plt.legend()
plt.savefig("plots/lstm_prediction.png")
plt.show()

# Plot loss
plt.figure(figsize=(10, 4))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title("LSTM Training Loss")
plt.legend()
plt.savefig("plots/lstm_loss.png")
plt.show()

print("Plots saved to plots/")