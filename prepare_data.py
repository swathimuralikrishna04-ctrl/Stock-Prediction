import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib

def load_csv(path):
    # Read CSV and select the correct header row
    df = pd.read_csv(path, header=1)  # use the second row as header

    # Rename columns to standard names
    rename_map = {
        df.columns[0]: 'Date',
        df.columns[1]: 'Open',
        df.columns[2]: 'High',
        df.columns[3]: 'Low',
        df.columns[4]: 'Close',
        df.columns[5]: 'Volume'
    }
    df.rename(columns=rename_map, inplace=True)

    # Convert Date column
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df.set_index('Date', inplace=True)

    # Convert numeric columns
    numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

    # Drop rows with missing values
    df.dropna(inplace=True)

    return df


def make_close_sequences(series: pd.Series, seq_len=60):
    """
    series: pd.Series of Close prices
    returns: X (samples, seq_len, 1), y (samples,)
    """
    values = series.values.reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(values)             # save this scaler
    X, y = [], []
    for i in range(len(scaled) - seq_len):
        X.append(scaled[i:i+seq_len, 0])
        y.append(scaled[i+seq_len, 0])
    X = np.array(X)
    y = np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    return X, y, scaler

if __name__ == "__main__":
    # Ensure required directories exist
    import os
    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    # Load the CSV file
    df = load_csv("data/AAPL.csv")

    # Create sequences
    X, y, scaler = make_close_sequences(df['Close'], seq_len=60)

    # Save the scaler
    joblib.dump(scaler, "models/scaler_close.pkl")

    print("X.shape", X.shape, "y.shape", y.shape)

