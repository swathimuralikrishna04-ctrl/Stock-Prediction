# train_lstm.py
import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import joblib
import tensorflow as tf

def build_model(seq_len, units=50):
    model = Sequential()
    model.add(LSTM(units, return_sequences=True, input_shape=(seq_len,1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units//2))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

def train(X, y, seq_len=60, epochs=50, batch_size=32):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, shuffle=False)
    model = build_model(seq_len)
    es = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                        epochs=epochs, batch_size=batch_size, callbacks=[es])
    os.makedirs("models", exist_ok=True)
    model.save("models/lstm_close.h5")
    print("Saved model to models/lstm_close.h5")
    return model, history

if __name__ == "__main__":
    import prepare_data
    df = prepare_data.load_csv("data/AAPL.csv")
    X, y, scaler = prepare_data.make_close_sequences(df['Close'], seq_len=60)
    joblib.dump(scaler, "models/scaler_close.pkl")
    model, history = train(X, y, seq_len=60, epochs=50)
