# app.py (Streamlit)
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from forecast_util import iterative_forecast

st.title("Simple Stock Close Price Predictor (LSTM)")

ticker = st.text_input("Ticker", value="AAPL")
seq_len = st.number_input("Sequence length (days)", value=60, min_value=10, max_value=500)
forecast_days = st.number_input("Forecast days", value=7, min_value=1, max_value=30)

if st.button("Fetch & Forecast"):
    with st.spinner("Downloading data..."):
        df = yf.download(ticker, period="3y", progress=False)[['Close']].dropna()
    if len(df) < seq_len+1:
        st.error("Not enough data for chosen sequence length.")
    else:
        st.subheader(f"Last {len(df)} rows for {ticker}")
        st.line_chart(df['Close'])
        # prepare last seq
        scaler = joblib.load("models/scaler_close.pkl")
        scaled = scaler.transform(df['Close'].values.reshape(-1,1)).flatten()
        last_seq = scaled[-seq_len:]
        preds = iterative_forecast("models/lstm_close.h5", "models/scaler_close.pkl", last_seq, n_days=forecast_days)
        future_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=forecast_days, freq='B')  # business days
        pred_df = pd.Series(preds, index=future_dates, name="Predicted Close")
        st.subheader("Forecast")
        st.line_chart(pd.concat([df['Close'].tail(200), pred_df]))
        st.table(pred_df.reset_index().rename(columns={'index':'Date','Predicted Close':'Close'}))
