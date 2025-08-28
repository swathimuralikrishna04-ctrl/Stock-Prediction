# data_fetch.py
import yfinance as yf
import os

def fetch_to_csv(ticker: str, start=None, end=None, period='max', out_dir='data'):
    os.makedirs(out_dir, exist_ok=True)
    if start or end:
        df = yf.download(ticker, start=start, end=end, progress=False)
    else:
        df = yf.Ticker(ticker).history(period=period)
    df = df[['Open','High','Low','Close','Volume']].dropna()
    out_path = f"{out_dir}/{ticker}.csv"
    df.to_csv(out_path)
    print(f"Saved {len(df)} rows to {out_path}")
    return out_path

if __name__ == "__main__":
    fetch_to_csv("AAPL", start="2015-01-01")