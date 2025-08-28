# forecast_util.py
import numpy as np
import joblib
from tensorflow.keras.models import load_model

def iterative_forecast(model_path, scaler_path, last_seq, n_days=7):
    model = load_model(model_path, compile=False)  # <- keep only this

    scaler = joblib.load(scaler_path)
    seq = last_seq.copy()  # should be 1D scaled array of length seq_len
    preds_scaled = []
    for _ in range(n_days):
        x = seq.reshape(1, seq.shape[0], 1)
        p = model.predict(x)
        preds_scaled.append(p[0, 0])
        seq = np.append(seq[1:], p)[-seq.shape[0]:]  # update sequence

    preds = scaler.inverse_transform(np.array(preds_scaled).reshape(-1, 1))
    preds = preds.flatten()
    return preds
