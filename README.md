# Stock Prediction App

This project predicts stock prices using historical data and machine learning models (LSTM). It fetches stock data, prepares it for analysis, trains a predictive model, and visualizes future stock trends.

---

## Features
- Fetches real-time stock market data using `yfinance`
- Preprocesses and cleans data for modeling
- Implements LSTM-based time series forecasting
- Visualizes predicted vs actual stock prices
- Modular and easy-to-extend architecture

Example output:

<img width="958" height="545" alt="image" src="https://github.com/user-attachments/assets/092dcc4a-e5b8-498f-b642-ea6f4fc52967" />

<img width="958" height="600" alt="image" src="https://github.com/user-attachments/assets/fde4e280-c030-438c-a43d-336c698f170f" />


---

## Project Structure

<img width="810" height="366" alt="image" src="https://github.com/user-attachments/assets/2b090814-675a-4a0d-9e0d-cc3b0edf0bcd" />

---


## Installation

1. **Clone this repository:**

```bash
git clone https://github.com/your-username/stock-prediction_app.git
cd stock-prediction_app
```

2.  Create a virtual environment and activate it:

```bash
python -m venv venv
.\venv\Scripts\activate   # On Windows
```
# OR
```bash
source venv/bin/activate  # On macOS/Linux
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Usage
1. Prepare the data:
```bash
pip install -r requirements.txt
```

2. Train the LSTM model:
```bash
python train_lstm.py
```

3. Run the application:
```bash
python app.py
```

---

## Dependencies

1. numpy
2. pandas
3. matplotlib
4. scikit-learn
5. tensorflow / keras
6. yfinance

---

## Contributing

Contributions are welcome! Here's how you can help:

1. Fork the repository
2. Create a new branch (`git checkout -b feature-branch`)
3. Make your changes
4. Commit your changes (`git commit -m "Add new feature"`)
5. Push to your branch (`git push origin feature-branch`)
6. Open a Pull Request

Please ensure your contributions follow the project's coding style and include appropriate tests if applicable.

---

## License

This project is licensed under the [MIT License](LICENSE).


