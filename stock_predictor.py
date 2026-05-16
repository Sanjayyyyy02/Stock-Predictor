import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Stock Predictor", page_icon="📈", layout="centered")
st.title("📈 NSE Stock Price Predictor")
st.caption("Powered by Neural Network — Enter any NSE stock ticker to predict tomorrow's price")

col1, col2 = st.columns([3, 1])
with col1:
    ticker_input = st.text_input(
        "Enter NSE Ticker Symbol",
        placeholder="e.g. RELIANCE, INFY, HDFCBANK, ZOMATO"
    ).strip().upper()
with col2:
    predict_btn = st.button("🔍 Predict", use_container_width=True)

st.markdown("---")

def fetch_data(ticker):
    df = yf.download(f"{ticker}.NS", period="4y", progress=False)
    df.dropna(inplace=True)
    return df

def create_features(df, sequence_length=60):
    close = df['Close'].squeeze()
    returns = close.pct_change().dropna().values

    scaler = MinMaxScaler(feature_range=(-1, 1))
    returns_scaled = scaler.fit_transform(returns.reshape(-1, 1)).flatten()
    close_prices = close.values[1:]

    X, y = [], []
    for i in range(sequence_length, len(returns_scaled)):
        X.append(returns_scaled[i-sequence_length:i])
        y.append(returns_scaled[i])

    X, y = np.array(X), np.array(y)
    split = int(len(X) * 0.8)

    return (X[:split], X[split:], y[:split], y[split:],
            scaler, close_prices)

def train_and_predict(df):
    X_train, X_test, y_train, y_test, scaler, close_prices = create_features(df)

    model = MLPRegressor(
        hidden_layer_sizes=(128, 64, 32),
        activation='relu',
        solver='adam',
        max_iter=500,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20
    )
    model.fit(X_train, y_train)

    y_pred_scaled = model.predict(X_test).reshape(-1, 1)
    y_actual_scaled = y_test.reshape(-1, 1)

    y_pred_returns = scaler.inverse_transform(y_pred_scaled).flatten()
    y_actual_returns = scaler.inverse_transform(y_actual_scaled).flatten()

    test_start = len(close_prices) - len(y_test)
    base_prices = close_prices[test_start-1: test_start-1+len(y_test)]
    y_pred_prices = base_prices * (1 + y_pred_returns)
    y_actual_prices = base_prices * (1 + y_actual_returns)

    mae = mean_absolute_error(y_actual_prices, y_pred_prices)
    r2 = r2_score(y_actual_prices, y_pred_prices)

    close = df['Close'].squeeze()
    current_price = float(close.iloc[-1])
    last_60_returns = close.pct_change().dropna().values[-60:]
    last_60_scaled = scaler.transform(last_60_returns.reshape(-1, 1)).flatten()
    pred_scaled = model.predict(last_60_scaled.reshape(1, -1))[0]
    pred_return = scaler.inverse_transform([[pred_scaled]])[0][0]
    predicted_price = current_price * (1 + pred_return)

    return {
        'mae': mae,
        'r2': r2,
        'current_price': current_price,
        'predicted_price': predicted_price,
        'pred_return': pred_return,
        'y_actual': y_actual_prices,
        'y_pred': y_pred_prices
    }

if predict_btn and ticker_input:
    with st.spinner(f"Fetching data for {ticker_input}..."):
        try:
            df = fetch_data(ticker_input)
            if len(df) < 100:
                st.error("❌ Not enough data. Try a different ticker.")
                st.stop()
        except Exception as e:
            st.error(f"❌ Could not fetch data. Error: {e}")
            st.stop()

    close = df['Close'].squeeze()
    current_price = float(close.iloc[-1])
    prev_price = float(close.iloc[-2])
    day_change = current_price - prev_price
    day_pct = (day_change / prev_price) * 100

    st.subheader(f"📊 {ticker_input}.NS")
    m1, m2, m3 = st.columns(3)
    m1.metric("Current Price", f"₹{current_price:.2f}",
              f"{day_change:+.2f} ({day_pct:+.2f}%)")
    m2.metric("52W High", f"₹{float(close.max()):.2f}")
    m3.metric("52W Low", f"₹{float(close.min()):.2f}")

    st.markdown("**Price History (4 Years)**")
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(df.index, close, color='#1f77b4', linewidth=1.2)
    ax.fill_between(df.index, close, alpha=0.1, color='#1f77b4')
    ax.set_ylabel("Price (₹)")
    ax.grid(True, alpha=0.3)
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("---")
    st.markdown("**🧠 Training Neural Network...**")
    progress = st.progress(0, text="Preparing data...")
    progress.progress(30, text="Creating features...")
    progress.progress(60, text="Training model...")

    with st.spinner("Training in progress — usually under 30 seconds..."):
        results = train_and_predict(df)

    progress.progress(100, text="Done!")
    st.markdown("---")
    st.subheader("🎯 Prediction Results")

    direction = "📈 UP" if results['pred_return'] > 0 else "📉 DOWN"
    color = "green" if results['pred_return'] > 0 else "red"

    r1, r2_col = st.columns(2)
    r1.metric("Tomorrow's Predicted Price",
              f"₹{results['predicted_price']:.2f}",
              f"{results['pred_return']*100:+.2f}%")
    r2_col.metric("Model Accuracy (R²)",
                  f"{results['r2']:.4f}",
                  "Higher is better (max 1.0)")

    st.markdown(f"### Expected Direction: :{color}[{direction}]")
    st.caption(f"Mean Absolute Error: ₹{results['mae']:.2f}")

    st.markdown("**Actual vs Predicted (Test Set)**")
    fig2, ax2 = plt.subplots(figsize=(12, 4))
    ax2.plot(results['y_actual'], label='Actual',
             color='#1f77b4', linewidth=1.5)
    ax2.plot(results['y_pred'], label='Predicted',
             color='orange', linewidth=1.5, linestyle='--')
    ax2.set_ylabel("Price (₹)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close()

    st.markdown("---")
    st.warning("⚠️ For educational purposes only. Not financial advice.")

elif predict_btn and not ticker_input:
    st.warning("Please enter a ticker symbol first!")
