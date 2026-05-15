# Modern Streamlit Dashboard Upgrade 🚀

import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AI Stock Prediction",
    page_icon="📈",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
.main {
    background-color: #0E1117;
}

h1, h2, h3, h4 {
    color: white;
}

.stMetric {
    background-color: #1E1E1E;
    padding: 15px;
    border-radius: 15px;
    text-align: center;
    box-shadow: 0px 0px 10px rgba(255,255,255,0.1);
}

.css-1d391kg {
    background-color: #111827;
}

.block-container {
    padding-top: 2rem;
}
</style>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------
st.sidebar.title("📊 AI Stock Dashboard")
st.sidebar.write("Deep Learning Based Financial Forecasting")

stocks = {
    "Apple": "AAPL",
    "Tesla": "TSLA",
    "Microsoft": "MSFT",
    "Reliance": "RELIANCE.NS",
    "Ashok Leyland": "ASHOKLEY.NS"
}

selected_stock = st.sidebar.selectbox(
    "Select Suggested Company",
    list(stocks.keys()),
    index=0
)

custom_ticker = st.sidebar.text_input(
    "Or Enter Custom Ticker",
    ""
)

company_map = {
    "tcs": "TCS.NS",
    "infosys": "INFY.NS",
    "google": "GOOGL",
    "amazon": "AMZN",
    "nvidia": "NVDA",
    "meta": "META"
}

if custom_ticker:

    search = custom_ticker.lower()

    if search in company_map:
        ticker = company_map[search]
    else:
        ticker = custom_ticker.upper()

    st.sidebar.success(f"Showing data for: {ticker}")

else:
    ticker = stocks[selected_stock]

timeframe = st.sidebar.selectbox(
    "Select Timeframe",
    ["1y", "2y", "5y", "max"]
)

theme = st.sidebar.radio(
    "Theme",
    ["Dark", "Light"]
)
if theme == "Dark":
    plot_template = "plotly_dark"
else:
    plot_template = "plotly"
if theme == "Dark":
    background_color = "#0E1117"
    text_color = "white"
else:
    background_color = "white"
    text_color = "black"

st.markdown(f"""
    <style>
    .stApp {{
        background-color: {background_color};
        color: {text_color};
    }}
    </style>
""", unsafe_allow_html=True)

# ---------------- LOAD DATA ----------------
data = yf.download(
    ticker,
    period=timeframe
)

if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)

# ---------------- TITLE ----------------
st.title("📈 AI-Powered Stock Market Prediction System")
st.markdown("### Real-Time Financial Analytics Dashboard")

# ---------------- TOP METRICS ----------------
latest_close = float(data['Close'].iloc[-1])
latest_high = float(data['High'].iloc[-1])
latest_low = float(data['Low'].iloc[-1])
latest_volume = int(data['Volume'].iloc[-1])

col1, col2, col3, col4 = st.columns(4)

col1.metric("Current Price", f"₹{latest_close:.2f}")
col2.metric("Day High", f"₹{latest_high:.2f}")
col3.metric("Day Low", f"₹{latest_low:.2f}")
col4.metric("Volume", f"{latest_volume:,}")

st.divider()

# ---------------- CLOSING PRICE CHART ----------------
st.subheader("📉 Closing Price Trend")

st.line_chart(data['Close'])

# ---------------- MOVING AVERAGES ----------------
data['MA50'] = data['Close'].rolling(50).mean()
data['MA100'] = data['Close'].rolling(100).mean()

st.subheader("📊 Moving Average Analysis")

st.line_chart(data[['Close', 'MA50', 'MA100']])

# ---------------- CANDLESTICK CHART ----------------
st.subheader("🕯️ Candlestick Chart")

fig = go.Figure(data=[go.Candlestick(
    x=data.index,
    open=data['Open'],
    high=data['High'],
    low=data['Low'],
    close=data['Close']
)])

fig.update_layout(
    template=plot_template,
    height=600,
    xaxis_title='Date',
    yaxis_title='Price',
    title='Live Candlestick Chart'
)

st.plotly_chart(fig, use_container_width=True)

# ---------------- VOLUME CHART ----------------
st.subheader("📦 Trading Volume")

volume_fig = go.Figure()

volume_fig.add_trace(
    go.Bar(
        x=data.index,
        y=data['Volume'],
        name='Volume'
    )
)

volume_fig.update_layout(
    template='plotly_dark',
    height=400
)

st.plotly_chart(volume_fig, use_container_width=True)

# ---------------- LOAD MODEL ----------------
model = load_model("models/stock_prediction_model.h5")

# ---------------- AI PREDICTION ----------------
st.subheader("🤖 AI Next-Day Stock Prediction")

closing_price = data[['Close']]

scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(closing_price)

last_60_days = scaled_data[-60:]

X_test = []
X_test.append(last_60_days)
X_test = np.array(X_test)

prediction = model.predict(X_test)
prediction = scaler.inverse_transform(prediction)

predicted_price = prediction[0][0]

st.success(f"Predicted Next Day Price: ₹{predicted_price:.2f}")

st.metric(
    label="Predicted Next Day Price",
    value=f"₹{predicted_price:.2f}"
)
if predicted_price > latest_close:
    st.success("📈 BUY Recommendation")
else:
    st.error("📉 SELL Recommendation")


# ---------------- DOWNLOAD BUTTON ----------------
st.subheader("⬇️ Download Dataset")

csv = data.to_csv().encode('utf-8')

st.download_button(
    label="Download Stock Dataset",
    data=csv,
    file_name='stock_data.csv',
    mime='text/csv'
)

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown("### 🚀 Developed using Python, TensorFlow, Streamlit & LSTM")
