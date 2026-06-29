import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

st.set_page_config(page_title="Stock Price Predictor", layout="wide")
st.title("📈 Stock Price Prediction App")

ticker = st.text_input("Enter Stock Ticker", "AAPL")
period = st.selectbox("Select Period", ["1mo", "3mo", "6mo", "1y", "2y"], index=3)

if st.button("Predict"):
    with st.spinner("Fetching data..."):
        df = yf.download(ticker, period=period)
        if df.empty:
            st.error("No data found for this ticker.")
        else:
            st.success(f"Data fetched for {ticker}")
            st.subheader("Historical Closing Prices")
            fig, ax = plt.subplots(figsize=(10,4))
            df['Close'].plot(ax=ax)
            st.pyplot(fig)

            st.subheader("Prediction")
            st.info("Placeholder: load your LSTM model here.")