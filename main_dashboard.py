import streamlit as st
import json
import pandas as pd
import os

st.set_page_config(page_title="ZAYA – Live AI Dashboard", layout="wide")

st.title("🐋 ZAYA – Live Trading Dashboard")

# --- قراءة ملف Bybit ---
bybit_file = "data/bybit_orderbook.json"
binance_file = "data/binance_depth.json"

col1, col2 = st.columns(2)

with col1:
    st.header("Bybit – Orderbook")
    if os.path.exists(bybit_file):
        with open(bybit_file, "r") as f:
            data = json.load(f)

        bids = pd.DataFrame(data["bids"], columns=["price", "size"])
        asks = pd.DataFrame(data["asks"], columns=["price", "size"])

        st.subheader("Bids")
        st.dataframe(bids)

        st.subheader("Asks")
        st.dataframe(asks)
    else:
        st.info("في انتظار بيانات Bybit ...")


with col2:
    st.header("Binance – Depth")
    if os.path.exists(binance_file):
        with open(binance_file, "r") as f:
            data = json.load(f)

        bids = pd.DataFrame(data["bids"], columns=["price", "size"])
        asks = pd.DataFrame(data["asks"], columns=["price", "size"])

        st.subheader("Bids")
        st.dataframe(bids)

        st.subheader("Asks")
        st.dataframe(asks)
    else:
        st.info("في انتظار بيانات Binance ...")

