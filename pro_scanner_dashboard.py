# app.py
import streamlit as st
import pandas as pd

import config
from core.context_builder import build_context
from core.alerts import build_pre_signal

st.set_page_config(
    page_title="Shadow Scalper AI",
    layout="wide"
)

st.sidebar.title("⚙️ إعدادات السكالبينج")

symbol = st.sidebar.selectbox("اختر العملة", config.SYMBOLS, index=0)

refresh_sec = st.sidebar.slider("تحديث كل (ثانية)", 5, 60, 15)

st.title("⚡ Crypto Scalping AI Dashboard – Orderbook / Whales / Bots")

# تحديث تلقائي
st_autorefresh = st.experimental_rerun  # placeholder for older versions
st.write(f"سيتم التحديث يدويًا بالضغط على زر إعادة التشغيل من واجهة Streamlit.")

if st.button("🔄 تحديث الآن"):
    st.experimental_rerun()

# نبني السياق
try:
    ctx = build_context(symbol)
except Exception as e:
    st.error(f"خطأ في جلب البيانات: {e}")
    st.stop()

alert = build_pre_signal(ctx)

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("العملة")
    st.metric("Symbol", ctx["symbol"])

with col2:
    st.subheader("Orderbook Imbalance")
    st.metric("%", f"{ctx['imbalance']:.2f}")

with col3:
    st.subheader("Pre-Signal Level")
    color_map = {
        "none": "⚪ لا يوجد",
        "low": "🟡 ضعيف",
        "medium": "🟠 متوسط",
        "high": "🔴 قوي"
    }
    st.metric("Level", color_map.get(alert["level"], alert["level"]))

st.markdown("---")

### قسم دفتر الأوامر
ob_col1, ob_col2 = st.columns(2)

with ob_col1:
    st.subheader("📘 Bid Side (Bybit)")
    bids_df = pd.DataFrame(ctx["orderbook"]["bids"], columns=["price", "size"])
    st.dataframe(bids_df)

with ob_col2:
    st.subheader("📕 Ask Side (Bybit)")
    asks_df = pd.DataFrame(ctx["orderbook"]["asks"], columns=["price", "size"])
    st.dataframe(asks_df)

st.markdown("---")

### قسم الحيطان والبوتات
wb_col1, wb_col2, wb_col3 = st.columns(3)

with wb_col1:
    st.subheader("🧱 Walls")
    walls = ctx["walls"]
    st.write("Bid Walls:")
    st.write(walls["bid_walls"][:5])
    st.write("Ask Walls:")
    st.write(walls["ask_walls"][:5])

with wb_col2:
    st.subheader("🤖 Bot Fingerprint")
    bot_grid = ctx["bot_grid"]
    st.write(f"Grid Bids: {bot_grid['grid_bids']}")
    st.write(f"Grid Asks: {bot_grid['grid_asks']}")

with wb_col3:
    st.subheader("🐋 Whale Trades (Binance)")
    whales = ctx["whales"]
    if whales:
        whales_df = pd.DataFrame(whales)
        st.dataframe(whales_df)
    else:
        st.write("لا توجد صفقات حيتان في آخر الداتا.")

st.markdown("---")

### قسم آخر الصفقات من Binance
st.subheader("📊 Recent Trades (Binance)")
trades_df = pd.DataFrame(ctx["trades"])
st.dataframe(trades_df)
st.caption("is_buyer_maker = True يعني الصفقة بيع من صانع السوق (ضغط بيع).")
