import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import streamlit as st

# محاولة استيراد ccxt
HAS_CCXT = True
try:
    import ccxt
except Exception:
    HAS_CCXT = False
    ccxt = None

# ===================== إعداد صفحة Streamlit =====================
st.set_page_config(
    page_title="Multi-Symbol Scanner",
    layout="wide"
)

st.title("📊 Trading Scanner – Multi Symbol (Binance)")

if not HAS_CCXT:
    st.error("مكتبة ccxt غير مثبتة. من فضلك ثبّتها أولًا:\n\n`pip install ccxt`")
    st.stop()

# ===================== إعدادات عامة =====================

EXCHANGE_ID = "binance"
SYMBOLS = [
    "ADA/USDT",
    "BNB/USDT",
    "BTC/USDT",
    "DOGE/USDT",
    "ETH/USDT",
    "SOL/USDT",
    "XRP/USDT",
]

TIMEFRAME_MAP = {
    "1 دقيقة": "1m",
    "5 دقائق": "5m",
    "15 دقيقة": "15m",
    "4 ساعات": "4h",
}

BACKTEST_WINDOWS = {
    "آخر يوم": 1,
    "آخر أسبوع": 7,
    "آخر شهر": 30,
}

# إنشاء كائن المنصة
exchange_class = getattr(ccxt, EXCHANGE_ID)
exchange = exchange_class({"enableRateLimit": True})


# ===================== دوال المؤشرات =====================

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).fillna(0)
    loss = -1 * delta.clip(upper=0).fillna(0)
    avg_gain = gain.rolling(period, min_periods=1).mean()
    avg_loss = loss.rolling(period, min_periods=1).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi_series = 100 - (100 / (1 + rs))
    return rsi_series.fillna(50)


def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """حساب ADX بأسلوب مبسط"""
    high = df["high"]
    low = df["low"]
    close = df["close"]

    plus_dm = high.diff()
    minus_dm = -low.diff()

    plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0.0)
    minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0.0)

    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    tr_smooth = tr.rolling(period, min_periods=period).sum()
    plus_dm_smooth = pd.Series(plus_dm).rolling(period, min_periods=period).sum()
    minus_dm_smooth = pd.Series(minus_dm).rolling(period, min_periods=period).sum()

    plus_di = 100 * (plus_dm_smooth / tr_smooth.replace(0, np.nan))
    minus_di = 100 * (minus_dm_smooth / tr_smooth.replace(0, np.nan))

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx_val = dx.rolling(period, min_periods=period).mean()
    return adx_val.fillna(0)


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


# ===================== دوال لجلب البيانات =====================

def fetch_ohlcv(symbol: str, timeframe: str, days: int) -> pd.DataFrame:
    """
    جلب بيانات OHLCV من باينانس لعدد معين من الأيام تقريبًا.
    - limit في باينانس عادة 1000 شمعة، فنستخدم since ونلف في حلقات إذا احتجنا أكثر.
    هنا سنأخذ حتى 1000 شمعة كحد أقصى لكل رمز لتبسيط البداية.
    """
    secs_per_candle = exchange.parse_timeframe(timeframe)
    candles_per_day = int(24 * 60 * 60 / secs_per_candle)
    needed_candles = min(1000, days * candles_per_day)

    now = exchange.milliseconds()
    since = now - needed_candles * secs_per_candle * 1000

    ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=needed_candles)
    df = pd.DataFrame(
        ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"]
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


# ===================== نظام النقاط (Scoring) =====================

def compute_scores(df: pd.DataFrame,
                   rsi_period: int,
                   adx_period: int,
                   adx_trend_threshold: float,
                   w_structure: float,
                   w_sentiment: float) -> pd.DataFrame:
    """
    يحسب:
    - RSI
    - ADX
    - Market Structure Score (0–100)
    - Sentiment Score (0–100)
    - Decision Score (0–100)
    """
    df = df.copy()
    df["rsi"] = rsi(df["close"], rsi_period)
    df["adx"] = adx(df, adx_period)

    # Market structure: نعتمد على EMA50 و EMA200 + ميل الاتجاه
    df["ema50"] = ema(df["close"], 50)
    df["ema200"] = ema(df["close"], 200)

    # قوة الاتجاه = الفرق النسبي بين ema50 و ema200
    trend_raw = (df["ema50"] - df["ema200"]) / df["ema200"].replace(0, np.nan)
    # نحولها لنطاق -1 إلى 1 عن طريق tanh ثم لـ 0–100
    structure_score = (np.tanh(trend_raw * 5) + 1) * 50
    df["structure_score"] = structure_score.clip(0, 100)

    # Sentiment: مزيج من RSI + علاقة السعر بالـ EMA50
    price_vs_ema = (df["close"] - df["ema50"]) / df["ema50"].replace(0, np.nan)
    price_sent = (np.tanh(price_vs_ema * 5) + 1) * 50  # 0–100
    rsi_norm = df["rsi"]  # أصلاً من 0–100
    df["sentiment_score"] = (0.6 * rsi_norm + 0.4 * price_sent).clip(0, 100)

    # Decision Score = تركيب السوق + المشاعر + شرط ADX (لو الاتجاه قوي نزود)
    base_score = w_structure * df["structure_score"] + w_sentiment * df["sentiment_score"]

    # Bonus/penalty من ADX
    bonus_factor = np.where(df["adx"] >= adx_trend_threshold, 1.1, 0.9)
    df["decision_score"] = (base_score * bonus_factor).clip(0, 100)

    return df


# ===================== باك تست بسيط على أساس Decision Score =====================

def run_backtest(df_scores: pd.DataFrame,
                 entry_threshold: float,
                 exit_threshold: float,
                 initial_balance: float = 1000.0,
                 risk_per_trade_usd: float = 50.0) -> dict:
    """
    استراتيجية بسيطة:
    - دخول شراء إذا Decision Score >= entry_threshold ولا يوجد مركز.
    - خروج (إغلاق) إذا Decision Score <= exit_threshold.
    لا يوجد وقف خسارة/هدف محددين هنا – الهدف الآن تقييم الفكرة.
    """
    balance = initial_balance
    position = None
    trades = []

    for i in range(len(df_scores)):
        row = df_scores.iloc[i]
        price = float(row["close"])
        ts = row["timestamp"]
        score = float(row["decision_score"])

        # خروج
        if position is not None:
            if score <= exit_threshold:
                # إغلاق
                qty = position["qty"]
                entry_price = position["entry_price"]
                pnl = (price - entry_price) * qty
                balance += pnl
                trades.append({
                    "time": ts,
                    "type": "EXIT",
                    "entry_price": entry_price,
                    "exit_price": price,
                    "qty": qty,
                    "pnl": pnl,
                    "balance": balance
                })
                position = None
                continue

        # دخول
        if position is None and score >= entry_threshold:
            qty = risk_per_trade_usd / price
            position = {
                "entry_price": price,
                "qty": qty
            }
            trades.append({
                "time": ts,
                "type": "ENTRY",
                "entry_price": price,
                "exit_price": None,
                "qty": qty,
                "pnl": 0.0,
                "balance": balance
            })

    trades_df = pd.DataFrame(trades)

    if trades_df.empty:
        final_balance = balance
        total_return = 0.0
        win_rate = 0.0
    else:
        final_balance = trades_df["balance"].iloc[-1]
        total_return = (final_balance - initial_balance) / initial_balance * 100
        closed = trades_df[trades_df["type"] == "EXIT"]
        if closed.empty:
            win_rate = 0.0
        else:
            win_rate = (closed["pnl"] > 0).mean() * 100

    return {
        "final_balance": final_balance,
        "total_return": total_return,
        "trades_count": len(trades_df),
        "win_rate": win_rate,
    }


# ===================== واجهة الإعدادات (Sidebar) =====================

st.sidebar.header("⚙️ الإعدادات")

tf_label = st.sidebar.selectbox("الفريم الزمني", list(TIMEFRAME_MAP.keys()), index=1)
timeframe = TIMEFRAME_MAP[tf_label]

bt_window_label = st.sidebar.selectbox("فترة الباك تست", list(BACKTEST_WINDOWS.keys()), index=1)
bt_days = BACKTEST_WINDOWS[bt_window_label]

st.sidebar.markdown("---")

rsi_period = st.sidebar.number_input("RSI Period", 5, 50, 8)
rsi_entry = st.sidebar.slider("RSI مناسب للشراء (كعامل مشاعر)", 0, 100, 40)
rsi_exit = st.sidebar.slider("RSI مناسب للخروج (كعامل مشاعر)", 0, 100, 60)

adx_period = st.sidebar.number_input("ADX Period", 5, 50, 14)
adx_trend_threshold = st.sidebar.slider("ADX Trend Threshold", 0, 100, 20)

st.sidebar.markdown("---")

w_structure = st.sidebar.slider("وزن بنية السوق", 0.0, 1.0, 0.5, 0.05)
w_sentiment = 1.0 - w_structure
st.sidebar.write(f"وزن المشاعر = {w_sentiment:.2f}")

entry_threshold = st.sidebar.slider("Decision دخول (٪)", 0, 100, 70)
exit_threshold = st.sidebar.slider("Decision خروج (٪)", 0, 100, 40)

initial_balance = st.sidebar.number_input("رصيد افتراضي للباك تست (USDT)", 100.0, 100000.0, 1000.0, 100.0)
risk_per_trade = st.sidebar.number_input("حجم الصفقة في الباك تست (USDT)", 5.0, 1000.0, 50.0, 5.0)

st.sidebar.markdown("---")
run_button = st.sidebar.button("🚀 تحديث وبدء المسح + باك تست")


# ===================== تشغيل المسح =====================

if not run_button:
    st.info("اضبط الإعدادات في الشريط الجانبي ثم اضغط على زر **🚀 تحديث وبدء المسح + باك تست**.")
    st.stop()

rows = []

for sym in SYMBOLS:
    try:
        df = fetch_ohlcv(sym, timeframe, bt_days)
        if len(df) < 50:
            st.warning(f"بيانات {sym} قليلة للفترة المطلوبة، تم تجاهلها.")
            continue

        df_scores = compute_scores(
            df,
            rsi_period=rsi_period,
            adx_period=adx_period,
            adx_trend_threshold=adx_trend_threshold,
            w_structure=w_structure,
            w_sentiment=w_sentiment
        )

        bt_result = run_backtest(
            df_scores,
            entry_threshold=entry_threshold,
            exit_threshold=exit_threshold,
            initial_balance=initial_balance,
            risk_per_trade_usd=risk_per_trade
        )

        last = df_scores.iloc[-1]

        decision = float(last["decision_score"])
        structure = float(last["structure_score"])
        sentiment = float(last["sentiment_score"])
        rsi_now = float(last["rsi"])
        adx_now = float(last["adx"])
        price_now = float(last["close"])

        signal = "✅ دخول محتمل" if decision >= entry_threshold and adx_now >= adx_trend_threshold else "⏸ لا"

        rows.append({
            "العملة": sym.replace("/", ""),
            "بنية السوق %": round(structure, 1),
            "مشاعر السوق %": round(sentiment, 1),
            "قرار الدخول %": round(decision, 1),
            "السعر": round(price_now, 6),
            "RSI": round(rsi_now, 1),
            "ADX": round(adx_now, 1),
            f"PnL {bt_window_label} %": round(bt_result["total_return"], 2),
            "عدد الصفقات": bt_result["trades_count"],
            "Win Rate %": round(bt_result["win_rate"], 1),
            "إشارة": signal,
        })

    except Exception as e:
        st.error(f"خطأ في {sym}: {e}")

if not rows:
    st.error("لم يتم توليد أي صفوف. راجع الإعدادات أو الفريم/الفترة.")
    st.stop()

table_df = pd.DataFrame(rows)

st.subheader(f"📋 لوحة مسح العملات – الفريم: {tf_label} – الفترة: {bt_window_label}")
st.dataframe(
    table_df.set_index("العملة"),
    use_container_width=True
)

st.caption("💡 كل القيم محسوبة من شموع حقيقية عبر Binance (ccxt). يمكنك تغيير الإعدادات ثم إعادة التشغيل للحصول على نتائج مختلفة.")
