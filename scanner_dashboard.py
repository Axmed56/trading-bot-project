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

# إعداد صفحة Streamlit
st.set_page_config(page_title="Crypto Scanner", layout="wide")
st.title("📊 ماسح العملات – بنية السوق / مشاعر السوق / VWAP / Stochastic / قرار الدخول")

if not HAS_CCXT:
    st.error("مكتبة ccxt غير مثبتة. شغّل:\n\npip install ccxt\n\nثم أعد المحاولة.")
    st.stop()

# إعداد المنصة والرموز
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

exchange_class = getattr(ccxt, EXCHANGE_ID)
exchange = exchange_class({"enableRateLimit": True})

# ----------------- مؤشرات فنية -----------------

def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).fillna(0)
    loss = -1 * delta.clip(upper=0).fillna(0)
    avg_gain = gain.rolling(period, min_periods=1).mean()
    avg_loss = loss.rolling(period, min_periods=1).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    r = 100 - (100 / (1 + rs))
    return r.fillna(50)

def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
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

def vwap(df: pd.DataFrame) -> pd.Series:
    """VWAP بسيط على كامل الفترة (ينفع للمضاربة اليومية على الكريبتو)."""
    pv = df["close"] * df["volume"]
    cum_pv = pv.cumsum()
    cum_vol = df["volume"].cumsum().replace(0, np.nan)
    return (cum_pv / cum_vol).fillna(method="bfill").fillna(method="ffill")

def stochastic(df: pd.DataFrame, k_period: int = 5, d_period: int = 3) -> pd.DataFrame:
    """
    Stochastic (K%D) -> يرجع DataFrame فيه stoch_k, stoch_d
    """
    low_min = df["low"].rolling(window=k_period, min_periods=1).min()
    high_max = df["high"].rolling(window=k_period, min_periods=1).max()
    stoch_k = 100 * (df["close"] - low_min) / (high_max - low_min).replace(0, np.nan)
    stoch_k = stoch_k.fillna(50)
    stoch_d = stoch_k.rolling(window=d_period, min_periods=1).mean()
    return pd.DataFrame({"stoch_k": stoch_k, "stoch_d": stoch_d})

# ----------------- جلب البيانات -----------------

def fetch_ohlcv(symbol: str, timeframe: str, days: int) -> pd.DataFrame:
    secs = exchange.parse_timeframe(timeframe)
    candles_per_day = int(24 * 60 * 60 / secs)
    needed = min(1000, days * candles_per_day)

    now = exchange.milliseconds()
    since = now - needed * secs * 1000

    data = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=needed)
    df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df

# ----------------- نظام النقاط + باك تست -----------------

def compute_scores(df: pd.DataFrame,
                   rsi_period: int,
                   adx_period: int,
                   adx_trend_th: float,
                   k_period: int,
                   d_period: int,
                   w_structure: float,
                   w_sentiment: float,
                   w_vwap: float,
                   w_stoch: float) -> pd.DataFrame:
    df = df.copy()

    # مؤشرات أساسية
    df["rsi"] = rsi(df["close"], rsi_period)
    df["adx"] = adx(df, adx_period)
    df["ema50"] = ema(df["close"], 50)
    df["ema200"] = ema(df["close"], 200)
    df["vwap"] = vwap(df)

    stoch_df = stochastic(df, k_period=k_period, d_period=d_period)
    df["stoch_k"] = stoch_df["stoch_k"]
    df["stoch_d"] = stoch_df["stoch_d"]

    # بنية السوق: الفرق النسبي بين EMA50 و EMA200
    trend_raw = (df["ema50"] - df["ema200"]) / df["ema200"].replace(0, np.nan)
    structure_score = (np.tanh(trend_raw * 5) + 1) * 50   # 0–100
    df["structure_score"] = structure_score.clip(0, 100)

    # مشاعر السوق: مزيج من RSI + مكان السعر من EMA50
    price_vs_ema = (df["close"] - df["ema50"]) / df["ema50"].replace(0, np.nan)
    price_sent = (np.tanh(price_vs_ema * 5) + 1) * 50  # 0–100
    rsi_norm = df["rsi"]  # أصلاً من 0–100
    df["sentiment_score"] = (0.6 * rsi_norm + 0.4 * price_sent).clip(0, 100)

    # VWAP Score: كلما كان السعر قريب من VWAP يكون أفضل (فرصة إعادة تسعير أو continuation)
    vwap_dist = (df["close"] - df["vwap"]) / df["vwap"].replace(0, np.nan)  # نسبي
    # نخلي المسافة الصغيرة أفضل: 100 عند 0% مسافة وتنقص كلما بعد
    vwap_score = 100 - (vwap_dist.abs() * 1000)  # 0.1% مسافة → خصم 100 نقطة
    df["vwap_score"] = vwap_score.clip(0, 100)

    # Stoch Score: التطرف أفضل (فرصة انعكاس)، النصف محايد
    stoch_k = df["stoch_k"]
    # بعيد عن 50 أفضل (فرصة)، قريب من 50 ممل
    stoch_score = (stoch_k - 50).abs() * 2  # من 0 إلى 100
    df["stoch_score"] = stoch_score.clip(0, 100)

    # تطبيع الأوزان (تتجمع = 1)
    weights = np.array([w_structure, w_sentiment, w_vwap, w_stoch], dtype=float)
    if weights.sum() == 0:
        weights = np.array([0.25, 0.25, 0.25, 0.25])
    else:
        weights = weights / weights.sum()

    ws, wse, wv, wst = weights

    # Decision base
    base = (
        ws * df["structure_score"] +
        wse * df["sentiment_score"] +
        wv * df["vwap_score"] +
        wst * df["stoch_score"]
    )

    # Bonus/penalty من ADX
    bonus = np.where(df["adx"] >= adx_trend_th, 1.1, 0.9)
    df["decision_score"] = (base * bonus).clip(0, 100)

    return df

def run_backtest(df_scores: pd.DataFrame,
                 entry_th: float,
                 exit_th: float,
                 initial_balance: float,
                 risk_per_trade: float) -> dict:
    balance = initial_balance
    position = None
    trades = []

    for i in range(len(df_scores)):
        row = df_scores.iloc[i]
        price = float(row["close"])
        score = float(row["decision_score"])
        ts = row["timestamp"]

        # خروج
        if position is not None and score <= exit_th:
            qty = position["qty"]
            entry_price = position["entry_price"]
            pnl = (price - entry_price) * qty
            balance += pnl
            trades.append(
                {"time": ts, "type": "EXIT", "pnl": pnl, "balance": balance}
            )
            position = None
            continue

        # دخول
        if position is None and score >= entry_th:
            qty = risk_per_trade / price
            position = {"entry_price": price, "qty": qty}
            trades.append(
                {"time": ts, "type": "ENTRY", "pnl": 0.0, "balance": balance}
            )

    trades_df = pd.DataFrame(trades)

    if trades_df.empty:
        return {"final_balance": balance, "total_return": 0.0,
                "trades_count": 0, "win_rate": 0.0}

    final_balance = trades_df["balance"].iloc[-1]
    total_return = (final_balance - initial_balance) / initial_balance * 100

    exits = trades_df[trades_df["type"] == "EXIT"]
    if exits.empty:
        win_rate = 0.0
    else:
        win_rate = (exits["pnl"] > 0).mean() * 100

    return {
        "final_balance": final_balance,
        "total_return": total_return,
        "trades_count": len(trades_df),
        "win_rate": win_rate,
    }

# ----------------- واجهة الإعدادات -----------------

st.sidebar.header("⚙️ الإعدادات")

timeframe = st.sidebar.selectbox(
    "الفريم",
    ["1m", "5m", "15m", "4h"],
    index=2
)

backtest_days = st.sidebar.selectbox(
    "فترة الباك تست",
    [1, 7, 30],
    index=1,
    format_func=lambda x: "يوم" if x == 1 else ("أسبوع" if x == 7 else "شهر")
)

st.sidebar.markdown("### RSI / ADX")
rsi_period = st.sidebar.number_input("RSI Period", 5, 50, 8)
adx_period = st.sidebar.number_input("ADX Period", 5, 50, 14)
adx_trend_th = st.sidebar.slider("عتبة ADX لقوة الاتجاه", 0, 100, 20)

st.sidebar.markdown("### Stochastic")
k_period = st.sidebar.number_input("Stoch K Period", 3, 50, 5)
d_period = st.sidebar.number_input("Stoch D Period", 2, 50, 3)

st.sidebar.markdown("### أوزان القرار (0–1)")
w_structure = st.sidebar.slider("وزن بنية السوق", 0.0, 1.0, 0.35, 0.05)
w_sentiment = st.sidebar.slider("وزن مشاعر السوق", 0.0, 1.0, 0.25, 0.05)
w_vwap = st.sidebar.slider("وزن VWAP", 0.0, 1.0, 0.25, 0.05)
w_stoch = st.sidebar.slider("وزن Stoch", 0.0, 1.0, 0.15, 0.05)

st.sidebar.markdown("### حدود الدخول / الخروج")
entry_th = st.sidebar.slider("قرار الدخول ≥", 0, 100, 70)
exit_th = st.sidebar.slider("قرار الخروج ≤", 0, 100, 40)

st.sidebar.markdown("### إعدادات الباك تست")
initial_balance = st.sidebar.number_input("رصيد افتراضي للباك تست", 100.0, 100000.0, 1000.0, 100.0)
risk_per_trade = st.sidebar.number_input("حجم الصفقة في الباك تست (USDT)", 5.0, 2000.0, 50.0, 5.0)

run_button = st.sidebar.button("🚀 تشغيل المسح + الباك تست")

if not run_button:
    st.info("عدّل الإعدادات ثم اضغط على زر 🚀 لتحديث الجدول.")
    st.stop()

# ----------------- تشغيل المسح لكل عملة -----------------

rows = []

for sym in SYMBOLS:
    try:
        df = fetch_ohlcv(sym, timeframe, backtest_days)
        if len(df) < 50:
            st.warning(f"بيانات {sym} قليلة، تم تجاهلها.")
            continue

        df_scores = compute_scores(
            df,
            rsi_period=rsi_period,
            adx_period=adx_period,
            adx_trend_th=adx_trend_th,
            k_period=k_period,
            d_period=d_period,
            w_structure=w_structure,
            w_sentiment=w_sentiment,
            w_vwap=w_vwap,
            w_stoch=w_stoch
        )

        bt = run_backtest(
            df_scores,
            entry_th=entry_th,
            exit_th=exit_th,
            initial_balance=initial_balance,
            risk_per_trade=risk_per_trade
        )

        last = df_scores.iloc[-1]

        structure = float(last["structure_score"])
        sentiment = float(last["sentiment_score"])
        decision = float(last["decision_score"])
        price_now = float(last["close"])
        rsi_now = float(last["rsi"])
        adx_now = float(last["adx"])
        vwap_now = float(last["vwap"])
        vwap_dist_pct = (price_now - vwap_now) / vwap_now * 100
        stoch_k_now = float(last["stoch_k"])

        signal = "✅ دخول" if (decision >= entry_th and adx_now >= adx_trend_th) else "⏸ لا"

        rows.append({
            "العملة": sym.replace("/", ""),
            "بنية السوق %": round(structure, 1),
            "مشاعر السوق %": round(sentiment, 1),
            "قرار الدخول %": round(decision, 1),
            "السعر": round(price_now, 6),
            "VWAP": round(vwap_now, 6),
            "بعد عن VWAP %": round(vwap_dist_pct, 3),
            "Stoch K": round(stoch_k_now, 1),
            "RSI": round(rsi_now, 1),
            "ADX": round(adx_now, 1),
            "عدد الصفقات": bt["trades_count"],
            "Win Rate %": round(bt["win_rate"], 1),
            "إشارة": signal,
            f"PnL آخر {backtest_days} يوم %": round(bt["total_return"], 2),
        })

    except Exception as e:
        st.error(f"خطأ في {sym}: {e}")

if not rows:
    st.error("لا توجد بيانات كافية لعرض نتائج.")
else:
    table = pd.DataFrame(rows)
    st.subheader(f"📋 لوحة المسح – الفريم: {timeframe} – الفترة: {backtest_days} يوم")
    st.dataframe(table.set_index("العملة"), use_container_width=True)
    st.caption("كل القيم مبنية على شموع حقيقية من Binance + باك تست داخلي بسيط + VWAP + Stochastic.")
