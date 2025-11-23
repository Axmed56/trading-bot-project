import json
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# محاولة استيراد ccxt (Binance + Bybit)
try:
    import ccxt
    HAS_CCXT = True
except Exception:
    ccxt = None
    HAS_CCXT = False

# ============= إعداد صفحة Streamlit =============

st.set_page_config(
    page_title="ZAYA – AI Trading Terminal",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ============= تهيئة Session State =============

if "prev_orderbook" not in st.session_state:
    st.session_state["prev_orderbook"] = None

if "top15" not in st.session_state:
    st.session_state["top15"] = None

if "symbol_override" not in st.session_state:
    st.session_state["symbol_override"] = None

if "timeframe_override" not in st.session_state:
    st.session_state["timeframe_override"] = None

# ============= CSS – ألوان وخلفية =============

st.markdown(
    """
<style>
    .main {
        background-color: #050816;
        color: #FFFFFF;
    }
    .stApp {
        background: #050816;
        color: #FFFFFF;
    }
    .block-container {
        padding-top: 1rem;
    }
    .metric-container {
        background: rgba(15, 23, 42, 0.9);
        border-radius: 12px;
        padding: 16px;
        border-left: 4px solid #6366F1;
        box-shadow: 0 4px 10px rgba(0,0,0,0.4);
    }
    .decision-card {
        background: rgba(15, 23, 42, 0.95);
        border-radius: 16px;
        padding: 18px;
        border: 1px solid rgba(99, 102, 241, 0.7);
        box-shadow: 0 8px 20px rgba(0,0,0,0.5);
    }
    .header-gradient {
        background: linear-gradient(90deg, #38bdf8 0%, #a855f7 50%, #f97316 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 1.6rem;
    }
    .signal-long {
        background: linear-gradient(135deg, #059669 0%, #10B981 100%);
        color: white;
        padding: 10px;
        border-radius: 8px;
        text-align: center;
    }
    .signal-short {
        background: linear-gradient(135deg, #DC2626 0%, #EF4444 100%);
        color: white;
        padding: 10px;
        border-radius: 8px;
        text-align: center;
    }
    .signal-neutral {
        background: linear-gradient(135deg, #6B7280 0%, #9CA3AF 100%);
        color: white;
        padding: 10px;
        border-radius: 8px;
        text-align: center;
    }
</style>
""",
    unsafe_allow_html=True,
)

# ============= قوائم عامة =============

SYMBOLS = [
    "BTC/USDT",
    "ETH/USDT",
    "BNB/USDT",
    "SOL/USDT",
    "LTC/USDT",
    "XRP/USDT",
    "ADA/USDT",
    "LINK/USDT",
    "AVAX/USDT",
    "DOGE/USDT",
    "TON/USDT",
    "TRX/USDT",
]

TIMEFRAMES = ["1m", "3m", "5m", "15m", "1h"]

STRATEGIES = {
    "سكالبينج: VWAP / EMA / RSI / Stoch": "core_scalp",
    "فيبوناتشي + VWAP + ADX": "fibo_swing",
    "دفتر أوامر + اتجاه": "liquidity_trend",
}

STRATEGY_MAP_FILE = "strategy_map.json"

# ============= دوال مساعدة =============


def tf_to_rule(tf: str) -> str:
    if tf.endswith("m"):
        return f"{int(tf[:-1])}min"
    if tf.endswith("h"):
        return f"{int(tf[:-1])}h"
    if tf.endswith("d"):
        return f"{int(tf[:-1])}D"
    return "1min"


def load_strategy_map():
    try:
        with open(STRATEGY_MAP_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_strategy_map(mapping: dict):
    try:
        with open(STRATEGY_MAP_FILE, "w", encoding="utf-8") as f:
            json.dump(mapping, f, ensure_ascii=False, indent=2)
    except Exception as e:
        st.warning(f"تعذر حفظ خريطة الإستراتيجيات: {e}")


def fetch_ohlcv(symbol: str, timeframe: str, limit: int = 500) -> pd.DataFrame:
    """جلب الشموع من Binance أو توليد بيانات صناعية."""
    if HAS_CCXT:
        try:
            ex = ccxt.binance({"enableRateLimit": True})
            data = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
            df = pd.DataFrame(
                data,
                columns=["timestamp", "open", "high", "low", "close", "volume"],
            )
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            return df
        except Exception as e:
            st.warning(f"تعذر جلب OHLCV من Binance: {e}")

    # بيانات صناعية لاختبار الواجهة
    base_price = 30000.0
    timestamps = pd.date_range(
        end=pd.Timestamp.utcnow(),
        periods=limit,
        freq=tf_to_rule(timeframe),
    )
    prices = base_price + np.cumsum(np.random.normal(0, 50, size=limit))
    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": prices + np.random.normal(0, 10, size=limit),
            "high": prices + np.abs(np.random.normal(0, 15, size=limit)),
            "low": prices - np.abs(np.random.normal(0, 15, size=limit)),
            "close": prices + np.random.normal(0, 10, size=limit),
            "volume": np.random.randint(10, 1000, size=limit),
        }
    )
    return df


def fetch_bybit_orderbook(symbol: str, depth: int = 10) -> pd.DataFrame:
    """جلب دفتر الأوامر من Bybit أو توليد دفتر صناعي."""
    if HAS_CCXT:
        try:
            ex = ccxt.bybit({"enableRateLimit": True})
            ex.load_markets()
            ob = ex.fetch_order_book(symbol, limit=depth)
            bids = ob.get("bids", [])
            asks = ob.get("asks", [])
            rows = []
            for price, size in bids[:depth]:
                rows.append({"side": "bid", "price": price, "size": size})
            for price, size in asks[:depth]:
                rows.append({"side": "ask", "price": price, "size": size})
            df = pd.DataFrame(rows)
            return (
                df.sort_values(["side", "price"], ascending=[False, False])
                .reset_index(drop=True)
            )
        except Exception as e:
            st.warning(f"تعذر جلب دفتر الأوامر من Bybit: {e}")

    # دفتر صناعي
    mid = 30000
    prices_bid = [mid - i * 5 for i in range(1, depth + 1)]
    prices_ask = [mid + i * 5 for i in range(1, depth + 1)]
    sizes_bid = np.random.randint(1, 20, size=depth)
    sizes_ask = np.random.randint(1, 20, size=depth)
    rows = []
    for p, s in zip(prices_bid, sizes_bid):
        rows.append({"side": "bid", "price": p, "size": s})
    for p, s in zip(prices_ask, sizes_ask):
        rows.append({"side": "ask", "price": p, "size": s})
    df = pd.DataFrame(rows)
    return (
        df.sort_values(["side", "price"], ascending=[False, False])
        .reset_index(drop=True)
    )


# ============= مؤشرات فنية =============


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def rsi(series: pd.Series, period: int = 9) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).fillna(0)
    loss = -delta.clip(upper=0).fillna(0)
    avg_gain = gain.rolling(window=period, min_periods=1).mean()
    avg_loss = loss.rolling(window=period, min_periods=1).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi_val = 100 - (100 / (1 + rs))
    return rsi_val.fillna(50)


def stoch_kd(df: pd.DataFrame, k_period: int = 5, d_period: int = 3):
    low_min = df["low"].rolling(window=k_period, min_periods=1).min()
    high_max = df["high"].rolling(window=k_period, min_periods=1).max()
    k = (df["close"] - low_min) / (high_max - low_min + 1e-9) * 100
    d = k.rolling(window=d_period, min_periods=1).mean()
    return k, d


def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    tr = true_range(df)
    return tr.rolling(window=period, min_periods=1).mean()


def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["high"]
    low = df["low"]

    plus_dm = high.diff()
    minus_dm = low.diff().abs()

    plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0.0)
    minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0.0)

    tr = true_range(df)
    atr_val = tr.rolling(window=period, min_periods=1).mean()

    plus_di = 100 * (
        pd.Series(plus_dm).rolling(window=period, min_periods=1).sum()
        / (atr_val + 1e-9)
    )
    minus_di = 100 * (
        pd.Series(minus_dm).rolling(window=period, min_periods=1).sum()
        / (atr_val + 1e-9)
    )

    dx = (abs(plus_di - minus_di) / (plus_di + minus_di + 1e-9)) * 100
    adx_val = dx.rolling(window=period, min_periods=1).mean()
    return adx_val.fillna(0)


def vwap(df: pd.DataFrame) -> pd.Series:
    pv = (df["close"] * df["volume"]).cumsum()
    vol = df["volume"].cumsum().replace(0, np.nan)
    return (pv / vol).ffill()


def detect_candles(df: pd.DataFrame) -> pd.Series:
    """نمط بسيط: Doji / Hammer / Shooting Star / Engulfing."""
    body = (df["close"] - df["open"]).abs()
    range_ = df["high"] - df["low"]
    upper_wick = df["high"] - df[["open", "close"]].max(axis=1)
    lower_wick = df[["open", "close"]].min(axis=1) - df["low"]

    pattern = pd.Series("None", index=df.index)

    # Doji
    pattern = np.where(body < (range_ * 0.1), "Doji", pattern)

    # Hammer
    hammer = (lower_wick > body * 2) & (upper_wick < body)
    pattern = np.where(hammer, "Hammer", pattern)

    # Shooting Star
    star = (upper_wick > body * 2) & (lower_wick < body)
    pattern = np.where(star, "ShootingStar", pattern)

    # Engulfing
    prev_open = df["open"].shift(1)
    prev_close = df["close"].shift(1)
    bull_engulf = (
        (df["close"] > df["open"])
        & (prev_close < prev_open)
        & (df["close"] >= prev_open)
        & (df["open"] <= prev_close)
    )
    bear_engulf = (
        (df["close"] < df["open"])
        & (prev_close > prev_open)
        & (df["close"] <= prev_open)
        & (df["open"] >= prev_close)
    )

    pattern = np.where(bull_engulf, "BullEngulf", pattern)
    pattern = np.where(bear_engulf, "BearEngulf", pattern)

    return pd.Series(pattern, index=df.index)


# ============= فيبوناتشي =============


def fib_swing_levels(df: pd.DataFrame, lookback: int = 80) -> pd.Series:
    """حساب آخر سوينج high/low ومستويات فيبوناتشي الرئيسية."""
    if len(df) < lookback:
        lookback = len(df)
    window = df.iloc[-lookback:]
    swing_high = window["high"].max()
    swing_low = window["low"].min()
    diff = swing_high - swing_low

    levels = {
        "fib_0": swing_low,
        "fib_23": swing_high - 0.236 * diff,
        "fib_38": swing_high - 0.382 * diff,
        "fib_50": swing_high - 0.5 * diff,
        "fib_61": swing_high - 0.618 * diff,
        "fib_78": swing_high - 0.786 * diff,
        "fib_100": swing_high,
    }
    return pd.Series(levels)


def tag_fib_zone(df: pd.DataFrame, fib: pd.Series) -> pd.Series:
    """تحديد مناطق فيبوناتشي (منطقة شراء / بيع)."""
    close = df["close"]
    buy_zone = close.between(fib["fib_50"], fib["fib_61"])
    sell_zone = close.between(fib["fib_23"], fib["fib_38"])
    zone = pd.Series("None", index=df.index)
    zone = np.where(buy_zone, "FibBuy", zone)
    zone = np.where(sell_zone, "FibSell", zone)
    return pd.Series(zone, index=df.index)


# ============= دفتر أوامر / حيتان / بوتات =============


def analyze_orderbook(df_ob: pd.DataFrame) -> dict:
    bids = df_ob[df_ob["side"] == "bid"]
    asks = df_ob[df_ob["side"] == "ask"]

    bid_vol = float(bids["size"].sum()) if not bids.empty else 0.0
    ask_vol = float(asks["size"].sum()) if not asks.empty else 0.0

    total = bid_vol + ask_vol
    if total == 0:
        imbalance = 0.0
    else:
        imbalance = (bid_vol - ask_vol) / total * 100

    if imbalance > 15:
        regime = "Bullish Liquidity"
    elif imbalance < -15:
        regime = "Bearish Liquidity"
    else:
        regime = "Balanced / Neutral"

    big_levels = (
        df_ob[df_ob["size"] >= df_ob["size"].quantile(0.9)]
        if not df_ob.empty
        else pd.DataFrame()
    )

    return {
        "bid_vol": bid_vol,
        "ask_vol": ask_vol,
        "imbalance_pct": imbalance,
        "regime": regime,
        "whale_levels": big_levels,
    }


def analyze_orderbook_dynamics(current: pd.DataFrame, prev: pd.DataFrame):
    """
    مقارنة دفتر الأوامر الحالي بالقديم:
    - مستويات قل حجمها أو اختفت = إلغاءات محتملة
    - مستويات زاد حجمها أو ظهرت = أوامر جديدة
    """
    if prev is None or prev.empty or current.empty:
        return {
            "cancel_levels": 0,
            "cancel_volume": 0.0,
            "new_levels": 0,
            "new_volume": 0.0,
        }

    merged = current.merge(
        prev,
        on=["side", "price"],
        how="outer",
        suffixes=("_cur", "_prev"),
    ).fillna(0.0)

    cancels = merged[merged["size_prev"] > merged["size_cur"]]
    news = merged[merged["size_cur"] > merged["size_prev"]]

    return {
        "cancel_levels": int(len(cancels)),
        "cancel_volume": float((cancels["size_prev"] - cancels["size_cur"]).sum()),
        "new_levels": int(len(news)),
        "new_volume": float((news["size_cur"] - news["size_prev"]).sum()),
    }


def near_signal_alert(last_row: pd.Series) -> str or None:
    rsi_val = float(last_row["rsi9"])
    adx_val = float(last_row["adx"])
    vwap_dev = float(last_row["vwap_dev_pct"])

    alerts = []

    if 25 <= rsi_val <= 30 or 65 <= rsi_val <= 70:
        alerts.append("RSI يقترب من منطقة قرار سريعة.")

    if adx_val >= 25:
        alerts.append("قوة الاتجاه (ADX) مرتفعة – الحركة القادمة قد تكون قوية.")

    if abs(vwap_dev) <= 0.5:
        alerts.append("السعر قريب جدًا من VWAP – نقطة توازن حرجة.")

    if not alerts:
        return None
    return " | ".join(alerts)


# ============= إستراتيجية السكالبينج متعددة الإشارات =============


def generate_multi_signals(
    df: pd.DataFrame,
    rsi_buy_zone=(28, 45),
    rsi_sell_zone=(55, 72),
    stoch_over_sold=20,
    stoch_over_bought=80,
    adx_min=18,
    vwap_min_pct=0.5,
    vwap_max_pct=1.2,
) -> pd.DataFrame:
    df = df.copy()

    df["ema50"] = ema(df["close"], 50)
    df["ema200"] = ema(df["close"], 200)
    df["trend"] = np.where(df["ema50"] > df["ema200"], 1, -1)

    df["rsi9"] = rsi(df["close"], 9)
    df["stoch_k"], df["stoch_d"] = stoch_kd(df, 5, 3)
    df["adx"] = adx(df, 14)
    df["atr14"] = atr(df, 14)
    df["vwap"] = vwap(df)
    df["vwap_dev_pct"] = (df["close"] - df["vwap"]) / df["vwap"] * 100
    df["candle_pattern"] = detect_candles(df)

    fib = fib_swing_levels(df, lookback=min(120, len(df)))
    df["fib_zone"] = tag_fib_zone(df, fib)

    df["ret"] = df["close"].pct_change().fillna(0)
    df["cvd"] = (df["ret"] * df["volume"]).cumsum()

    df["signal"] = 0

    buy_core = (
        (df["trend"] == 1)
        & (df["rsi9"].between(rsi_buy_zone[0], rsi_buy_zone[1]))
        & (df["stoch_k"] < stoch_over_sold)
        & (df["adx"] >= adx_min)
        & (df["vwap_dev_pct"].abs().between(vwap_min_pct, vwap_max_pct))
    )

    sell_core = (
        (df["trend"] == -1)
        & (df["rsi9"].between(rsi_sell_zone[0], rsi_sell_zone[1]))
        & (df["stoch_k"] > stoch_over_bought)
        & (df["adx"] >= adx_min)
        & (df["vwap_dev_pct"].abs().between(vwap_min_pct, vwap_max_pct))
    )

    buy_fib_boost = buy_core & (df["fib_zone"] == "FibBuy")
    sell_fib_boost = sell_core & (df["fib_zone"] == "FibSell")

    df.loc[buy_core, "signal"] = 1
    df.loc[sell_core, "signal"] = -1

    df["signal_fib_boost"] = 0
    df.loc[buy_fib_boost, "signal_fib_boost"] = 1
    df.loc[sell_fib_boost, "signal_fib_boost"] = -1

    return df


# ============= AI Trading Decision =============


def compute_ai_decision(last_row: pd.Series, ob_info: dict) -> dict:
    trend = 1 if last_row["trend"] == 1 else -1
    rsi_val = float(last_row["rsi9"])
    stoch = float(last_row["stoch_k"])
    adx_val = float(last_row["adx"])
    vwap_dev = float(last_row["vwap_dev_pct"])
    fib_zone = last_row.get("fib_zone", "None")
    candle = last_row.get("candle_pattern", "None")

    # Trend Score
    trend_score = 60 if trend == 1 else 40

    # RSI
    if 28 <= rsi_val <= 45:
        rsi_score = 70
    elif 55 <= rsi_val <= 72:
        rsi_score = 30
    else:
        rsi_score = 50

    # Stoch
    if stoch < 20:
        stoch_score = 70
    elif stoch > 80:
        stoch_score = 30
    else:
        stoch_score = 50

    # ADX
    if adx_val >= 25:
        adx_score = 70
    elif adx_val < 15:
        adx_score = 40
    else:
        adx_score = 55

    # VWAP
    dev_abs = abs(vwap_dev)
    if 0.5 <= dev_abs <= 1.2:
        vwap_score = 70
    elif dev_abs > 3:
        vwap_score = 35
    else:
        vwap_score = 50

    # Fibo
    if fib_zone == "FibBuy":
        fib_score = 70
    elif fib_zone == "FibSell":
        fib_score = 30
    else:
        fib_score = 50

    # Candle
    strong_bull = ["Hammer", "BullEngulf"]
    strong_bear = ["ShootingStar", "BearEngulf"]
    if candle in strong_bull:
        candle_score = 70
    elif candle in strong_bear:
        candle_score = 30
    else:
        candle_score = 50

    regime = ob_info.get("regime", "Balanced / Neutral")
    if regime.startswith("Bullish"):
        ob_score = 65
    elif regime.startswith("Bearish"):
        ob_score = 35
    else:
        ob_score = 50

    components = [
        (trend_score, 1.0),
        (rsi_score, 1.0),
        (stoch_score, 0.8),
        (adx_score, 0.8),
        (vwap_score, 1.0),
        (fib_score, 0.7),
        (candle_score, 0.7),
        (ob_score, 0.8),
    ]
    num = sum(s * w for s, w in components)
    den = sum(w for _, w in components)
    score = num / den if den else 50

    if score >= 70 and trend == 1:
        label = "Strong Long"
    elif score >= 55 and trend == 1:
        label = "Long Bias"
    elif score <= 30 and trend == -1:
        label = "Strong Short"
    elif score <= 45 and trend == -1:
        label = "Short Bias"
    else:
        label = "No Clear Trade"

    return {
        "score": round(score, 1),
        "label": label,
        "trend": "Bullish" if trend == 1 else "Bearish",
    }


# ============= Backtest بسيط =============


def run_backtest(
    df: pd.DataFrame,
    sl_pct: float,
    tp_factor: float,
    max_trades: int,
    max_loss_pct: float,
    initial_balance: float = 1000.0,
) -> pd.DataFrame:
    balance = initial_balance
    equity_peak = initial_balance
    trades = []
    active_trades = 0

    for i in range(1, len(df)):
        if active_trades >= max_trades:
            break

        row = df.iloc[i]
        prev = df.iloc[i - 1]
        signal = int(prev["signal"])
        price = float(row["close"])

        if signal == 0:
            continue

        side = "buy" if signal == 1 else "sell"
        entry_price = float(prev["close"])

        # SL/TP نسبية لاستخدامها في التقارير فقط (هنا نغلق على الشمعة التالية)
        if side == "buy":
            pnl_pct = (price - entry_price) / entry_price * 100
        else:
            pnl_pct = (entry_price - price) / entry_price * 100

        balance *= (1 + pnl_pct / 100)
        equity_peak = max(equity_peak, balance)

        trades.append(
            {
                "time": row["timestamp"],
                "side": side,
                "entry_price": entry_price,
                "exit_price": price,
                "pnl_pct": pnl_pct,
                "balance": balance,
            }
        )
        active_trades += 1

        dd_total_pct = (equity_peak - balance) / equity_peak * 100
        if dd_total_pct >= max_loss_pct:
            break

    return pd.DataFrame(trades)


# ============= رادار أفضل 15 عملة =============


def score_symbol_for_scalping(symbol: str, timeframe: str = "3m"):
    try:
        df = fetch_ohlcv(symbol, timeframe, limit=400)
        df = df.sort_values("timestamp").reset_index(drop=True)
        sig = generate_multi_signals(df)
        last = sig.iloc[-1]
        fake_ob = {"regime": "Balanced / Neutral"}
        ai = compute_ai_decision(last, fake_ob)

        avg_vol = df["volume"].tail(100).mean()
        volatility = df["close"].pct_change().rolling(50).std().iloc[-1]

        vol_boost = min(float(volatility * 1000), 20.0)
        final_score = ai["score"] + vol_boost

        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "ai_score": round(ai["score"], 1),
            "label": ai["label"],
            "avg_volume": round(float(avg_vol), 2),
            "volatility": round(float(volatility * 100), 3),
            "final_score": round(final_score, 2),
        }
    except Exception:
        return None


# ============= الهيدر =============

col_logo, col_title = st.columns([1, 5])
with col_logo:
    st.markdown("<h1 style='text-align:center;'>⚡🐇</h1>", unsafe_allow_html=True)
with col_title:
    st.markdown(
        "<h1 class='header-gradient'>ZAYA – AI Trading Terminal</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='color:#94A3B8;'>لوحة تحكم للتداول الذكي – تحليل لحظي، حيتان، وباك تست سريع.</p>",
        unsafe_allow_html=True,
    )

st.markdown("---")

# ============= Sidebar – إعدادات =============

with st.sidebar:
    st.markdown(
        "<h2 style='color:#6366F1;'>⚙️ إعدادات التداول</h2>",
        unsafe_allow_html=True,
    )

    strategy_map = load_strategy_map()

    default_symbol = (
        st.session_state["symbol_override"]
        if st.session_state["symbol_override"] in SYMBOLS
        else SYMBOLS[0]
    )
    default_tf = (
        st.session_state["timeframe_override"]
        if st.session_state["timeframe_override"] in TIMEFRAMES
        else "5m"
    )

    symbol = st.selectbox("العملة", SYMBOLS, index=SYMBOLS.index(default_symbol))
    timeframe = st.selectbox(
        "الفريم الزمني", TIMEFRAMES, index=TIMEFRAMES.index(default_tf)
    )
    n_candles = st.slider("عدد الشموع", 300, 1500, 600, step=100)

    default_strategy_key = strategy_map.get(symbol, list(STRATEGIES.keys())[0])
    strategy_name = st.selectbox(
        "الإستراتيجية الافتراضية",
        list(STRATEGIES.keys()),
        index=list(STRATEGIES.keys()).index(default_strategy_key),
    )

    if st.button("💾 حفظ الإستراتيجية لهذا الزوج"):
        strategy_map[symbol] = strategy_name
        save_strategy_map(strategy_map)
        st.success("تم حفظ الإستراتيجية الافتراضية لهذا الزوج.")

    st.markdown("---")
    st.markdown(
        "<h3 style='color:#8B5CF6;'>🧪 إعدادات الباك تست</h3>",
        unsafe_allow_html=True,
    )

    initial_balance = st.number_input(
        "الرصيد الابتدائي (USDT)", 100.0, 1_000_000.0, 1000.0, step=100.0
    )
    max_trades = st.slider("أقصى عدد صفقات في المحاكاة", 5, 200, 30, step=5)
    max_loss_pct = st.slider("أقصى سحب من الرصيد %", 1.0, 80.0, 20.0, step=1.0)
    sl_pct = st.number_input("Stop Loss % (للتقارير فقط)", 0.05, 10.0, 0.3, step=0.05)
    tp_factor = st.number_input("TP Factor (كم ضعف SL)", 1.0, 10.0, 2.0, step=0.1)

    st.markdown("---")
    st.markdown(
        "<h3 style='color:#EC4899;'>📊 إعدادات المؤشرات</h3>",
        unsafe_allow_html=True,
    )

    rsi_buy_min = st.number_input("RSI شراء – أقل قيمة", 0.0, 100.0, 28.0)
    rsi_buy_max = st.number_input("RSI شراء – أعلى قيمة", 0.0, 100.0, 45.0)
    rsi_sell_min = st.number_input("RSI بيع – أقل قيمة", 0.0, 100.0, 55.0)
    rsi_sell_max = st.number_input("RSI بيع – أعلى قيمة", 0.0, 100.0, 72.0)
    stoch_over_sold = st.number_input("Stoch منطقة تشبع بيع", 0.0, 100.0, 20.0)
    stoch_over_bought = st.number_input("Stoch منطقة تشبع شراء", 0.0, 100.0, 80.0)
    adx_min = st.number_input("أقل قيمة ADX للدخول", 0.0, 100.0, 18.0)

    ob_depth = st.slider("عمق دفتر الأوامر (Bybit)", 5, 50, 20, step=5)

    st.markdown("---")
    st.markdown(
        "<h3 style='color:#22c55e;'>🔥 رادار أفضل 15 عملة</h3>",
        unsafe_allow_html=True,
    )
    scan_btn = st.button("🔍 مسح السوق الآن")

    st.markdown("---")
    run_backtest_btn = st.button("🚀 تشغيل الباك تست على الزوج الحالي")

# ============= تحميل البيانات الأساسية =============

with st.spinner("🔄 جاري تحميل البيانات وتحليلها..."):
    df_ohlcv = fetch_ohlcv(symbol, timeframe, limit=n_candles)
    df_ohlcv = df_ohlcv.sort_values("timestamp").reset_index(drop=True)

    df_sig = generate_multi_signals(
        df_ohlcv,
        rsi_buy_zone=(rsi_buy_min, rsi_buy_max),
        rsi_sell_zone=(rsi_sell_min, rsi_sell_max),
        stoch_over_sold=stoch_over_sold,
        stoch_over_bought=stoch_over_bought,
        adx_min=adx_min,
        vwap_min_pct=0.5,
        vwap_max_pct=1.2,
    )

    df_ob = fetch_bybit_orderbook(symbol, depth=ob_depth)
    if not df_ob.empty:
        ob_info = analyze_orderbook(df_ob)
        ob_dyn = analyze_orderbook_dynamics(df_ob, st.session_state["prev_orderbook"])
        st.session_state["prev_orderbook"] = df_ob.copy()
    else:
        ob_info = {
            "bid_vol": 0,
            "ask_vol": 0,
            "imbalance_pct": 0,
            "regime": "N/A",
            "whale_levels": pd.DataFrame(),
        }
        ob_dyn = {
            "cancel_levels": 0,
            "cancel_volume": 0.0,
            "new_levels": 0,
            "new_volume": 0.0,
        }

    last = df_sig.iloc[-1]
    ai_decision = compute_ai_decision(last, ob_info)

# ============= الصف الأول – نظرة عامة + AI + مخاطر =============

st.markdown(
    "<h2 style='color:#F59E0B;'>📊 نظرة عامة على سوق العملة المختارة</h2>",
    unsafe_allow_html=True,
)

col_a1, col_a2, col_a3 = st.columns([1.3, 1.0, 1.0])

# بطاقة نظرة عامة
with col_a1:
    st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
    st.markdown(f"### 🪙 {symbol} – {timeframe}")

    price = float(last["close"])
    vwap_now = float(last["vwap"])
    vwap_dev_now = float(last["vwap_dev_pct"])
    trend_now = "صاعد" if last["trend"] == 1 else "هابط"

    c1, c2, c3 = st.columns(3)
    c1.metric("السعر الحالي", f"{price:,.4f}")
    c2.metric("قيمة VWAP", f"{vwap_now:,.4f}", f"{vwap_dev_now:+.2f}%")
    c3.metric("اتجاه السوق", trend_now)

    c4, c5, c6 = st.columns(3)
    c4.metric("RSI(9)", f"{float(last['rsi9']):.1f}")
    c5.metric("Stoch K", f"{float(last['stoch_k']):.1f}")
    c6.metric("ADX", f"{float(last['adx']):.1f}")

    st.markdown("</div>", unsafe_allow_html=True)

# بطاقة AI Decision
with col_a2:
    st.markdown("<div class='decision-card'>", unsafe_allow_html=True)
    st.markdown("### 🤖 قرار التداول الذكي")

    score = ai_decision["score"]
    label = ai_decision["label"]

    if score >= 70:
        icon = "🟢"
    elif score >= 55:
        icon = "🟡"
    elif score >= 45:
        icon = "🟠"
    else:
        icon = "🔴"

    st.metric("تقييم الذكاء الاصطناعي", f"{score:.1f} / 100 {icon}", ai_decision["trend"])

    # ترجمة الحالات الخمسة
    if label == "Strong Long":
        st.markdown(
            "<div class='signal-long'><h3>✅ شراء قوي</h3>"
            "<p>انحياز واضح للشراء حسب هيكل السوق والمؤشرات الحالية.</p></div>",
            unsafe_allow_html=True,
        )
    elif label == "Long Bias":
        st.markdown(
            "<div class='signal-long' "
            "style='background:linear-gradient(135deg,#0EA5E9 0%,#3B82F6 100%);'>"
            "<h3>🟢 أفضلية شراء</h3>"
            "<p>فرص الشراء أقوى من البيع، مع ضرورة إدارة للمخاطر.</p></div>",
            unsafe_allow_html=True,
        )
    elif label == "Strong Short":
        st.markdown(
            "<div class='signal-short'><h3>❌ بيع قوي</h3>"
            "<p>انحياز واضح للبيع ومخاطر استمرار الهبوط.</p></div>",
            unsafe_allow_html=True,
        )
    elif label == "Short Bias":
        st.markdown(
            "<div class='signal-short' "
            "style='background:linear-gradient(135deg,#F97316 0%,#EF4444 100%);'>"
            "<h3>🟠 أفضلية بيع</h3>"
            "<p>فرص البيع أفضل، لكن يفضّل تأكيد إضافي قبل الدخول.</p></div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            "<div class='signal-neutral'><h3>⏸ لا توجد صفقة واضحة</h3>"
            "<p>الإشارات الحالية متضاربة أو ضعيفة – الأفضل الانتظار.</p></div>",
            unsafe_allow_html=True,
        )

    alert_text = near_signal_alert(last)
    if alert_text:
        st.warning(f"🔔 {alert_text}")

    st.markdown("</div>", unsafe_allow_html=True)

# بطاقة إدارة المخاطر
with col_a3:
    st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
    st.markdown("### 📉 إدارة المخاطر")

    if max_loss_pct <= 10:
        risk_level = "🟢 منخفض"
    elif max_loss_pct <= 25:
        risk_level = "🟡 متوسط"
    else:
        risk_level = "🔴 مرتفع"

    st.metric("مستوى المخاطرة اليومي", risk_level)
    st.metric("الرصيد الابتدائي", f"{initial_balance:,.0f} USDT")
    st.metric("أقصى عدد صفقات في المحاكاة", f"{max_trades}")

    st.caption(f"إعدادات SL: {sl_pct:.2f}%  |  TP = {tp_factor:.1f} × SL")
    st.caption(f"الإستراتيجية الحالية: {strategy_name}")
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")

# ============= الصف الثاني – الشارت + دفتر الأوامر =============

col_b1, col_b2 = st.columns([2, 1])

# الشارت + فيبوناتشي + إشارات
with col_b1:
    st.markdown("### 📈 الشارت – السعر + VWAP + فيبوناتشي + الإشارات")

    fig = go.Figure()

    fig.add_trace(
        go.Candlestick(
            x=df_sig["timestamp"],
            open=df_sig["open"],
            high=df_sig["high"],
            low=df_sig["low"],
            close=df_sig["close"],
            name="السعر",
            increasing_line_color="#10B981",
            decreasing_line_color="#EF4444",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df_sig["timestamp"],
            y=df_sig["vwap"],
            mode="lines",
            name="VWAP",
            line=dict(width=2, color="#8B5CF6"),
        )
    )

    fib_series = fib_swing_levels(df_sig, lookback=min(120, len(df_sig)))
    fib_colors = ["#6366F1", "#8B5CF6", "#EC4899", "#F59E0B", "#EF4444"]
    fib_levels = ["fib_23", "fib_38", "fib_50", "fib_61", "fib_78"]
    fib_names = ["23.6%", "38.2%", "50%", "61.8%", "78.6%"]

    for i, (level, name) in enumerate(zip(fib_levels, fib_names)):
        fig.add_hline(
            y=fib_series[level],
            line_dash="dash",
            line_width=1,
            line_color=fib_colors[i],
            opacity=0.7,
            annotation_text=name,
            annotation_position="right",
            annotation_font_size=10,
        )

    longs = df_sig[df_sig["signal"] == 1]
    shorts = df_sig[df_sig["signal"] == -1]

    fig.add_trace(
        go.Scatter(
            x=longs["timestamp"],
            y=longs["close"],
            mode="markers",
            name="إشارة شراء",
            marker=dict(
                symbol="triangle-up",
                size=11,
                color="#10B981",
                line=dict(width=2, color="white"),
            ),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=shorts["timestamp"],
            y=shorts["close"],
            mode="markers",
            name="إشارة بيع",
            marker=dict(
                symbol="triangle-down",
                size=11,
                color="#EF4444",
                line=dict(width=2, color="white"),
            ),
        )
    )

    fig.update_layout(
        plot_bgcolor="rgba(15,23,42,0.9)",
        paper_bgcolor="rgba(15,23,42,0.9)",
        font=dict(color="#E2E8F0"),
        height=600,
        margin=dict(l=10, r=10, t=40, b=40),
        xaxis_title="الوقت",
        yaxis_title="السعر (USDT)",
        xaxis_rangeslider_visible=False,
    )

    st.plotly_chart(fig)

# دفتر الأوامر + الحيتان + نشاط البوتات
with col_b2:
    st.markdown("### 📊 دفتر الأوامر (Bybit + حيتان + نشاط)")

    if df_ob.empty:
        st.info("لا يوجد دفتر أوامر متاح حاليًا لهذا الزوج.")
    else:
        st.markdown("<div class='metric-container'>", unsafe_allow_html=True)

        imbalance = ob_info["imbalance_pct"]
        regime = ob_info["regime"]

        col_ob1, col_ob2 = st.columns(2)
        with col_ob1:
            side_txt = "مشترين" if imbalance > 0 else "بائعين"
            st.metric("ميل دفتر الأوامر", f"{imbalance:+.1f}%", side_txt)
        with col_ob2:
            regime_color = (
                "#10B981"
                if "Bullish" in regime
                else "#EF4444"
                if "Bearish" in regime
                else "#e5e7eb"
            )
            st.markdown(
                f"<p style='color:{regime_color};font-weight:bold;'>{regime}</p>",
                unsafe_allow_html=True,
            )

        st.markdown("</div>", unsafe_allow_html=True)

        # نشاط الأوامر (تقريب لوجود بوتات تلغي/تنقل أوامرها)
        st.markdown("#### 🤖 نشاط الأوامر (تقريب للبوتات)")
        col_d1, col_d2 = st.columns(2)
        col_d1.metric(
            "مستويات تم تقليلها/إلغاؤها",
            f"{ob_dyn['cancel_levels']}",
        )
        col_d2.metric(
            "إجمالي حجم الإلغاءات",
            f"{ob_dyn['cancel_volume']:.2f}",
        )

        if ob_dyn["cancel_levels"] > 10 and abs(imbalance) < 10:
            st.warning(
                "🚨 إلغاءات كثيرة مع توازن في دفتر الأوامر – قد يكون هناك نشاط بوتات أو Spoofing."
            )

        st.markdown("#### خريطة عمق دفتر الأوامر")

        df_heat = df_ob.copy()
        df_heat["price_str"] = df_heat["price"].round(2).astype(str)

        fig_ob = px.bar(
            df_heat,
            x="size",
            y="price_str",
            color="side",
            orientation="h",
            color_discrete_map={"bid": "#10B981", "ask": "#EF4444"},
            title="الأحجام عند مستويات الأسعار",
        )

        fig_ob.update_layout(
            plot_bgcolor="rgba(15,23,42,0.9)",
            paper_bgcolor="rgba(15,23,42,0.9)",
            font=dict(color="#E2E8F0"),
            height=320,
            showlegend=False,
            yaxis_title="مستوى السعر",
            xaxis_title="الحجم",
        )

        st.plotly_chart(fig_ob)

        st.markdown("#### 🐋 مستويات يُحتمل أنها حيتان")
        whale_df = ob_info["whale_levels"]
        if whale_df is not None and not whale_df.empty:
            whale_display = whale_df.copy()
            whale_display["price"] = whale_display["price"].round(2)
            whale_display["size"] = whale_display["size"].round(2)
            whale_display["side"] = whale_display["side"].map(
                {"bid": "🟢 شراء", "ask": "🔴 بيع"}
            )
            st.dataframe(
                whale_display[["side", "price", "size"]],
            )
        else:
            st.info("لا توجد مستويات ضخمة واضحة في هذه اللقطة (Snapshot).")

st.markdown("---")

# ============= تبويبات: جدول الإشارات + Backtest =============

tab1, tab2 = st.tabs(["📋 جدول الإشارات", "🧪 نتائج الباك تست"])

with tab1:
    st.markdown("### 📋 الإشارات وأنماط الشموع – آخر 50 شمعة")

    last_signals = df_sig[
        [
            "timestamp",
            "close",
            "trend",
            "rsi9",
            "stoch_k",
            "adx",
            "vwap_dev_pct",
            "fib_zone",
            "candle_pattern",
            "signal",
        ]
    ].tail(50)

    last_signals = last_signals.copy()
    last_signals["trend"] = last_signals["trend"].map({1: "🟢 صاعد", -1: "🔴 هابط"})
    last_signals["signal"] = last_signals["signal"].map(
        {1: "🟢 شراء", -1: "🔴 بيع", 0: "⚪ محايد"}
    )
    last_signals["fib_zone"] = last_signals["fib_zone"].replace(
        {"FibBuy": "🟢 شراء", "FibSell": "🔴 بيع", "None": "⚪"}
    )

    last_signals["close"] = last_signals["close"].round(4)
    last_signals["rsi9"] = last_signals["rsi9"].round(1)
    last_signals["stoch_k"] = last_signals["stoch_k"].round(1)
    last_signals["adx"] = last_signals["adx"].round(1)
    last_signals["vwap_dev_pct"] = last_signals["vwap_dev_pct"].round(2)

    st.dataframe(last_signals)

with tab2:
    st.markdown("### 🧪 نتائج المحاكاة (باك تست سريع)")

    if run_backtest_btn:
        with st.spinner("جاري تشغيل المحاكاة على الزوج الحالي..."):
            bt_df = run_backtest(
                df_sig,
                sl_pct=sl_pct,
                tp_factor=tp_factor,
                max_trades=max_trades,
                max_loss_pct=max_loss_pct,
                initial_balance=initial_balance,
            )

        if bt_df.empty:
            st.info("لم يتم توليد صفقات في هذه المحاكاة – راجع شروط الإشارات.")
        else:
            trades_count = len(bt_df)
            wins = bt_df[bt_df["pnl_pct"] > 0]
            losses = bt_df[bt_df["pnl_pct"] < 0]
            win_rate = len(wins) / trades_count * 100 if trades_count > 0 else 0
            total_return_pct = (bt_df["balance"].iloc[-1] / initial_balance - 1) * 100
            avg_pnl = bt_df["pnl_pct"].mean()
            max_win = bt_df["pnl_pct"].max()
            max_loss = bt_df["pnl_pct"].min()

            col_bt1, col_bt2, col_bt3, col_bt4 = st.columns(4)
            col_bt1.metric("عدد الصفقات", trades_count)
            col_bt2.metric("معدل الصفقات الرابحة", f"{win_rate:.1f}%")
            col_bt3.metric("إجمالي العائد", f"{total_return_pct:+.2f}%")
            col_bt4.metric("متوسط ربح/خسارة لكل صفقة", f"{avg_pnl:+.2f}%")

            col_bt5, col_bt6, col_bt7, col_bt8 = st.columns(4)
            col_bt5.metric("عدد الصفقات الرابحة", len(wins))
            col_bt6.metric("عدد الصفقات الخاسرة", len(losses))
            col_bt7.metric("أكبر صفقة رابحة", f"{max_win:+.2f}%")
            col_bt8.metric("أكبر صفقة خاسرة", f"{max_loss:+.2f}%")

            # بدون شارت – أرقام فقط كما طلبت
            st.markdown("#### ملخص نصي")
            st.write(
                f"- لو كنت بدأت بـ **{initial_balance:.0f} USDT** "
                f"لَأصبح رصيدك الآن حوالي **{bt_df['balance'].iloc[-1]:.2f} USDT**.\n"
                f"- عدد الأيام/الساعات يعتمد على عدد الشموع في الفريم الذي اخترته."
            )

    else:
        st.info("اضغط على زر **🚀 تشغيل الباك تست على الزوج الحالي** من الشريط الجانبي لرؤية النتائج.")

st.markdown("---")

# ============= رادار أفضل 15 عملة =============

st.markdown("## 🔥 أفضل 15 فرصة سكالبينج (حسب مسح السوق)")

if scan_btn:
    with st.spinner("جاري مسح السوق لاختيار أفضل 15 عملة..."):
        rows = []
        for sym in SYMBOLS:
            for tf in ["3m", "5m"]:
                row = score_symbol_for_scalping(sym, tf)
                if row is not None:
                    rows.append(row)
        if rows:
            df_top = (
                pd.DataFrame(rows)
                .sort_values("final_score", ascending=False)
                .head(15)
                .reset_index(drop=True)
            )
            st.session_state["top15"] = df_top
        else:
            st.session_state["top15"] = None

top15 = st.session_state.get("top15", None)
if top15 is None:
    st.info(
        "لم يتم مسح السوق بعد. اضغط على زر **🔍 مسح السوق الآن** في الشريط الجانبي لعرض أفضل 15 عملة."
    )
else:
    df_show = top15.copy()
    df_show["حالة السوق"] = df_show["label"].replace(
        {
            "Strong Long": "شراء قوي",
            "Long Bias": "أفضلية شراء",
            "Strong Short": "بيع قوي",
            "Short Bias": "أفضلية بيع",
            "No Clear Trade": "لا توجد صفقة واضحة",
        }
    )
    df_show = df_show[
        ["symbol", "timeframe", "final_score", "حالة السوق", "avg_volume", "volatility"]
    ]
    df_show.columns = [
        "الزوج",
        "الفريم",
        "Score نهائي",
        "حالة السوق",
        "متوسط الحجم",
        "تقلب % (تقريبية)",
    ]
    st.dataframe(df_show)

    idx = st.selectbox(
        "اختر زوجًا من القائمة لفتحه في لوحة التحليل:",
        options=df_show.index,
        format_func=lambda i: f"{df_show.loc[i, 'الزوج']} – {df_show.loc[i, 'الفريم']}",
    )
    if st.button("📌 فتح هذا الزوج في الأعلى"):
        sel_row = top15.loc[idx]
        st.session_state["symbol_override"] = sel_row["symbol"]
        st.session_state["timeframe_override"] = sel_row["timeframe"]
        st.experimental_rerun()

# ============= فوتر =============

st.markdown(
    """
<div style='text-align:center;color:#94A3B8;padding:16px 0;'>
    ⚡ <b>ZAYA – AI Trading Terminal</b>  
    <br/>هذا النظام أداة مساعدة لاتخاذ القرار، وليس توصية استثمارية مباشرة.
</div>
""",
    unsafe_allow_html=True,
)
