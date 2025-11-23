import json
import math
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# محاولة استيراد ccxt (لباينانس و بايبيت REST)
try:
    import ccxt
    HAS_CCXT = True
except Exception:
    ccxt = None
    HAS_CCXT = False

# ===================== إعداد صفحة Streamlit =====================

st.set_page_config(
    page_title="ZAYA – AI Trading Terminal",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ===================== تنسيق بصري عام =====================

st.markdown("""
<style>
    .main {
        background-color: #020617;
        color: #F9FAFB;
    }
    .stApp {
        background: radial-gradient(circle at top, #1f2937 0, #020617 55%);
    }
    .block-container {
        padding-top: 1rem;
    }
    .metric-container {
        background: rgba(15,23,42,0.90);
        border-radius: 14px;
        padding: 14px 16px;
        border: 1px solid rgba(148, 163, 184, 0.35);
        box-shadow: 0 18px 45px rgba(0,0,0,0.55);
    }
    .decision-card {
        background: radial-gradient(circle at top left, #0f172a 0, #020617 60%);
        border-radius: 16px;
        padding: 18px 18px 16px 18px;
        border: 1px solid rgba(129, 140, 248, 0.5);
        box-shadow: 0 22px 60px rgba(15,23,42,0.95);
    }
    .header-gradient {
        background: linear-gradient(90deg, #38bdf8 0%, #a855f7 40%, #f97316 80%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
    }
    .ai-state-pill {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 6px 12px;
        border-radius: 999px;
        background: rgba(15,23,42,0.95);
        border: 1px solid rgba(148,163,184,0.4);
        font-size: 0.9rem;
    }
    .ai-state-pill span.label {
        font-weight: 600;
        letter-spacing: 0.03em;
    }
    .signal-long {
        background: linear-gradient(135deg, #16a34a 0%, #22c55e 100%);
        color: white;
        padding: 10px;
        border-radius: 10px;
        text-align: center;
        font-weight: 600;
    }
    .signal-short {
        background: linear-gradient(135deg, #dc2626 0%, #f97316 100%);
        color: white;
        padding: 10px;
        border-radius: 10px;
        text-align: center;
        font-weight: 600;
    }
    .signal-neutral {
        background: linear-gradient(135deg, #6b7280 0%, #9ca3af 100%);
        color: white;
        padding: 10px;
        border-radius: 10px;
        text-align: center;
        font-weight: 600;
    }
    .logo-circle {
        width: 52px;
        height: 52px;
        border-radius: 999px;
        background: radial-gradient(circle at 30% 0, #f97316 0, #e11d48 40%, #0f172a 100%);
        display: flex;
        align-items: center;
        justify-content: center;
        box-shadow: 0 16px 40px rgba(15,23,42,0.85);
    }
</style>
""", unsafe_allow_html=True)

# ===================== الهيدر الرئيسي =====================

col_logo, col_title, col_ai = st.columns([0.8, 3, 2])

with col_logo:
    st.markdown(
        "<div class='logo-circle'><span style='font-size: 26px;'>🐇</span></div>",
        unsafe_allow_html=True
    )

with col_title:
    st.markdown("<h1 class='header-gradient'>ZAYA – AI Trading Terminal</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p style='color:#CBD5F5;font-size:0.95rem;'>"
        "لوحة تحكم ذكية لقراءة السوق، دفتر الأوامر، والحيتان – في الزمن الحقيقي."
        "</p>",
        unsafe_allow_html=True
    )

with col_ai:
    st.markdown(
        "<div class='ai-state-pill'>"
        "<span>🧠</span>"
        "<span class='label'>ZAYA AI – قراءة السوق المباشرة</span>"
        "</div>",
        unsafe_allow_html=True
    )

st.markdown("---")

# ===================== إعدادات عامة =====================

SYMBOLS = [
    "BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT",
    "LTC/USDT", "XRP/USDT", "ADA/USDT", "LINK/USDT",
    "AVAX/USDT", "DOGE/USDT", "TON/USDT", "TRX/USDT",
]

TIMEFRAMES = ["1m", "3m", "5m", "15m", "1h"]

STRATEGIES = {
    "سكالبينج VWAP + مؤشرات": "core_scalp",
    "فيبوناتشي + VWAP + ADX": "fibo_swing",
    "ترند + سيولة دفتر أوامر": "liquidity_trend",
}

STRATEGY_MAP_FILE = "strategy_map.json"


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


# ===================== جلب البيانات =====================

def fetch_ohlcv(symbol: str, timeframe: str, limit: int = 500) -> pd.DataFrame:
    """جلب الشموع من Binance (أولوية)، أو توليد بيانات صناعية."""
    # نحاول Binance أولا
    if HAS_CCXT:
        try:
            ex = ccxt.binance({"enableRateLimit": True})
            data = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
            df = pd.DataFrame(
                data,
                columns=["timestamp", "open", "high", "low", "close", "volume"]
            )
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            return df
        except Exception as e:
            st.warning(f"تعذر جلب OHLCV من Binance: {e}")

    # بيانات صناعية
    base_price = 30000.0
    timestamps = pd.date_range(
        end=pd.Timestamp.utcnow(),
        periods=limit,
        freq=tf_to_rule(timeframe)
    )
    prices = base_price + np.cumsum(np.random.normal(0, 50, size=limit))
    df = pd.DataFrame({
        "timestamp": timestamps,
        "open": prices + np.random.normal(0, 10, size=limit),
        "high": prices + np.abs(np.random.normal(0, 15, size=limit)),
        "low": prices - np.abs(np.random.normal(0, 15, size=limit)),
        "close": prices + np.random.normal(0, 10, size=limit),
        "volume": np.random.randint(10, 1000, size=limit)
    })
    return df


def fetch_bybit_orderbook(symbol: str, depth: int = 10) -> pd.DataFrame:
    """جلب دفتر الأوامر من Bybit عبر REST أو توليد بيانات صناعية."""
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
            return df.sort_values(["side", "price"], ascending=[False, False]).reset_index(drop=True)
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
    return df.sort_values(["side", "price"], ascending=[False, False]).reset_index(drop=True)


# ===================== مؤشرات فنية =====================

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
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs()
    ], axis=1).max(axis=1)
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

    plus_di = 100 * (pd.Series(plus_dm).rolling(window=period, min_periods=1).sum() / (atr_val + 1e-9))
    minus_di = 100 * (pd.Series(minus_dm).rolling(window=period, min_periods=1).sum() / (atr_val + 1e-9))

    dx = (abs(plus_di - minus_di) / (plus_di + minus_di + 1e-9)) * 100
    adx_val = dx.rolling(window=period, min_periods=1).mean()
    return adx_val.fillna(0)


def vwap(df: pd.DataFrame) -> pd.Series:
    pv = (df["close"] * df["volume"]).cumsum()
    vol = df["volume"].cumsum().replace(0, np.nan)
    return (pv / vol).ffill()


def detect_candles(df: pd.DataFrame) -> pd.Series:
    """نموذج بسيط لقراءة الشموع."""
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
    bull_engulf = (df["close"] > df["open"]) & (prev_close < prev_open) & \
                  (df["close"] >= prev_open) & (df["open"] <= prev_close)
    bear_engulf = (df["close"] < df["open"]) & (prev_close > prev_open) & \
                  (df["close"] <= prev_open) & (df["open"] >= prev_close)

    pattern = np.where(bull_engulf, "BullEngulf", pattern)
    pattern = np.where(bear_engulf, "BearEngulf", pattern)

    return pd.Series(pattern, index=df.index)


# ===================== فيبوناتشي =====================

def fib_swing_levels(df: pd.DataFrame, lookback: int = 80) -> pd.Series:
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
    close = df["close"]
    buy_zone = (close.between(fib["fib_50"], fib["fib_61"]))
    sell_zone = (close.between(fib["fib_23"], fib["fib_38"]))
    zone = pd.Series("None", index=df.index)
    zone = np.where(buy_zone, "FibBuy", zone)
    zone = np.where(sell_zone, "FibSell", zone)
    return pd.Series(zone, index=df.index)


# ===================== تحليل دفتر الأوامر / الحيتان =====================

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

    if imbalance > 18:
        regime = "سيولة شرائية (حيتان شراء)"
    elif imbalance < -18:
        regime = "سيولة بيعية (حيتان بيع)"
    else:
        regime = "توازن / نطاق"

    big_levels = df_ob[df_ob["size"] >= df_ob["size"].quantile(0.9)] if not df_ob.empty else pd.DataFrame()

    return {
        "bid_vol": bid_vol,
        "ask_vol": ask_vol,
        "imbalance_pct": imbalance,
        "regime": regime,
        "whale_levels": big_levels
    }


def near_signal_alert(last_row: pd.Series) -> str | None:
    rsi_val = float(last_row["rsi9"])
    adx_val = float(last_row["adx"])
    vwap_dev = float(last_row["vwap_dev_pct"])

    alerts = []

    if 25 <= rsi_val <= 30 or 70 <= rsi_val <= 75:
        alerts.append("RSI يقترب من منطقة قرار حادة.")

    if adx_val >= 25:
        alerts.append("قوة الاتجاه (ADX) مرتفعة – الحركة القادمة غالبًا قوية.")

    if abs(vwap_dev) <= 0.4:
        alerts.append("السعر حول VWAP – منطقة توازن (احتمال انطلاق حركة).")

    if not alerts:
        return None
    return " | ".join(alerts)


# ===================== إستراتيجية السكالبينج والإشارات =====================

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
        (df["trend"] == 1) &
        (df["rsi9"].between(rsi_buy_zone[0], rsi_buy_zone[1])) &
        (df["stoch_k"] < stoch_over_sold) &
        (df["adx"] >= adx_min) &
        (df["vwap_dev_pct"].abs().between(vwap_min_pct, vwap_max_pct))
    )

    sell_core = (
        (df["trend"] == -1) &
        (df["rsi9"].between(rsi_sell_zone[0], rsi_sell_zone[1])) &
        (df["stoch_k"] > stoch_over_bought) &
        (df["adx"] >= adx_min) &
        (df["vwap_dev_pct"].abs().between(vwap_min_pct, vwap_max_pct))
    )

    df.loc[buy_core, "signal"] = 1
    df.loc[sell_core, "signal"] = -1

    df["signal_fib_boost"] = 0
    df.loc[buy_core & (df["fib_zone"] == "FibBuy"), "signal_fib_boost"] = 1
    df.loc[sell_core & (df["fib_zone"] == "FibSell"), "signal_fib_boost"] = -1

    return df


# ===================== AI Trading Brain =====================

def classify_market_state(last: pd.Series, ob_info: dict) -> dict:
    """
    ترجمة حالة السوق إلى ٥ حالات عربية:
    1) اتجاه صاعد قوي
    2) اتجاه هابط قوي
    3) تذبذب حاد / فوضوي
    4) نطاق هادئ / تجميع
    5) منطقة انعكاس محتملة
    """
    trend = "صاعد" if last["trend"] == 1 else "هابط"
    rsi_val = float(last["rsi9"])
    adx_val = float(last["adx"])
    vwap_dev = float(last["vwap_dev_pct"])
    fib_zone = last.get("fib_zone", "None")
    regime = ob_info.get("regime", "توازن / نطاق")
    imbalance = ob_info.get("imbalance_pct", 0.0)

    # مبدئيًا نحدد state بالأولوية
    # 1) انعكاس محتمل
    if fib_zone in ("FibBuy", "FibSell") and 30 <= adx_val <= 45 and abs(vwap_dev) < 1.0:
        state = "منطقة انعكاس محتملة"
        color = "🟣"
        short = "السعر قريب من مستويات فيبوناتشي قوية مع اتجاه ليس عنيفًا."
    # 2) اتجاه صاعد قوي
    elif trend == "صاعد" and adx_val >= 25 and rsi_val > 55 and imbalance > 5:
        state = "اتجاه صاعد قوي"
        color = "🟢"
        short = "قوة شراء واضحة، ودفتر الأوامر يميل للمشترين."
    # 3) اتجاه هابط قوي
    elif trend == "هابط" and adx_val >= 25 and rsi_val < 45 and imbalance < -5:
        state = "اتجاه هابط قوي"
        color = "🔴"
        short = "قوة بيع واضحة، ودفتر الأوامر يميل للبائعين."
    # 4) تذبذب حاد
    elif adx_val < 15 and abs(vwap_dev) > 2.0:
        state = "تذبذب عالي / فوضى"
        color = "🟠"
        short = "الحركة متذبذبة وسريعة، بدون اتجاه واضح."
    # 5) نطاق هادئ
    else:
        state = "نطاق هادئ / تجميع"
        color = "🔵"
        short = "السوق في حالة توازن نسبي، مناسب لمراقبة الاختراقات."

    descr = (
        f"الاتجاه العام الآن: **{trend}** · قوة الاتجاه (ADX): **{adx_val:.1f}** · RSI: **{rsi_val:.1f}**\n"
        f"انحراف السعر عن VWAP: **{vwap_dev:+.2f}%** · وضع السيولة: **{regime}**"
    )

    return {
        "state": state,
        "icon": color,
        "short": short,
        "descr": descr
    }


def compute_ai_decision(last_row: pd.Series, ob_info: dict) -> dict:
    trend = 1 if last_row["trend"] == 1 else -1
    rsi_val = float(last_row["rsi9"])
    stoch = float(last_row["stoch_k"])
    adx_val = float(last_row["adx"])
    vwap_dev = float(last_row["vwap_dev_pct"])
    fib_zone = last_row.get("fib_zone", "None")
    candle = last_row.get("candle_pattern", "None")

    regime = ob_info.get("regime", "توازن / نطاق")

    # Trend
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
        stoch_score = 65
    elif stoch > 80:
        stoch_score = 35
    else:
        stoch_score = 50

    # ADX
    if adx_val >= 25:
        adx_score = 70
    elif adx_val < 15:
        adx_score = 45
    else:
        adx_score = 55

    # VWAP
    dev_abs = abs(vwap_dev)
    if 0.5 <= dev_abs <= 1.5:
        vwap_score = 70
    elif dev_abs > 3:
        vwap_score = 40
    else:
        vwap_score = 50

    # Fibo
    if fib_zone == "FibBuy":
        fib_score = 65
    elif fib_zone == "FibSell":
        fib_score = 35
    else:
        fib_score = 50

    # Candle
    strong_bull = ["Hammer", "BullEngulf"]
    strong_bear = ["ShootingStar", "BearEngulf"]
    if candle in strong_bull:
        candle_score = 65
    elif candle in strong_bear:
        candle_score = 35
    else:
        candle_score = 50

    # Orderbook
    if "شرائية" in regime:
        ob_score = 65
    elif "بيعية" in regime:
        ob_score = 35
    else:
        ob_score = 50

    components = [
        (trend_score, 1.0),
        (rsi_score, 1.0),
        (stoch_score, 0.8),
        (adx_score, 0.9),
        (vwap_score, 0.9),
        (fib_score, 0.6),
        (candle_score, 0.6),
        (ob_score, 0.8),
    ]
    num = sum(s * w for s, w in components)
    den = sum(w for _, w in components)
    score = num / den if den else 50

    if score >= 72 and trend == 1:
        label = "Strong Long"
    elif score >= 58 and trend == 1:
        label = "Long Bias"
    elif score <= 28 and trend == -1:
        label = "Strong Short"
    elif score <= 42 and trend == -1:
        label = "Short Bias"
    else:
        label = "No Clear Trade"

    return {
        "score": round(score, 1),
        "label": label,
        "trend": "اتجاه صاعد" if trend == 1 else "اتجاه هابط"
    }


# ===================== Backtest بسيط =====================

def run_backtest(
    df: pd.DataFrame,
    sl_pct: float,
    tp_factor: float,
    max_trades: int,
    max_loss_pct: float,
    initial_balance: float = 1000.0
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

        sl_price = entry_price * (1 - sl_pct / 100) if side == "buy" else entry_price * (1 + sl_pct / 100)
        tp_price = entry_price * (1 + sl_pct * tp_factor / 100) if side == "buy" else entry_price * (1 - sl_pct * tp_factor / 100)

        exit_price = price

        if side == "buy":
            pnl_pct = (exit_price - entry_price) / entry_price * 100
        else:
            pnl_pct = (entry_price - exit_price) / entry_price * 100

        balance *= (1 + pnl_pct / 100)
        equity_peak = max(equity_peak, balance)

        trades.append({
            "time": row["timestamp"],
            "side": side,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "pnl_pct": pnl_pct,
            "balance": balance
        })
        active_trades += 1

        dd_total_pct = (equity_peak - balance) / equity_peak * 100
        if dd_total_pct >= max_loss_pct:
            break

    return pd.DataFrame(trades)


# ===================== الشريط الجانبي (الإعدادات) =====================

with st.sidebar:
    st.markdown("<h2 style='color:#38bdf8;'>⚙️ إعدادات التداول</h2>", unsafe_allow_html=True)

    symbol = st.selectbox("الزوج", SYMBOLS, index=0)
    timeframe = st.selectbox("الفريم الزمني", TIMEFRAMES, index=2)
    n_candles = st.slider("عدد الشموع", 300, 1500, 600, step=100)

    strategy_map = load_strategy_map()
    default_strategy_key = strategy_map.get(symbol, list(STRATEGIES.keys())[0])
    strategy_name = st.selectbox(
        "إستراتيجية الدخول",
        list(STRATEGIES.keys()),
        index=list(STRATEGIES.keys()).index(default_strategy_key)
    )

    if st.button("💾 حفظ الإستراتيجية لهذا الزوج", type="secondary", key="save_strategy", help="حفظ الإعداد الحالي لهذا الزوج", width="stretch"):
        strategy_map[symbol] = strategy_name
        save_strategy_map(strategy_map)
        st.success("تم حفظ الإستراتيجية لهذا الزوج.")

    st.markdown("---")
    st.markdown("<h3 style='color:#a855f7;'>🧪 إعدادات الباك تست</h3>", unsafe_allow_html=True)

    initial_balance = st.number_input("الرصيد الابتدائي (USDT)", 100.0, 1_000_000.0, 1000.0, step=100.0)
    max_trades = st.slider("أقصى عدد صفقات في الاختبار", 5, 200, 30, step=5)
    max_loss_pct = st.slider("أقصى سحب من الرصيد %", 1.0, 80.0, 20.0, step=1.0)
    sl_pct = st.number_input("نسبة وقف الخسارة %", 0.05, 10.0, 0.3, step=0.05)
    tp_factor = st.number_input("عامل الهدف (x SL)", 1.0, 10.0, 2.0, step=0.1)

    st.markdown("---")
    st.markdown("<h3 style='color:#f97316;'>📊 إعدادات المؤشرات</h3>", unsafe_allow_html=True)

    rsi_buy_min = st.number_input("RSI شراء من", 0.0, 100.0, 28.0)
    rsi_buy_max = st.number_input("RSI شراء إلى", 0.0, 100.0, 45.0)
    rsi_sell_min = st.number_input("RSI بيع من", 0.0, 100.0, 55.0)
    rsi_sell_max = st.number_input("RSI بيع إلى", 0.0, 100.0, 72.0)
    stoch_over_sold = st.number_input("Stoch منطقة تشبع بيع", 0.0, 100.0, 20.0)
    stoch_over_bought = st.number_input("Stoch منطقة تشبع شراء", 0.0, 100.0, 80.0)
    adx_min = st.number_input("أدنى ADX لاعتبار الاتجاه قوي", 0.0, 100.0, 18.0)

    ob_depth = st.slider("عمق دفتر الأوامر (مستويات)", 5, 50, 20, step=5)

    run_backtest_btn = st.button("🚀 تشغيل اختبار الإستراتيجية", type="primary", key="run_bt", width="stretch")

# ===================== تحميل البيانات =====================

with st.spinner("🔄 تحميل البيانات والتحليل..."):
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
    ob_info = analyze_orderbook(df_ob) if not df_ob.empty else {
        "bid_vol": 0, "ask_vol": 0, "imbalance_pct": 0,
        "regime": "لا توجد بيانات دفتر أوامر", "whale_levels": pd.DataFrame()
    }

    last = df_sig.iloc[-1]
    ai_decision = compute_ai_decision(last, ob_info)
    ai_state = classify_market_state(last, ob_info)

# ===================== صف: AI + نظرة عامة + مخاطرة =====================

st.markdown("### 🧠 قراءة ZAYA AI لحالة السوق الآن")

col_ai_main, col_overview, col_risk = st.columns([1.7, 1.4, 1.1])

with col_ai_main:
    st.markdown("<div class='decision-card'>", unsafe_allow_html=True)
    st.markdown(
        f"<p style='font-size:0.9rem;color:#e5e7eb;margin-bottom:4px;'>"
        f"{ai_state['icon']} <strong>{ai_state['state']}</strong>"
        f"</p>",
        unsafe_allow_html=True
    )
    st.markdown(
        f"<p style='font-size:0.85rem;color:#cbd5f5;margin-bottom:6px;'>{ai_state['short']}</p>",
        unsafe_allow_html=True
    )
    st.markdown(
        f"<p style='font-size:0.8rem;color:#9ca3af;'>{ai_state['descr']}</p>",
        unsafe_allow_html=True
    )
    alert_text = near_signal_alert(last)
    if alert_text:
        st.warning("🔔 " + alert_text)
    st.markdown("</div>", unsafe_allow_html=True)

with col_overview:
    st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
    st.markdown(f"#### 🪙 {symbol} – {timeframe}")
    price = float(last["close"])
    vwap_now = float(last["vwap"])
    vwap_dev_now = float(last["vwap_dev_pct"])
    trend_now = "صاعد" if last["trend"] == 1 else "هابط"

    c1, c2, c3 = st.columns(3)
    c1.metric("السعر الحالي", f"{price:,.2f}")
    c2.metric("VWAP", f"{vwap_now:,.2f}", f"{vwap_dev_now:+.2f}%")
    c3.metric("الاتجاه", trend_now)

    c4, c5, c6 = st.columns(3)
    c4.metric("RSI(9)", f"{float(last['rsi9']):.1f}")
    c5.metric("Stoch K", f"{float(last['stoch_k']):.1f}")
    c6.metric("ADX", f"{float(last['adx']):.1f}")
    st.markdown("</div>", unsafe_allow_html=True)

with col_risk:
    st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
    st.markdown("#### 📉 ملف المخاطرة")

    if max_loss_pct <= 10:
        risk_level = "منخفض"
        risk_icon = "🟢"
    elif max_loss_pct <= 25:
        risk_level = "متوسط"
        risk_icon = "🟡"
    else:
        risk_level = "مرتفع"
        risk_icon = "🔴"

    st.metric("مستوى المخاطرة", f"{risk_icon} {risk_level}")
    st.metric("الرصيد الابتدائي", f"{initial_balance:,.0f} USDT")
    st.metric("أقصى عدد صفقات", f"{max_trades}")
    st.caption(f"SL: {sl_pct:.2f}% · TP: {tp_factor:.1f}x SL")
    st.caption(f"إستراتيجية: {strategy_name}")
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")

# ===================== الصف الثاني – الشارت + دفتر الأوامر =====================

col_chart, col_orderbook = st.columns([2, 1])

with col_chart:
    st.markdown("### 📈 الشارت + إشارات الدخول + فيبوناتشي")

    fig = go.Figure()

    fig.update_layout(
        plot_bgcolor='rgba(15,23,42, 0.9)',
        paper_bgcolor='rgba(2,6,23, 0.95)',
        font=dict(color='#E5E7EB'),
        height=600,
        margin=dict(l=10, r=10, t=40, b=40),
        xaxis_title="الوقت",
        yaxis_title="السعر (USDT)",
        xaxis_rangeslider_visible=False,
    )

    fig.add_trace(go.Candlestick(
        x=df_sig["timestamp"],
        open=df_sig["open"],
        high=df_sig["high"],
        low=df_sig["low"],
        close=df_sig["close"],
        name="السعر",
        increasing_line_color='#22c55e',
        decreasing_line_color='#ef4444'
    ))

    fig.add_trace(go.Scatter(
        x=df_sig["timestamp"],
        y=df_sig["vwap"],
        mode="lines",
        name="VWAP",
        line=dict(width=2, color='#a855f7')
    ))

    fib_series = fib_swing_levels(df_sig, lookback=min(120, len(df_sig)))
    fib_colors = ['#38bdf8', '#a855f7', '#facc15', '#f97316', '#ef4444']
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
            annotation_font_size=10
        )

    longs = df_sig[df_sig["signal"] == 1]
    shorts = df_sig[df_sig["signal"] == -1]

    fig.add_trace(go.Scatter(
        x=longs["timestamp"],
        y=longs["close"],
        mode="markers",
        name="إشارة شراء",
        marker=dict(symbol="triangle-up", size=11, color="#22c55e", line=dict(width=1.5, color="white"))
    ))
    fig.add_trace(go.Scatter(
        x=shorts["timestamp"],
        y=shorts["close"],
        mode="markers",
        name="إشارة بيع",
        marker=dict(symbol="triangle-down", size=11, color="#ef4444", line=dict(width=1.5, color="white"))
    ))

    st.plotly_chart(fig, width="stretch")

with col_orderbook:
    st.markdown("### 📊 دفتر الأوامر – Bybit (Top Levels)")

    if df_ob.empty:
        st.info("لا توجد بيانات دفتر أوامر متاحة حاليًا.")
    else:
        st.markdown("<div class='metric-container'>", unsafe_allow_html=True)

        imbalance = ob_info['imbalance_pct']
        regime = ob_info['regime']

        col_ob1, col_ob2 = st.columns(2)
        with col_ob1:
            st.metric("توازن السيولة", f"{imbalance:+.1f}%")
        with col_ob2:
            st.markdown(
                f"<p style='color:#e5e7eb;font-size:0.85rem;margin-top:4px;'><strong>{regime}</strong></p>",
                unsafe_allow_html=True
            )

        st.markdown("</div>", unsafe_allow_html=True)

        df_heat = df_ob.copy()
        df_heat["price_str"] = df_heat["price"].round(2).astype(str)

        fig_ob = px.bar(
            df_heat,
            x="size",
            y="price_str",
            color="side",
            orientation='h',
            color_discrete_map={'bid': '#22c55e', 'ask': '#ef4444'},
            title="عمق دفتر الأوامر (أحجام عند كل سعر)"
        )

        fig_ob.update_layout(
            plot_bgcolor='rgba(15,23,42, 0.9)',
            paper_bgcolor='rgba(2,6,23, 0.95)',
            font=dict(color='#E5E7EB'),
            height=320,
            showlegend=False,
            yaxis_title="السعر",
            xaxis_title="الحجم"
        )

        st.plotly_chart(fig_ob, width="stretch")

        st.markdown("#### 🐋 مستويات الحيتان (أكبر أوامر)")
        whale_df = ob_info["whale_levels"]
        if whale_df is not None and not whale_df.empty:
            whale_display = whale_df.copy()
            whale_display["price"] = whale_display["price"].round(2)
            whale_display["size"] = whale_display["size"].round(4)
            whale_display["side"] = whale_display["side"].map({"bid": "🟢 شراء", "ask": "🔴 بيع"})

            st.dataframe(
                whale_display,
                column_config={
                    "side": "النوع",
                    "price": "السعر",
                    "size": "الحجم"
                },
                hide_index=True,
                height=200,
                width="stretch"
            )
        else:
            st.info("لا توجد أوامر ضخمة مميزة الآن.")

st.markdown("---")

# ===================== الصف الثالث – جدول الإشارات + Backtest =====================

tab1, tab2 = st.tabs(["📋 جدول الإشارات", "🧪 نتائج الباك تست"])

with tab1:
    st.markdown("### 📋 آخر 50 شمعة – إشارات وحالة المؤشرات")

    last_signals = df_sig[[
        "timestamp", "close", "trend", "rsi9", "stoch_k", "adx",
        "vwap_dev_pct", "fib_zone", "candle_pattern", "signal"
    ]].tail(50).copy()

    last_signals["trend"] = last_signals["trend"].map({1: "🟢 صاعد", -1: "🔴 هابط"})
    last_signals["signal"] = last_signals["signal"].map({1: "🟢 شراء", -1: "🔴 بيع", 0: "⚪ محايد"})
    last_signals["fib_zone"] = last_signals["fib_zone"].map({
        "FibBuy": "🟢 منطقة شراء",
        "FibSell": "🔴 منطقة بيع",
        "None": "⚪"
    })

    last_signals_display = last_signals.copy()
    last_signals_display["close"] = last_signals_display["close"].round(4)
    last_signals_display["rsi9"] = last_signals_display["rsi9"].round(1)
    last_signals_display["stoch_k"] = last_signals_display["stoch_k"].round(1)
    last_signals_display["adx"] = last_signals_display["adx"].round(1)
    last_signals_display["vwap_dev_pct"] = last_signals_display["vwap_dev_pct"].round(2)

    st.dataframe(
        last_signals_display,
        column_config={
            "timestamp": "الوقت",
            "close": "السعر",
            "trend": "الاتجاه",
            "rsi9": "RSI(9)",
            "stoch_k": "Stoch K",
            "adx": "ADX",
            "vwap_dev_pct": "انحراف VWAP %",
            "fib_zone": "منطقة فيبوناتشي",
            "candle_pattern": "نمط الشمعة",
            "signal": "الإشارة"
        },
        hide_index=True,
        height=420,
        width="stretch"
    )

with tab2:
    st.markdown("### 🧪 نتائج اختبار الإستراتيجية")

    if run_backtest_btn:
        with st.spinner("تشغيل المحاكاة..."):
            bt_df = run_backtest(
                df_sig,
                sl_pct=sl_pct,
                tp_factor=tp_factor,
                max_trades=max_trades,
                max_loss_pct=max_loss_pct,
                initial_balance=initial_balance
            )

        if bt_df.empty:
            st.info("لم يتم توليد صفقات – جرّب تخفيف شروط الإشارات أو زيادة عدد الشموع.")
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
            col_bt2.metric("معدل الربح", f"{win_rate:.1f}%")
            col_bt3.metric("إجمالي العائد", f"{total_return_pct:+.2f}%")
            col_bt4.metric("متوسط ربح/خسارة صفقة", f"{avg_pnl:+.2f}%")

            col_bt5, col_bt6, col_bt7, col_bt8 = st.columns(4)
            col_bt5.metric("عدد الصفقات الرابحة", len(wins))
            col_bt6.metric("عدد الصفقات الخاسرة", len(losses))
            col_bt7.metric("أكبر ربح صفقة", f"{max_win:+.2f}%")
            col_bt8.metric("أكبر خسارة صفقة", f"{max_loss:+.2f}%")

            # ملخص رقمي فقط (بدون شارت كما طلبت)
            st.markdown("#### 💡 ملخص رقمي سريع")
            st.write(
                f"- لو بدأت بـ **{initial_balance:.0f} USDT** كان الرصيد سيكون الآن تقريبًا **{bt_df['balance'].iloc[-1]:.2f} USDT**.\n"
                f"- تم تنفيذ **{trades_count}** صفقة خلال الفترة المختبرة.\n"
                f"- نسبة الربح: **{win_rate:.1f}%** من إجمالي عدد الصفقات."
            )

            st.markdown("#### 📋 تفاصيل الصفقات")
            bt_display = bt_df.copy()
            bt_display["entry_price"] = bt_display["entry_price"].round(4)
            bt_display["exit_price"] = bt_display["exit_price"].round(4)
            bt_display["pnl_pct"] = bt_display["pnl_pct"].round(2)
            bt_display["balance"] = bt_display["balance"].round(2)
            bt_display["side"] = bt_display["side"].map({"buy": "🟢 شراء", "sell": "🔴 بيع"})

            st.dataframe(
                bt_display,
                column_config={
                    "time": "الوقت",
                    "side": "نوع الصفقة",
                    "entry_price": "سعر الدخول",
                    "exit_price": "سعر الخروج",
                    "pnl_pct": "الربح %",
                    "balance": "الرصيد بعد الصفقة"
                },
                hide_index=True,
                height=320,
                width="stretch"
            )
    else:
        st.info("لرؤية نتائج الإستراتيجية، اضغط زر **🚀 تشغيل اختبار الإستراتيجية** من الشريط الجانبي.")

# ===================== الفوتر =====================

st.markdown("---")
st.markdown(
    """
    <div style='text-align:center;color:#6b7280;padding:12px;font-size:0.8rem;'>
        ⚡ <b>ZAYA AI Trading Terminal</b> – أداة تحليل ذكية، وليست نصيحة استثمارية. 
        التداول عالي المخاطر، استخدم إدارة رأس مال صارمة دائمًا.
    </div>
    """,
    unsafe_allow_html=True
)
