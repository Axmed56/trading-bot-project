import os
import json
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# ========================= إعداد صفحة Streamlit =========================

st.set_page_config(
    page_title="Trading Lab",
    layout="wide"
)

# ========================= مسارات الملفات =========================

OHLCV_PATH = "ohlcv.csv"
STRATEGY_STORE_PATH = "symbol_strategies.json"

# ========================= تحميل بيانات الشموع =========================

if not os.path.exists(OHLCV_PATH):
    st.error(f"ملف الشموع غير موجود: {OHLCV_PATH}")
    st.stop()

ohlcv_raw = pd.read_csv(OHLCV_PATH, parse_dates=["timestamp"])
if ohlcv_raw.empty:
    st.error("ملف الشموع فارغ.")
    st.stop()

ohlcv_raw = ohlcv_raw.sort_values("timestamp").reset_index(drop=True)

# ========================= تحميل/تهيئة ملف حفظ الإستراتيجيات =========================

if "saved_strategies" not in st.session_state:
    if os.path.exists(STRATEGY_STORE_PATH):
        try:
            with open(STRATEGY_STORE_PATH, "r", encoding="utf-8") as f:
                st.session_state.saved_strategies = json.load(f)
        except Exception:
            st.session_state.saved_strategies = {}
    else:
        st.session_state.saved_strategies = {}

# ========================= دوال المؤشرات =========================

def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period, min_periods=1).mean()

def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).fillna(0)
    loss = -1 * delta.clip(upper=0).fillna(0)
    avg_gain = gain.rolling(period, min_periods=1).mean()
    avg_loss = loss.rolling(period, min_periods=1).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi_series = 100 - (100 / (1 + rs))
    return rsi_series.fillna(50)

def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def vwap_from_df(df: pd.DataFrame) -> pd.Series:
    typical_price = (df["high"] + df["low"] + df["close"]) / 3.0
    vol = df["volume"].replace(0, np.nan)
    cum_vp = (typical_price * vol).cumsum()
    cum_vol = vol.cumsum()
    vwap = cum_vp / cum_vol
    return vwap.fillna(method="bfill").fillna(method="ffill")

def bollinger(series: pd.Series, period: int = 20, std_factor: float = 2.0):
    mid = series.rolling(period, min_periods=1).mean()
    std = series.rolling(period, min_periods=1).std().fillna(0)
    upper = mid + std_factor * std
    lower = mid - std_factor * std
    return mid, upper, lower

# ========================= دوال مساعدة =========================

TF_SECONDS = {
    "5T": 5 * 60,
    "15T": 15 * 60,
    "4h": 4 * 60 * 60,
}

TF_LABELS = {
    "5T": "5 دقائق",
    "15T": "15 دقيقة",
    "4h": "4 ساعات",
}

def infer_base_tf_seconds(df: pd.DataFrame) -> float:
    df_sorted = df.sort_values("timestamp")
    deltas = df_sorted["timestamp"].diff().dropna()
    if deltas.empty:
        return 60.0
    median_delta = deltas.median()
    return median_delta.total_seconds()

def human_tf(sec: float) -> str:
    if abs(sec - 60) < 1:
        return "1 دقيقة"
    if abs(sec - 300) < 1:
        return "5 دقائق"
    if abs(sec - 900) < 1:
        return "15 دقيقة"
    if abs(sec - 3600) < 1:
        return "1 ساعة"
    if abs(sec - 14400) < 1:
        return "4 ساعات"
    if abs(sec - 86400) < 1:
        return "يومي"
    return f"{sec:.0f} ثانية"

def resample_ohlcv(df: pd.DataFrame, tf_rule: str):
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp")

    base_secs = infer_base_tf_seconds(df)
    req_secs = TF_SECONDS[tf_rule]

    note = ""
    if req_secs < base_secs:
        note = (
            f"⚠️ الفريم الأصلي للبيانات: {human_tf(base_secs)} "
            f"أكبر من الفريم المطلوب ({TF_LABELS[tf_rule]}). "
            f"يتم استخدام الفريم الأصلي كما هو."
        )
        return df.reset_index(drop=True), note

    df_idx = df.set_index("timestamp")
    agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }
    df_res = df_idx.resample(tf_rule).agg(agg).dropna().reset_index()
    return df_res, note

def generate_signals(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    df = df.copy()

    # EMA20 أساسي للرؤية
    df["ema20"] = ema(df["close"], 20)

    # VWAP
    df["vwap"] = vwap_from_df(df)

    # إعداد أعمدة المؤشرات الأخرى بقيم افتراضية
    df["sma_fast"] = np.nan
    df["sma_slow"] = np.nan
    df["rsi"] = np.nan
    df["macd"] = np.nan
    df["macd_signal"] = np.nan
    df["macd_hist"] = np.nan
    df["bb_mid"] = np.nan
    df["bb_upper"] = np.nan
    df["bb_lower"] = np.nan

    # SMA
    if cfg["use_sma"]:
        df["sma_fast"] = sma(df["close"], cfg["sma_fast"])
        df["sma_slow"] = sma(df["close"], cfg["sma_slow"])

    # RSI
    if cfg["use_rsi"]:
        df["rsi"] = rsi(df["close"], cfg["rsi_period"])

    # MACD
    if cfg["use_macd_filter"]:
        macd_line, macd_signal_line, macd_hist = macd(
            df["close"],
            fast=cfg["macd_fast"],
            slow=cfg["macd_slow"],
            signal=cfg["macd_signal"],
        )
        df["macd"] = macd_line
        df["macd_signal"] = macd_signal_line
        df["macd_hist"] = macd_hist

    # Bollinger
    if cfg["use_bollinger"]:
        mid, upper, lower = bollinger(
            df["close"], cfg["bb_period"], cfg["bb_std_factor"]
        )
        df["bb_mid"] = mid
        df["bb_upper"] = upper
        df["bb_lower"] = lower

    df["signal"] = 0

    # ✅ لو كل المؤشرات مقفولة → مفيش سيجنال
    if not (
        cfg["use_rsi"]
        or cfg["use_macd_filter"]
        or cfg["use_vwap_filter"]
        or cfg["use_sma"]
        or cfg["use_bollinger"]
    ):
        return df

    # منطق ثابت: EMA20 + VWAP + RSI (لو مفعّل)
    cross_up = (df["close"] > df["ema20"]) & (
        df["close"].shift(1) <= df["ema20"].shift(1)
    )
    cross_down = (df["close"] < df["ema20"]) & (
        df["close"].shift(1) >= df["ema20"].shift(1)
    )

    buy_cond = cross_up
    sell_cond = cross_down

    if cfg["use_rsi"]:
        buy_cond = buy_cond & (df["rsi"] >= cfg["rsi_buy"])
        sell_cond = sell_cond & (df["rsi"] <= cfg["rsi_sell"])

    if cfg["use_macd_filter"]:
        buy_cond = buy_cond & (df["macd_hist"] > 0)
        sell_cond = sell_cond & (df["macd_hist"] < 0)

    if cfg["use_vwap_filter"]:
        tol = cfg["vwap_tolerance"]
        buy_cond = buy_cond & (df["close"] >= df["vwap"] * (1 - tol))
        sell_cond = sell_cond & (df["close"] <= df["vwap"] * (1 + tol))

    df.loc[buy_cond, "signal"] = 1
    df.loc[sell_cond, "signal"] = -1

    return df

def calculate_position_size(price: float, amount_usd: float) -> float:
    if price <= 0:
        return 0.0
    return amount_usd / price

def run_backtest(df: pd.DataFrame, cfg: dict):
    df = df.copy()
    df = generate_signals(df, cfg)

    # ✅ لو الدخول الآلي مقفول → مفيش صفقات
    if not cfg["enable_trading"]:
        trades_df = pd.DataFrame(
            columns=["time", "type", "side", "entry_price", "exit_price", "amount", "pnl", "balance"]
        )
        return df, trades_df

    balance = cfg["initial_balance"]
    position = None
    trades = []

    if df["signal"].abs().sum() == 0:
        trades_df = pd.DataFrame(
            columns=["time", "type", "side", "entry_price", "exit_price", "amount", "pnl", "balance"]
        )
        return df, trades_df

    for i in range(len(df)):
        row = df.iloc[i]
        ts = row["timestamp"]
        price = float(row["close"])
        signal = int(row["signal"])

        if position is not None:
            side = position["side"]
            entry_price = position["entry_price"]
            amount = position["amount"]
            sl = position["stop_loss"]
            tp = position["take_profit"]

            exit_reason = None
            if side == "buy":
                if price <= sl:
                    exit_reason = "STOPLOSS"
                elif price >= tp:
                    exit_reason = "TAKEPROFIT"
            else:
                if price >= sl:
                    exit_reason = "STOPLOSS"
                elif price <= tp:
                    exit_reason = "TAKEPROFIT"

            if exit_reason is not None:
                direction = 1 if side == "buy" else -1
                pnl = (price - entry_price) * amount * direction
                balance += pnl

                trades.append({
                    "time": ts,
                    "type": exit_reason,
                    "side": side,
                    "entry_price": entry_price,
                    "exit_price": price,
                    "amount": amount,
                    "pnl": pnl,
                    "balance": balance
                })
                position = None
                continue

        if position is None and signal != 0:
            side = "buy" if signal == 1 else "sell"
            amount = calculate_position_size(price, cfg["amount_usd"])
            if amount <= 0:
                continue

            if side == "buy":
                sl = price * (1 - cfg["stop_loss_pct"])
                tp = price * (1 + cfg["take_profit_pct"])
            else:
                sl = price * (1 + cfg["stop_loss_pct"])
                tp = price * (1 - cfg["take_profit_pct"])

            position = {
                "side": side,
                "entry_price": price,
                "amount": amount,
                "stop_loss": sl,
                "take_profit": tp
            }

            trades.append({
                "time": ts,
                "type": "ENTRY",
                "side": side,
                "entry_price": price,
                "exit_price": None,
                "amount": amount,
                "pnl": 0.0,
                "balance": balance
            })

    trades_df = pd.DataFrame(trades)
    return df, trades_df

def compute_stats(trades_df: pd.DataFrame, initial_balance: float) -> dict:
    if trades_df.empty:
        return {
            "final_balance": initial_balance,
            "total_return": 0.0,
            "max_drawdown": 0.0,
            "total_trades": 0,
            "win_trades": 0,
            "loss_trades": 0,
            "buy_count": 0,
            "sell_count": 0,
            "win_rate": 0.0,
            "avg_profit": 0.0,
            "avg_loss": 0.0,
        }

    final_balance = trades_df["balance"].iloc[-1]
    total_return = ((final_balance - initial_balance) / initial_balance) * 100

    max_balance = trades_df["balance"].cummax()
    drawdown = (max_balance - trades_df["balance"]) / max_balance * 100
    max_drawdown = drawdown.max()

    total_trades = len(trades_df)
    wins = trades_df[trades_df["pnl"] > 0]
    losses = trades_df[trades_df["pnl"] < 0]

    win_trades = len(wins)
    loss_trades = len(losses)
    win_rate = (win_trades / total_trades * 100) if total_trades > 0 else 0.0
    avg_profit = wins["pnl"].mean() if not wins.empty else 0.0
    avg_loss = losses["pnl"].mean() if not losses.empty else 0.0

    buy_count = len(trades_df[trades_df["side"] == "buy"])
    sell_count = len(trades_df[trades_df["side"] == "sell"])

    return {
        "final_balance": final_balance,
        "total_return": total_return,
        "max_drawdown": max_drawdown,
        "total_trades": total_trades,
        "win_trades": win_trades,
        "loss_trades": loss_trades,
        "buy_count": buy_count,
        "sell_count": sell_count,
        "win_rate": win_rate,
        "avg_profit": avg_profit,
        "avg_loss": avg_loss,
    }

def build_chart(df_signals: pd.DataFrame, trades_df: pd.DataFrame, title: str):
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.65, 0.35],
        subplot_titles=(title, "Equity / PnL")
    )

    fig.add_trace(
        go.Candlestick(
            x=df_signals["timestamp"],
            open=df_signals["open"],
            high=df_signals["high"],
            low=df_signals["low"],
            close=df_signals["close"],
            name="Price",
            increasing_line_color="#26A69A",
            decreasing_line_color="#EF5350",
            increasing_fillcolor="rgba(38,166,154,0.6)",
            decreasing_fillcolor="rgba(239,83,80,0.6)"
        ),
        row=1,
        col=1
    )

    fig.add_trace(
        go.Scatter(
            x=df_signals["timestamp"],
            y=df_signals["ema20"],
            mode="lines",
            name="EMA20"
        ),
        row=1,
        col=1
    )

    fig.add_trace(
        go.Scatter(
            x=df_signals["timestamp"],
            y=df_signals["vwap"],
            mode="lines",
            line=dict(dash="dot"),
            name="VWAP"
        ),
        row=1,
        col=1
    )

    if "bb_upper" in df_signals.columns and df_signals["bb_upper"].notna().any():
        fig.add_trace(
            go.Scatter(
                x=df_signals["timestamp"],
                y=df_signals["bb_upper"],
                mode="lines",
                line=dict(width=1, dash="dot"),
                name="BB Upper"
            ),
            row=1,
            col=1
        )
        fig.add_trace(
            go.Scatter(
                x=df_signals["timestamp"],
                y=df_signals["bb_lower"],
                mode="lines",
                line=dict(width=1, dash="dot"),
                name="BB Lower"
            ),
            row=1,
            col=1
        )

    if not trades_df.empty:
        entries = trades_df[trades_df["type"] == "ENTRY"]
        exits = trades_df[trades_df["type"].isin(["STOPLOSS", "TAKEPROFIT"])]

        if not entries.empty:
            fig.add_trace(
                go.Scatter(
                    x=entries["time"],
                    y=entries["entry_price"],
                    mode="markers",
                    marker=dict(symbol="diamond", size=11, color="#00E676", line=dict(width=1, color="#1B5E20")),
                    name="Entry"
                ),
                row=1,
                col=1
            )

        if not exits.empty:
            sl = exits[exits["type"] == "STOPLOSS"]
            tp = exits[exits["type"] == "TAKEPROFIT"]

            if not sl.empty:
                fig.add_trace(
                    go.Scatter(
                        x=sl["time"],
                        y=sl["exit_price"],
                        mode="markers",
                        marker=dict(symbol="x", size=10, color="#FF1744", line=dict(width=1)),
                        name="Stop Loss"
                    ),
                    row=1,
                    col=1
                )

            if not tp.empty:
                fig.add_trace(
                    go.Scatter(
                        x=tp["time"],
                        y=tp["exit_price"],
                        mode="markers",
                        marker=dict(symbol="triangle-down", size=10, color="#2962FF", line=dict(width=1)),
                        name="Take Profit"
                    ),
                    row=1,
                    col=1
                )

        fig.add_trace(
            go.Scatter(
                x=trades_df["time"],
                y=trades_df["balance"],
                mode="lines+markers",
                name="Balance"
            ),
            row=2,
            col=1
        )

        fig.add_trace(
            go.Scatter(
                x=trades_df["time"],
                y=trades_df["pnl"].cumsum(),
                mode="lines",
                line=dict(dash="dot"),
                name="PnL Cum"
            ),
            row=2,
            col=1
        )

    fig.update_layout(
        height=500,
        hovermode="x unified",
        xaxis_rangeslider_visible=False,
        showlegend=True,
    )
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Equity", row=2, col=1)

    return fig

# ========================= واجهة المستخدم – الصف العلوي =========================

top_left, top_mid, top_right = st.columns([1.5, 1.2, 1.3])

with top_left:
    st.subheader("Symbol / Account")
    symbol_name = st.text_input("رمز العملة", value=st.session_state.get("symbol_name", "BTC/USDT"), key="symbol_name")
    initial_balance = st.number_input(
        "رصيد افتراضي (USDT)",
        min_value=100.0,
        max_value=100000.0,
        value=float(st.session_state.get("initial_balance", 1000.0)),
        step=100.0,
        key="initial_balance"
    )
    amount_usd = st.number_input(
        "حجم الصفقة (USDT)",
        min_value=5.0,
        max_value=float(st.session_state.get("initial_balance", 1000.0)),
        value=float(st.session_state.get("amount_usd", 50.0)),
        step=5.0,
        key="amount_usd"
    )

with top_mid:
    st.subheader("Moving Averages / Bollinger")
    use_sma = st.checkbox("SMA On", value=st.session_state.get("use_sma", True), key="use_sma")
    sma_fast = st.number_input("SMA Fast", min_value=2, max_value=200, value=int(st.session_state.get("sma_fast", 10)), key="sma_fast")
    sma_slow = st.number_input("SMA Slow", min_value=5, max_value=500, value=int(st.session_state.get("sma_slow", 30)), key="sma_slow")

    use_bollinger = st.checkbox("Bollinger Bands", value=st.session_state.get("use_bollinger", False), key="use_bollinger")
    bb_period = st.number_input("BB Period", min_value=5, max_value=100, value=int(st.session_state.get("bb_period", 20)), key="bb_period")
    bb_std_factor = st.number_input("BB Std Dev", min_value=0.5, max_value=4.0, value=float(st.session_state.get("bb_std_factor", 2.0)), step=0.1, key="bb_std_factor")

with top_right:
    st.subheader("Filters / Risk")
    enable_trading = st.checkbox("تفعيل الدخول الآلي (Backtest)", value=st.session_state.get("enable_trading", False), key="enable_trading")

    use_rsi = st.checkbox("RSI Filter", value=st.session_state.get("use_rsi", True), key="use_rsi")
    rsi_period = st.number_input("RSI Period", min_value=5, max_value=14, value=int(st.session_state.get("rsi_period", 8)), key="rsi_period")
    rsi_buy = st.number_input("RSI Buy ≥", min_value=0, max_value=100, value=int(st.session_state.get("rsi_buy", 40)), key="rsi_buy")
    rsi_sell = st.number_input("RSI Sell ≤", min_value=0, max_value=100, value=int(st.session_state.get("rsi_sell", 60)), key="rsi_sell")

    use_macd_filter = st.checkbox("MACD Filter", value=st.session_state.get("use_macd_filter", False), key="use_macd_filter")
    use_vwap_filter = st.checkbox("VWAP Filter", value=st.session_state.get("use_vwap_filter", True), key="use_vwap_filter")
    vwap_tolerance_pct = st.number_input("VWAP Tolerance %", 0.0, 5.0, float(st.session_state.get("vwap_tolerance_pct", 0.5)), 0.1, key="vwap_tolerance_pct")

    stop_loss_pct = st.number_input("SL %", min_value=0.1, max_value=20.0, value=float(st.session_state.get("stop_loss_pct", 2.0)), step=0.1, key="stop_loss_pct")
    take_profit_pct = st.number_input("TP %", min_value=0.1, max_value=50.0, value=float(st.session_state.get("take_profit_pct", 4.0)), step=0.1, key="take_profit_pct")

# ========================= أزرار التحكم =========================

control_col1, control_col2, control_col3 = st.columns([1, 1, 2])
with control_col1:
    run_bt = st.button("🚀 Run Backtest")
with control_col2:
    save_strategy_btn = st.button("💾 Save Strategy for Symbol")
with control_col3:
    load_strategy_btn = st.button("🔁 Load Saved Strategy for Symbol")

def apply_loaded_cfg(cfg_loaded: dict):
    st.session_state.initial_balance = cfg_loaded.get("initial_balance", 1000.0)
    st.session_state.amount_usd = cfg_loaded.get("amount_usd", 50.0)

    st.session_state.use_sma = cfg_loaded.get("use_sma", True)
    st.session_state.sma_fast = cfg_loaded.get("sma_fast", 10)
    st.session_state.sma_slow = cfg_loaded.get("sma_slow", 30)

    st.session_state.use_bollinger = cfg_loaded.get("use_bollinger", False)
    st.session_state.bb_period = cfg_loaded.get("bb_period", 20)
    st.session_state.bb_std_factor = cfg_loaded.get("bb_std_factor", 2.0)

    st.session_state.use_rsi = cfg_loaded.get("use_rsi", True)
    st.session_state.rsi_period = cfg_loaded.get("rsi_period", 8)
    st.session_state.rsi_buy = cfg_loaded.get("rsi_buy", 40)
    st.session_state.rsi_sell = cfg_loaded.get("rsi_sell", 60)

    st.session_state.use_macd_filter = cfg_loaded.get("use_macd_filter", False)
    st.session_state.use_vwap_filter = cfg_loaded.get("use_vwap_filter", True)
    st.session_state.vwap_tolerance_pct = cfg_loaded.get("vwap_tolerance", 0.005) * 100.0

    st.session_state.stop_loss_pct = cfg_loaded.get("stop_loss_pct", 0.02) * 100.0
    st.session_state.take_profit_pct = cfg_loaded.get("take_profit_pct", 0.04) * 100.0

if load_strategy_btn:
    sym = st.session_state.symbol_name
    saved_all = st.session_state.saved_strategies
    if sym in saved_all:
        apply_loaded_cfg(saved_all[sym])
        st.success(f"تم تحميل إعدادات الإستراتيجية المحفوظة لـ {sym}")
        st.experimental_rerun()
    else:
        st.warning("لا يوجد إعداد محفوظ لهذه العملة حتى الآن.")

sym = st.session_state.symbol_name
with control_col3:
    if sym in st.session_state.saved_strategies:
        st.write(f"✅ يوجد إعداد محفوظ لـ {sym}")
    else:
        st.write(f"ℹ️ لا يوجد إعداد محفوظ لـ {sym}")

if save_strategy_btn:
    cfg_to_save = {
        "initial_balance": float(st.session_state.initial_balance),
        "amount_usd": float(st.session_state.amount_usd),
        "use_sma": bool(st.session_state.use_sma),
        "sma_fast": int(st.session_state.sma_fast),
        "sma_slow": int(st.session_state.sma_slow),
        "use_bollinger": bool(st.session_state.use_bollinger),
        "bb_period": int(st.session_state.bb_period),
        "bb_std_factor": float(st.session_state.bb_std_factor),
        "use_rsi": bool(st.session_state.use_rsi),
        "rsi_period": int(st.session_state.rsi_period),
        "rsi_buy": int(st.session_state.rsi_buy),
        "rsi_sell": int(st.session_state.rsi_sell),
        "use_macd_filter": bool(st.session_state.use_macd_filter),
        "macd_fast": 12,
        "macd_slow": 26,
        "macd_signal": 9,
        "use_vwap_filter": bool(st.session_state.use_vwap_filter),
        "vwap_tolerance": float(st.session_state.vwap_tolerance_pct) / 100.0,
        "stop_loss_pct": float(st.session_state.stop_loss_pct) / 100.0,
        "take_profit_pct": float(st.session_state.take_profit_pct) / 100.0,
    }

    st.session_state.saved_strategies[st.session_state.symbol_name] = cfg_to_save
    try:
        with open(STRATEGY_STORE_PATH, "w", encoding="utf-8") as f:
            json.dump(st.session_state.saved_strategies, f, ensure_ascii=False, indent=2)
        st.success("تم حفظ إعداد الإستراتيجية لهذه العملة.")
    except Exception as e:
        st.error(f"فشل حفظ الإعدادات: {e}")

cfg = {
    "initial_balance": float(st.session_state.initial_balance),
    "amount_usd": float(st.session_state.amount_usd),
    "enable_trading": bool(st.session_state.enable_trading),
    "use_sma": bool(st.session_state.use_sma),
    "sma_fast": int(st.session_state.sma_fast),
    "sma_slow": int(st.session_state.sma_slow),
    "use_bollinger": bool(st.session_state.use_bollinger),
    "bb_period": int(st.session_state.bb_period),
    "bb_std_factor": float(st.session_state.bb_std_factor),
    "use_rsi": bool(st.session_state.use_rsi),
    "rsi_period": int(st.session_state.rsi_period),
    "rsi_buy": int(st.session_state.rsi_buy),
    "rsi_sell": int(st.session_state.rsi_sell),
    "use_macd_filter": bool(st.session_state.use_macd_filter),
    "macd_fast": 12,
    "macd_slow": 26,
    "macd_signal": 9,
    "use_vwap_filter": bool(st.session_state.use_vwap_filter),
    "vwap_tolerance": float(st.session_state.vwap_tolerance_pct) / 100.0,
    "stop_loss_pct": float(st.session_state.stop_loss_pct) / 100.0,
    "take_profit_pct": float(st.session_state.take_profit_pct) / 100.0,
}

base_secs = infer_base_tf_seconds(ohlcv_raw)
st.caption(f"📌 الفريم الأصلي لملف ohlcv.csv تقريبًا: {human_tf(base_secs)}")

st.markdown("---")
tabs = st.tabs([f"{TF_LABELS[tf_key]}" for tf_key in ["5T", "15T", "4h"]])

for idx, tf_key in enumerate(["5T", "15T", "4h"]):
    label = TF_LABELS[tf_key]
    with tabs[idx]:
        df_tf, note = resample_ohlcv(ohlcv_raw, tf_key)

        if note:
            st.warning(note)

        if run_bt:
            df_signals, trades_df = run_backtest(df_tf, cfg)
        else:
            df_signals = generate_signals(df_tf, cfg)
            trades_df = pd.DataFrame(
                columns=["time", "type", "side", "entry_price", "exit_price", "amount", "pnl", "balance"]
            )

        stats = compute_stats(trades_df, float(st.session_state.initial_balance))

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Trades", f"{stats['total_trades']}")
        c2.metric("Win Rate", f"{stats['win_rate']:.1f}%")
        c3.metric("PnL %", f"{stats['total_return']:.2f}%")
        c4.metric("DD Max", f"{stats['max_drawdown']:.2f}%")

        fig = build_chart(df_signals, trades_df, f"{st.session_state.symbol_name} – {label}")
        st.plotly_chart(fig, width="stretch")

        st.caption(f"شموع حقيقية بعد إعادة التجميع على فريم: {label}")

        st.markdown("جدول الصفقات")
        if not trades_df.empty:
            st.dataframe(
                trades_df.sort_values("time").reset_index(drop=True),
                width="stretch",
                height=250
            )
        else:
            st.info("لا توجد صفقات لهذا الفريم مع الإعدادات الحالية.")
