import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# ---------------- محاولة استيراد ccxt ----------------
HAS_CCXT = True
try:
    import ccxt
except Exception:
    HAS_CCXT = False
    ccxt = None

# ---------------- إعداد صفحة ستريملت ----------------
st.set_page_config(page_title="Scalping Dashboard", layout="wide")
st.title("⚡ Crypto Scalping Dashboard – EMA / VWAP / Stoch / RSI / CVD / ADX")

if not HAS_CCXT:
    st.error("مكتبة ccxt غير مثبتة. نفّذ:\n\npip install ccxt\n\nثم أعد تشغيل التطبيق.")
    st.stop()

# ---------------- إعداد المنصة والرموز ----------------
EXCHANGE_ID = "binance"
SYMBOLS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "DOGE/USDT", "XRP/USDT"]

exchange_class = getattr(ccxt, EXCHANGE_ID)
exchange = exchange_class({"enableRateLimit": True})


# ---------------- دوال المؤشرات الفنية ----------------
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
    plus_dm_smooth = pd.Series(plus_dm, index=df.index).rolling(period, min_periods=period).sum()
    minus_dm_smooth = pd.Series(minus_dm, index=df.index).rolling(period, min_periods=period).sum()

    plus_di = 100 * (plus_dm_smooth / tr_smooth.replace(0, np.nan))
    minus_di = 100 * (minus_dm_smooth / tr_smooth.replace(0, np.nan))

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx_val = dx.rolling(period, min_periods=period).mean()
    return adx_val.fillna(0)


def vwap(df: pd.DataFrame) -> pd.Series:
    pv = df["close"] * df["volume"]
    cum_pv = pv.cumsum()
    cum_vol = df["volume"].cumsum().replace(0, np.nan)
    vw = cum_pv / cum_vol
    # استخدام bfill/ffill بدل method لتفادي FutureWarning
    vw = vw.bfill().ffill()
    return vw


def stochastic(df: pd.DataFrame, k_period: int = 5, d_period: int = 3) -> pd.DataFrame:
    low_min = df["low"].rolling(window=k_period, min_periods=1).min()
    high_max = df["high"].rolling(window=k_period, min_periods=1).max()
    stoch_k = 100 * (df["close"] - low_min) / (high_max - low_min).replace(0, np.nan)
    stoch_k = stoch_k.fillna(50)
    stoch_d = stoch_k.rolling(window=d_period, min_periods=1).mean()
    return pd.DataFrame({"stoch_k": stoch_k, "stoch_d": stoch_d})


def compute_cvd(df: pd.DataFrame) -> pd.Series:
    delta = (df["close"] - df["open"]) * df["volume"]
    cvd = delta.cumsum()
    if cvd.max() == cvd.min():
        return pd.Series(0.0, index=df.index)
    return cvd


# ---------------- جلب بيانات الشموع ----------------
def fetch_ohlcv(symbol: str, timeframe: str, candles: int = 500) -> pd.DataFrame:
    """
    جلب عدد ثابت من الشموع من باينانس.
    """
    data = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=candles)
    df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


# ---------------- حساب المؤشرات لكل فريم ----------------
def compute_indicators_for_tf(df: pd.DataFrame, rsi_period: int, adx_period: int, k_period: int, d_period: int) -> pd.DataFrame:
    d = df.copy()
    d["rsi"] = rsi(d["close"], rsi_period)
    d["ema50"] = ema(d["close"], 50)
    d["ema200"] = ema(d["close"], 200)
    d["vwap"] = vwap(d)
    stoch_df = stochastic(d, k_period=k_period, d_period=d_period)
    d["stoch_k"] = stoch_df["stoch_k"]
    d["stoch_d"] = stoch_df["stoch_d"]
    d["adx"] = adx(d, adx_period)
    d["cvd"] = compute_cvd(d)
    return d


# ---------------- باك تست وفق إعدادات السكالبينج ----------------
def run_scalping_backtest(
    df: pd.DataFrame,
    trend_dir_series: pd.Series,
    rsi_buy_min: float,
    rsi_buy_max: float,
    rsi_sell_min: float,
    rsi_sell_max: float,
    stoch_oversold: float,
    stoch_overbought: float,
    adx_min: float,
    vwap_dev_min: float,
    vwap_dev_max: float,
    sl_pct: float,
    tp_factor: float,
    initial_balance: float,
    risk_per_trade_pct: float,
    max_trades_per_day: int,
    daily_loss_limit_pct: float,
):
    """
    Backtest لصفقات Long و Short وفق القواعد:
    - اتجاه EMA50/EMA200 (trend_dir_series: 1 أو -1)
    - RSI Zones
    - Stoch Reversal
    - CVD sign
    - ADX Filter
    - VWAP Deviation
    - SL/TP + Max trades/day + Daily loss limit
    """
    df = df.copy()
    df["trend_dir"] = trend_dir_series.reindex(df.index).ffill().bfill()

    balance = float(initial_balance)
    position = None
    trades = []
    equity = []

    sl_factor = sl_pct / 100.0
    tp_factor_total = sl_factor * tp_factor  # TP = SL% * factor

    current_day = None
    trades_today = 0
    day_start_equity = balance
    stop_for_today = False

    for i in range(1, len(df)):
        row_prev = df.iloc[i - 1]
        row = df.iloc[i]

        ts = row["timestamp"]
        price = float(row["close"])
        high = float(row["high"])
        low = float(row["low"])
        rsi_now = float(row["rsi"])
        stoch_k_now = float(row["stoch_k"])
        stoch_k_prev = float(row_prev["stoch_k"])
        adx_now = float(row["adx"])
        vwap_now = float(row["vwap"])
        cvd_now = float(row["cvd"])
        trend_dir = int(row["trend_dir"]) if not np.isnan(row["trend_dir"]) else 0

        vwap_dev = abs((price - vwap_now) / vwap_now * 100) if vwap_now != 0 else 0.0

        # إدارة اليوم
        day = ts.date()
        if current_day is None or day != current_day:
            current_day = day
            trades_today = 0
            day_start_equity = balance
            stop_for_today = False

        # حد الخسارة اليومي
        day_change_pct = (balance - day_start_equity) / day_start_equity * 100 if day_start_equity > 0 else 0
        if day_change_pct <= -daily_loss_limit_pct:
            stop_for_today = True

        equity.append({"time": ts, "balance": balance})

        # إدارة صفقة مفتوحة
        if position is not None:
            side = position["side"]
            entry_price = position["entry_price"]
            qty = position["qty"]
            sl = position["sl"]
            tp = position["tp"]

            exit_reason = None
            exit_price = None

            if side == "long":
                if high >= tp:
                    exit_reason = "TP"
                    exit_price = tp
                elif low <= sl:
                    exit_reason = "SL"
                    exit_price = sl
                else:
                    if abs((price - vwap_now) / vwap_now * 100) < 0.05:
                        exit_reason = "VWAP"
                        exit_price = price
            else:  # short
                if low <= tp:
                    exit_reason = "TP"
                    exit_price = tp
                elif high >= sl:
                    exit_reason = "SL"
                    exit_price = sl
                else:
                    if abs((price - vwap_now) / vwap_now * 100) < 0.05:
                        exit_reason = "VWAP"
                        exit_price = price

            if exit_reason is not None:
                if side == "long":
                    pnl = (exit_price - entry_price) * qty
                else:
                    pnl = (entry_price - exit_price) * qty

                balance += pnl
                trades.append(
                    {
                        "entry_time": position["entry_time"],
                        "exit_time": ts,
                        "side": side,
                        "entry_price": entry_price,
                        "exit_price": exit_price,
                        "qty": qty,
                        "pnl": pnl,
                        "pnl_pct": pnl / (entry_price * qty) * 100 if entry_price * qty > 0 else 0,
                        "reason": exit_reason,
                        "balance": balance,
                    }
                )
                position = None

        if stop_for_today:
            continue

        if position is not None:
            continue

        if trades_today >= max_trades_per_day:
            continue

        # ---------- إشارات LONG / SHORT ----------
        long_signal = False
        short_signal = False

        if adx_now >= adx_min and vwap_dev_min <= vwap_dev <= vwap_dev_max:
            stoch_long_cond = (stoch_k_prev < stoch_oversold) and (stoch_k_now > stoch_oversold)
            stoch_short_cond = (stoch_k_prev > stoch_overbought) and (stoch_k_now < stoch_overbought)

            cvd_long = cvd_now > 0
            cvd_short = cvd_now < 0

            rsi_long_zone = (rsi_now >= rsi_buy_min) and (rsi_now <= rsi_buy_max)
            rsi_short_zone = (rsi_now >= rsi_sell_min) and (rsi_now <= rsi_sell_max)

            if trend_dir == 1 and stoch_long_cond and cvd_long and rsi_long_zone:
                long_signal = True
            if trend_dir == -1 and stoch_short_cond and cvd_short and rsi_short_zone:
                short_signal = True

        if long_signal or short_signal:
            risk_usd = balance * (risk_per_trade_pct / 100.0)
            if risk_usd <= 0 or price <= 0:
                continue

            qty = risk_usd / price
            entry_price = price

            if long_signal:
                side = "long"
                sl = entry_price * (1 - sl_factor)
                tp = entry_price * (1 + tp_factor_total)
            else:
                side = "short"
                sl = entry_price * (1 + sl_factor)
                tp = entry_price * (1 - tp_factor_total)

            position = {
                "side": side,
                "entry_price": entry_price,
                "qty": qty,
                "sl": sl,
                "tp": tp,
                "entry_time": ts,
            }
            trades_today += 1

    trades_df = pd.DataFrame(trades)
    equity_df = pd.DataFrame(equity)

    df["bt_mark"] = ""
    if not trades_df.empty:
        entry_times = set(pd.to_datetime(trades_df["entry_time"]))
        exit_times = set(pd.to_datetime(trades_df["exit_time"]))
        df.loc[df["timestamp"].isin(entry_times), "bt_mark"] = "ENTRY"
        df.loc[df["timestamp"].isin(exit_times), "bt_mark"] = "EXIT"

    if trades_df.empty:
        stats = {
            "trades_count": 0,
            "win_rate": 0.0,
            "total_return_pct": 0.0,
            "avg_pnl_pct": 0.0,
        }
    else:
        wins = trades_df[trades_df["pnl"] > 0]
        win_rate = len(wins) / len(trades_df) * 100 if len(trades_df) > 0 else 0
        total_return_pct = (equity_df["balance"].iloc[-1] - initial_balance) / initial_balance * 100
        avg_pnl_pct = trades_df["pnl_pct"].mean()

        stats = {
            "trades_count": int(len(trades_df)),
            "win_rate": float(win_rate),
            "total_return_pct": float(total_return_pct),
            "avg_pnl_pct": float(avg_pnl_pct),
        }

    return df, trades_df, equity_df, stats


# ---------------- Optimizer بسيط لـ 3m / 5m / 15m ----------------
def optimize_strategies_for_timeframes(
    symbol: str,
    tfs: list,
    base_rsi_cfg: dict,
    base_stoch_cfg: dict,
    adx_period: int,
    initial_balance: float,
    risk_per_trade_pct: float,
    max_trades_per_day: int,
    daily_loss_limit_pct: float,
):
    """
    يبحث عن أفضل كومبو (VWAP, SL, TP, ADX_min) لكل فريم.
    """
    results = []
    # فريم الترند ثابت 15m
    trend_tf = "15m"
    df_trend = fetch_ohlcv(symbol, trend_tf, candles=300)
    df_trend = compute_indicators_for_tf(
        df_trend,
        rsi_period=base_rsi_cfg["period"],
        adx_period=adx_period,
        k_period=base_stoch_cfg["k_period"],
        d_period=base_stoch_cfg["d_period"],
    )
    df_trend["trend_dir"] = np.where(df_trend["ema50"] > df_trend["ema200"], 1, -1)
    trend_small = df_trend[["timestamp", "trend_dir"]].copy()

    # نطاقات البحث
    VWAP_MIN_OPTIONS = [0.3, 0.5]
    VWAP_MAX_OPTIONS = [1.2, 1.8]
    SL_OPTIONS = [0.2, 0.35]
    TP_FACTORS = [2.0, 3.0]
    ADX_MIN_OPTIONS = [12.0, 18.0]

    for tf in tfs:
        df = fetch_ohlcv(symbol, tf, candles=800)
        df = compute_indicators_for_tf(
            df,
            rsi_period=base_rsi_cfg["period"],
            adx_period=adx_period,
            k_period=base_stoch_cfg["k_period"],
            d_period=base_stoch_cfg["d_period"],
        )
        df = pd.merge_asof(
            df.sort_values("timestamp"),
            trend_small.sort_values("timestamp"),
            on="timestamp",
            direction="backward",
        )

        for vmin in VWAP_MIN_OPTIONS:
            for vmax in VWAP_MAX_OPTIONS:
                if vmax <= vmin:
                    continue
                for sl in SL_OPTIONS:
                    for tp_fac in TP_FACTORS:
                        for adx_min in ADX_MIN_OPTIONS:
                            df_bt, trades_df, equity_df, stats = run_scalping_backtest(
                                df,
                                df["trend_dir"],
                                rsi_buy_min=base_rsi_cfg["buy_min"],
                                rsi_buy_max=base_rsi_cfg["buy_max"],
                                rsi_sell_min=base_rsi_cfg["sell_min"],
                                rsi_sell_max=base_rsi_cfg["sell_max"],
                                stoch_oversold=base_stoch_cfg["oversold"],
                                stoch_overbought=base_stoch_cfg["overbought"],
                                adx_min=adx_min,
                                vwap_dev_min=vmin,
                                vwap_dev_max=vmax,
                                sl_pct=sl,
                                tp_factor=tp_fac,
                                initial_balance=initial_balance,
                                risk_per_trade_pct=risk_per_trade_pct,
                                max_trades_per_day=max_trades_per_day,
                                daily_loss_limit_pct=daily_loss_limit_pct,
                            )
                            results.append(
                                {
                                    "symbol": symbol,
                                    "timeframe": tf,
                                    "vwap_min": vmin,
                                    "vwap_max": vmax,
                                    "sl_pct": sl,
                                    "tp_factor": tp_fac,
                                    "adx_min": adx_min,
                                    "trades": stats["trades_count"],
                                    "win_rate": stats["win_rate"],
                                    "total_return_pct": stats["total_return_pct"],
                                    "avg_pnl_pct": stats["avg_pnl_pct"],
                                }
                            )
    return pd.DataFrame(results)


# ---------------- إعدادات الواجهة (Sidebar) ----------------

st.sidebar.header("⚙️ إعدادات السكالبينج")

symbol = st.sidebar.selectbox("اختر العملة", SYMBOLS, index=1)

scalp_tf = st.sidebar.selectbox("فريم الدخول (Scalp TF)", ["1m", "3m"], index=1)
trend_tf = "15m"  # ثابت طبقًا لاستراتيجيتك

candles_scalp = st.sidebar.slider("عدد الشموع (الفريم السكالبينج)", 300, 2000, 800, 100)

st.sidebar.markdown("### RSI9 Zones")
rsi_period = 9
rsi_buy_min = st.sidebar.number_input("RSI Buy Min", 0.0, 100.0, 28.0, 1.0)
rsi_buy_max = st.sidebar.number_input("RSI Buy Max", 0.0, 100.0, 45.0, 1.0)
rsi_sell_min = st.sidebar.number_input("RSI Sell Min", 0.0, 100.0, 55.0, 1.0)
rsi_sell_max = st.sidebar.number_input("RSI Sell Max", 0.0, 100.0, 72.0, 1.0)

st.sidebar.markdown("### Stochastic (بديل Stoch RSI)")
k_period = st.sidebar.number_input("Stoch K Period", 3, 50, 5)
d_period = st.sidebar.number_input("Stoch D Period", 2, 50, 3)
stoch_oversold = st.sidebar.number_input("Oversold Level", 0.0, 100.0, 20.0, 1.0)
stoch_overbought = st.sidebar.number_input("Overbought Level", 0.0, 100.0, 80.0, 1.0)

st.sidebar.markdown("### ADX")
adx_period = st.sidebar.number_input("ADX Period", 5, 50, 14)
adx_min = st.sidebar.number_input("ADX Min (قوة الاتجاه)", 0.0, 100.0, 18.0, 1.0)

st.sidebar.markdown("### VWAP Deviation (%)")
vwap_dev_min = st.sidebar.number_input("VWAP Min %", 0.0, 5.0, 0.5, 0.1)
vwap_dev_max = st.sidebar.number_input("VWAP Max %", 0.1, 10.0, 1.2, 0.1)

st.sidebar.markdown("### Risk / Money Management")
initial_balance = st.sidebar.number_input("Initial Balance (USDT)", 100.0, 100000.0, 1000.0, 50.0)
risk_per_trade_pct = st.sidebar.number_input("Risk per Trade %", 0.1, 5.0, 1.0, 0.1)
max_trades_per_day = st.sidebar.number_input("Max Trades per Day", 1, 100, 30, 1)
daily_loss_limit_pct = st.sidebar.number_input("Daily Loss Limit %", 0.1, 10.0, 2.0, 0.1)

st.sidebar.markdown("### SL / TP")
sl_pct = st.sidebar.number_input("Stop Loss %", 0.05, 2.0, 0.2, 0.05)
tp_factor = st.sidebar.number_input("TP Factor (x SL)", 1.0, 5.0, 2.0, 0.5)

run_btn = st.sidebar.button("🚀 Run Scan + Backtest")

# زر خاص بالأوبتيميزر
st.sidebar.markdown("---")
optimize_btn = st.sidebar.button("🔍 ابحث عن أفضل إعدادات على 3m / 5m / 15m")


# ---------------- Tabs لعرض النتائج ----------------
tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "📈 Chart & Signals", "🧪 Backtest Results", "🔍 Strategy Optimizer"])


# ================== تشغيل Scan + Backtest عادي ==================
if run_btn:
    with st.spinner("⏳ جاري جلب البيانات من Binance وحساب المؤشرات..."):
        df_scalp = fetch_ohlcv(symbol, scalp_tf, candles=candles_scalp)
        df_scalp = compute_indicators_for_tf(df_scalp, rsi_period, adx_period, k_period, d_period)

        df_trend = fetch_ohlcv(symbol, trend_tf, candles=300)
        df_trend = compute_indicators_for_tf(df_trend, rsi_period, adx_period, k_period, d_period)
        df_trend["trend_dir"] = np.where(df_trend["ema50"] > df_trend["ema200"], 1, -1)
        df_trend_small = df_trend[["timestamp", "trend_dir"]].copy()

        df_scalp = pd.merge_asof(
            df_scalp.sort_values("timestamp"),
            df_trend_small.sort_values("timestamp"),
            on="timestamp",
            direction="backward",
        )

        df_bt, trades_df, equity_df, stats = run_scalping_backtest(
            df_scalp,
            df_scalp["trend_dir"],
            rsi_buy_min=rsi_buy_min,
            rsi_buy_max=rsi_buy_max,
            rsi_sell_min=rsi_sell_min,
            rsi_sell_max=rsi_sell_max,
            stoch_oversold=stoch_oversold,
            stoch_overbought=stoch_overbought,
            adx_min=adx_min,
            vwap_dev_min=vwap_dev_min,
            vwap_dev_max=vwap_dev_max,
            sl_pct=sl_pct,
            tp_factor=tp_factor,
            initial_balance=initial_balance,
            risk_per_trade_pct=risk_per_trade_pct,
            max_trades_per_day=max_trades_per_day,
            daily_loss_limit_pct=daily_loss_limit_pct,
        )

    # ---------------- Overview Tab ----------------
    with tab1:
        st.subheader(f"📊 نظرة عامة – {symbol} – Scalping {scalp_tf}")

        last = df_bt.iloc[-1]
        price_now = float(last["close"])
        rsi_now = float(last["rsi"])
        stoch_k_now = float(last["stoch_k"])
        adx_now = float(last["adx"])
        vwap_now = float(last["vwap"])
        trend_dir_now = int(last["trend_dir"]) if not np.isnan(last["trend_dir"]) else 0
        cvd_now = float(last["cvd"])
        vwap_dev_now = abs((price_now - vwap_now) / vwap_now * 100) if vwap_now != 0 else 0.0

        current_long = (
            trend_dir_now == 1
            and vwap_dev_min <= vwap_dev_now <= vwap_dev_max
            and adx_now >= adx_min
            and cvd_now > 0
            and rsi_buy_min <= rsi_now <= rsi_buy_max
            and stoch_k_now < stoch_overbought
        )
        current_short = (
            trend_dir_now == -1
            and vwap_dev_min <= vwap_dev_now <= vwap_dev_max
            and adx_now >= adx_min
            and cvd_now < 0
            and rsi_sell_min <= rsi_now <= rsi_sell_max
            and stoch_k_now > stoch_oversold
        )

        if current_long:
            live_signal = "✅ Long Bias"
        elif current_short:
            live_signal = "✅ Short Bias"
        else:
            live_signal = "⏸ No Clear Trade"

        c1, c2, c3 = st.columns(3)
        c1.metric("Current Price", f"{price_now:.4f}")
        c2.metric("VWAP", f"{vwap_now:.4f}", f"{vwap_dev_now:.2f}% from VWAP")
        c3.metric("Trend (15m)", "Bullish" if trend_dir_now == 1 else "Bearish", live_signal)

        c4, c5, c6 = st.columns(3)
        c4.metric("RSI(9)", f"{rsi_now:.1f}")
        c5.metric("Stoch K", f"{stoch_k_now:.1f}")
        c6.metric("ADX", f"{adx_now:.1f}")

        st.markdown(f"**Live Scalping Signal:** {live_signal}")

        st.markdown("### Backtest Summary")
        if stats["trades_count"] == 0:
            st.warning("لا توجد صفقات في الباك تست بهذه الإعدادات.")
        else:
            c7, c8, c9, c10 = st.columns(4)
            c7.metric("Trades", stats["trades_count"])
            c8.metric("Win Rate %", f"{stats['win_rate']:.1f}")
            c9.metric("Total Return %", f"{stats['total_return_pct']:.2f}")
            c10.metric("Avg PnL per Trade %", f"{stats['avg_pnl_pct']:.2f}")

    # ---------------- Chart & Signals Tab (TradingView-style) ----------------
    with tab2:
        st.subheader("📈 شارت السكالبينج – Candles + EMA + VWAP + Entries/Exits")

        df_view = df_bt.tail(300)

        fig = go.Figure()

        # Candles
        fig.add_trace(
            go.Candlestick(
                x=df_view["timestamp"],
                open=df_view["open"],
                high=df_view["high"],
                low=df_view["low"],
                close=df_view["close"],
                name="Price",
                increasing_line_color="#00C853",
                decreasing_line_color="#FF3D00",
            )
        )

        # EMA50 / EMA200 / VWAP
        fig.add_trace(
            go.Scatter(
                x=df_view["timestamp"],
                y=df_view["ema50"],
                mode="lines",
                name="EMA50",
                line=dict(color="#2962FF", width=1.5),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=df_view["timestamp"],
                y=df_view["ema200"],
                mode="lines",
                name="EMA200",
                line=dict(color="#FF6D00", width=1.5),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=df_view["timestamp"],
                y=df_view["vwap"],
                mode="lines",
                name="VWAP",
                line=dict(color="#FFB300", width=1.5, dash="dot"),
            )
        )

        # Entry / Exit markers
        entries = df_view[df_view["bt_mark"] == "ENTRY"]
        exits = df_view[df_view["bt_mark"] == "EXIT"]

        fig.add_trace(
            go.Scatter(
                x=entries["timestamp"],
                y=entries["close"],
                mode="markers",
                name="ENTRY",
                marker=dict(symbol="triangle-up", size=12, line=dict(width=1), color="#00E676"),
            )
        )

        fig.add_trace(
            go.Scatter(
                x=exits["timestamp"],
                y=exits["close"],
                mode="markers",
                name="EXIT",
                marker=dict(symbol="triangle-down", size=12, line=dict(width=1), color="#FF5252"),
            )
        )

        fig.update_layout(
            height=650,
            xaxis_title="Time",
            yaxis_title="Price",
            xaxis_rangeslider_visible=False,
            template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=40, r=20, t=40, b=40),
        )

        st.plotly_chart(fig, width="stretch")

    # ---------------- Backtest Results Tab ----------------
    with tab3:
        st.subheader("🧪 نتائج الباك تست (Scalping Rules)")

        if trades_df.empty:
            st.warning("لا توجد صفقات مسجلة لهذه الإعدادات.")
        else:
            st.markdown("**سجل الصفقات:**")
            st.dataframe(trades_df, width="stretch")

            st.markdown("**منحنى الرصيد:**")
            eq = equity_df.set_index("time")
            st.line_chart(eq["balance"])


# ================== Strategy Optimizer Tab ==================
with tab4:
    st.subheader("🔍 Strategy Optimizer – البحث عن أفضل إعدادات على 3m / 5m / 15m")

    st.markdown(
        """
        هذا الجزء يقوم بعمل Backtest أوتوماتيكي على الفريمات:
        **3m – 5m – 15m** لنفس العملة،
        ويبحث في مجموعة إعدادات مختلفة (VWAP / SL / TP / ADX)،
        ثم يعرض أفضل النتائج حسب **Total Return %**.
        """
    )

    if not optimize_btn:
        st.info("اضغط على زر 🔍 في الشريط الجانبي لبدء البحث عن أفضل إعدادات.")
    else:
        with st.spinner("⏳ جاري تشغيل الأوبتيميزر على 3m / 5m / 15m..."):
            base_rsi_cfg = {
                "period": rsi_period,
                "buy_min": rsi_buy_min,
                "buy_max": rsi_buy_max,
                "sell_min": rsi_sell_min,
                "sell_max": rsi_sell_max,
            }
            base_stoch_cfg = {
                "k_period": k_period,
                "d_period": d_period,
                "oversold": stoch_oversold,
                "overbought": stoch_overbought,
            }

            tfs_to_optimize = ["3m", "5m", "15m"]
            opt_df = optimize_strategies_for_timeframes(
                symbol,
                tfs_to_optimize,
                base_rsi_cfg,
                base_stoch_cfg,
                adx_period=adx_period,
                initial_balance=initial_balance,
                risk_per_trade_pct=risk_per_trade_pct,
                max_trades_per_day=max_trades_per_day,
                daily_loss_limit_pct=daily_loss_limit_pct,
            )

        if opt_df.empty:
            st.warning("لم يتم تسجيل أي صفقات في الأوبتيميزر بهذه النطاقات. جرّب توسيع الإعدادات.")
        else:
            st.success("تم الانتهاء من الأوبتيميزر. هذه أفضل النتائج:")

            # ترتيب من الأعلى للأدنى حسب العائد
            opt_sorted = opt_df.sort_values("total_return_pct", ascending=False).reset_index(drop=True)

            # أفضل 5 فقط لعرض مختصر
            top5 = opt_sorted.head(5)

            st.markdown("### 🏆 أفضل 5 إعدادات إجمالًا")
            st.dataframe(top5, width="stretch")

            # عرض أفضل إعداد لكل فريم
            st.markdown("### 📌 أفضل إعداد لكل فريم (3m / 5m / 15m)")
            best_per_tf = opt_sorted.groupby("timeframe").head(1).reset_index(drop=True)
            st.dataframe(best_per_tf, width="stretch")

            st.markdown(
                """
                يمكنك أخذ الإعدادات (VWAP Min/Max, SL, TP Factor, ADX Min)
                من الصفوف الأعلى، وكتابتها يدويًا في إعدادات السكالبينج في الشريط الجانبي،
                ثم إعادة تشغيل الباك تست على الفريم الذي تريده.
                """
            )
