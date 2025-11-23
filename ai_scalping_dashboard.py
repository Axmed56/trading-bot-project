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
st.set_page_config(page_title="AI Scalping Optimizer", layout="wide")
st.title("⚡ Crypto AI Scalping Dashboard – EMA / VWAP / Stoch / RSI / CVD / ADX")

if not HAS_CCXT:
    st.error("مكتبة ccxt غير مثبتة. نفّذ:\n\npip install ccxt\n\nثم أعد تشغيل التطبيق.")
    st.stop()

# ---------------- إعداد المنصة ----------------
EXCHANGE_ID = "binance"
exchange_class = getattr(ccxt, EXCHANGE_ID)
exchange = exchange_class({"enableRateLimit": True})

# أفضل 15 عملة USDT تقريبية (تقدر تعدلها لاحقًا)
TOP_15_SYMBOLS = [
    "BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "XRP/USDT",
    "ADA/USDT", "DOGE/USDT", "TON/USDT", "LINK/USDT", "TRX/USDT",
    "AVAX/USDT", "NEAR/USDT", "OP/USDT", "LTC/USDT", "UNI/USDT"
]

SCALP_TFS = ["1m", "3m"]
TREND_TF = "15m"
OPT_TFS = ["3m", "5m", "15m"]  # الفريمات التي سيتم الأوبتيميز عليها

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
    # لتفادي FutureWarning:
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

# ---------------- جلب بيانات OHLCV (مع كاش) ----------------
@st.cache_data(show_spinner=False)
def fetch_ohlcv(symbol: str, timeframe: str, candles: int = 800) -> pd.DataFrame:
    data = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=candles)
    df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df

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

# ---------------- دالة الباك تست (كما بنيناها) ----------------
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
    df = df.copy()
    df["trend_dir"] = trend_dir_series.reindex(df.index).ffill().bfill()

    balance = float(initial_balance)
    position = None
    trades = []
    equity = []

    sl_factor = sl_pct / 100.0
    tp_factor_total = sl_factor * tp_factor

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

        day_change_pct = (balance - day_start_equity) / day_start_equity * 100 if day_start_equity > 0 else 0
        if day_change_pct <= -daily_loss_limit_pct:
            stop_for_today = True

        equity.append({"time": ts, "balance": balance})

        # إدارة مركز مفتوح
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
            else:
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

        # إشارات LONG / SHORT
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

# ---------------- Global AI Optimizer (أفضل 15 عملة × 3 فريمات) ----------------
def ai_global_optimizer(
    symbols,
    timeframes,
    total_trials,
    base_rsi_cfg,
    base_stoch_cfg,
    adx_period,
    initial_balance,
    risk_per_trade_pct,
    max_trades_per_day,
    daily_loss_limit_pct,
):
    """
    توزيع ~total_trials على (symbol, timeframe) عشوائيًا،
    وتجربة إعدادات مختلفة (VWAP / SL / TP / ADX_min)،
    ثم إرجاع جدول بالنتائج.
    """
    results = []

    # تجهيز بيانات الترند (15m) مرة واحدة لكل رمز
    trend_data = {}
    for sym in symbols:
        df_trend = fetch_ohlcv(sym, TREND_TF, candles=400)
        df_trend = compute_indicators_for_tf(
            df_trend,
            rsi_period=base_rsi_cfg["period"],
            adx_period=adx_period,
            k_period=base_stoch_cfg["k_period"],
            d_period=base_stoch_cfg["d_period"],
        )
        df_trend["trend_dir"] = np.where(df_trend["ema50"] > df_trend["ema200"], 1, -1)
        trend_data[sym] = df_trend[["timestamp", "trend_dir"]].copy()

    # تجهيز بيانات السعر لكل رمز/فريم
    market_data = {}
    for sym in symbols:
        for tf in timeframes:
            df = fetch_ohlcv(sym, tf, candles=800)
            df = compute_indicators_for_tf(
                df,
                rsi_period=base_rsi_cfg["period"],
                adx_period=adx_period,
                k_period=base_stoch_cfg["k_period"],
                d_period=base_stoch_cfg["d_period"],
            )
            trend_df = trend_data[sym]
            df_merged = pd.merge_asof(
                df.sort_values("timestamp"),
                trend_df.sort_values("timestamp"),
                on="timestamp",
                direction="backward",
            )
            market_data[(sym, tf)] = df_merged

    n_combos = len(symbols) * len(timeframes)
    trials_per_combo = max(1, total_trials // n_combos)

    progress_bar = st.progress(0)
    total_loops = n_combos * trials_per_combo
    loop_counter = 0

    for sym in symbols:
        for tf in timeframes:
            df_sym = market_data[(sym, tf)]

            for _ in range(trials_per_combo):
                loop_counter += 1
                progress_bar.progress(min(loop_counter / total_loops, 1.0))

                # عشوائية المعاملات داخل نطاقات منطقية
                vmin = float(np.round(np.random.uniform(0.2, 1.5), 2))
                vmax = float(np.round(np.random.uniform(vmin + 0.3, vmin + 2.0), 2))
                vmax = min(vmax, 4.0)

                sl = float(np.round(np.random.uniform(0.1, 0.5), 2))  # 0.1% – 0.5%
                tp_fac = float(np.round(np.random.uniform(1.5, 3.5), 2))
                adx_min = float(np.round(np.random.uniform(10.0, 30.0), 1))

                df_bt, trades_df, equity_df, stats = run_scalping_backtest(
                    df_sym,
                    df_sym["trend_dir"],
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

                if stats["trades_count"] < 10:
                    # تجاهل الاستراتيجيات اللي عدد صفقاتها قليل جدا
                    continue

                # Score بسيط يفضّل الربحية و الـ Win Rate ويعاقب Drawdown لو حبينا نضيفه لاحقًا
                score = (stats["win_rate"] * 0.6) + (stats["total_return_pct"] * 0.3) + (stats["avg_pnl_pct"] * 0.1)

                results.append(
                    {
                        "symbol": sym,
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
                        "score": score,
                    }
                )

    progress_bar.progress(1.0)
    return pd.DataFrame(results)

# ---------------- واجهة الإعدادات (Sidebar) ----------------
st.sidebar.header("⚙️ إعدادات السكالبينج الأساسية")

symbol = st.sidebar.selectbox("اختر عملة للمشاهدة (للشارت فقط)", TOP_15_SYMBOLS, index=1)
scalp_tf = st.sidebar.selectbox("فريم الدخول (Scalp TF)", SCALP_TFS, index=1)
candles_scalp = st.sidebar.slider("عدد الشموع (الفريم السكالبينج)", 300, 2000, 800, 100)

st.sidebar.markdown("### RSI9 Zones")
rsi_period = 9
rsi_buy_min = st.sidebar.number_input("RSI Buy Min", 0.0, 100.0, 28.0, 1.0)
rsi_buy_max = st.sidebar.number_input("RSI Buy Max", 0.0, 100.0, 45.0, 1.0)
rsi_sell_min = st.sidebar.number_input("RSI Sell Min", 0.0, 100.0, 55.0, 1.0)
rsi_sell_max = st.sidebar.number_input("RSI Sell Max", 0.0, 100.0, 72.0, 1.0)

st.sidebar.markdown("### Stochastic")
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

st.sidebar.markdown("---")
run_btn = st.sidebar.button("🚀 Run Scan + Backtest (للعملة المختارة)")
st.sidebar.markdown("---")
ai_trials = 5000  # حسب اختيارك
ai_btn = st.sidebar.button(f"🤖 تشغيل AI Optimizer – أفضل 15 عملة – {ai_trials} محاولة")

# ---------------- Tabs ----------------
tab1, tab2, tab3 = st.tabs(["📊 Overview & Chart", "🧪 Backtest (Single Symbol)", "🤖 AI Global Optimizer"])

# ---------- Scan + Backtest لعملة واحدة (نفس فكرة الداش القديم) ----------
if run_btn:
    with st.spinner("⏳ جاري جلب البيانات من Binance وحساب المؤشرات..."):
        df_scalp = fetch_ohlcv(symbol, scalp_tf, candles=candles_scalp)
        df_scalp = compute_indicators_for_tf(df_scalp, rsi_period, adx_period, k_period, d_period)

        df_trend = fetch_ohlcv(symbol, TREND_TF, candles=400)
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

        st.subheader("📈 الشارت (Candles + EMA + VWAP + Entries/Exits)")
        df_view = df_bt.tail(300)

        fig = go.Figure()
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

        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("🧪 نتائج الباك تست (للعملة المختارة)")
        if trades_df.empty:
            st.warning("لا توجد صفقات مسجلة لهذه الإعدادات.")
        else:
            st.markdown("**سجل الصفقات:**")
            st.dataframe(trades_df, use_container_width=True)
            st.markdown("**منحنى الرصيد:**")
            eq = equity_df.set_index("time")
            st.line_chart(eq["balance"])

# ---------- AI Global Optimizer ----------
with tab3:
    st.subheader("🤖 AI Global Optimizer – أفضل 15 عملة × 3 فريمات (3m / 5m / 15m)")

    st.markdown(
        f"""
        هذا المحرك يقوم بعمل **بحث آلي** على أفضل 15 عملة USDT،
        وعلى فريمات: **3m / 5m / 15m**،
        مع حوالي **{ai_trials} محاولة** مختلفة لإعدادات (VWAP / SL / TP / ADX).
        
        النتيجة:
        - أفضل الاستراتيجيات مرتبة حسب **Score** (Win Rate + الربحية + جودة الصفقة).
        - أفضل إعداد لكل عملة/فريم.
        """
    )

    if ai_btn:
        with st.spinner("⏳ جاري تشغيل الأوبتيميزر على أفضل 15 عملة..."):
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

            opt_df = ai_global_optimizer(
                symbols=TOP_15_SYMBOLS,
                timeframes=OPT_TFS,
                total_trials=ai_trials,
                base_rsi_cfg=base_rsi_cfg,
                base_stoch_cfg=base_stoch_cfg,
                adx_period=adx_period,
                initial_balance=initial_balance,
                risk_per_trade_pct=risk_per_trade_pct,
                max_trades_per_day=max_trades_per_day,
                daily_loss_limit_pct=daily_loss_limit_pct,
            )

        if opt_df.empty:
            st.warning("لم يتم تسجيل أي نتيجة نافعة (عدد صفقات قليل جدًا). وسّع النطاقات أو قلل القيود.")
        else:
            st.success("✅ تم الانتهاء من الأوبتيميزر. النتائج أدناه:")

            # ترتيب حسب score
            opt_sorted = opt_df.sort_values("score", ascending=False).reset_index(drop=True)

            st.markdown("### 🏆 أفضل 10 استراتيجيات عالميًا (كل العملات، كل الفريمات)")
            st.dataframe(opt_sorted.head(10), use_container_width=True)

            st.markdown("### 📌 أفضل إعداد لكل (عملة / فريم)")
            best_per_pair_tf = opt_sorted.sort_values("score", ascending=False).groupby(
                ["symbol", "timeframe"], as_index=False
            ).first()
            st.dataframe(best_per_pair_tf, use_container_width=True)

            st.markdown(
                """
                يمكنك نسخ قيم:
                - vwap_min / vwap_max  
                - sl_pct / tp_factor  
                - adx_min  

                ولصقها في إعدادات السكالبينج في الشريط الجانبي، ثم تشغيل الباك تست 
                على العملة والفريم المطلوبين لمراجعة النتائج على الشارت بالتفصيل.
                """
            )
    else:
        st.info("اضغط على زر 🤖 AI Optimizer من الشريط الجانبي لبدء البحث على أفضل 15 عملة.")
