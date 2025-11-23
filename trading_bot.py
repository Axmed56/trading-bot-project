import pandas as pd
import numpy as np
from datetime import datetime
import os

# ================= إعدادات البوت الأساسية =================

DEFAULT_CONFIG = {
    "symbol": "BTC/USDT",
    "timeframe": "1h",
    "initial_balance": 1000.0,   # رصيد افتراضي يبدأ منه البوت
    "amount_usd": 50.0,          # قيمة الصفقة الواحدة بالدولار
    "sma_fast": 10,
    "sma_slow": 30,
    "rsi_period": 14,
    "rsi_buy": 40,               # فلتر شراء بالـ RSI
    "rsi_sell": 60,              # فلتر بيع بالـ RSI
    "stop_loss_pct": 0.02,       # 2% وقف خسارة
    "take_profit_pct": 0.04      # 4% جني ربح
}


# ================= مؤشرات فنية بسيطة =================

def sma(series, period):
    return series.rolling(period, min_periods=1).mean()


def rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).fillna(0)
    loss = -1 * delta.clip(upper=0).fillna(0)
    avg_gain = gain.rolling(period, min_periods=1).mean()
    avg_loss = loss.rolling(period, min_periods=1).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi_series = 100 - (100 / (1 + rs))
    return rsi_series.fillna(50)


# ================= تحميل / إنشاء بيانات OHLCV =================

def make_synthetic_ohlcv(path="ohlcv.csv", periods=300):
    """
    إنشاء بيانات صناعية لو مفيش ملف ohlcv.csv
    """
    base = 30000.0
    timestamps = [pd.Timestamp('2025-01-01') + pd.Timedelta(hours=i) for i in range(periods)]
    prices = (np.sin(np.linspace(0, 8 * np.pi, periods)) * 500) + base + np.linspace(-300, 300, periods)

    df = pd.DataFrame({
        "timestamp": timestamps,
        "open": prices + np.random.normal(0, 10, periods),
        "high": prices + np.random.normal(20, 10, periods),
        "low":  prices - np.random.normal(20, 10, periods),
        "close": prices + np.random.normal(0, 10, periods),
        "volume": np.random.randint(10, 1000, periods)
    })

    df.to_csv(path, index=False, date_format="%Y-%m-%d %H:%M:%S")
    print(f"✅ تم إنشاء بيانات صناعية في الملف: {path}")
    return df


def load_ohlcv(path="ohlcv.csv"):
    """
    تحميل بيانات OHLCV من CSV، أو إنشاء بيانات صناعية لو الملف غير موجود
    """
    if not os.path.exists(path):
        print(f"⚠️ الملف {path} غير موجود، سيتم إنشاء بيانات صناعية للاختبار...")
        return make_synthetic_ohlcv(path)

    df = pd.read_csv(path, parse_dates=["timestamp"])
    required = ["timestamp", "open", "high", "low", "close", "volume"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"❌ ملف {path} لا يحتوي على العمود المطلوب: {col}")
    print(f"✅ تم تحميل {len(df)} شمعة من {path}")
    return df


# ================= توليد الإشارات =================

def generate_signals(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    df = df.copy()
    df["sma_fast"] = sma(df["close"], cfg["sma_fast"])
    df["sma_slow"] = sma(df["close"], cfg["sma_slow"])
    df["rsi"] = rsi(df["close"], cfg["rsi_period"])

    df["signal"] = 0

    cross_up = (df["sma_fast"] > df["sma_slow"]) & (df["sma_fast"].shift(1) <= df["sma_slow"].shift(1))
    cross_down = (df["sma_fast"] < df["sma_slow"]) & (df["sma_fast"].shift(1) >= df["sma_slow"].shift(1))

    df.loc[cross_up & (df["rsi"] > cfg["rsi_buy"]), "signal"] = 1
    df.loc[cross_down & (df["rsi"] < cfg["rsi_sell"]), "signal"] = -1

    return df


# ================= حساب حجم الصفقة =================

def calculate_position_size(price: float, amount_usd: float) -> float:
    if price <= 0:
        return 0.0
    return amount_usd / price


# ================= محرك المحاكاة (Paper Trading Backtest) =================

def run_paper_backtest(ohlcv_path="ohlcv.csv", trades_out_path="backtest_trades.csv", cfg=None):
    if cfg is None:
        cfg = DEFAULT_CONFIG

    df = load_ohlcv(ohlcv_path)
    df = generate_signals(df, cfg)

    balance = float(cfg.get("initial_balance", 1000.0))
    print(f"💰 الرصيد الابتدائي (افتراضي): {balance:.2f} USDT")

    position = None   # إما dict فيه تفاصيل الصفقة المفتوحة أو None
    trades = []       # قائمة الصفقات المنفذة

    for i in range(len(df)):
        row = df.iloc[i]
        price = float(row["close"])
        signal = int(row["signal"])
        ts = row["timestamp"]

        # إدارة مركز مفتوح (لو موجود)
        if position is not None:
            side = position["side"]
            entry_price = position["entry_price"]
            amount = position["amount"]
            sl = position["stop_loss"]
            tp = position["take_profit"]

            # تحقق من SL/TP
            exit_reason = None
            if side == "buy":
                if price <= sl:
                    exit_reason = "STOPLOSS"
                elif price >= tp:
                    exit_reason = "TAKEPROFIT"
            else:  # sell
                if price >= sl:
                    exit_reason = "STOPLOSS"
                elif price <= tp:
                    exit_reason = "TAKEPROFIT"

            if exit_reason is not None:
                # حساب الربح/الخسارة
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

                print(f"📤 {exit_reason} | {side} @ {entry_price:.2f} → {price:.2f} | pnl={pnl:.2f} | balance={balance:.2f}")
                position = None
                continue  # ننتقل للشمعة التالية بعد الإغلاق

        # فتح مركز جديد لو مفيش صفقة مفتوحة
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
                "amount": amount,
                "pnl": 0.0,
                "balance": balance
            })

            print(f"📥 ENTRY {side.upper()} @ {price:.2f} | amount={amount:.6f} | SL={sl:.2f} | TP={tp:.2f}")

    # تحويل الصفقات إلى DataFrame وحفظها
    trades_df = pd.DataFrame(trades)
    trades_df.to_csv(trades_out_path, index=False, date_format="%Y-%m-%d %H:%M:%S")
    print(f"\n✅ تم حفظ نتائج المحاكاة في: {trades_out_path}")
    print(f"🔚 الرصيد النهائي: {balance:.2f} USDT")

    return trades_df


# ================= نقطة الدخول الرئيسية =================

if __name__ == "__main__":
    print("🚀 تشغيل محاكاة التداول (Paper Trading) برصيد افتراضي...")
    run_paper_backtest()
