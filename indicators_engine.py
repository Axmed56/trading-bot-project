import json
import time
import logging
from collections import deque
from typing import Dict, Deque, List

# =========================
# إعداد اللوج
# =========================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
)

LIVE_FEED_FILE = "live_feed.json"
METRICS_FILE = "metrics.json"

# الأزواج التي نراقبها
WATCHED_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]

# الفريمات بالثواني
TIMEFRAMES = {
    "1m": 60,
    "5m": 5 * 60,
    "15m": 15 * 60,
}

# عدد الشموع القصوى لكل فريم (تاريخ داخلي فقط)
MAX_BARS = 400

# price_history[symbol][tf] = deque([{"time": ts, "close": price}, ...])
price_history: Dict[str, Dict[str, Deque[dict]]] = {
    s: {tf: deque(maxlen=MAX_BARS) for tf in TIMEFRAMES.keys()}
    for s in WATCHED_SYMBOLS
}

# last_bucket[symbol][tf] = رقم الباكت الحالي (دقيقة/5 دقائق/15...)
last_bucket: Dict[str, Dict[str, int]] = {
    s: {tf: None for tf in TIMEFRAMES.keys()}
    for s in WATCHED_SYMBOLS
}

# =========================
# دوال المؤشرات
# =========================

def calc_ema(values: List[float], period: int) -> float:
    """حساب EMA لآخر قيمة فقط."""
    if len(values) < period:
        return None
    k = 2 / (period + 1)
    ema = values[0]
    for v in values[1:]:
        ema = v * k + ema * (1 - k)
    return ema


def calc_rsi(values: List[float], period: int = 14) -> float:
    """حساب RSI لآخر قيمة فقط."""
    if len(values) <= period:
        return None

    gains = []
    losses = []
    for i in range(1, len(values)):
        change = values[i] - values[i - 1]
        if change > 0:
            gains.append(change)
            losses.append(0.0)
        else:
            gains.append(0.0)
            losses.append(-change)

    avg_gain = sum(gains[-period:]) / period
    avg_loss = sum(losses[-period:]) / period

    if avg_loss == 0:
        return 100.0

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def build_multi_tf_summary(tf_data: Dict[str, dict]) -> dict:
    """
    تلخيص إشارة الـ AI من الفريمات الثلاثة.
    نعتمد 5m كأساس، ونستخدم 1m و 15m كفلتر.
    """
    base = tf_data.get("5m", {})
    rsi5 = base.get("rsi")
    ema9_5 = base.get("ema9")
    ema26_5 = base.get("ema26")

    if rsi5 is None or ema9_5 is None or ema26_5 is None:
        return {
            "trend": None,
            "signal": None,
            "confidence": None,
            "source_tf": "5m",
        }

    trend = "Sideways"
    signal = "HOLD"
    confidence = 50.0

    if ema9_5 > ema26_5 and rsi5 > 55:
        trend = "Bullish"
        signal = "BUY"
        confidence = min(95.0, 55.0 + (rsi5 - 55) * 1.2)
    elif ema9_5 < ema26_5 and rsi5 < 45:
        trend = "Bearish"
        signal = "SELL"
        confidence = min(95.0, 55.0 + (45 - rsi5) * 1.2)

    # تعزيز الثقة إذا 1m و 15m متفقين مع 5m
    tf1 = tf_data.get("1m", {})
    tf15 = tf_data.get("15m", {})

    same_dir = 0
    for tf in (tf1, tf15):
        if tf.get("trend") == trend and tf.get("signal") == signal:
            same_dir += 1

    if same_dir == 1:
        confidence += 5.0
    elif same_dir == 2:
        confidence += 10.0

    confidence = max(0.0, min(99.0, confidence))

    return {
        "trend": trend,
        "signal": signal,
        "confidence": round(confidence, 1),
        "source_tf": "5m",
    }


def update_price_history(symbol: str, ts: float, price: float):
    """تحديث الهستوري لكل فريم بناءً على الـ timestamp."""
    for tf_name, tf_seconds in TIMEFRAMES.items():
        bucket = int(ts // tf_seconds)

        if last_bucket[symbol][tf_name] is None:
            # أول مرة
            last_bucket[symbol][tf_name] = bucket
            price_history[symbol][tf_name].append({"time": ts, "close": price})
        else:
            if bucket == last_bucket[symbol][tf_name]:
                # نفس الباكت → نحدّث آخر شمعة
                if price_history[symbol][tf_name]:
                    price_history[symbol][tf_name][-1]["close"] = price
            else:
                # دخلنا في باكت جديد → نفتح شمعة جديدة
                last_bucket[symbol][tf_name] = bucket
                price_history[symbol][tf_name].append({"time": ts, "close": price})


def compute_metrics_for_symbol(symbol: str) -> dict:
    """حساب مؤشرات الفريمات الثلاثة + الملخص."""
    tf_metrics = {}

    for tf_name in TIMEFRAMES.keys():
        closes = [bar["close"] for bar in price_history[symbol][tf_name]]
        if len(closes) < 30:
            tf_metrics[tf_name] = {
                "rsi": None,
                "ema9": None,
                "ema26": None,
                "trend": None,
                "signal": None,
            }
            continue

        rsi = calc_rsi(closes, period=14)
        ema9 = calc_ema(closes, period=9)
        ema26 = calc_ema(closes, period=26)

        if rsi is None or ema9 is None or ema26 is None:
            tf_metrics[tf_name] = {
                "rsi": None,
                "ema9": None,
                "ema26": None,
                "trend": None,
                "signal": None,
            }
            continue

        # اتجاه بسيط بناءً على EMA
        if ema9 > ema26 and rsi > 55:
            trend = "Bullish"
            signal = "BUY"
        elif ema9 < ema26 and rsi < 45:
            trend = "Bearish"
            signal = "SELL"
        else:
            trend = "Sideways"
            signal = "HOLD"

        tf_metrics[tf_name] = {
            "rsi": round(rsi, 2),
            "ema9": round(ema9, 2),
            "ema26": round(ema26, 2),
            "trend": trend,
            "signal": signal,
        }

    summary = build_multi_tf_summary(tf_metrics)

    return {
        "timeframes": tf_metrics,
        "summary": summary,
    }


# =========================
# الحلقة الرئيسية
# =========================

def main_loop():
    logging.info("🚀 Starting Indicators Engine (1m + 5m + 15m)...")

    while True:
        try:
            try:
                with open(LIVE_FEED_FILE, "r", encoding="utf-8") as f:
                    live_data = json.load(f)
            except FileNotFoundError:
                logging.warning(f"{LIVE_FEED_FILE} not found. Waiting for WebSocket feed...")
                time.sleep(2)
                continue
            except json.JSONDecodeError:
                logging.warning(f"{LIVE_FEED_FILE} is not valid JSON yet. Retrying...")
                time.sleep(1)
                continue

            metrics_out = {}

            for symbol in WATCHED_SYMBOLS:
                raw = live_data.get(symbol)
                if not raw:
                    continue

                price = raw.get("binance_price")
                ts = raw.get("timestamp")
                if price is None or ts is None:
                    continue

                update_price_history(symbol, ts, price)
                metrics_out[symbol] = compute_metrics_for_symbol(symbol)

            output = {
                "generated_at": time.time(),
                "metrics": metrics_out,
            }

            with open(METRICS_FILE, "w", encoding="utf-8") as f:
                json.dump(output, f, ensure_ascii=False, indent=2)

            time.sleep(2)

        except KeyboardInterrupt:
            logging.info("⛔ Indicators Engine stopped by user.")
            break
        except Exception as e:
            logging.exception(f"Indicators engine error: {e}")
            time.sleep(3)


if __name__ == "__main__":
    main_loop()
