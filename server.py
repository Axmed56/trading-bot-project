from flask import Flask, jsonify, render_template
from datetime import datetime, timezone
import json
import os
import logging
import time

app = Flask(__name__, static_folder=".", template_folder=".")

LIVE_FEED_FILE = "live_feed.json"
METRICS_FILE = "metrics.json"
DASHBOARD_HTML = "zaya_futures_dashboard.html"  # اسم ملف الداشبورد اللي عندك

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
)

def safe_load_json(path):
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


@app.route("/")
def index():
    # نعرض نفس ملف الـ HTML اللي شغال عندك الآن
    return render_template(DASHBOARD_HTML)


@app.route("/api/live-feed")
def api_live_feed():
    live = safe_load_json(LIVE_FEED_FILE) or {}
    metrics_blob = safe_load_json(METRICS_FILE) or {}

    metrics = metrics_blob.get("metrics", {})
    gen_ts = metrics_blob.get("generated_at")

    merged = {}

    for symbol, data in live.items():
        sym = dict(data)  # binance_price, bybit_price, timestamp, spread

        m = metrics.get(symbol, {})
        tfs = m.get("timeframes", {})
        summary = m.get("summary", {})

        # مؤشرات الفريمات الثلاثة
        rsi_1m = tfs.get("1m", {}).get("rsi")
        rsi_5m = tfs.get("5m", {}).get("rsi")
        rsi_15m = tfs.get("15m", {}).get("rsi")

        ema9_5m = tfs.get("5m", {}).get("ema9")
        ema26_15m = tfs.get("15m", {}).get("ema26")

        # إشارة AI المجمعة من الثلاث فريمات
        ai_trend = summary.get("trend")
        ai_signal = summary.get("signal")
        ai_conf = summary.get("confidence")

        # قيم أساسية تستخدمها الداشبورد الحالية
        sym["rsi"] = rsi_5m
        sym["ema9"] = ema9_5m
        sym["ema26"] = ema26_15m
        sym["ai_trend"] = ai_trend
        sym["ai_signal"] = ai_signal
        sym["ai_confidence"] = ai_conf

        # نحفظ كمان القيم التفصيلية لو حبيت تستخدمها مستقبلاً
        sym["rsi_1m"] = rsi_1m
        sym["rsi_5m"] = rsi_5m
        sym["rsi_15m"] = rsi_15m
        sym["ema9_5m"] = ema9_5m
        sym["ema26_15m"] = ema26_15m

        merged[symbol] = sym

    # آخر تحديث
    if gen_ts:
        last_update_ts = gen_ts
    else:
        try:
            last_update_ts = max(
                (d.get("timestamp", time.time()) for d in live.values()),
                default=time.time(),
            )
        except Exception:
            last_update_ts = time.time()

    last_update_iso = datetime.fromtimestamp(last_update_ts, timezone.utc).isoformat()

    resp = {
        "last_update": last_update_iso,
        "symbols": merged,
    }

    return jsonify(resp)


if __name__ == "__main__":
    logging.info("🚀 Starting ZAYA Futures Dashboard API on http://127.0.0.1:5005")
    app.run(host="0.0.0.0", port=5005)
