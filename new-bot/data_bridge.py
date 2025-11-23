import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from flask import Flask, jsonify, send_from_directory
from flask_cors import CORS

# ==========================
# إعداد اللوج
# ==========================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-5s | %(message)s",
)
logger = logging.getLogger("data_bridge")

# ==========================
# المسارات
# ==========================
BASE_DIR = Path(__file__).resolve().parent
LIVE_FEED_PATH = BASE_DIR / "live_feed.json"

app = Flask(__name__)
CORS(app)  # في حالة احتجنا نفتح من جهاز آخر في نفس الشبكة


def read_live_feed() -> dict:
    """قراءة ملف live_feed.json وإرجاعه كقاموس."""
    if LIVE_FEED_PATH.exists():
        try:
            with LIVE_FEED_PATH.open("r", encoding="utf-8") as f:
                data = json.load(f)
                return data if isinstance(data, dict) else {}
        except Exception as e:
            logger.error(f"Error reading live_feed.json: {e}")
            return {}
    return {}


@app.route("/api/live-feed")
def api_live_feed():
    """
    API رئيسية ترجع أحدث البيانات للداشبورد.
    يضيف حقل last_update_utc لكل زوج.
    """
    data = read_live_feed()
    for sym, row in data.items():
        ts = row.get("timestamp")
        if isinstance(ts, (int, float)):
            dt = datetime.fromtimestamp(ts, tz=timezone.utc)
            row["last_update_utc"] = dt.isoformat()
        else:
            row["last_update_utc"] = None
    return jsonify(data)


@app.route("/")
def index():
    """إرسال ملف الداشبورد HTML."""
    return send_from_directory(BASE_DIR, "dashboard.html")


if __name__ == "__main__":
    logger.info("🚀 Starting ZAYA Futures Dashboard API on http://127.0.0.1:5005")
    app.run(host="127.0.0.1", port=5005, debug=False)
