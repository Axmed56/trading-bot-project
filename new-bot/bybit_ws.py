import time
import json
import logging
from pathlib import Path

import ccxt

# ==========================
#  إعداد اللوج
# ==========================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-5s | %(message)s",
)
logger = logging.getLogger("bybit_ws")

# ==========================
#  إعداد Bybit (فيوتشـر)
# ==========================
bybit = ccxt.bybit({
    "options": {
        "defaultType": "future",  # نتعامل مع عقود USDT Perpetual
    }
})

# الأزواج التي نتابعها
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]

# ملف الـ JSON المشترك مع Binance والداش بورد
BASE_DIR = Path(__file__).resolve().parent
LIVE_FEED_PATH = BASE_DIR / "live_feed.json"


def atomic_write_json(path: Path, data: dict) -> None:
    """
    كتابة آمنة لملف JSON:
    نكتب في ملف مؤقت ثم نستبدله بالملف النهائي
    عشان مايحصلش corruption لو حصل قطع مفاجئ.
    """
    tmp_path = path.with_suffix(".json.tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    tmp_path.replace(path)


def load_live_feed() -> dict:
    """قراءة محتوى live_feed.json لو موجود، أو إنشاء قاموس فاضي."""
    if LIVE_FEED_PATH.exists():
        try:
            with LIVE_FEED_PATH.open("r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading live_feed.json: {e}")
            return {}
    return {}


def save_live_feed(data: dict) -> None:
    """حفظ التحديثات في live_feed.json"""
    try:
        atomic_write_json(LIVE_FEED_PATH, data)
    except Exception as e:
        logger.error(f"Error saving live_feed.json: {e}")


def fetch_bybit_price(symbol: str) -> float | None:
    """
    جلب آخر سعر من Bybit لعقد USDT Perpetual
    في ccxt صيغة الرمز بتكون مثلاً: BTCUSDT:USDT
    """
    try:
        market_symbol = f"{symbol}:USDT"
        ticker = bybit.fetch_ticker(market_symbol)
        price = ticker.get("last")
        if price is None:
            logger.warning(f"[BYBIT][{symbol}] last price is None")
        return float(price) if price is not None else None
    except Exception as e:
        logger.error(f"[BYBIT][{symbol}] fetch error: {e}")
        return None


def main_loop():
    logger.info("🚀 Starting Bybit price watcher (REST polling every ~2s)...")

    while True:
        try:
            live_data = load_live_feed()
            now_ts = time.time()

            for sym in SYMBOLS:
                byb_price = fetch_bybit_price(sym)
                if byb_price is None:
                    # لو السعر فاضي، نسيب آخر قيمة زي ما هي
                    continue

                if sym not in live_data:
                    live_data[sym] = {}

                live_data[sym]["bybit_price"] = byb_price
                live_data[sym]["timestamp"] = now_ts

                # لو عندي سعر Binance، أحسب السبريد (Bybit - Binance)
                bin_price = live_data[sym].get("binance_price")
                if bin_price is not None:
                    spread = round(byb_price - float(bin_price), 4)
                    live_data[sym]["spread"] = spread

                logger.info(f"[BYBIT][{sym}] price={byb_price}")

            save_live_feed(live_data)

        except Exception as e:
            logger.error(f"MAIN LOOP error: {e}")

        # كل 2 ثانية تقريبًا
        time.sleep(2)


if __name__ == "__main__":
    main_loop()
