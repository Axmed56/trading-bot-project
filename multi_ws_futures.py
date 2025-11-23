import asyncio
import json
import logging
import os
import time
from pathlib import Path

import ccxt

# -----------------------
# إعداد اللوج
# -----------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-5s | %(message)s",
)

logger = logging.getLogger("multi_ws_futures")

# -----------------------
# إعداد الإكسشينجات (فيوتشر)
# -----------------------
binance = ccxt.binance({
    "options": {
        "defaultType": "future"
    }
})

bybit = ccxt.bybit({
    "options": {
        "defaultType": "future"
    }
})

# الأزواج اللي هنراقبها (موحّدة)
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]

# تحويل لأسماء ccxt للفيوتشر
BINANCE_MAP = {
    "BTCUSDT": "BTC/USDT:USDT",
    "ETHUSDT": "ETH/USDT:USDT",
    "SOLUSDT": "SOL/USDT:USDT",
}

BYBIT_MAP = {
    "BTCUSDT": "BTC/USDT:USDT",
    "ETHUSDT": "ETH/USDT:USDT",
    "SOLUSDT": "SOL/USDT:USDT",
}

# ملف الـ JSON اللي هيتشارك مع الداشبورد
BASE_DIR = Path(__file__).resolve().parent
LIVE_FEED_PATH = BASE_DIR / "data" / "live_feed.json"


def atomic_write_json(path: Path, data: dict) -> None:
    """
    كتابة آمنة للـ JSON:
    نكتب في ملف مؤقت ثم نستبدله بالملف الأساسي
    علشان ما يحصلش corruption أثناء القراءة.
    """
    tmp_path = path.with_suffix(".json.tmp")

    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    os.replace(tmp_path, path)


async def fetch_ticker(exchange, symbol_ccxt: str) -> float | None:
    try:
        ticker = exchange.fetch_ticker(symbol_ccxt)
        return float(ticker["last"])
    except Exception as e:
        logger.warning(f"[{exchange.id.upper()}][{symbol_ccxt}] fetch error: {e}")
        return None


async def main_loop():
    logger.info("🚀 Starting simple futures price collector (Binance + Bybit)")

    snapshot: dict[str, dict] = {}

    while True:
        try:
            now_ts = time.time()

            for sym in SYMBOLS:
                b_sym = BINANCE_MAP[sym]
                y_sym = BYBIT_MAP[sym]

                bin_price = await asyncio.to_thread(fetch_ticker, binance, b_sym)
                byb_price = await asyncio.to_thread(fetch_ticker, bybit, y_sym)

                # resolve futures
                bin_price = await fetch_ticker(binance, b_sym)
                byb_price = await fetch_ticker(bybit, y_sym)

                if bin_price is not None:
                    logger.info(f"[BINANCE][{sym}] price={bin_price}")
                if byb_price is not None:
                    logger.info(f"[BYBIT  ][{sym}] price={byb_price}")

                if bin_price is None and byb_price is None:
                    # مفيش داتا خالص – ما نحدّثش هذا الزوج
                    continue

                spread = None
                if bin_price is not None and byb_price is not None:
                    spread = round(byb_price - bin_price, 4)

                snapshot[sym] = {
                    "binance_price": bin_price,
                    "bybit_price": byb_price,
                    "spread": spread,
                    "timestamp": now_ts,
                }

            # كتابة اللقطة في ملف مشترك مع الداشبورد
            if snapshot:
                atomic_write_json(LIVE_FEED_PATH, snapshot)
                logger.info(f"💾 saved snapshot for {len(snapshot)} symbols")

        except Exception as e:
            logger.error(f"MAIN LOOP error: {e}")

        # كل 2 ثانية تحديث (تقدر تعدّلها)
        await asyncio.sleep(2)


if __name__ == "__main__":
    asyncio.run(main_loop())
