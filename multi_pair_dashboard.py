# multi_pair_dashboard.py
import asyncio
import json
import logging
from typing import List

import websockets

# =========================
# إعداد اللوجينج
# =========================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)

# =========================
# إعداد الرموز (Top 3 من السكانر)
# لو حابب تغيرهم بعدين: عدّل الليست دي بس
# =========================
BINANCE_SYMBOLS: List[str] = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
BYBIT_SYMBOLS:   List[str] = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]

# Binance USDT-M Futures WebSocket
BINANCE_WS_TEMPLATE = "wss://fstream.binance.com/stream?streams={streams}"

# Bybit Linear Perps WebSocket
BYBIT_WS_URL = "wss://stream.bybit.com/v5/public/linear"


# =========================
# 1) Binance Futures – ORDERBOOK + TRADES + KLINES
# =========================
import aiohttp

async def binance_consumer():
    urls = []

    for sym in BINANCE_SYMBOLS:
        s = sym.lower()
        urls.append(f"wss://fstream.binance.com/ws/{s}@trade")
        urls.append(f"wss://fstream.binance.com/ws/{s}@kline_1m")
        urls.append(f"wss://fstream.binance.com/ws/{s}@depth5@100ms")  # FIXED ✔️

    async def connect_single(url):
        while True:
            try:
                logging.info("📡 [BINANCE] Connecting to %s", url)
                async with websockets.connect(
                    url,
                    ping_interval=20,
                    ping_timeout=20,
                ) as ws:

                    logging.info("✅ [BINANCE] Connected: %s", url)

                    async for raw in ws:
                        try:
                            msg = json.loads(raw)
                        except:
                            continue

                        # -----------------
                        # Depth (orderbook)
                        # -----------------
                        if "bids" in msg and "asks" in msg:
                            bids = msg.get("bids", [])
                            asks = msg.get("asks", [])
                            best_bid = bids[0] if bids else None
                            best_ask = asks[0] if asks else None

                            logging.info(
                                "📘 [BINANCE][ORDERBOOK] best_bid=%s | best_ask=%s",
                                best_bid,
                                best_ask,
                            )

                        # -----------------
                        # Trades
                        # -----------------
                        if msg.get("e") == "trade":
                            side = "Sell" if msg.get("m") else "Buy"
                            logging.info(
                                "💹 [BINANCE][TRADE] %s price=%s qty=%s",
                                side, msg.get("p"), msg.get("q")
                            )

                        # -----------------
                        # Kline
                        # -----------------
                        if msg.get("e") == "kline":
                            k = msg.get("k", {})
                            logging.info(
                                "🕯 [BINANCE][KLINE 1m] O:%s H:%s L:%s C:%s V:%s",
                                k.get("o"),
                                k.get("h"),
                                k.get("l"),
                                k.get("c"),
                                k.get("v"),
                            )

            except Exception as e:
                logging.warning("⚠️ [BINANCE] error: %s – reconnecting in 3s", e)
                await asyncio.sleep(3)

    # تشغيل كل WebSocket في كوروتين منفصل
    tasks = [asyncio.create_task(connect_single(url)) for url in urls]
    await asyncio.gather(*tasks)

# =========================
# 2) Bybit Consumer (Linear Perps)
# =========================
def build_bybit_topics() -> List[str]:
    topics = []
    for sym in BYBIT_SYMBOLS:
        topics.append(f"orderbook.1.{sym}")
        topics.append(f"publicTrade.{sym}")
        topics.append(f"kline.1.{sym}")
    return topics


async def bybit_consumer():
    topics = build_bybit_topics()
    while True:
        try:
            logging.info("📡 [BYBIT] Connecting: %s", BYBIT_WS_URL)
            async with websockets.connect(
                BYBIT_WS_URL,
                ping_interval=None,  # هنستخدم بروتوكول Bybit ping الخاص
                ping_timeout=None,
                max_queue=None,
            ) as ws:
                sub_msg = {
                    "req_id": "sub-multi",
                    "op": "subscribe",
                    "args": topics,
                }
                await ws.send(json.dumps(sub_msg))
                logging.info("✅ [BYBIT] Subscribed to %d topics.", len(topics))

                async for raw in ws:
                    try:
                        msg = json.loads(raw)
                    except json.JSONDecodeError:
                        logging.warning("[BYBIT] Non-JSON message: %s", raw)
                        continue

                    # رسائل System أو ردود Ping/Pong
                    if msg.get("op") in ("pong", "ping", "subscribe"):
                        continue

                    topic = msg.get("topic", "")
                    data = msg.get("data")
                    if not topic or not data:
                        continue

                    parts = topic.split(".")
                    if len(parts) < 2:
                        continue

                    channel = parts[0]   # orderbook / publicTrade / kline
                    sym = parts[-1]      # BTCUSDT

                    # ----- ORDERBOOK -----
                    if channel == "orderbook":
                        book = data[0] if isinstance(data, list) else data
                        bids = book.get("b", [])
                        asks = book.get("a", [])
                        best_bid = bids[0] if bids else None
                        best_ask = asks[0] if asks else None
                        logging.info(
                            "📘 [BYBIT][%s][ORDERBOOK] best_bid=%s | best_ask=%s",
                            sym,
                            best_bid,
                            best_ask,
                        )

                    # ----- TRADES -----
                    elif channel == "publicTrade":
                        trades = data if isinstance(data, list) else [data]
                        for t in trades:
                            side = t.get("S")   # Buy / Sell
                            price = t.get("p")
                            size = t.get("v")
                            logging.info(
                                "💹 [BYBIT][%s][TRADE] side=%s price=%s size=%s",
                                sym,
                                side,
                                price,
                                size,
                            )

                    # ----- KLINE -----
                    elif channel == "kline":
                        k = data[0] if isinstance(data, list) else data
                        logging.info(
                            "🕯 [BYBIT][%s][KLINE 1m] O:%s H:%s L:%s C:%s V:%s",
                            sym,
                            k.get("open"),
                            k.get("high"),
                            k.get("low"),
                            k.get("close"),
                            k.get("volume"),
                        )

        except Exception as e:
            logging.warning("⚠️ [BYBIT] error: %s – reconnecting in 3s...", e)
            await asyncio.sleep(3)


# =========================
# 3) MAIN – تشغيل الاثنين معًا
# =========================
async def main():
    logging.info("🚀 Starting Multi-Pair Dual-Exchange Dashboard (Futures)")
    tasks = [
        asyncio.create_task(binance_consumer()),
        asyncio.create_task(bybit_consumer()),
    ]
    await asyncio.gather(*tasks)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.warning("🛑 KeyboardInterrupt received. Exiting...")
