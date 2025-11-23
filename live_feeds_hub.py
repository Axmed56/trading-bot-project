import asyncio
import json
import logging
import time
from typing import List, Dict, Any

import websockets

# ========== إعداد اللوجينج ==========
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)

# ========== إعدادات عامة ==========
SYMBOL = "BTCUSDT"

# Binance WS (عمق + صفقات + كندل 1 دقيقة)
BINANCE_WS_URL = (
    "wss://stream.binance.com:9443/stream"
    "?streams=btcusdt@depth5@100ms/btcusdt@trade/btcusdt@kline_1m"
)

# Bybit WS (Linear USDT Perp)
BYBIT_WS_URL = "wss://stream.bybit.com/v5/public/linear"

BYBIT_TOPICS = [
    f"orderbook.1.{SYMBOL}",
    f"publicTrade.{SYMBOL}",
    f"kline.1.{SYMBOL}",
]

PING_INTERVAL = 15
MAX_RECONNECT_DELAY = 60

SNAPSHOT_FILE = "live_snapshot.json"

# ========== الحالة المشتركة بين المنصتين ==========
state: Dict[str, Any] = {
    "ts": None,
    "binance": {
        "orderbook": None,
        "last_trade": None,
        "kline_1m": None,
    },
    "bybit": {
        "orderbook": None,
        "last_trade": None,
        "kline_1m": None,
    },
}


def save_snapshot():
    """كتابة الحالة الحالية في ملف JSON يقرؤه الداش."""
    try:
        state["ts"] = time.time()
        with open(SNAPSHOT_FILE, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logging.error("Error saving snapshot: %s", e)


# ========== Binance Client ==========


async def binance_client():
    reconnect_tries = 0

    while True:
        try:
            logging.info("📡 [BINANCE] Connecting: %s", BINANCE_WS_URL)
            async with websockets.connect(BINANCE_WS_URL, ping_interval=None, ping_timeout=None) as ws:
                logging.info("✅ [BINANCE] Connected.")
                reconnect_tries = 0

                async for raw in ws:
                    try:
                        msg = json.loads(raw)
                    except json.JSONDecodeError:
                        logging.warning("BINANCE non-JSON: %s", raw)
                        continue

                    stream = msg.get("stream", "")
                    data = msg.get("data", {})

                    # عمق دفتر الأوامر
                    if stream.endswith("@depth5@100ms"):
                        bids = data.get("bids", [])
                        asks = data.get("asks", [])
                        best_bid = bids[0] if bids else None
                        best_ask = asks[0] if asks else None

                        state["binance"]["orderbook"] = {
                            "bids": bids,
                            "asks": asks,
                            "best_bid": best_bid,
                            "best_ask": best_ask,
                        }

                        logging.info(
                            "📘 [BINANCE] ORDERBOOK | best_bid=%s | best_ask=%s",
                            best_bid,
                            best_ask,
                        )
                        save_snapshot()

                    # الصفقات
                    elif stream.endswith("@trade"):
                        side = "Buy" if data.get("m") is False else "Sell"
                        price = data.get("p")
                        qty = data.get("q")
                        ts = data.get("T")

                        state["binance"]["last_trade"] = {
                            "side": side,
                            "price": price,
                            "qty": qty,
                            "ts": ts,
                        }

                        logging.info(
                            "💹 [BINANCE] TRADE | side=%s price=%s qty=%s",
                            side,
                            price,
                            qty,
                        )
                        save_snapshot()

                    # كندل 1 دقيقة
                    elif stream.endswith("@kline_1m"):
                        k = data.get("k", {})
                        if not k:
                            continue

                        kline = {
                            "start": k.get("t"),
                            "end": k.get("T"),
                            "open": k.get("o"),
                            "high": k.get("h"),
                            "low": k.get("l"),
                            "close": k.get("c"),
                            "volume": k.get("v"),
                            "is_closed": k.get("x"),
                        }
                        state["binance"]["kline_1m"] = kline

                        if kline["is_closed"]:
                            logging.info(
                                "🕯 [BINANCE] KLINE 1m | O:%s H:%s L:%s C:%s V:%s",
                                kline["open"],
                                kline["high"],
                                kline["low"],
                                kline["close"],
                                kline["volume"],
                            )
                            save_snapshot()

                    else:
                        # رسائل أخرى لا نهتم بها الآن
                        pass

        except Exception as e:
            logging.error("❌ [BINANCE] WebSocket error: %s", e, exc_info=True)

        delay = min(2 ** reconnect_tries, MAX_RECONNECT_DELAY)
        reconnect_tries += 1
        logging.warning("🔁 [BINANCE] Reconnecting in %s seconds...", delay)
        await asyncio.sleep(delay)


# ========== Bybit Client (نفس الكود اللي اشتغل عندك مع تعديل بسيط) ==========


class BybitWSClient:
    def __init__(self, url: str, topics: List[str]):
        self.url = url
        self.topics = topics
        self.ws = None
        self._reconnect_tries = 0
        self._stop = False

    async def run_forever(self):
        while not self._stop:
            try:
                logging.info("📡 [BYBIT] Connecting: %s", self.url)
                async with websockets.connect(
                    self.url,
                    ping_interval=None,
                    ping_timeout=None,
                    max_queue=None,
                ) as ws:
                    self.ws = ws
                    self._reconnect_tries = 0
                    logging.info("✅ [BYBIT] Connected. Subscribing...")

                    await self.subscribe()

                    consumer_task = asyncio.create_task(self._consume_loop())
                    ping_task = asyncio.create_task(self._ping_loop())

                    done, pending = await asyncio.wait(
                        [consumer_task, ping_task],
                        return_when=asyncio.FIRST_EXCEPTION,
                    )

                    for task in pending:
                        task.cancel()

                    for task in done:
                        exc = task.exception()
                        if exc:
                            raise exc

            except asyncio.CancelledError:
                logging.warning("🛑 [BYBIT] Run loop cancelled.")
                break
            except Exception as e:
                logging.error("❌ [BYBIT] WebSocket error: %s", e, exc_info=True)

            delay = min(2 ** self._reconnect_tries, MAX_RECONNECT_DELAY)
            self._reconnect_tries += 1
            logging.warning("🔁 [BYBIT] Reconnecting in %s seconds...", delay)
            await asyncio.sleep(delay)

    async def subscribe(self):
        if not self.ws:
            raise RuntimeError("Bybit WebSocket is not connected")

        sub_msg = {
            "req_id": f"sub-{int(time.time() * 1000)}",
            "op": "subscribe",
            "args": self.topics,
        }
        payload = json.dumps(sub_msg)
        logging.info("📨 [BYBIT] Sending subscribe: %s", payload)
        await self.ws.send(payload)

    async def _ping_loop(self):
        try:
            while True:
                if self.ws is None:
                    await asyncio.sleep(PING_INTERVAL)
                    continue

                ping_msg = {
                    "req_id": f"ping-{int(time.time() * 1000)}",
                    "op": "ping",
                }
                await self.ws.send(json.dumps(ping_msg))
                logging.debug("📡 [BYBIT] Ping sent")
                await asyncio.sleep(PING_INTERVAL)
        except asyncio.CancelledError:
            logging.debug("[BYBIT] Ping loop cancelled.")
        except Exception as e:
            logging.error("[BYBIT] Ping loop error: %s", e)

    async def _consume_loop(self):
        try:
            async for raw in self.ws:
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    logging.warning("[BYBIT] non-JSON: %s", raw)
                    continue

                await self._handle_message(msg)
        except asyncio.CancelledError:
            logging.debug("[BYBIT] Consumer loop cancelled.")
        except Exception as e:
            logging.error("[BYBIT] Consumer loop error: %s", e)
            raise

    async def _handle_message(self, msg: dict):
        if msg.get("op") in ("pong", "ping"):
            logging.debug("🔄 [BYBIT] Pong/Ping: %s", msg)
            return

        if msg.get("op") == "subscribe":
            logging.info("✅ [BYBIT] Subscribed successfully: %s", msg)
            return

        topic = msg.get("topic")
        if not topic:
            logging.debug("[BYBIT] System message: %s", msg)
            return

        if topic.startswith("orderbook."):
            await self._handle_orderbook(msg)
        elif topic.startswith("publicTrade."):
            await self._handle_trade(msg)
        elif topic.startswith("kline."):
            await self._handle_kline(msg)
        else:
            logging.debug("[BYBIT] Other topic (%s): %s", topic, msg)

    async def _handle_orderbook(self, msg: dict):
        data = msg.get("data")
        if not data:
            return

        book = data[0] if isinstance(data, list) else data
        bids = book.get("b", [])
        asks = book.get("a", [])

        best_bid = bids[0] if bids else None
        best_ask = asks[0] if asks else None

        state["bybit"]["orderbook"] = {
            "bids": bids,
            "asks": asks,
            "best_bid": best_bid,
            "best_ask": best_ask,
        }

        logging.info(
            "📘 [BYBIT] ORDERBOOK %s | best_bid=%s | best_ask=%s",
            msg.get("topic"),
            best_bid,
            best_ask,
        )
        save_snapshot()

    async def _handle_trade(self, msg: dict):
        data = msg.get("data")
        if not data:
            return

        for trade in data:
            side = trade.get("S")
            price = trade.get("p")
            size = trade.get("v")
            ts = trade.get("T")

            state["bybit"]["last_trade"] = {
                "side": side,
                "price": price,
                "qty": size,
                "ts": ts,
            }

            logging.info(
                "💹 [BYBIT] TRADE %s | side=%s price=%s size=%s",
                msg.get("topic"),
                side,
                price,
                size,
            )
            save_snapshot()

    async def _handle_kline(self, msg: dict):
        data = msg.get("data")
        if not data:
            return

        k = data[0] if isinstance(data, list) else data
        kline = {
            "start": k.get("start"),
            "end": k.get("end"),
            "open": k.get("open"),
            "high": k.get("high"),
            "low": k.get("low"),
            "close": k.get("close"),
            "volume": k.get("volume"),
            "confirm": k.get("confirm"),
        }

        state["bybit"]["kline_1m"] = kline

        if kline["confirm"]:
            logging.info(
                "🕯 [BYBIT] KLINE 1m | O:%s H:%s L:%s C:%s V:%s",
                kline["open"],
                kline["high"],
                kline["low"],
                kline["close"],
                kline["volume"],
            )
            save_snapshot()

    def stop(self):
        self._stop = True


async def bybit_client():
    client = BybitWSClient(BYBIT_WS_URL, BYBIT_TOPICS)
    await client.run_forever()


# ========== تشغيل الاثنين معًا ==========


async def main():
    tasks = [
        asyncio.create_task(binance_client()),
        asyncio.create_task(bybit_client()),
    ]
    await asyncio.gather(*tasks)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.warning("KeyboardInterrupt received. Exiting...")
