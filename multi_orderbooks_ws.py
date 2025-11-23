import asyncio
import json
import logging
import time
from typing import List

import websockets

# =========================
# إعداد اللوجينج (Logging)
# =========================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)

# =========================
# إعدادات Bybit
# =========================

BYBIT_SYMBOL = "BTCUSDT"
BYBIT_WS_URL = "wss://stream.bybit.com/v5/public/linear"

BYBIT_TOPICS = [
    f"orderbook.1.{BYBIT_SYMBOL}",   # أفضل سعر Bid/Ask
    f"publicTrade.{BYBIT_SYMBOL}",   # الصفقات
    f"kline.1.{BYBIT_SYMBOL}",       # شموع 1 دقيقة
]

BYBIT_PING_INTERVAL = 15
BYBIT_MAX_RECONNECT_DELAY = 60


class BybitWSClient:
    """
    عميل WebSocket لبايبيت:
    - يتصل على v5 public (linear)
    - يشترك في:
        orderbook.1
        publicTrade
        kline.1
    - يعيد الاتصال تلقائياً لو حصل Error
    """

    def __init__(self, url: str, topics: List[str]):
        self.url = url
        self.topics = topics
        self.ws = None
        self._reconnect_tries = 0
        self._last_pong_ts = None
        self._stop = False

    async def run_forever(self):
        while not self._stop:
            try:
                logging.info("📡 [BYBIT] Connecting: %s", self.url)
                async with websockets.connect(
                    self.url,
                    ping_interval=None,   # هندير الـ ping يدوي
                    ping_timeout=None,
                    max_queue=None,
                ) as ws:
                    self.ws = ws
                    self._reconnect_tries = 0
                    logging.info("✅ [BYBIT] Connected. Subscribing to topics...")

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

            delay = min(2 ** self._reconnect_tries, BYBIT_MAX_RECONNECT_DELAY)
            self._reconnect_tries += 1
            logging.warning("🔁 [BYBIT] Reconnecting in %s seconds...", delay)
            await asyncio.sleep(delay)

    async def subscribe(self):
        if not self.ws:
            raise RuntimeError("Bybit WebSocket is not connected")

        sub_msg = {
            "req_id": f"bybit-sub-{int(time.time() * 1000)}",
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
                    await asyncio.sleep(BYBIT_PING_INTERVAL)
                    continue

                ping_msg = {
                    "req_id": f"bybit-ping-{int(time.time() * 1000)}",
                    "op": "ping",
                }
                await self.ws.send(json.dumps(ping_msg))
                logging.debug("📡 [BYBIT] Ping sent")

                await asyncio.sleep(BYBIT_PING_INTERVAL)
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
                    logging.warning("[BYBIT] Received non-JSON message: %s", raw)
                    continue

                await self._handle_message(msg)
        except asyncio.CancelledError:
            logging.debug("[BYBIT] Consumer loop cancelled.")
        except Exception as e:
            logging.error("[BYBIT] Consumer loop error: %s", e)
            raise

    async def _handle_message(self, msg: dict):
        if msg.get("op") in ("pong", "ping"):
            logging.debug("🔄 [BYBIT] Pong/Ping message: %s", msg)
            self._last_pong_ts = time.time()
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

        logging.info(
            "📘 [BYBIT] ORDERBOOK %s | best_bid=%s | best_ask=%s",
            msg.get("topic"),
            best_bid,
            best_ask,
        )

    async def _handle_trade(self, msg: dict):
        data = msg.get("data")
        if not data:
            return

        for trade in data:
            side = trade.get("S")      # Buy / Sell
            price = trade.get("p")
            size = trade.get("v")
            ts = trade.get("T")

            logging.info(
                "💹 [BYBIT] TRADE %s | side=%s price=%s size=%s ts=%s",
                msg.get("topic"),
                side,
                price,
                size,
                ts,
            )

    async def _handle_kline(self, msg: dict):
        data = msg.get("data")
        if not data:
            return

        k = data[0] if isinstance(data, list) else data

        start = k.get("start")
        end = k.get("end")
        open_ = k.get("open")
        high = k.get("high")
        low = k.get("low")
        close = k.get("close")
        volume = k.get("volume")

        logging.info(
            "🕯 [BYBIT] KLINE %s | O:%s H:%s L:%s C:%s V:%s | %s -> %s",
            msg.get("topic"),
            open_,
            high,
            low,
            close,
            volume,
            start,
            end,
        )

    def stop(self):
        self._stop = True


# =========================
# إعدادات Binance
# =========================

BINANCE_SYMBOL = "btcusdt"  # لازم حروف صغيرة في URL
# stream مركب: depth + trades + kline_1m
BINANCE_WS_URL = (
    "wss://stream.binance.com:9443/stream?"
    f"streams={BINANCE_SYMBOL}@depth5@100ms/"
    f"{BINANCE_SYMBOL}@trade/"
    f"{BINANCE_SYMBOL}@kline_1m"
)

BINANCE_MAX_RECONNECT_DELAY = 60


class BinanceWSClient:
    """
    عميل WebSocket لبينانس:
    - يستخدم multi-stream:
        depth5@100ms
        trade
        kline_1m
    - يطبع أفضل Bid/Ask + صفقات + شموع
    """

    def __init__(self, url: str):
        self.url = url
        self.ws = None
        self._reconnect_tries = 0
        self._stop = False

    async def run_forever(self):
        while not self._stop:
            try:
                logging.info("📡 [BINANCE] Connecting: %s", self.url)
                async with websockets.connect(
                    self.url,
                    ping_interval=20,
                    ping_timeout=10,
                    max_queue=None,
                ) as ws:
                    self.ws = ws
                    self._reconnect_tries = 0
                    logging.info("✅ [BINANCE] Connected.")

                    await self._consume_loop()

            except asyncio.CancelledError:
                logging.warning("🛑 [BINANCE] Run loop cancelled.")
                break
            except Exception as e:
                logging.error("❌ [BINANCE] WebSocket error: %s", e, exc_info=True)

            delay = min(2 ** self._reconnect_tries, BINANCE_MAX_RECONNECT_DELAY)
            self._reconnect_tries += 1
            logging.warning("🔁 [BINANCE] Reconnecting in %s seconds...", delay)
            await asyncio.sleep(delay)

    async def _consume_loop(self):
        try:
            async for raw in self.ws:
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    logging.warning("[BINANCE] Received non-JSON message: %s", raw)
                    continue

                await self._handle_message(msg)
        except asyncio.CancelledError:
            logging.debug("[BINANCE] Consumer loop cancelled.")
        except Exception as e:
            logging.error("[BINANCE] Consumer loop error: %s", e)
            raise

    async def _handle_message(self, msg: dict):
        """
        بنية Binance multi-stream:
        {
          "stream": "btcusdt@depth5@100ms",
          "data": { ... }
        }
        """
        stream = msg.get("stream")
        data = msg.get("data")

        if not stream or not data:
            logging.debug("[BINANCE] System message: %s", msg)
            return

        # depth
        if "@depth" in stream:
            await self._handle_depth(stream, data)
        # trade
        elif "@trade" in stream:
            await self._handle_trade(stream, data)
        # kline
        elif "@kline_" in stream:
            await self._handle_kline(stream, data)
        else:
            logging.debug("[BINANCE] Other stream (%s): %s", stream, msg)

    async def _handle_depth(self, stream: str, data: dict):
        bids = data.get("bids", [])
        asks = data.get("asks", [])

        best_bid = bids[0] if bids else None
        best_ask = asks[0] if asks else None

        logging.info(
            "📘 [BINANCE] ORDERBOOK %s | best_bid=%s | best_ask=%s",
            stream,
            best_bid,
            best_ask,
        )

    async def _handle_trade(self, stream: str, data: dict):
        price = data.get("p")
        qty = data.get("q")
        is_buyer_maker = data.get("m")  # True=Sell side, False=Buy side

        side = "Sell" if is_buyer_maker else "Buy"

        logging.info(
            "💹 [BINANCE] TRADE %s | side=%s price=%s qty=%s",
            stream,
            side,
            price,
            qty,
        )

    async def _handle_kline(self, stream: str, data: dict):
        k = data.get("k", {})
        open_ = k.get("o")
        high = k.get("h")
        low = k.get("l")
        close = k.get("c")
        volume = k.get("v")
        start = k.get("t")
        end = k.get("T")
        is_closed = k.get("x")

        if is_closed:
            logging.info(
                "🕯 [BINANCE] KLINE %s | O:%s H:%s L:%s C:%s V:%s | %s -> %s",
                stream,
                open_,
                high,
                low,
                close,
                volume,
                start,
                end,
            )

    def stop(self):
        self._stop = True


# =========================
# تشغيل الاتنين معًا
# =========================

async def main():
    bybit_client = BybitWSClient(BYBIT_WS_URL, BYBIT_TOPICS)
    binance_client = BinanceWSClient(BINANCE_WS_URL)

    logging.info("🚀 Starting Bybit + Binance WebSocket listeners...")
    await asyncio.gather(
        bybit_client.run_forever(),
        binance_client.run_forever(),
    )


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.warning("KeyboardInterrupt received. Exiting...")
