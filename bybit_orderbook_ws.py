import asyncio
import json
import logging
import time
from typing import List, Dict, Any

import websockets

# =========================
# إعدادات عامة (Config)
# =========================

SYMBOL = "BTCUSDT"

# لو عاوز Level 25 بدل 1 غيّر دي في TOPICS فقط
BYBIT_WS_URL = "wss://stream.bybit.com/v5/public/linear"

TOPICS = [
    f"orderbook.1.{SYMBOL}",   # أفضل عمق (Level 1 Orderbook)
    f"publicTrade.{SYMBOL}",   # الصفقات
    f"kline.1.{SYMBOL}",       # شموع دقيقة واحدة
]

PING_INTERVAL = 15
MAX_RECONNECT_DELAY = 60

STATE_FILE = "bybit_public_state.json"

# حالة السوق التي سيتم مشاركتها مع الداشبورد
STATE: Dict[str, Any] = {
    "symbol": SYMBOL,
    "best_bid": None,         # [price, size]
    "best_ask": None,         # [price, size]
    "spread": None,           # فرق السعر
    "last_trade": None,       # {side, price, size, ts}
    "last_kline": None,       # {open, high, low, close, volume, start, end}
    "orderflow_bias": None,   # "Buy", "Sell", "Neutral"
    "last_update_ts": None,   # time.time()
}


# =========================
# إعداد اللوجينج (Logging)
# =========================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)


def save_state() -> None:
    """حفظ STATE في ملف JSON ليستخدمه الداشبورد."""
    try:
        STATE["last_update_ts"] = time.time()
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(STATE, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logging.error("⚠️ Error saving state: %s", e)


class BybitWSClient:
    """
    عميل WebSocket لبايبيت:
    - اتصال + اشتراك في المواضيع
    - Ping/Pong للحفاظ على الجلسة
    - إعادة اتصال تلقائية في حالة الفشل
    """

    def __init__(self, url: str, topics: List[str]):
        self.url = url
        self.topics = topics
        self.ws = None
        self._reconnect_tries = 0
        self._last_pong_ts = None
        self._stop = False

    async def run_forever(self):
        """حلقة تشغيل رئيسية مع إعادة اتصال تلقائية."""
        while not self._stop:
            try:
                logging.info("🔌 Connecting to Bybit WebSocket: %s", self.url)
                async with websockets.connect(
                    self.url,
                    ping_interval=None,   # نحن ندير Ping يدويًا
                    ping_timeout=None,
                    max_queue=None,
                ) as ws:
                    self.ws = ws
                    self._reconnect_tries = 0
                    logging.info("✅ Connected. Subscribing to topics...")

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
                logging.warning("🛑 Run loop cancelled.")
                break
            except Exception as e:
                logging.error("❌ WebSocket error: %s", e, exc_info=True)

            delay = min(2 ** self._reconnect_tries, MAX_RECONNECT_DELAY)
            self._reconnect_tries += 1
            logging.warning("🔁 Reconnecting in %s seconds...", delay)
            await asyncio.sleep(delay)

    async def subscribe(self):
        """إرسال رسالة الاشتراك في المواضيع المحددة."""
        if not self.ws:
            raise RuntimeError("WebSocket is not connected")

        sub_msg = {
            "req_id": f"sub-{int(time.time() * 1000)}",
            "op": "subscribe",
            "args": self.topics,
        }
        payload = json.dumps(sub_msg)
        logging.info("📨 Sending subscribe: %s", payload)
        await self.ws.send(payload)

    async def _ping_loop(self):
        """إرسال Ping دوري للحفاظ على الاتصال."""
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
                logging.debug("📡 Ping sent")
                await asyncio.sleep(PING_INTERVAL)
        except asyncio.CancelledError:
            logging.debug("Ping loop cancelled.")
        except Exception as e:
            logging.error("Ping loop error: %s", e)

    async def _consume_loop(self):
        """استقبال ومعالجة كل رسائل الـ WebSocket."""
        try:
            async for raw in self.ws:
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    logging.warning("Received non-JSON message: %s", raw)
                    continue

                await self._handle_message(msg)
        except asyncio.CancelledError:
            logging.debug("Consumer loop cancelled.")
        except Exception as e:
            logging.error("Consumer loop error: %s", e)
            raise

    async def _handle_message(self, msg: dict):
        """توجيه الرسالة حسب نوعها/موضوعها."""
        if msg.get("op") in ("pong", "ping"):
            logging.debug("🔄 Pong/Ping message: %s", msg)
            self._last_pong_ts = time.time()
            return

        if msg.get("op") == "subscribe":
            logging.info("✅ Subscribed successfully: %s", msg)
            return

        topic = msg.get("topic")
        if not topic:
            logging.debug("System message: %s", msg)
            return

        if topic.startswith("orderbook."):
            await self._handle_orderbook(msg)
        elif topic.startswith("publicTrade."):
            await self._handle_trade(msg)
        elif topic.startswith("kline."):
            await self._handle_kline(msg)
        else:
            logging.debug("Other topic (%s): %s", topic, msg)

    async def _handle_orderbook(self, msg: dict):
        """معالجة بيانات دفتر الأوامر (Level 1)."""
        data = msg.get("data")
        if not data:
            return

        book = data[0] if isinstance(data, list) else data
        bids = book.get("b", [])
        asks = book.get("a", [])

        best_bid = bids[0] if bids else None
        best_ask = asks[0] if asks else None

        if best_bid:
            STATE["best_bid"] = [float(best_bid[0]), float(best_bid[1])]
        if best_ask:
            STATE["best_ask"] = [float(best_ask[0]), float(best_ask[1])]

        if STATE["best_bid"] and STATE["best_ask"]:
            STATE["spread"] = STATE["best_ask"][0] - STATE["best_bid"][0]

        logging.info(
            "📘 ORDERBOOK %s | best_bid=%s | best_ask=%s | spread=%.2f",
            msg.get("topic"),
            STATE["best_bid"],
            STATE["best_ask"],
            STATE["spread"] if STATE["spread"] is not None else 0.0,
        )

        save_state()

    async def _handle_trade(self, msg: dict):
        """معالجة بيانات الصفقات (publicTrade)."""
        data = msg.get("data")
        if not data:
            return

        for trade in data:
            side = trade.get("S")   # Buy / Sell
            price = float(trade.get("p"))
            size = float(trade.get("v"))
            ts = trade.get("T")

            STATE["last_trade"] = {
                "side": side,
                "price": price,
                "size": size,
                "ts": ts,
            }

            # منطق بسيط لتحليل تدفق الأوامر
            if size >= 1.0:
                STATE["orderflow_bias"] = "Buy" if side == "Buy" else "Sell"
            elif STATE["orderflow_bias"] is None:
                STATE["orderflow_bias"] = "Neutral"

            logging.info(
                "💹 TRADE %s | side=%s price=%.2f size=%.4f ts=%s",
                msg.get("topic"),
                side,
                price,
                size,
                ts,
            )

        save_state()

    async def _handle_kline(self, msg: dict):
        """معالجة بيانات الشموع (kline)."""
        data = msg.get("data")
        if not data:
            return

        k = data[0] if isinstance(data, list) else data

        last_kline = {
            "start": k.get("start"),
            "end": k.get("end"),
            "open": float(k.get("open")),
            "high": float(k.get("high")),
            "low": float(k.get("low")),
            "close": float(k.get("close")),
            "volume": float(k.get("volume")),
        }
        STATE["last_kline"] = last_kline

        logging.info(
            "🕯 KLINE %s | O:%s H:%s L:%s C:%s V:%s | %s -> %s",
            msg.get("topic"),
            last_kline["open"],
            last_kline["high"],
            last_kline["low"],
            last_kline["close"],
            last_kline["volume"],
            last_kline["start"],
            last_kline["end"],
        )

        save_state()

    def stop(self):
        self._stop = True


async def main():
    client = BybitWSClient(BYBIT_WS_URL, TOPICS)
    await client.run_forever()


if __name__ == "__main__":
    try:
        logging.info("⏳ Starting Bybit public WS client ...")
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.warning("KeyboardInterrupt received. Exiting...")
