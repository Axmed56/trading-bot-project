import asyncio
import json
import logging
import websockets

# ================== إعداد اللوج ==================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)

# ================== الإعدادات الأساسية ==================
# هنستخدم Binance Spot WebSocket لأنه مستقر وعنوانه صحيح
BINANCE_WS_BASE = "wss://stream.binance.com:9443/stream?streams="
# Bybit Futures public market data
BYBIT_WS_URL = "wss://stream.bybit.com/v5/public/linear"

# رموز المراقبة (ممكن تغيّرها بعدين)
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]


# ============== بناء ستريم بينانس (أمر واحد لعدة Streams) ==============
def build_binance_stream_url():
    streams = []
    for s in SYMBOLS:
        base = s.lower()  # binance ws بيتعامل بحروف صغيرة
        streams.append(f"{base}@trade")         # صفقات
        streams.append(f"{base}@kline_1m")      # شموع دقيقة
        streams.append(f"{base}@depth5@100ms")  # دفتر أوامر 5 مستويات
    stream_str = "/".join(streams)
    return BINANCE_WS_BASE + stream_str


# ================== Binance WebSocket ==================
async def binance_ws():
    url = build_binance_stream_url()
    logging.info(f"🌐 [BINANCE] Connecting WS: {url}")

    while True:
        try:
            async with websockets.connect(url, ping_interval=20, ping_timeout=10) as ws:
                logging.info("✅ [BINANCE] Connected")
                while True:
                    msg = await ws.recv()
                    data = json.loads(msg)

                    stream = data.get("stream", "")
                    payload = data.get("data", {})

                    if "@trade" in stream:
                        symbol = payload.get("s", "")
                        side = "Buy" if payload.get("m") is False else "Sell"
                        price = float(payload.get("p", 0))
                        qty = float(payload.get("q", 0))
                        logging.info(
                            f"💹 [BINANCE][{symbol}][TRADE] side={side} price={price} qty={qty}"
                        )

                    elif "@kline_1m" in stream:
                        k = payload.get("k", {})
                        symbol = k.get("s", "")
                        o = float(k.get("o", 0))
                        h = float(k.get("h", 0))
                        l = float(k.get("l", 0))
                        c = float(k.get("c", 0))
                        v = float(k.get("v", 0))
                        logging.info(
                            f"🕯 [BINANCE][{symbol}][KLINE 1m] "
                            f"O:{o} H:{h} L:{l} C:{c} V:{v}"
                        )

                    elif "@depth5@" in stream:
                        symbol = payload.get("s", "")
                        bids = payload.get("b", [])
                        asks = payload.get("a", [])
                        best_bid = float(bids[0][0]) if bids else None
                        best_ask = float(asks[0][0]) if asks else None
                        logging.info(
                            f"📘 [BINANCE][{symbol}][ORDERBOOK] best_bid={best_bid} | best_ask={best_ask}"
                        )

        except Exception as e:
            logging.error(f"❌ [BINANCE] WS error: {e}. إعادة المحاولة خلال 5 ثوانٍ...")
            await asyncio.sleep(5)


# ================== Bybit WebSocket ==================
async def bybit_ws():
    logging.info(f"🌐 [BYBIT] Connecting WS: {BYBIT_WS_URL}")

    # تجهيز رموز Bybit (نفسها لكن بصيغة bybit مثل BTCUSDT)
    # Bybit topics:
    # orderbook: orderbook.1.BTCUSDT
    # trades: publicTrade.BTCUSDT
    # kline: kline.1.BTCUSDT
    orderbook_topics = [f"orderbook.1.{s}" for s in SYMBOLS]
    trade_topics = [f"publicTrade.{s}" for s in SYMBOLS]
    kline_topics = [f"kline.1.{s}" for s in SYMBOLS]

    subscribe_args = orderbook_topics + trade_topics + kline_topics

    while True:
        try:
            async with websockets.connect(BYBIT_WS_URL, ping_interval=20, ping_timeout=10) as ws:
                logging.info("✅ [BYBIT] Connected WS")

                sub_msg = {
                    "op": "subscribe",
                    "args": subscribe_args
                }
                await ws.send(json.dumps(sub_msg))
                logging.info(f"📨 [BYBIT] Subscribed to {len(subscribe_args)} streams")

                while True:
                    msg = await ws.recv()
                    data = json.loads(msg)

                    topic = data.get("topic", "")
                    if not topic:
                        # غالبًا رسائل system/ping
                        continue

                    # -------- Orderbook --------
                    if topic.startswith("orderbook.1."):
                        symbol = topic.split(".")[2]
                        ob = data.get("data", {})
                        bids = ob.get("b", [])
                        asks = ob.get("a", [])
                        best_bid = float(bids[0][0]) if bids else None
                        best_ask = float(asks[0][0]) if asks else None
                        logging.info(
                            f"📘 [BYBIT][{symbol}][ORDERBOOK] best_bid={best_bid} | best_ask={best_ask}"
                        )

                    # -------- Trades --------
                    elif topic.startswith("publicTrade."):
                        symbol = topic.split(".")[1]
                        trades = data.get("data", [])
                        for t in trades:
                            side = t.get("S", "")
                            price = float(t.get("p", 0))
                            qty = float(t.get("v", 0))
                            logging.info(
                                f"💹 [BYBIT][{symbol}][TRADE] side={side} price={price} size={qty}"
                            )

                    # -------- Klines --------
                    elif topic.startswith("kline.1."):
                        symbol = topic.split(".")[2]
                        klines = data.get("data", [])
                        for k in klines:
                            o = float(k.get("o", 0))
                            h = float(k.get("h", 0))
                            l = float(k.get("l", 0))
                            c = float(k.get("c", 0))
                            v = float(k.get("v", 0))
                            logging.info(
                                f"🕯 [BYBIT][{symbol}][KLINE 1m] "
                                f"O:{o} H:{h} L:{l} C:{c} V:{v}"
                            )

        except Exception as e:
            logging.error(f"❌ [BYBIT] WS error: {e}. إعادة المحاولة خلال 5 ثوانٍ...")
            await asyncio.sleep(5)


# ================== نقطة التشغيل الرئيسية ==================
async def main():
    logging.info("🚀 Starting Dual Market Data Stream (Binance Spot + Bybit Futures)")
    await asyncio.gather(
        binance_ws(),
        bybit_ws()
    )


if __name__ == "__main__":
    asyncio.run(main())
