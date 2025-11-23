# futures_ai_dashboard.py
# =====================================================
# Live Futures AI Dashboard (Binance + Bybit + AI)
# =====================================================
import asyncio
import json
import logging
import time
from typing import Dict, Tuple

import websockets

from ai_brain import MarketSnapshot, SimpleOrderflowAI

# ---------- إعداد اللوج ----------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
log = logging.getLogger("FUTURES_AI_DASH")

# ---------- إعدادات عامة ----------
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]  # تقدر تعدلها بعدين

# Binance FUTURES WS
BINANCE_FUTURES_WS = "wss://fstream.binance.com/stream"

# Bybit Linear Futures WS
BYBIT_LINEAR_WS = "wss://stream.bybit.com/v5/public/linear"

# AI Layer
ai = SimpleOrderflowAI()

# حالة السوق الحالية لكل (exchange, symbol)
snapshots: Dict[Tuple[str, str], MarketSnapshot] = {}
last_actions: Dict[Tuple[str, str], str] = {}


# =====================================================
# Utilities
# =====================================================
def get_snapshot(exchange: str, symbol: str) -> MarketSnapshot:
    key = (exchange, symbol)
    if key not in snapshots:
        snapshots[key] = MarketSnapshot(symbol=symbol, exchange=exchange)
    return snapshots[key]


def maybe_log_ai(exchange: str, symbol: str):
    snap = get_snapshot(exchange, symbol)
    res = ai.evaluate(snap)
    action = res["action"]
    score = res["score"]
    reason = res["reason"]

    key = (exchange, symbol)
    prev = last_actions.get(key)
    last_actions[key] = action

    # آيكونات مختلفة لكل حالة
    icon_map = {
        "BUY_5M": "🟢",
        "SELL_5M": "🔴",
        "NO_TRADE": "⚪",
    }
    icon = icon_map.get(action, "⚪")

    # نطبع دائمًا تحليلات الـ AI كي تراها على الشاشة
    log.info(
        f"{icon} [AI][{exchange}][{symbol}] action={action} | score={score:.2f} | {reason}"
    )



# =====================================================
# Binance Futures Handler
# =====================================================
async def binance_futures_handler():
    """
    WebSocket لـ Binance Futures USDT-M:
    - depth5@100ms
    - aggTrade
    - kline_1m
    """
    # build streams
    streams = []
    for s in SYMBOLS:
        s_lower = s.lower()
        streams.append(f"{s_lower}@depth5@100ms")
        streams.append(f"{s_lower}@aggTrade")
        streams.append(f"{s_lower}@kline_1m")

    streams_param = "/".join(streams)
    url = f"{BINANCE_FUTURES_WS}?streams={streams_param}"

    while True:
        try:
            log.info(f"📡 [BINANCE] Connecting Futures WS: {url}")
            async with websockets.connect(url, ping_interval=20, ping_timeout=20) as ws:
                log.info("✅ [BINANCE] Connected.")

                async for msg in ws:
                    try:
                        data = json.loads(msg)
                    except Exception as e:
                        log.warning(f"[BINANCE] JSON parse error: {e}")
                        continue

                    stream = data.get("stream", "")
                    payload = data.get("data", {})

                    if not stream or not payload:
                        continue

                    # استنتاج الرمز من اسم الـ stream
                    # example: btcusdt@depth5@100ms
                    try:
                        stream_symbol = stream.split("@")[0].upper()
                    except Exception:
                        stream_symbol = "UNKNOWN"

                    # Orderbook
                    if "@depth5" in stream:
                        bids = payload.get("b", [])
                        asks = payload.get("a", [])
                        best_bid = best_ask = None
                        best_bid_size = best_ask_size = None

                        if bids:
                            best_bid = float(bids[0][0])
                            best_bid_size = float(bids[0][1])
                        if asks:
                            best_ask = float(asks[0][0])
                            best_ask_size = float(asks[0][1])

                        snap = get_snapshot("BINANCE", stream_symbol)
                        snap.best_bid = best_bid
                        snap.best_bid_size = best_bid_size
                        snap.best_ask = best_ask
                        snap.best_ask_size = best_ask_size
                        snap.last_update_ts = time.time()

                        log.info(
                            f"📘 [BINANCE][{stream_symbol}][ORDERBOOK] "
                            f"best_bid={best_bid} | best_ask={best_ask}"
                        )

                        maybe_log_ai("BINANCE", stream_symbol)

                    # Trades
                    elif "@aggtrade" in stream or "@aggTrade" in stream or "@trade" in stream:
                        price = float(payload.get("p", 0.0))
                        qty = float(payload.get("q", 0.0))
                        # m = True يعني buyer هو الـ maker => صفقة بيع
                        is_buyer_maker = payload.get("m", False)
                        side = "Sell" if is_buyer_maker else "Buy"

                        snap = get_snapshot("BINANCE", stream_symbol)
                        snap.last_price = price
                        snap.last_qty = qty
                        snap.last_side = side
                        snap.last_update_ts = time.time()

                        log.info(
                            f"💹 [BINANCE][{stream_symbol}][TRADE] "
                            f"side={side} price={price} qty={qty}"
                        )

                        maybe_log_ai("BINANCE", stream_symbol)

                    # Candles 1m
                    elif "@kline_1m" in stream:
                        k = payload.get("k", {})
                        o = k.get("o")
                        h = k.get("h")
                        l = k.get("l")
                        c = k.get("c")
                        v = k.get("v")
                        log.info(
                            f"🕯 [BINANCE][{stream_symbol}][KLINE 1m] "
                            f"O:{o} H:{h} L:{l} C:{c} V:{v}"
                        )

        except Exception as e:
            log.warning(f"🛑 [BINANCE] WS error: {e}. Reconnecting in 5s...")
            await asyncio.sleep(5)


# =====================================================
# Bybit Linear Futures Handler
# =====================================================
async def bybit_linear_handler():
    """
    WebSocket لـ Bybit Linear Futures:
    - orderbook.1.SYMBOL
    - publicTrade.SYMBOL
    - kline.1.SYMBOL
    """
    while True:
        try:
            log.info(f"📡 [BYBIT] Connecting Linear Futures WS: {BYBIT_LINEAR_WS}")
            async with websockets.connect(
                BYBIT_LINEAR_WS, ping_interval=20, ping_timeout=20
            ) as ws:
                log.info("✅ [BYBIT] Connected.")

                # تحضير الـ topics
                topics = []
                for s in SYMBOLS:
                    topics.append(f"orderbook.1.{s}")
                    topics.append(f"publicTrade.{s}")
                    topics.append(f"kline.1.{s}")

                sub_msg = {
                    "op": "subscribe",
                    "args": topics,
                }
                await ws.send(json.dumps(sub_msg))
                log.info(f"📨 [BYBIT] Subscribed to: {topics}")

                async for msg in ws:
                    try:
                        data = json.loads(msg)
                    except Exception as e:
                        log.warning(f"[BYBIT] JSON parse error: {e}")
                        continue

                    topic = data.get("topic")
                    if not topic:
                        # رسائل Ping/Pong أو Info
                        continue

                    # topic example: orderbook.1.BTCUSDT
                    parts = topic.split(".")
                    if len(parts) < 2:
                        continue

                    t_type = parts[0]  # orderbook / publicTrade / kline
                    symbol = parts[-1]  # BTCUSDT

                    payload = data.get("data")
                    if payload is None:
                        continue

                    # Bybit أحياناً تبعت data كـ dict أو list
                    if isinstance(payload, list):
                        if not payload:
                            continue
                        payload = payload[0]

                    # Orderbook
                    if t_type == "orderbook":
                        bids = payload.get("b", [])
                        asks = payload.get("a", [])
                        best_bid = best_ask = None
                        best_bid_size = best_ask_size = None

                        if bids:
                            best_bid = float(bids[0][0])
                            best_bid_size = float(bids[0][1])
                        if asks:
                            best_ask = float(asks[0][0])
                            best_ask_size = float(asks[0][1])

                        snap = get_snapshot("BYBIT", symbol)
                        snap.best_bid = best_bid
                        snap.best_bid_size = best_bid_size
                        snap.best_ask = best_ask
                        snap.best_ask_size = best_ask_size
                        snap.last_update_ts = time.time()

                        log.info(
                            f"📘 [BYBIT][{symbol}][ORDERBOOK] "
                            f"best_bid={best_bid} | best_ask={best_ask}"
                        )

                        maybe_log_ai("BYBIT", symbol)

                    # Trades
                    elif t_type == "publicTrade":
                        # قد تكون قائمة من التريدات
                        trades = data.get("data", [])
                        if isinstance(trades, dict):
                            trades = [trades]

                        for tr in trades:
                            price = float(tr.get("p", 0.0))
                            qty = float(tr.get("v", 0.0))
                            side = tr.get("S", "Buy")  # "Buy" أو "Sell"

                            snap = get_snapshot("BYBIT", symbol)
                            snap.last_price = price
                            snap.last_qty = qty
                            snap.last_side = side
                            snap.last_update_ts = time.time()

                            log.info(
                                f"💹 [BYBIT][{symbol}][TRADE] "
                                f"side={side} price={price} size={qty}"
                            )

                            maybe_log_ai("BYBIT", symbol)

                    # Kline 1m
                    elif t_type == "kline":
                        klines = data.get("data", [])
                        if isinstance(klines, dict):
                            klines = [klines]

                        for k in klines:
                            o = k.get("o")
                            h = k.get("h")
                            l = k.get("l")
                            c = k.get("c")
                            v = k.get("v")
                            log.info(
                                f"🕯 [BYBIT][{symbol}][KLINE 1m] "
                                f"O:{o} H:{h} L:{l} C:{c} V:{v}"
                            )

        except Exception as e:
            log.warning(f"🛑 [BYBIT] WS error: {e}. Reconnecting in 5s...")
            await asyncio.sleep(5)


# =====================================================
# Main
# =====================================================
async def main():
    log.info("🚀 Starting Futures AI Dashboard (Binance + Bybit + AI)")
    tasks = [
        asyncio.create_task(binance_futures_handler()),
        asyncio.create_task(bybit_linear_handler()),
    ]
    await asyncio.gather(*tasks)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        log.warning("🛑 Stopped by user.")
