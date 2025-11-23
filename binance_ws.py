import asyncio
import json
import websockets
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s"
)

BINANCE_FUTURES_WS = "wss://fstream.binance.com/ws"

SYMBOLS = [
    "btcusdt",
    "ethusdt",
    "solusdt"
]

async def subscribe(ws):
    subs = []
    for sym in SYMBOLS:
        subs.append({
            "method": "SUBSCRIBE",
            "params": [
                f"{sym}@depth5@100ms",
                f"{sym}@trade",
                f"{sym}@kline_1m"
            ],
            "id": 1
        })
    for sub in subs:
        await ws.send(json.dumps(sub))
        await asyncio.sleep(0.1)

async def handle_message(msg):
    try:
        data = json.loads(msg)

        # ORDERBOOK
        if "b" in data and "a" in data:
            bids = data["b"][0][0]
            asks = data["a"][0][0]
            logging.info(f"📘 [BINANCE][ORDERBOOK] bid={bids} | ask={asks}")

        # TRADES
        elif data.get("e") == "trade":
            logging.info(f"💹 [BINANCE][TRADE] side={'Buy' if data['m']==False else 'Sell'} "
                         f"price={data['p']} qty={data['q']}")

        # KLINES
        elif data.get("e") == "kline":
            k = data["k"]
            logging.info(
                f"🕯 [BINANCE][{k['s']} KLINE 1m] "
                f"O:{k['o']} H:{k['h']} L:{k['l']} C:{k['c']} V:{k['v']}"
            )
    except Exception:
        pass

async def main():
    async with websockets.connect(BINANCE_FUTURES_WS) as ws:
        await subscribe(ws)
        logging.info("🚀 Binance Futures WebSocket Connected.")
        while True:
            msg = await ws.recv()
            await handle_message(msg)

if __name__ == "__main__":
    asyncio.run(main())
