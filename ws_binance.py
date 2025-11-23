import json
import asyncio
import websockets
import os

os.makedirs("data", exist_ok=True)

OUTPUT = "data/binance_depth.json"

BINANCE_WS = "wss://stream.binance.com:9443/ws/btcusdt@depth5@100ms"


async def run():
    async with websockets.connect(BINANCE_WS) as ws:
        print("Connected → Binance depth stream")

        while True:
            msg = await ws.recv()
            data = json.loads(msg)

            output = {
                "bids": data.get("b", []),
                "asks": data.get("a", []),
                "timestamp": data.get("E")
            }

            with open(OUTPUT, "w") as f:
                json.dump(output, f)

            print("📡 Binance depth updated")


asyncio.run(run())
