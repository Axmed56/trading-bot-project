import asyncio
import json
import websockets
from pathlib import Path

BINANCE_WS = "wss://fstream.binance.com/ws"

# الأزواج اللي هنجرب بيها
SYMBOLS = ["btcusdt", "ethusdt", "solusdt"]

BASE_DIR = Path(__file__).resolve().parent
BINANCE_FILE = BASE_DIR / "binance_prices.json"


async def save_json(path, data):
    tmp = path.with_suffix(".tmp")
    with tmp.open("w") as f:
        json.dump(data, f, indent=2)
    tmp.replace(path)


async def subscribe_binance():
    stream = "/".join([f"{sym}@ticker" for sym in SYMBOLS])
    url = f"{BINANCE_WS}/{stream}"

    async with websockets.connect(url) as ws:
        prices = {}

        while True:
            msg = await ws.recv()
            data = json.loads(msg)

            sym = data["s"].lower()
            price = float(data["c"])

            prices[sym] = {
                "price": price,
                "timestamp": data["E"]
            }

            await save_json(BINANCE_FILE, prices)
            await asyncio.sleep(0.01)


if __name__ == "__main__":
    asyncio.run(subscribe_binance())
