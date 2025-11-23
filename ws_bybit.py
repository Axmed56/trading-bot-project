import json
import websocket
import time
import os

SYMBOL = "BTCUSDT"
DEPTH = 25

OUTPUT_FILE = "data/bybit_orderbook.json"
WS_ENDPOINT = "wss://stream.bybit.com/v5/public/linear"
TOPIC = f"orderbook.{DEPTH}.{SYMBOL}"

os.makedirs("data", exist_ok=True)

orderbook = {"bids": [], "asks": [], "timestamp": None}


def on_open(ws):
    sub_msg = {"op": "subscribe", "args": [TOPIC]}
    ws.send(json.dumps(sub_msg))
    print(f"Subscribed → {TOPIC}")


def on_message(ws, message):
    global orderbook
    msg = json.loads(message)

    if msg.get("topic") != TOPIC:
        return

    data = msg.get("data", {})
    orderbook["bids"] = data.get("b", [])
    orderbook["asks"] = data.get("a", [])
    orderbook["timestamp"] = msg.get("ts")

    with open(OUTPUT_FILE, "w") as f:
        json.dump(orderbook, f)

    bid_vol = sum(float(b[1]) for b in orderbook["bids"])
    ask_vol = sum(float(a[1]) for a in orderbook["asks"])
    imbalance = (bid_vol - ask_vol) / (bid_vol + ask_vol + 1e-9) * 100

    print("===============================")
    print("📡 Bybit Live")
    print(f"Bid Vol: {bid_vol:.2f} | Ask Vol: {ask_vol:.2f} | Imb: {imbalance:+.2f}%")


def on_error(ws, error):
    print("ERROR:", error)


def on_close(ws, code, msg):
    print("Closed:", code, msg)


def run():
    ws = websocket.WebSocketApp(
        WS_ENDPOINT,
        on_open=on_open,
        on_message=on_message,
        on_close=on_close,
        on_error=on_error
    )
    ws.run_forever()


if __name__ == "__main__":
    run()
