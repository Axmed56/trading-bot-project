import json
import websocket
import threading
import time


# ============================
# 📌 اعدادات
# ============================
SYMBOL = "BTCUSDT"
DEPTH = 25
WS_URL = "wss://stream.bybit.com/v5/public/linear"

# نخزن اخر دفتر أوامر
last_orderbook = {
    "bids": [],
    "asks": []
}


# ============================
# 📌 دالة تحليل دفتر الاوامر
# ============================
def analyze(bids, asks):
    bid_vol = sum([float(x[1]) for x in bids])
    ask_vol = sum([float(x[1]) for x in asks])

    imbalance = 0
    if bid_vol + ask_vol != 0:
        imbalance = (bid_vol - ask_vol) / (bid_vol + ask_vol) * 100

    if imbalance > 15:
        regime = "🟢 اتجاه شراء"
    elif imbalance < -15:
        regime = "🔴 اتجاه بيع"
    else:
        regime = "⚪ متعادل"

    return bid_vol, ask_vol, imbalance, regime


# ============================
# 📌 Print summary
# ============================
def print_report(bids, asks):
    bid_vol, ask_vol, imbalance, regime = analyze(bids, asks)

    print("\n" + "="*60)
    print(f"📡 Live Orderbook – {SYMBOL}")
    print(f"Bid Volume (Top {DEPTH}): {bid_vol:.2f}")
    print(f"Ask Volume (Top {DEPTH}): {ask_vol:.2f}")
    print(f"Imbalance %: {imbalance:+.2f}%")
    print(f"Market Regime: {regime}")


# ============================
# 📌 WebSocket Handlers
# ============================
def on_open(ws):
    print(f"⚡ Connected → Subscribing to orderbook.{DEPTH}.{SYMBOL}")

    sub_msg = {
        "op": "subscribe",
        "args": [f"orderbook.{DEPTH}.{SYMBOL}"]
    }
    ws.send(json.dumps(sub_msg))


def on_message(ws, message):
    global last_orderbook

    try:
        data = json.loads(message)
    except:
        return

    if "data" not in data:
        return

    book = data["data"][0]

    bids = book.get("b", [])
    asks = book.get("a", [])

    # Clean as (price, size)
    bids = [(float(p), float(s)) for p, s, *_ in bids]
    asks = [(float(p), float(s)) for p, s, *_ in asks]

    last_orderbook["bids"] = bids
    last_orderbook["asks"] = asks

    print_report(bids, asks)


def on_error(ws, error):
    print("❌ WebSocket Error:", error)


def on_close(ws, code, msg):
    print("🔴 WebSocket Closed:", code, msg)


# ============================
# 📌 تشغيل WebSocket
# ============================
def start_ws():
    websocket.enableTrace(False)

    ws = websocket.WebSocketApp(
        WS_URL,
        on_open=on_open,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close
    )

    ws.run_forever()
