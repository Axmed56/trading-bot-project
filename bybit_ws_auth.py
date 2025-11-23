import json
import time
import hmac
import hashlib
import websocket

API_KEY = "TudTqZX9rNfU8FW1BC"
API_SECRET = "mqjhr30Lmb8VdbqzcdtBmA67XgGd2W0JhBVY"

SYMBOL = "BTCUSDT"
DEPTH = 25

ws_url = "wss://stream.bybit.com/v5/public/linear"


def generate_signature(timestamp: str):
    """توليد التوقيع حسب شروط Bybit v5"""
    message = timestamp + API_KEY
    signature = hmac.new(
        API_SECRET.encode(), 
        message.encode(), 
        hashlib.sha256
    ).hexdigest()
    return signature


def on_open(ws):
    print("🔗 Connected → Authenticating…")

    ts = str(int(time.time() * 1000))
    sign = generate_signature(ts)

    auth = {
        "op": "auth",
        "args": [API_KEY, ts, sign]
    }

    ws.send(json.dumps(auth))
    print("🔐 Auth Sent")

    # بعد نجاح المصادقة هنبعت الاشتراك
    sub = {
        "op": "subscribe",
        "args": [f"orderbook.{DEPTH}.{SYMBOL}"]
    }
    ws.send(json.dumps(sub))
    print(f"📡 Subscribed to: orderbook.{DEPTH}.{SYMBOL}")


def on_message(ws, message):
    data = json.loads(message)

    if "success" in data and data["success"] and data.get("op") == "auth":
        print("✅ Authentication Success!")

    if "topic" in data:
        if data["topic"].startswith("orderbook"):
            orderbook = data["data"]

            bids = orderbook["b"][:3]
            asks = orderbook["a"][:3]

            print("\n📘 BYBIT ORDERBOOK UPDATE")
            print("Top 3 Bids:", bids)
            print("Top 3 Asks:", asks)


def on_error(ws, error):
    print("❌ Error:", error)


def on_close(ws, close_code, close_msg):
    print("🔴 WebSocket Closed:", close_code, close_msg)


if __name__ == "__main__":
    ws = websocket.WebSocketApp(
        ws_url,
        on_open=on_open,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close
    )
    ws.run_forever()
