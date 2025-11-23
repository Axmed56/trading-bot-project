# bybit_orderbook_pybit.py
import time
from pybit.unified_trading import WebSocket

# ===== إعدادات عامة =====
SYMBOL = "BTCUSDT"     # عقد بيربتشوال USDT على Bybit
DEPTH = 25             # عدد مستويات دفتر الأوامر

# لو عاوز تستخدم API Keys (مش مطلوب للـ public orderbook)
API_KEY = ""           # اختياري
API_SECRET = ""        # اختياري

def handle_orderbook(message: dict):
    """
    كول باك بيشتغل مع كل تحديث لدفتر الأوامر.
    الرسالة بتيجي من Bybit فورمات v5 unified.
    """
    if "data" not in message:
        return

    data_list = message.get("data", [])
    if not data_list:
        return

    data = data_list[0]

    bids = data.get("b", [])
    asks = data.get("a", [])

    if not bids or not asks:
        return

    # كل عنصر: [price, size, ...]
    top_bid = bids[0]
    top_ask = asks[0]

    bid_price, bid_size = float(top_bid[0]), float(top_bid[1])
    ask_price, ask_size = float(top_ask[0]), float(top_ask[1])

    print("\n=== BYBIT ORDERBOOK UPDATE ===")
    print(f"Top Bid : {bid_price:.2f}  | Size: {bid_size:.4f}")
    print(f"Top Ask : {ask_price:.2f}  | Size: {ask_size:.4f}")

def main():
    print("⏳ Connecting to Bybit (Unified Trading WS)…")

    # channel_type مهم جدًّا:
    #   - "linear"  لعقود USDT perpetual زي BTCUSDT
    #   - "inverse" لعقود inverse
    #   - "spot"    لو Spot
    ws = WebSocket(
        testnet=False,            # لو عايز testnet خليه True
        channel_type="linear",    # إحنا شغالين على USDT Perp
        api_key=API_KEY or None,  # مش شرط في public
        api_secret=API_SECRET or None,
    )

    # دي هي الطريقة الصح لـ pybit unified بدل ما نبعث topic يدوي
    ws.orderbook_stream(
        depth=DEPTH,
        symbol=SYMBOL,
        callback=handle_orderbook
    )

    print(f"✅ Subscribed to Bybit orderbook depth={DEPTH} for {SYMBOL}")
    print("👂 Listening for live updates... (Ctrl + C للإيقاف)")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Stopped by user.")


if __name__ == "__main__":
    main()
