from pybit.unified_trading import WebSocket
import time

# Pair = العملة
SYMBOL = "BTCUSDT"

def on_orderbook(message):
    """ استلام الداتا المباشرة من بايبيت """
    if "data" not in message:
        return
    
    data = message["data"][0]
    bids = data.get("b", [])
    asks = data.get("a", [])

    if bids and asks:
        best_bid = bids[0]
        best_ask = asks[0]

        print("\n=== BYBIT ORDERBOOK UPDATE ===")
        print("Best Bid:", best_bid)
        print("Best Ask:", best_ask)

def main():
    print("⏳ Connecting to Bybit…")

    ws = WebSocket(
        testnet=False,  # لو عايز Testnet خليها True
        channel_type="linear"
    )

    # Orderbook depth 25
    ws.orderbook_stream(
        depth=25,
        symbol=SYMBOL,
        callback=on_orderbook
    )

    while True:
        time.sleep(1)

if __name__ == "__main__":
    main()
