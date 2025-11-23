from binance_ws import start_binance_ws
from bybit_ws import start_bybit_ws
import time

print("🚀 Starting Binance + Bybit WebSocket Feeds...")

start_binance_ws("btcusdt", 20)
start_bybit_ws("BTCUSDT", 25)

while True:
    time.sleep(1)
