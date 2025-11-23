from bybit_ws import start_bybit_orderbook
import time

start_bybit_orderbook("BTCUSDT", 25)

while True:
    time.sleep(1)
