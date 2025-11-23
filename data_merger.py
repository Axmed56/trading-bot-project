from collections import deque

binance_data = deque(maxlen=1)
bybit_data = deque(maxlen=1)

def update_binance(data):
    binance_data.append(data)

def update_bybit(data):
    bybit_data.append(data)

def merged_snapshot():
    return {
        "binance": binance_data[-1] if binance_data else None,
        "bybit": bybit_data[-1] if bybit_data else None
    }
