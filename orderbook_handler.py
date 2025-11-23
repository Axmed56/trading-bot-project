# orderbook_handler.py
import numpy as np

def analyze_orderbook(bids, asks):
    """تحليل دفتر الأوامر: حجم، توازن، حيتان"""
    bid_vol = sum([size for _, size in bids])
    ask_vol = sum([size for _, size in asks])

    total = bid_vol + ask_vol

    if total == 0:
        imbalance = 0.0
    else:
        imbalance = (bid_vol - ask_vol) / total * 100

    if imbalance > 15:
        regime = "📈 سوق صاعد (سيولة شراء)"
    elif imbalance < -15:
        regime = "📉 سوق هابط (سيولة بيع)"
    else:
        regime = "⚪ سوق متوازن"

    # كشف أكبر 10 مستويات (الحيتان)
    arr = [{"side": "bid", "price": p, "size": s} for p, s in bids] + \
          [{"side": "ask", "price": p, "size": s} for p, s in asks]

    arr_sorted = sorted(arr, key=lambda x: x["size"], reverse=True)
    whale_levels = arr_sorted[:10]

    return {
        "bid_vol": bid_vol,
        "ask_vol": ask_vol,
        "imbalance": imbalance,
        "regime": regime,
        "whales": whale_levels
    }
