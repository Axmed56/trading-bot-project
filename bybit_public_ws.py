from pybit.unified_trading import WebSocket   # أو حسب المكتبة التي تختارها

api_key = "TudTqZX9rNfU8FW1BC"
api_secret = "mqjhr30Lmb8VdbqzcdtBmA67XgGd2W0JhBVY"

ws = WebSocket(
    testnet=True,             # أو False حسب إذا كنت في بيئة الاختبار
    channel_type="linear",    # أو نوع السوق الذي تستهدفه
    api_key=api_key,
    api_secret=api_secret
)

def handle_message(msg):
    print("Received:", msg)

# مثال: الاشتراك بشمعة (kline) لفترة 1 دقيقة لرمز BTCUSDT
ws.kline_stream(
    callback=handle_message,
    symbol="BTCUSDT",
    interval=1
)

# أو حلقة تشغيل بسيطة
import time
while True:
    time.sleep(1)
