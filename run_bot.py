import logging
import time
from decision_engine import DecisionEngine
from execution_bot import ExecutionBot

# -----------------------------------------------------
# إعداد اللوجينج
# -----------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s"
)

logger = logging.getLogger("main_bot")


# -----------------------------------------------------
# 1) تهيئة الـ Execution Bot (التنفيذ على Bybit FUTURES)
# -----------------------------------------------------
API_KEY = "YOUR_BYBIT_TESTNET_API_KEY"
API_SECRET = "YOUR_BYBIT_TESTNET_SECRET"

executor = ExecutionBot(
    api_key=API_KEY,
    api_secret=API_SECRET,
    testnet=True,              # مهم جدًا – تجارب فقط
    default_leverage=10,
    risk_per_trade_usdt=5.0,
)


# -----------------------------------------------------
# 2) تهيئة Decision Engine (دماغ البوت)
# -----------------------------------------------------
def on_new_decision(symbol, decision, ctx):
    """
    لما قرار تداول يتأكد — نفتح صفقة فورًا.
    """

    logger.info(f"🚨 NEW DECISION FIRED → {symbol}: {decision}")

    if decision == "BUY":
        executor.open_position(symbol, "BUY", duration_sec=300)

    elif decision == "SELL":
        executor.open_position(symbol, "SELL", duration_sec=300)


decision_engine = DecisionEngine(
    confirmation_window=5,        # لازم الإشارة تفضل ثابتة 5 ثواني
    on_decision=on_new_decision
)


# -----------------------------------------------------
# 3) محاكاة إشارات AI (بديل مؤقت لغاية ما نربطه بالويب سكت)
# -----------------------------------------------------
# الإشارات التجريبية: ممكن تغير الرموز هنا:
symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]

fake_ai_signals = ["BUY", "BUY", "BUY", "NO_TRADE", "SELL", "SELL"]


# -----------------------------------------------------
# 4) حلقة التشغيل الرئيسية (تجريب)
# -----------------------------------------------------
if __name__ == "__main__":
    logger.info("🚀 BOT STARTED… waiting for signals")

    i = 0
    while True:
        # كل لفة بندي إشارة عشوائية للتجريب
        symbol = symbols[i % len(symbols)]
        signal = fake_ai_signals[i % len(fake_ai_signals)]

        logger.info(f"🧠 AI SIGNAL: {symbol} → {signal}")

        # نرسل الإشارة لموتور اتخاذ القرار
        decision_engine.update_signal(symbol, signal, meta={"source": "fake"})

        # مراقبة الصفقات المفتوحة
        executor.monitor_positions_once()

        i += 1
        time.sleep(1)
