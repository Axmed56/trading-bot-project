# ========== CONFIGURATION FILE ==========
# إعدادات عامة للبوت

TESTNET = False       # خليها True لو عاوز تجرّب على testnet
LOG_LEVEL = "INFO"

# Binance API (مش مطلوب علشان WebSocket public)
BINANCE_WS = "wss://fstream.binance.com/stream"

# Bybit API (public فقط)
BYBIT_WS = "wss://stream.bybit.com/v5/public/linear"

# الفريم الزمني المستخدم في التحليل
KLINE_INTERVAL = "1"

# العملات التي يتابعها البوت
TRACKED_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]

# مدة صلاحية قرار AI قبل إعادة التحليل (بالثواني)
AI_DECISION_TTL = 15
