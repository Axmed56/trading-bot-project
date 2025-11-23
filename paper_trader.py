# -*- coding: utf-8 -*-
"""
PAPER_TRADER v2
Binance Futures + Bybit Futures
RSI Strategy (1m timeframe) + Paper Trading
"""

import time
import logging
from datetime import datetime

import ccxt

# ================== إعداد اللوج ==================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# ================== إعدادات عامة ==================
SYMBOLS = [
    "BTC/USDT:USDT",
    "ETH/USDT:USDT",
    "SOL/USDT:USDT",
]

TIMEFRAME = "1m"
RSI_PERIOD = 14

LOOP_SLEEP_SECONDS = 60   # ثانية بين كل دورة (شمعة 1 دقيقة)

# إعدادات الحساب الوهمي
STARTING_CASH = 10000.0
RISK_PER_TRADE = 0.02  # 2% من رأس المال في كل صفقة

# حدود RSI
RSI_BUY_LEVEL = 30
RSI_SELL_LEVEL = 70


# ================== دوال مساعدة ==================
def init_exchanges():
    """تجهيز Binance Futures + Bybit USDT Perpetual"""
    binance = ccxt.binance({
        "enableRateLimit": True,
        "options": {
            "defaultType": "future",
        },
    })

    bybit = ccxt.bybit({
        "enableRateLimit": True,
        "options": {
            "defaultType": "future",   # USDT Perpetual
        },
    })

    logging.info("Loading markets from Binance Futures...")
    binance.load_markets()
    logging.info("Loading markets from Bybit Futures...")
    bybit.load_markets()

    return binance, bybit


def compute_rsi(closes, period=14):
    """حساب RSI بسيط من قائمة أسعار الإغلاق."""
    if len(closes) < period + 1:
        return None

    gains = []
    losses = []
    for i in range(1, period + 1):
        diff = closes[i] - closes[i - 1]
        if diff >= 0:
            gains.append(diff)
            losses.append(0.0)
        else:
            gains.append(0.0)
            losses.append(-diff)

    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period

    if avg_loss == 0:
        return 100.0

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def format_symbol(symbol: str) -> str:
    """تحويل 'BTC/USDT:USDT' إلى 'BTCUSDT' للعرض فقط."""
    left = symbol.split(":")[0]
    return left.replace("/", "")


# ================== كلاس الحساب الوهمي ==================
class PaperAccount:
    def __init__(self, starting_cash):
        self.cash = starting_cash
        # لكل رمز: dict يحتوي على position / entry_price / qty
        self.positions = {
            s: {"side": "FLAT", "entry_price": None, "qty": 0.0}
            for s in SYMBOLS
        }
        self.realized_pnl = 0.0

    def _position_value(self, symbol, price):
        pos = self.positions[symbol]
        if pos["side"] == "FLAT" or pos["qty"] == 0:
            return 0.0
        direction = 1 if pos["side"] == "LONG" else -1
        return direction * (price - pos["entry_price"]) * pos["qty"]

    def total_equity(self, prices):
        equity = self.cash + self.realized_pnl
        for symbol in SYMBOLS:
            price = prices.get(symbol)
            if price is None:
                continue
            equity += self._position_value(symbol, price)
        return equity

    def open_position(self, symbol, side, price):
        """فتح صفقة جديدة بنسبة RISK_PER_TRADE من رأس المال."""
        pos = self.positions[symbol]
        if pos["side"] != "FLAT":
            # يوجد صفقة مفتوحة بالفعل
            return

        risk_capital = self.cash * RISK_PER_TRADE
        if price <= 0:
            return

        qty = risk_capital / price
        if qty <= 0:
            return

        pos["side"] = side
        pos["entry_price"] = price
        pos["qty"] = qty

        logging.info(
            f"[{format_symbol(symbol)}] OPEN {side} @ {price:.2f} qty={qty:.6f}"
        )

    def close_position(self, symbol, price):
        """إغلاق الصفقة الحالية وحساب الربح/الخسارة."""
        pos = self.positions[symbol]
        if pos["side"] == "FLAT" or pos["qty"] == 0:
            return

        direction = 1 if pos["side"] == "LONG" else -1
        pnl = direction * (price - pos["entry_price"]) * pos["qty"]
        self.realized_pnl += pnl
        self.cash += 0  # لا نحرك الكاش هنا لأننا نتعامل بأن القيمة تظهر في الـ PnL

        logging.info(
            f"[{format_symbol(symbol)}] CLOSE {pos['side']} @ {price:.2f} "
            f"entry={pos['entry_price']:.2f} qty={pos['qty']:.6f} pnl={pnl:.2f}"
        )

        pos["side"] = "FLAT"
        pos["entry_price"] = None
        pos["qty"] = 0.0

    def position_side(self, symbol):
        return self.positions[symbol]["side"]


# ================== الحلقة الرئيسية ==================
def main():
    logging.info("Starting PAPER TRADER v2 (Binance + Bybit Futures, RSI Strategy)")

    binance, bybit = init_exchanges()
    account = PaperAccount(STARTING_CASH)

    while True:
        loop_prices_binance = {}
        loop_prices_bybit = {}

        loop_start = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        logging.info("-" * 80)
        logging.info(f"SNAPSHOT UTC: {loop_start}")

        for symbol in SYMBOLS:
            code = format_symbol(symbol)

            try:
                # ---------- بيانات Binance (شموع + سعر) ----------
                ohlcv = binance.fetch_ohlcv(symbol, timeframe=TIMEFRAME, limit=RSI_PERIOD + 1)
                closes = [c[4] for c in ohlcv]
                last_price_binance = closes[-1]

                # ---------- بيانات Bybit (سعر لحظي من الفيوتشر) ----------
                try:
                    ticker_bybit = bybit.fetch_ticker(symbol)
                    last_price_bybit = ticker_bybit.get("last")
                except Exception as e:
                    logging.error(f"[{code}] Bybit ticker error: {e}")
                    last_price_bybit = None

                loop_prices_binance[symbol] = last_price_binance
                if last_price_bybit is not None:
                    loop_prices_bybit[symbol] = last_price_bybit

                # ---------- حساب RSI من بيانات Binance ----------
                rsi = compute_rsi(closes, RSI_PERIOD)
                rsi_str = f"{rsi:.2f}" if rsi is not None else "N/A"

                # ---------- منطق الإشارة (بناءً على Binance RSI) ----------
                side = account.position_side(symbol)
                signal = "HOLD"

                if rsi is not None:
                    if side == "FLAT":
                        if rsi <= RSI_BUY_LEVEL:
                            signal = "BUY"
                            account.open_position(symbol, "LONG", last_price_binance)
                        elif rsi >= RSI_SELL_LEVEL:
                            signal = "SELL"
                            account.open_position(symbol, "SHORT", last_price_binance)
                    elif side == "LONG":
                        # خروج آمن من الشراء
                        if rsi >= 50:
                            signal = "EXIT_LONG"
                            account.close_position(symbol, last_price_binance)
                    elif side == "SHORT":
                        # خروج آمن من البيع
                        if rsi <= 50:
                            signal = "EXIT_SHORT"
                            account.close_position(symbol, last_price_binance)

                # ---------- طباعة السطر ----------
                if last_price_bybit is not None:
                    spread = last_price_bybit - last_price_binance
                    logging.info(
                        f"[{code}] BINANCE={last_price_binance:.2f}  "
                        f"BYBIT={last_price_bybit:.2f}  "
                        f"SPREAD={spread:.2f}  "
                        f"RSI={rsi_str}  "
                        f"pos={side}  signal={signal}"
                    )
                else:
                    logging.info(
                        f"[{code}] BINANCE={last_price_binance:.2f}  "
                        f"BYBIT=N/A  "
                        f"RSI={rsi_str}  "
                        f"pos={side}  signal={signal}"
                    )

            except Exception as e:
                logging.error(f"[{code}] main loop error: {e}")
                continue

        # ---------- إجمالي الحساب ----------
        equity = account.total_equity(loop_prices_binance)
        logging.info(
            f"EQUITY SNAPSHOT | cash={account.cash:.2f}  "
            f"realized_pnl={account.realized_pnl:.2f}  "
            f"total_equity={equity:.2f}"
        )

        # انتظار الشمعة التالية
        try:
            time.sleep(LOOP_SLEEP_SECONDS)
        except KeyboardInterrupt:
            logging.info("KeyboardInterrupt – stopping PAPER TRADER.")
            break


if __name__ == "__main__":
    main()
