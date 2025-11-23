# ai_brain.py
"""
Mini AI Brain v1
----------------
دماغ بسيطة بتاخد:
- symbol
- price
- rsi
- prev_rsi
- position (FLAT / LONG / SHORT)

وترجع قرار: BUY / SELL / HOLD
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class MarketContext:
    symbol: str
    price: float
    rsi: Optional[float]
    prev_rsi: Optional[float]
    position: str  # "FLAT" / "LONG" / "SHORT"


class AIBrain:
    def __init__(self):
        # ممكن بعدين نزود إعدادات هنا
        self.rsi_buy_level = 35.0
        self.rsi_sell_level = 65.0
        self.extreme_oversold = 25.0
        self.extreme_overbought = 75.0

    def decide(self, ctx: MarketContext) -> str:
        """يرجع إشارة: BUY / SELL / HOLD"""

        rsi = ctx.rsi
        prev_rsi = ctx.prev_rsi

        # لسه مفيش بيانات كفاية
        if rsi is None or prev_rsi is None:
            return "HOLD"

        # منطقة شراء قوية (خروج من تشبّع بيعي)
        if rsi < self.extreme_oversold and rsi > prev_rsi:
            if ctx.position != "LONG":
                return "BUY"

        # منطقة بيع قوية (خروج من تشبّع شرائي)
        if rsi > self.extreme_overbought and rsi < prev_rsi:
            if ctx.position != "SHORT":
                return "SELL"

        # سكالبنج أخف حوالين 35 / 65
        if rsi < self.rsi_buy_level and rsi > prev_rsi:
            if ctx.position == "FLAT":
                return "BUY"

        if rsi > self.rsi_sell_level and rsi < prev_rsi:
            if ctx.position == "FLAT":
                return "SELL"

        # مفيش إشارة واضحة
        return "HOLD"
