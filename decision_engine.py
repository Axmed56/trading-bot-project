"""
decision_engine.py
وحدة اتخاذ القرار:
- تستقبل إشارات الـ AI (BUY / SELL / NO_TRADE)
- تتأكد أن الإشارة ثابتة خلال فترة زمنية (confirmation_window)
- لو الإشارة اتأكدت => تصدر قرار تداول واضح للعملة
"""

import time
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Literal, Callable

SignalType = Literal["BUY", "SELL", "NO_TRADE"]


@dataclass
class SymbolState:
    last_signal: SignalType = "NO_TRADE"
    first_seen_ts: float = 0.0
    last_update_ts: float = 0.0
    active_decision: Optional[SignalType] = None
    # ممكن تستخدم الحقول دي لاحقًا لحساب نسبة ثبات الإشارة
    positive_duration: float = 0.0
    total_duration: float = 0.0


class DecisionEngine:
    """
    DecisionEngine:
    - confirmation_window: المدة بالثواني اللي لازم الإشارة تفضل فيها ثابتة (BUY أو SELL)
      عشان تتحول لقرار تداول حقيقي
    - on_decision: callback بيتنده لما قرار جديد يتأكد
      الشكل: on_decision(symbol, decision, context_dict)
    """

    def __init__(
        self,
        confirmation_window: float = 10.0,
        logger: Optional[logging.Logger] = None,
        on_decision: Optional[Callable[[str, SignalType, dict], None]] = None,
    ):
        self.confirmation_window = confirmation_window
        self.logger = logger or logging.getLogger("decision_engine")
        self.on_decision = on_decision
        self._states: Dict[str, SymbolState] = {}

    def _get_state(self, symbol: str) -> SymbolState:
        if symbol not in self._states:
            self._states[symbol] = SymbolState()
        return self._states[symbol]

    def reset_symbol(self, symbol: str) -> None:
        """إعادة تهيئة حالة رمز معين بعد إغلاق الصفقة مثلاً."""
        if symbol in self._states:
            self.logger.info(f"🔄 [Decision] Reset state for {symbol}")
            self._states[symbol] = SymbolState()

    def update_signal(
        self,
        symbol: str,
        signal: SignalType,
        meta: Optional[dict] = None,
    ) -> Optional[SignalType]:
        """
        تُستدعى في كل مرة الـ AI يطلع فيها إشارة جديدة للعملة.
        - symbol: مثال "BTCUSDT" أو "ETHUSDT" (يفضل نفس فورمات WebSocket / البوت)
        - signal: "BUY" / "SELL" / "NO_TRADE"
        - meta: ممكن تحط فيها price, spread, volume... إلخ
        ترجع:
        - decision: "BUY" أو "SELL" لما يتم تأكيد قرار جديد
        - None لو لسه مفيش قرار مؤكد
        """

        now = time.time()
        state = self._get_state(symbol)

        # تحديث إجمالي الزمن بين آخر إشارة والآن
        if state.last_update_ts > 0:
            delta = now - state.last_update_ts
            state.total_duration += delta
            if state.last_signal in ("BUY", "SELL"):
                state.positive_duration += delta

        # لو الإشارة الجديدة مختلفة عن السابقة بشكل جذري، نعيد نافذة التأكيد
        if signal != state.last_signal:
            state.first_seen_ts = now
            state.positive_duration = 0.0
            state.total_duration = 0.0
            self.logger.info(
                f"🧠 [Decision] New raw signal for {symbol}: {signal} (window restarted)"
            )

        state.last_signal = signal
        state.last_update_ts = now

        # لو NO_TRADE → لا قرار، ولازم نفضي أي قرار قديم
        if signal == "NO_TRADE":
            if state.active_decision is not None:
                self.logger.info(
                    f"⚪ [Decision] Signal back to NO_TRADE for {symbol}, clearing active decision."
                )
                state.active_decision = None
            return None

        # لو BUY أو SELL → نتحقق من ثبات الإشارة خلال confirmation_window
        elapsed = now - state.first_seen_ts

        if elapsed >= self.confirmation_window:
            # لو مفيش قرار حالي أو القرار الحالي مختلف عن الإشارة الحالية → نثبت قرار جديد
            if state.active_decision != signal:
                state.active_decision = signal
                context = {
                    "symbol": symbol,
                    "decision": signal,
                    "timestamp": now,
                    "elapsed_confirmation": elapsed,
                    "meta": meta or {},
                }

                self.logger.info(
                    f"✅ [Decision] CONFIRMED decision for {symbol}: {signal} "
                    f"(window={elapsed:.1f}s)"
                )

                if self.on_decision:
                    try:
                        self.on_decision(symbol, signal, context)
                    except Exception as e:
                        self.logger.error(
                            f"❌ [Decision] Error in on_decision callback for {symbol}: {e}",
                            exc_info=True,
                        )

                return signal

        # مفيش قرار مؤكد لسه
        return None


# تشغيل تجريبي بسيط لو شغلت الملف مباشرة
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
    )

    engine = DecisionEngine(confirmation_window=5.0)

    sym = "BTCUSDT"

    # مثال: إشارة BUY ثابتة لمدة 6 ثواني
    for i in range(7):
        engine.update_signal(sym, "BUY", meta={"price": 84500 + i})
        time.sleep(1.0)
