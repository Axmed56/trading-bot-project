import time

class ExecutionBot:
    def __init__(self, decision_engine):
        self.engine = decision_engine

    async def process_market_event(self, symbol, event_type, data):
        """
        يستقبل الإشارات القادمة من decision_engine
        """
        decision = self.engine.get_latest_decision(symbol)

        if decision is None:
            return
        
        # لو مفيش فرصة قوية AI مش هيدخل صفقة
        if decision["signal"] == "HOLD":
            return
        
        # عرض الإشارة فقط (بدون تنفيذ فعلي)
        print(f"\n🔥 EXECUTION SIGNAL => {symbol}")
        print(f"📌 ACTION : {decision['signal']}")
        print(f"⏳ HOLD FOR: {decision['duration']} seconds")
        print("-" * 50)
