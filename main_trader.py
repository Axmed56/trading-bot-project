import asyncio
from multi_ws_futures import start_multi_ws
from decision_engine import DecisionEngine
from execution_bot import ExecutionBot
from ai_brain import AIBrain

async def main():
    print("🚀 Unified Dual Futures Bot Started (Binance + Bybit + AI)…")

    ai = AIBrain()
    decision_engine = DecisionEngine(ai)
    executor = ExecutionBot(decision_engine)

    await start_multi_ws(decision_engine, executor)

if __name__ == "__main__":
    asyncio.run(main())
