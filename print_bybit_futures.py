import ccxt

bybit = ccxt.bybit({
    "options": {
        "defaultType": "linear"
    }
})

bybit.load_markets()

print("\n=== BYBIT USDT PERPETUAL FUTURES SYMBOLS ===")
for s, m in bybit.markets.items():
    if m.get("contract") and m.get("linear"):
        print(s, "| id:", m["id"])
