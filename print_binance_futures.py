import ccxt

binance = ccxt.binance({
    "options": {
        "defaultType": "future"
    }
})

binance.load_markets()

print("\n=== BINANCE USDT FUTURES SYMBOLS ===")
for s, m in binance.markets.items():
    if m.get("contract") and m.get("linear"):  # only USDT-M perpetual
        print(s, "| id:", m["id"])
