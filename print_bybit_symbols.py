import ccxt

bybit = ccxt.bybit()

bybit.load_markets()

print("\n=== BYBIT SYMBOLS ===")
for s, m in bybit.markets.items():
    print(f"{s}   |   id={m['id']}   |   type={m['type']}   |   spot={m['spot']}   |   contract={m['contract']}")
