# smart_scanner_futures.py
import logging
import ccxt
from typing import Dict, List

# =========================
# إعداد اللوجينج
# =========================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)


def load_futures_markets(
    ex: ccxt.Exchange,
    name: str,
) -> Dict[str, dict]:
    """
    تحميل أسواق الفيوتشر (USDT/USDC - Perpetual Swaps فقط)
    نعتمد على:
      - swap = True   → عقود مستمرة
      - contract = True → عقد مشتقات
      - expiry = None → مش عقد منتهي (مش ربع سنوي مثلاً)
      - quote in [USDT, USDC]
    نستخدم ex.symbol كـ مفتاح (مثلاً 'BTC/USDT:USDT')
    """
    logging.info("🔍[%s] Loading futures markets...", name)
    ex.load_markets()

    futures = {}
    for symbol, m in ex.markets.items():
        try:
            if not m.get("swap", False):
                continue
            if not m.get("contract", False):
                continue
            if m.get("expiry") is not None:
                # استبعاد العقود ذات تاريخ انتهاء
                continue
            if m.get("quote") not in ("USDT", "USDC"):
                continue

            futures[symbol] = m
        except Exception:
            # في حالة أي سوق غريب في البيانات نتجاهله بهدوء
            continue

    logging.info(
        "[%s] Loaded %s futures markets (USDT/USDC, perpetual).",
        name,
        len(futures),
    )
    return futures


def fetch_tickers_safe(ex: ccxt.Exchange, symbols: List[str], name: str) -> Dict[str, dict]:
    """
    جلب tickers لعدد من الرموز مع التعامل مع الأخطاء بشكل آمن.
    """
    logging.info("📈[%s] Fetching tickers for %d symbols...", name, len(symbols))
    tickers = {}
    if not symbols:
        return tickers

    # بعض المنصات تفضل التجزئة لو القائمة كبيرة – نطبق ذلك ببساطة
    batch_size = 100
    for i in range(0, len(symbols), batch_size):
        batch = symbols[i:i + batch_size]
        try:
            t = ex.fetch_tickers(batch)
            tickers.update(t)
        except Exception as e:
            logging.warning(
                "[%s] Error fetching tickers for batch (%s): %s",
                name,
                ", ".join(batch),
                e,
            )
    logging.info("[%s] Got %d tickers.", name, len(tickers))
    return tickers


def score_symbol(bin_t: dict, byb_t: dict) -> float:
    """
    حساب "درجة" لكل رمز بناءً على حجم التداول × السعر
    نستخدم بيانات المنصتين ونجمّعهم.
    """
    score = 0.0

    if bin_t:
        bv = bin_t.get("baseVolume") or 0
        last = bin_t.get("last") or 0
        try:
            score += float(bv) * float(last)
        except Exception:
            pass

    if byb_t:
        bv = byb_t.get("baseVolume") or 0
        last = byb_t.get("last") or 0
        try:
            score += float(bv) * float(last)
        except Exception:
            pass

    return score


def main():
    # =========================
    # تهيئة المنصات (Binance Futures + Bybit Linear Perps)
    # =========================
    logging.info("🚀 Starting Smart Futures Scanner (Binance + Bybit)")

    binance = ccxt.binance({
        "enableRateLimit": True,
        "options": {
            "defaultType": "future",  # USDT-M futures
        },
    })

    bybit = ccxt.bybit({
        "enableRateLimit": True,
        "options": {
            "defaultType": "swap",    # Linear perpetual swaps
        },
    })

    # =========================
    # تحميل أسواق الفيوتشر فقط (Perpetual – USDT/USDC)
    # =========================
    bin_futs = load_futures_markets(binance, "BINANCE")
    byb_futs = load_futures_markets(bybit, "BYBIT")

    if not bin_futs:
        logging.error("❌ No futures markets loaded from Binance.")
        return
    if not byb_futs:
        logging.error("❌ No futures markets loaded from Bybit.")
        return

    # =========================
    # إيجاد الرموز المشتركة
    # نستخدم الـ symbol الموحّد من ccxt (مثال: 'BTC/USDT:USDT')
    # =========================
    common_symbols = sorted(set(bin_futs.keys()) & set(byb_futs.keys()))

    # لو الاتنين محدّدين رموز مختلفة (مثلاً Bybit تستخدم 'BTC/USDT' و Binance 'BTC/USDT:USDT')
    # نعمل تطبيع بسيط: base + '/' + quote
    if not common_symbols:
        logging.warning("[WARN] No direct common symbols via unified 'symbol'. Trying base/quote normalization...")
        bin_norm = {}
        for s, m in bin_futs.items():
            key = f"{m.get('base')}/{m.get('quote')}"
            bin_norm[key] = s

        byb_norm = {}
        for s, m in byb_futs.items():
            key = f"{m.get('base')}/{m.get('quote')}"
            byb_norm[key] = s

        common_keys = sorted(set(bin_norm.keys()) & set(byb_norm.keys()))

        if not common_keys:
            print("[WARN] No common USDT futures markets between Binance & Bybit even after normalization.")
            return

        # نحول الـ keys المشتركة إلى رموز منصات فعلية
        common_symbols = []
        symbol_pairs = []
        for k in common_keys:
            b_sym = bin_norm[k]
            y_sym = byb_norm[k]
            common_symbols.append(k)  # للعرض
            symbol_pairs.append((b_sym, y_sym))

        # نشتغل بالـ pairs في الحساب
        use_pairs = True
        logging.info("✅ Found %d common futures markets (normalized).", len(symbol_pairs))
    else:
        # لو الاتنين عندهم نفس unified symbol
        use_pairs = False
        symbol_pairs = [(s, s) for s in common_symbols]
        logging.info("✅ Found %d common futures markets via unified symbol.", len(symbol_pairs))

    if not symbol_pairs:
        print("[WARN] No common USDT futures markets between Binance & Bybit.")
        return

    # =========================
    # جلب tickers للرموز المشتركة
    # =========================
    # لو use_pairs = False → نفس الـ symbol على الاتنين
    bin_symbols = [bp for (bp, _) in symbol_pairs]
    byb_symbols = [yp for (_, yp) in symbol_pairs]

    bin_tickers = fetch_tickers_safe(binance, list(set(bin_symbols)), "BINANCE")
    byb_tickers = fetch_tickers_safe(bybit, list(set(byb_symbols)), "BYBIT")

    # =========================
    # حساب درجات و ترتيب أفضل العملات
    # =========================
    scored = []
    for (b_sym, y_sym) in symbol_pairs:
        # مفتاح العرض
        if use_pairs:
            # display_key يكون base/quote (من normalized)
            b_m = bin_futs[b_sym]
            display_key = f"{b_m.get('base')}/{b_m.get('quote')}"
        else:
            display_key = b_sym

        b_t = bin_tickers.get(b_sym, {})
        y_t = byb_tickers.get(y_sym, {})

        s = score_symbol(b_t, y_t)
        if s <= 0:
            continue

        scored.append({
            "display": display_key,
            "bin_sym": b_sym,
            "byb_sym": y_sym,
            "score": s,
            "bin_t": b_t,
            "byb_t": y_t,
        })

    if not scored:
        print("[WARN] No markets with non-zero volume/price to rank.")
        return

    scored.sort(key=lambda x: x["score"], reverse=True)

    top_n = 3
    print()
    print(f"=== TOP {top_n} COMMON FUTURES MARKETS (Binance + Bybit, USDT/USDC Perpetual) ===")
    for i, row in enumerate(scored[:top_n], start=1):
        display = row["display"]
        b_sym = row["bin_sym"]
        y_sym = row["byb_sym"]
        b_m = bin_futs[b_sym]
        y_m = byb_futs[y_sym]

        b_id = b_m.get("id")
        y_id = y_m.get("id")

        b_t = row["bin_t"]
        y_t = row["byb_t"]

        b_last = b_t.get("last")
        y_last = y_t.get("last")

        b_vol = b_t.get("baseVolume")
        y_vol = y_t.get("baseVolume")

        print()
        print(f"#{i} → {display}")
        print(f"   [BINANCE] symbol={b_sym} | id={b_id} | last={b_last} | baseVol={b_vol}")
        print(f"   [BYBIT]   symbol={y_sym} | id={y_id} | last={y_last} | baseVol={y_vol}")
        print(f"   >>> SCORE (Liquidity & Activity): {row['score']:.2f}")

    print()
    print("✅ Scanner finished. These markets هم أفضل مرشحين للسكالبينج من حيث النشاط والسيولة على المنصتين.")
    print("تقدر تستخدم الـ symbols دي في البوت الرئيسي والـ WebSocket dashboard.")
    

if __name__ == "__main__":
    main()
