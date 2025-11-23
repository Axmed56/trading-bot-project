import ccxt
import logging
from typing import Dict, Any, List, Tuple

# =========================
# إعداد اللوجينج (Logging)
# =========================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)


def safe_get(d: Dict[str, Any], key: str, default=None):
    v = d.get(key, default)
    return default if v is None else v


def compute_spread_pct(ticker: Dict[str, Any]) -> float:
    """
    حساب السبريد كنسبة مئوية من متوسط السعر:
    spread% = (ask - bid) / mid * 100
    """
    bid = safe_get(ticker, "bid", 0.0)
    ask = safe_get(ticker, "ask", 0.0)
    if not bid or not ask:
        return 9999.0

    mid = (bid + ask) / 2.0
    if mid <= 0:
        return 9999.0

    return (ask - bid) / mid * 100.0


def compute_score(
    spread_pct: float,
    vol_binance: float,
    vol_bybit: float,
) -> float:
    """
    درجة السيولة:
    - كلما قلّ السبريد كان أفضل
    - كلما زاد حجم التداول كان أفضل
    سكوري بسيط: (vol_total / (1 + spread_pct))
    """
    total_vol = vol_binance + vol_bybit
    if spread_pct <= 0:
        spread_pct = 0.01
    return total_vol / (1.0 + spread_pct)


def classify_market(spread_pct: float, total_vol_usd: float) -> str:
    """
    تصنيف رمزي للحالة:
    - سبريد ضعيف + حجم كبير => ممتاز للمضاربة
    """
    if spread_pct < 0.02 and total_vol_usd > 50_000_000:
        return "🔥 سيولة عالية جدًا وسبريد شبه معدوم (Scalp ممتاز)"
    if spread_pct < 0.05 and total_vol_usd > 20_000_000:
        return "✅ مناسب للمضاربة السريعة"
    if spread_pct < 0.1 and total_vol_usd > 5_000_000:
        return "🟡 متوسط – يحتاج حذر في الدخول"
    return "⚪ سيولة/سبريد أقل من المطلوب للمضاربة المكثفة"


def fetch_tickers(exchange, name: str) -> Dict[str, Dict[str, Any]]:
    """
    جلب جميع tickers من المنصة مع حماية من الأخطاء.
    """
    try:
        logging.info("🔍[%s] Fetching tickers...", name)
        exchange.load_markets()
        tickers = exchange.fetch_tickers()
        return tickers
    except Exception as e:
        logging.error("[%s] Error fetching tickers: %s", name, e)
        return {}


def main():
    logging.info("🔍 [Scanner] Fetching tickers from Binance & Bybit...")

    # إنشاء العملاء
    binance = ccxt.binance({"enableRateLimit": True})
    bybit = ccxt.bybit({"enableRateLimit": True})

    # جلب كل tickers
    b_tickers = fetch_tickers(binance, "BINANCE")
    y_tickers = fetch_tickers(bybit, "BYBIT")

    if not b_tickers or not y_tickers:
        logging.error("لم يتمكن السكـانر من جلب البيانات من واحدة من المنصتين.")
        return

    common_markets: List[Tuple[str, Dict[str, Any], Dict[str, Any]]] = []

    for sym, bt in b_tickers.items():
        # نركز على أزواج USDT فقط
        if not sym.endswith("/USDT"):
            continue

        # ccxt يوحّد الرموز، لذلك Bybit أيضًا سيكون بنفس الشكل "BTC/USDT"
        if sym not in y_tickers:
            continue

        yt = y_tickers[sym]

        # حجم التداول بالدولار (quoteVolume غالبًا بالدولار لأزواج USDT)
        vol_b = float(safe_get(bt, "quoteVolume", 0.0) or 0.0)
        vol_y = float(safe_get(yt, "quoteVolume", 0.0) or 0.0)

        # استبعاد الأزواج الميتة تقريبًا
        if vol_b + vol_y < 500_000:  # أقل من نصف مليون دولار
            continue

        spread_pct = compute_spread_pct(bt)
        score = compute_score(spread_pct, vol_b, vol_y)

        common_markets.append(
            (sym, bt, yt, spread_pct, vol_b, vol_y, score)
        )

    if not common_markets:
        print("[WARN] No common USDT markets between Binance & Bybit.")
        return

    # ترتيب حسب السكور من الأكبر إلى الأصغر
    common_markets.sort(key=lambda x: x[6], reverse=True)

    top_n = 3
    print("\n=== Top 3 Markets (Binance + Bybit) ===\n")

    for i, (sym, bt, yt, spread_pct, vol_b, vol_y, score) in enumerate(
        common_markets[:top_n], start=1
    ):
        last_b = safe_get(bt, "last", 0.0)
        last_y = safe_get(yt, "last", 0.0)
        total_vol = vol_b + vol_y
        label = classify_market(spread_pct, total_vol)

        print(f"{i}) {sym}")
        print(f"   ▸ سعر آخر صفقة Binance : {last_b:.4f}")
        print(f"   ▸ سعر آخر صفقة Bybit   : {last_y:.4f}")
        print(f"   ▸ السبريد (من Binance) : {spread_pct:.4f}%")
        print(f"   ▸ حجم 24h Binance      : {vol_b:,.0f} USDT")
        print(f"   ▸ حجم 24h Bybit        : {vol_y:,.0f} USDT")
        print(f"   ▸ إجمالي الحجم        : {total_vol:,.0f} USDT")
        print(f"   ▸ تقييم السوق          : {label}")
        print(f"   ▸ Score داخلي          : {score:,.2f}")
        print("-" * 70)


if __name__ == "__main__":
    main()
