import json
import os
import time
from flask import Flask, jsonify, render_template_string

# مسار ملف الـ live_feed.json (نفس فولدر المشروع)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LIVE_FEED_PATH = os.path.join(BASE_DIR, "live_feed.json")

app = Flask(__name__)


def load_live_feed():
    """قراءة آخر بيانات من live_feed.json بأمان."""
    try:
        with open(LIVE_FEED_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        data = {}
    except json.JSONDecodeError:
        # لو الملف بيكتب حاليًا ومش متكامل نتجاهل القراءة الغلط
        data = {}
    return data


@app.route("/api/live")
def api_live():
    """API يرجع آخر أسعار من Binance + Bybit."""
    data = load_live_feed()
    return jsonify(
        {
            "symbols": data,          # dict: ETHUSDT / BTCUSDT / SOLUSDT ...
            "server_time": time.time()
        }
    )


# HTML بسيط للداش بورد
INDEX_HTML = """
<!doctype html>
<html lang="ar" dir="rtl">
<head>
    <meta charset="utf-8">
    <title>ZAYA – Realtime Futures Monitor</title>
    <style>
        body {
            background: #050816;
            color: #f5f5f5;
            font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            margin: 0;
            padding: 20px;
        }
        h1 {
            margin-bottom: 5px;
            color: #4ade80;
        }
        .subtitle {
            color: #9ca3af;
            margin-bottom: 20px;
            font-size: 14px;
        }
        .top-bar {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 10px;
        }
        .badge {
            padding: 4px 10px;
            border-radius: 999px;
            font-size: 12px;
            background: #111827;
            border: 1px solid #1f2937;
        }
        .badge span {
            color: #22c55e;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            background: #020617;
            border-radius: 12px;
            overflow: hidden;
        }
        thead {
            background: #111827;
        }
        th, td {
            padding: 10px 12px;
            font-size: 13px;
            text-align: center;
            border-bottom: 1px solid #1f2937;
        }
        th {
            font-weight: 600;
            color: #e5e7eb;
        }
        tr:last-child td {
            border-bottom: none;
        }
        .symbol {
            font-weight: 600;
        }
        .exchange {
            font-size: 11px;
            color: #9ca3af;
        }
        .pos {
            color: #22c55e;
        }
        .neg {
            color: #ef4444;
        }
        .neutral {
            color: #e5e7eb;
        }
        .updated {
            font-size: 12px;
            color: #9ca3af;
            margin-top: 6px;
        }
    </style>
</head>
<body>
    <div class="top-bar">
        <div>
            <h1>ZAYA – Live Futures Feed</h1>
            <div class="subtitle">Binance + Bybit | BTC / ETH / SOL | WebSocket → JSON → Dashboard</div>
        </div>
        <div>
            <div class="badge">
                وضع التشغيل: <span>مراقبة فقط (Paper / Signals)</span>
            </div>
            <div class="updated" id="updated">آخر تحديث: ...</div>
        </div>
    </div>

    <table>
        <thead>
            <tr>
                <th>الرمز</th>
                <th>سعر Binance</th>
                <th>سعر Bybit</th>
                <th>السبريد (فرق السعر)</th>
            </tr>
        </thead>
        <tbody id="symbols-body">
            <!-- يتم ملؤها من JavaScript -->
        </tbody>
    </table>

    <script>
        async function loadData() {
            try {
                const res = await fetch("/api/live");
                const json = await res.json();
                const data = json.symbols || {};
                const tbody = document.getElementById("symbols-body");
                tbody.innerHTML = "";

                const symbols = Object.keys(data).sort();
                symbols.forEach(symbol => {
                    const item = data[symbol] || {};
                    const bPrice = item.binance_price ?? null;
                    const yPrice = item.bybit_price ?? null;
                    const spread = item.spread ?? null;

                    const tr = document.createElement("tr");

                    const tdSymbol = document.createElement("td");
                    tdSymbol.className = "symbol";
                    tdSymbol.textContent = symbol;
                    tr.appendChild(tdSymbol);

                    const tdBinance = document.createElement("td");
                    tdBinance.textContent = bPrice !== null ? bPrice.toFixed(4) : "N/A";
                    tr.appendChild(tdBinance);

                    const tdBybit = document.createElement("td");
                    tdBybit.textContent = yPrice !== null ? yPrice.toFixed(4) : "N/A";
                    tr.appendChild(tdBybit);

                    const tdSpread = document.createElement("td");
                    let cls = "neutral";
                    if (spread !== null) {
                        if (spread > 0) cls = "pos";
                        if (spread < 0) cls = "neg";
                        tdSpread.textContent = spread.toFixed(4);
                    } else {
                        tdSpread.textContent = "N/A";
                    }
                    tdSpread.className = cls;
                    tr.appendChild(tdSpread);

                    tbody.appendChild(tr);
                });

                const updated = document.getElementById("updated");
                const t = new Date((json.server_time || Date.now()/1000) * 1000);
                updated.textContent = "آخر تحديث: " + t.toLocaleTimeString();
            } catch (err) {
                console.error("loadData error:", err);
            }
        }

        // أول تحميل ثم تحديث كل ثانية
        loadData();
        setInterval(loadData, 1000);
    </script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(INDEX_HTML)


if __name__ == "__main__":
    # تشغيل السيرفر
    app.run(host="127.0.0.1", port=5000, debug=False)
