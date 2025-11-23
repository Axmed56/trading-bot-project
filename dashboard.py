import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

def create_dashboard(
    ohlcv_path="ohlcv.csv",
    trades_path="backtest_trades.csv",
    output_path="advanced_dashboard.html",
    symbol="BTC/USDT",
    timeframe="1h",
    initial_balance=1000.0
):
    # 1) قراءة البيانات
    if not os.path.exists(ohlcv_path):
        print(f"❌ ملف الشموع غير موجود: {ohlcv_path}")
        return

    if not os.path.exists(trades_path):
        print(f"❌ ملف الصفقات غير موجود: {trades_path}")
        return

    ohlcv_df = pd.read_csv(ohlcv_path, parse_dates=["timestamp"])
    trades_df = pd.read_csv(trades_path, parse_dates=["time"])

    if ohlcv_df.empty:
        print("❌ ملف الشموع فارغ")
        return

    # 2) تجهيز بعض المعلومات الأساسية
    if not trades_df.empty:
        first_trade_time = trades_df["time"].iloc[0]
        last_trade_time = trades_df["time"].iloc[-1]
        duration_days = (last_trade_time - first_trade_time).days or 1
        final_balance = trades_df["balance"].iloc[-1]
        total_return = ((final_balance - initial_balance) / initial_balance) * 100
        max_balance = trades_df["balance"].cummax()
        drawdown = (max_balance - trades_df["balance"]) / max_balance * 100
        max_drawdown = drawdown.max()
    else:
        first_trade_time = ohlcv_df["timestamp"].iloc[0]
        last_trade_time = ohlcv_df["timestamp"].iloc[-1]
        duration_days = (last_trade_time - first_trade_time).days or 1
        final_balance = initial_balance
        total_return = 0.0
        max_drawdown = 0.0

    total_trades = len(trades_df)
    wins = trades_df[trades_df["pnl"] > 0]
    losses = trades_df[trades_df["pnl"] < 0]
    win_rate = (len(wins) / total_trades * 100) if total_trades > 0 else 0.0
    avg_profit = wins["pnl"].mean() if len(wins) > 0 else 0.0
    avg_loss = losses["pnl"].mean() if len(losses) > 0 else 0.0

    # 3) فصل صفقات الدخول والخروج
    entries = trades_df[trades_df["type"] == "ENTRY"]
    exits = trades_df[trades_df["type"].isin(["STOPLOSS", "TAKEPROFIT"])]

    # 4) إنشاء الشكل العام (3 صفوف)
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.55, 0.15, 0.30],
        subplot_titles=(
            f"{symbol} - {timeframe} | من {first_trade_time.strftime('%Y-%m-%d')} إلى {last_trade_time.strftime('%Y-%m-%d')}",
            "📊 الحجم",
            "📈 منحنى الرصيد / الأرباح"
        )
    )

    # 5) الشموع
    fig.add_trace(
        go.Candlestick(
            x=ohlcv_df["timestamp"],
            open=ohlcv_df["open"],
            high=ohlcv_df["high"],
            low=ohlcv_df["low"],
            close=ohlcv_df["close"],
            name="السعر"
        ),
        row=1,
        col=1
    )

    # 6) نقاط الدخول (BUY/SELL)
    if not entries.empty:
        fig.add_trace(
            go.Scatter(
                x=entries["time"],
                y=entries["entry_price"],
                mode="markers",
                marker=dict(
                    symbol="triangle-up",
                    size=10,
                    color="#00E676",
                    line=dict(width=1, color="#00C853")
                ),
                name="دخول صفقات"
            ),
            row=1,
            col=1
        )

    # 7) نقاط الخروج (SL/TP)
    if not exits.empty:
        colors_exit = exits["type"].map(
            {"STOPLOSS": "#FF5252", "TAKEPROFIT": "#2962FF"}
        ).fillna("#999999")

        fig.add_trace(
            go.Scatter(
                x=exits["time"],
                y=exits["exit_price"],
                mode="markers",
                marker=dict(
                    symbol="x",
                    size=9,
                    color=colors_exit
                ),
                name="خروج صفقات"
            ),
            row=1,
            col=1
        )

    # 8) الحجم
    if "volume" in ohlcv_df.columns:
        fig.add_trace(
            go.Bar(
                x=ohlcv_df["timestamp"],
                y=ohlcv_df["volume"],
                name="الحجم",
                opacity=0.6
            ),
            row=2,
            col=1
        )

    # 9) منحنى الرصيد
    if not trades_df.empty:
        fig.add_trace(
            go.Scatter(
                x=trades_df["time"],
                y=trades_df["balance"],
                mode="lines+markers",
                name="الرصيد",
                line=dict(width=2)
            ),
            row=3,
            col=1
        )

        fig.add_trace(
            go.Scatter(
                x=trades_df["time"],
                y=trades_df["pnl"].cumsum(),
                mode="lines",
                name="الأرباح المتراكمة",
                line=dict(width=2, dash="dash")
            ),
            row=3,
            col=1
        )

    # 10) نص الأداء
    performance_text = (
        f"<b>💰 الأداء المالي</b><br>"
        f"• الرصيد الابتدائي: ${initial_balance:,.2f}<br>"
        f"• الرصيد النهائي: ${final_balance:,.2f}<br>"
        f"• العائد الإجمالي: <b>{total_return:+.2f}%</b><br>"
        f"• أقصى سحب (Drawdown): {max_drawdown:.2f}%<br><br>"
        f"<b>📊 إحصائيات التداول</b><br>"
        f"• إجمالي الصفقات: {total_trades}<br>"
        f"• معدل الربح: {win_rate:.1f}%<br>"
        f"• متوسط الربح: ${avg_profit:.2f}<br>"
        f"• متوسط الخسارة: ${avg_loss:.2f}<br>"
        f"• المدة: {duration_days} يوم تقريباً"
    )

    fig.update_layout(
        title=dict(
            text=(
                f"<b>🎯 لوحة تحكم محاكاة التداول</b><br>"
                f"<span style='font-size:12px; color:gray;'>"
                f"{symbol} | {timeframe} | العائد: {total_return:+.2f}%"
                f"</span>"
            ),
            x=0.5,
            xanchor="center"
        ),
        template="plotly_white",
        hovermode="x unified",
        height=900,
        xaxis_rangeslider_visible=False,
        annotations=[
            dict(
                text=performance_text,
                x=0.01,
                y=0.99,
                xref="paper",
                yref="paper",
                showarrow=False,
                align="left",
                bgcolor="rgba(255,255,255,0.9)",
                bordercolor="rgba(0,0,0,0.1)",
                borderwidth=1,
                borderpad=10,
                font=dict(size=11)
            )
        ]
    )

    fig.update_yaxes(title_text="السعر", row=1, col=1)
    fig.update_yaxes(title_text="الحجم", row=2, col=1)
    fig.update_yaxes(title_text="الرصيد / الأرباح", row=3, col=1)
    fig.update_xaxes(title_text="الوقت", row=3, col=1)

    fig.write_html(output_path)
    print(f"✅ تم إنشاء لوحة التحكم في الملف: {output_path}")


if __name__ == "__main__":
    print("🚀 إنشاء داشبورد المحاكاة من ملفات ohlcv.csv و backtest_trades.csv ...")
    create_dashboard()
