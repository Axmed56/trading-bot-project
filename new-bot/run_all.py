
import subprocess
import sys
import time

SCRIPTS = [
    "binance_ws.py",
    "bybit_ws.py",
    "data_bridge.py",
]


def main():
    processes = []

    try:
        print("🚀 Starting ZAYA mini stack (Binance + Bybit + Dashboard)...")
        for script in SCRIPTS:
            p = subprocess.Popen([sys.executable, script])
            print(f"  ▶ started {script} (pid={p.pid})")
            processes.append(p)

        print("\nكل حاجة اشتغلت. افتح المتصفح على:")
        print("  http://127.0.0.1:5005")
        print("\nاضغط Ctrl + C في هذه الشاشة لإيقاف الكل.\n")

        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        print("\n⏹ Stopping all processes...")
    finally:
        for p in processes:
            p.terminate()
        print("تم الإيقاف.")


if __name__ == "__main__":
    main()
