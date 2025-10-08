import os
import asyncio
import threading
import time
import yfinance as yf
import pandas as pd
import requests
from quart import Quart, request
from telegram import Update
from telegram.ext import Application, CommandHandler
from sklearn.linear_model import LinearRegression
from hypercorn.asyncio import serve
from hypercorn.config import Config
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# ========================
# 🧠 Initial Setup
# ========================
app = Quart(__name__)
BOT_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
RENDER_URL = os.getenv("RENDER_EXTERNAL_URL") or "https://monica-option.onrender.com"
TWELVEDATA_KEY = os.getenv("TWELVEDATA_KEY")
PORT = int(os.getenv("PORT", "10000"))

if not BOT_TOKEN:
    raise RuntimeError("❌ BOT_TOKEN fehlt in Environment Variables")

application = Application.builder().token(BOT_TOKEN).build()
training_status = {"running": False, "accuracy": None, "message": ""}


# ========================
# 📊 Datenquelle mit Fallback
# ========================
def get_data(symbol="EURUSD=X", period="1mo", interval="1h"):
    """Holt Daten von Yahoo, wechselt bei Fehler zu TwelveData."""
    print(f"📡 Lade {symbol} von Yahoo...")
    try:
        df = yf.download(symbol, period=period, interval=interval)
        if df is not None and not df.empty:
            print("✅ Yahoo Finance Daten erhalten.")
            return df
        print("⚠️ Yahoo lieferte keine Daten, wechsle zu TwelveData...")
    except Exception as e:
        print(f"⚠️ Yahoo Finance Fehler: {e}")

    if not TWELVEDATA_KEY:
        print("❌ Kein TWELVEDATA_KEY in Environment gefunden – kann nicht wechseln.")
        return pd.DataFrame()

    try:
        url = f"https://api.twelvedata.com/time_series?symbol=EUR/USD&interval=1h&outputsize=500&apikey={TWELVEDATA_KEY}"
        r = requests.get(url, timeout=10)
        data = r.json()
        if "values" not in data:
            print(f"❌ TwelveData Fehler: {data}")
            return pd.DataFrame()

        df = pd.DataFrame(data["values"])
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close"})
        df = df.sort_values("datetime").set_index("datetime")
        print("✅ Daten von TwelveData geladen.")
        return df
    except Exception as e:
        print(f"❌ TwelveData ebenfalls fehlgeschlagen: {e}")
        return pd.DataFrame()


# ========================
# 📈 Trainingsfunktion
# ========================
async def train_model():
    global training_status
    training_status["running"] = True
    training_status["message"] = "📊 Training gestartet..."
    print(training_status["message"])

    try:
        df = get_data("EURUSD=X", period="1mo", interval="1h")
        if df.empty or len(df) < 10:
            training_status["message"] = "❌ Zu wenige oder keine Daten erhalten."
            training_status["running"] = False
            return

        df["Target"] = df["Close"].shift(-1)
        X = df[["Open", "High", "Low", "Close"]].iloc[:-1]
        y = df["Target"].iloc[:-1]

        model = LinearRegression()
        model.fit(X, y)
        acc = model.score(X, y)
        training_status["accuracy"] = round(acc * 100, 2)
        training_status["message"] = f"✅ Training fertig: {training_status['accuracy']}% Genauigkeit"
        print(training_status["message"])

    except Exception as e:
        training_status["message"] = f"❌ Fehler beim Training: {e}"
        print(training_status["message"])
    finally:
        training_status["running"] = False


# ========================
# 🤖 Telegram Befehle
# ========================
async def start(update, context):
    await update.message.reply_text("👋 Monica Option Bot aktiv.\nBefehle: /train /status /predict")

async def train(update, context):
    if training_status["running"]:
        await update.message.reply_text("⚙️ Training läuft bereits...")
    else:
        await update.message.reply_text("📊 Starte Training...")
        asyncio.create_task(train_model())

async def status(update, context):
    msg = f"📡 Status: {'läuft' if training_status['running'] else 'bereit'}"
    if training_status["accuracy"]:
        msg += f"\n🎯 Genauigkeit: {training_status['accuracy']}%"
    await update.message.reply_text(msg)

async def predict(update, context):
    df = get_data("EURUSD=X", period="1d", interval="1h")
    if df.empty:
        await update.message.reply_text("❌ Keine Daten verfügbar.")
        return
    last = df.iloc[-1]
    change = last["Close"] - last["Open"]
    signal = "📈 BUY" if change > 0 else "📉 SELL"
    await update.message.reply_text(f"{signal} — Δ {round(change,5)}")

application.add_handler(CommandHandler("start", start))
application.add_handler(CommandHandler("train", train))
application.add_handler(CommandHandler("status", status))
application.add_handler(CommandHandler("predict", predict))


# ========================
# 🌐 Quart API
# ========================
@app.route("/")
async def index():
    return "✅ Monica Option Bot läuft."

@app.route("/webhook", methods=["POST"])
async def webhook():
    data = await request.get_json()
    update = Update.de_json(data, application.bot)
    await application.process_update(update)
    return "OK"


# ========================
# 🔁 Automatisches Training
# ========================
async def auto_trainer():
    while True:
        print("⏱️ Starte automatisches Training (alle 6 Stunden)...")
        await train_model()
        await asyncio.sleep(6 * 60 * 60)  # 6 Stunden


# ========================
# 🧩 Code-Wächter
# ========================
class RestartHandler(FileSystemEventHandler):
    def __init__(self, loop):
        self.loop = loop
        self._last = 0

    def on_modified(self, event):
        if not event.src_path.endswith(".py"): return
        now = time.time()
        if now - self._last < 1.0: return
        self._last = now
        print(f"♻️ Änderung erkannt: {event.src_path} -> Neustart")
        time.sleep(0.5)
        os._exit(0)

def start_watchdog(loop):
    observer = Observer()
    observer.schedule(RestartHandler(loop), ".", recursive=True)
    observer.start()
    print("🔍 Watchdog läuft...")
    while True: time.sleep(1)


# ========================
# 🚀 Startpunkt
# ========================
async def main():
    print("🚀 Initialisiere Monica Option Bot...")
    await application.initialize()
    await application.bot.set_webhook(f"{RENDER_URL}/webhook")
    print(f"✅ Webhook gesetzt: {RENDER_URL}/webhook")

    loop = asyncio.get_running_loop()
    threading.Thread(target=start_watchdog, args=(loop,), daemon=True).start()
    asyncio.create_task(auto_trainer())

    config = Config()
    config.bind = [f"0.0.0.0:{PORT}"]
    await serve(app, config)

if __name__ == "__main__":
    asyncio.run(main())
