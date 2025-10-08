import os
import asyncio
import threading
import time
import pandas as pd
import yfinance as yf
from quart import Quart, request
from telegram import Update
from telegram.ext import Application, CommandHandler
from sklearn.linear_model import LinearRegression
from hypercorn.asyncio import serve
from hypercorn.config import Config
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import requests

# ========================
# 🧠 Initial Setup
# ========================
app = Quart(__name__)
BOT_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
RENDER_URL = os.getenv("RENDER_EXTERNAL_URL") or "https://monica-option-bot.onrender.com"
PORT = int(os.getenv("PORT", "10000"))

if not BOT_TOKEN:
    raise RuntimeError("❌ BOT_TOKEN fehlt in Environment Variables")
if not CHAT_ID:
    print("⚠️ WARNUNG: TELEGRAM_CHAT_ID nicht gesetzt – Benachrichtigungen deaktiviert.")

application = Application.builder().token(BOT_TOKEN).build()
training_status = {"running": False, "accuracy": None, "message": ""}

# ========================
# 📊 Datenbeschaffung mit Fallback
# ========================
async def get_eurusd_data(period="1mo", interval="1h", retries=3, delay=10):
    """Versucht zuerst Yahoo Finance, dann Apple Finance, mit automatischem Retry."""
    for attempt in range(retries):
        try:
            print(f"📡 Lade EURUSD von Yahoo (Versuch {attempt+1}/{retries})...")
            df = yf.download("EURUSD=X", period=period, interval=interval, progress=False)
            if not df.empty:
                return df
        except Exception as e:
            print(f"⚠️ Yahoo Finance Fehler: {e}")

        print(f"⚠️ Yahoo lieferte keine Daten, versuche Apple Finance...")
        try:
            url = f"https://query1.finance.apple.com/api/v1/quotes?symbols=EURUSD"
            resp = requests.get(url, timeout=10)
            data = resp.json().get("data", [])
            if data:
                price = float(data[0].get("last", 0))
                df = pd.DataFrame([{"Open": price, "High": price, "Low": price, "Close": price}])
                return df
        except Exception as e:
            print(f"⚠️ Apple Finance Fehler: {e}")

        print(f"🔁 Neuer Versuch in {delay} Sekunden...")
        await asyncio.sleep(delay)

    print("❌ Keine Kursdaten erhalten (Yahoo & Apple fehlgeschlagen).")
    return pd.DataFrame()

# ========================
# 🧠 Trainingsfunktion
# ========================
async def train_model():
    global training_status
    training_status["running"] = True
    training_status["message"] = "📊 Training gestartet..."
    print(training_status["message"])

    try:
        df = await get_eurusd_data()
        if df.empty or len(df) < 10:
            training_status["message"] = "❌ Keine ausreichenden Daten zum Trainieren."
            training_status["running"] = False
            if CHAT_ID:
                await application.bot.send_message(chat_id=CHAT_ID, text="⚠️ Keine Kursdaten erhalten. Training übersprungen.")
            return

        df["Target"] = df["Close"].shift(-1)
        X = df[["Open", "High", "Low", "Close"]].iloc[:-1]
        y = df["Target"].iloc[:-1]

        model = LinearRegression()
        model.fit(X, y)
        acc = model.score(X, y)
        training_status["accuracy"] = round(acc * 100, 2)
        training_status["message"] = f"✅ Training fertig: {training_status['accuracy']} % Genauigkeit"
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
    df = await get_eurusd_data(period="1d", interval="1h")
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
        if CHAT_ID:
            try:
                await application.bot.send_message(
                    chat_id=CHAT_ID,
                    text=f"🤖 Automatisches Training abgeschlossen.\n🎯 Genauigkeit: {training_status.get('accuracy', 'N/A')}%"
                )
            except Exception as e:
                print("Warnung: Telegram-Nachricht fehlgeschlagen:", e)
        await asyncio.sleep(6 * 60 * 60)  # 6 Stunden warten

# ========================
# 🧩 Watchdog (Code-Neustart)
# ========================
class RestartHandler(FileSystemEventHandler):
    def __init__(self, loop):
        super().__init__()
        self.loop = loop
        self._last = 0.0
    def on_modified(self, event):
        if not event.src_path.endswith(".py"):
            return
        now = time.time()
        if now - self._last < 1.0:
            return
        self._last = now
        print(f"♻️ Änderung erkannt: {event.src_path} → Neustart")
        if CHAT_ID:
            coro = application.bot.send_message(chat_id=CHAT_ID, text="🔄 Bot wird neu gestartet (Code-Update)...")
            try:
                fut = asyncio.run_coroutine_threadsafe(coro, self.loop)
                fut.result(timeout=5)
            except Exception as e:
                print("Warnung: Restart-Nachricht fehlgeschlagen:", e)
        time.sleep(0.5)
        os._exit(0)

def start_watchdog(loop):
    handler = RestartHandler(loop)
    observer = Observer()
    observer.schedule(handler, ".", recursive=True)
    observer.start()
    print("🔍 Watchdog läuft...")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

# ========================
# 🚀 Startpunkt
# ========================
async def main():
    print("🚀 Initialisiere Monica Option Bot...")
    await application.initialize()
    webhook_url = f"{RENDER_URL}/webhook"
    await application.bot.set_webhook(webhook_url)
    print(f"✅ Webhook gesetzt: {webhook_url}")

    if CHAT_ID:
        try:
            await application.bot.send_message(chat_id=CHAT_ID, text="✅ Monica Option Bot gestartet.")
        except Exception as e:
            print("Info: Startup-Message fehlgeschlagen:", e)

    loop = asyncio.get_running_loop()
    threading.Thread(target=start_watchdog, args=(loop,), daemon=True).start()
    asyncio.create_task(auto_trainer())

    config = Config()
    config.bind = [f"0.0.0.0:{PORT}"]
    await serve(app, config)

if __name__ == "__main__":
    asyncio.run(main())
