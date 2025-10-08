import os
import asyncio
import threading
import time
import yfinance as yf
import pandas as pd
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
PORT = int(os.getenv("PORT", "10000"))

if not BOT_TOKEN:
    raise RuntimeError("❌ BOT_TOKEN fehlt in Environment Variables")
if not CHAT_ID:
    print("⚠️ WARNUNG: TELEGRAM_CHAT_ID nicht gesetzt – Benachrichtigungen deaktiviert.")

application = Application.builder().token(BOT_TOKEN).build()
training_status = {"running": False, "accuracy": None, "message": ""}

# ========================
# 📈 Trainingsfunktion
# ========================
async def train_model():
    global training_status
    training_status["running"] = True
    training_status["message"] = "📊 Training gestartet..."
    print(training_status["message"])

    try:
        df = yf.download("EURUSD=X", period="1mo", interval="1h")
        df.dropna(inplace=True)
        if len(df) < 10:
            training_status["message"] = "❌ Zu wenige Daten."
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
    df = yf.download("EURUSD=X", period="1d", interval="1h")
    if df.empty:
        await update.message.reply_text("❌ Keine Daten.")
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
# 🧩 Code-Wächter (Auto-Neustart)
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
        print(f"♻️ Änderung erkannt: {event.src_path} -> Neustart")
        if CHAT_ID:
            coro = application.bot.send_message(chat_id=CHAT_ID, text="🔄 Bot wird neu gestartet (Code-Update)...")
            try:
                fut = asyncio.run_coroutine_threadsafe(coro, self.loop)
                fut.result(timeout=5)
            except Exception as e:
                print("Warnung: konnte Restart-Nachricht nicht senden:", e)
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

    # Watchdog starten
    loop = asyncio.get_running_loop()
    threading.Thread(target=start_watchdog, args=(loop,), daemon=True).start()

    # Auto-Training starten
    asyncio.create_task(auto_trainer())

    config = Config()
    config.bind = [f"0.0.0.0:{PORT}"]
    await serve(app, config)


if __name__ == "__main__":
    asyncio.run(main())
