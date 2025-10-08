import os
import asyncio
import threading
import time
import yfinance as yf
import pandas as pd
import requests
import matplotlib.pyplot as plt
from io import BytesIO
from quart import Quart, request
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes
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
model = LinearRegression()

# ========================
# 📦 Datenquelle
# ========================
def fetch_data(symbol="EURUSD=X", period="1mo", interval="1h"):
    """Versucht zuerst Yahoo, dann TwelveData."""
    try:
        print(f"📡 Lade {symbol} von Yahoo...")
        df = yf.download(symbol, period=period, interval=interval)
        if df is not None and not df.empty:
            return df
        print("⚠️ Yahoo Finance leer – versuche TwelveData...")
    except Exception as e:
        print("⚠️ Yahoo Finance Fehler:", e)

    if not TWELVEDATA_KEY:
        print("❌ Kein TWELVEDATA_API_KEY gesetzt.")
        return pd.DataFrame()

    try:
        url = f"https://api.twelvedata.com/time_series?symbol=EUR/USD&interval={interval}&apikey={TWELVEDATA_KEY}&outputsize=500"
        resp = requests.get(url)
        data = resp.json()
        if "values" not in data:
            print("⚠️ TwelveData ungültige Antwort:", data)
            return pd.DataFrame()
        df = pd.DataFrame(data["values"])
        df["datetime"] = pd.to_datetime(df["datetime"])
        df.set_index("datetime", inplace=True)
        df = df.astype(float)
        df.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close"}, inplace=True)
        print(f"✅ TwelveData erfolgreich geladen ({len(df)} Zeilen).")
        return df
    except Exception as e:
        print("❌ TwelveData Fehler:", e)
        return pd.DataFrame()

# ========================
# 📈 Trainingsfunktion
# ========================
async def train_model():
    global training_status, model
    training_status["running"] = True
    training_status["message"] = "📊 Training gestartet..."
    print(training_status["message"])

    try:
        df = fetch_data(period="1mo", interval="1h")
        if df.empty or len(df) < 10:
            training_status["message"] = "❌ Zu wenige oder keine Daten."
            training_status["running"] = False
            return

        df["Target"] = df["Close"].shift(-1)
        X = df[["Open", "High", "Low", "Close"]].iloc[:-1]
        y = df["Target"].iloc[:-1]

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

# --- 🧠 Neue PREDICT-Funktion mit Chart ---
async def predict(update: Update, context: ContextTypes.DEFAULT_TYPE):
    symbol = "EURUSD=X"

    try:
        df = fetch_data(symbol, period="15m", interval="1m")
        if df.empty:
            await update.message.reply_text("⚠️ Keine Kursdaten erhalten.")
            return

        last = df.iloc[-1]
        open_price = float(last["Open"])
        close_price = float(last["Close"])
        change = close_price - open_price
        timestamp = last.name.strftime("%Y-%m-%d %H:%M:%S UTC")
        action = "BUY 📈" if change > 0 else "SELL 📉"

        msg = (
            f"🕒 {timestamp}\n"
            f"💱 EUR/USD (1m)\n"
            f"Open: `{open_price:.5f}`\n"
            f"Close: `{close_price:.5f}`\n"
            f"Δ: `{change:.5f}`\n\n"
            f"➡️ Empfehlung: **{action}**"
        )

        # --- 📊 Chart generieren ---
        plt.figure(figsize=(6, 3))
        plt.plot(df.index, df["Close"], label="Kurs (Close)", linewidth=1.8)
        plt.title("EUR/USD – letzte 10 Minuten")
        plt.xlabel("Zeit")
        plt.ylabel("Preis")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend()

        buf = BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format="png")
        buf.seek(0)
        plt.close()

        await update.message.reply_photo(photo=buf, caption=msg, parse_mode="Markdown")

    except Exception as e:
        await update.message.reply_text(f"❌ Fehler bei /predict: {e}")

# --- Telegram Handler ---
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
        await asyncio.sleep(6 * 60 * 60)

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

    loop = asyncio.get_running_loop()
    threading.Thread(target=start_watchdog, args=(loop,), daemon=True).start()
    asyncio.create_task(auto_trainer())

    config = Config()
    config.bind = [f"0.0.0.0:{PORT}"]
    await serve(app, config)

if __name__ == "__main__":
    asyncio.run(main())
