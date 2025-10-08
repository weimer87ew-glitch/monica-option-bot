import os
import asyncio
import pandas as pd
import yfinance as yf
import requests
from quart import Quart, request
from telegram import Update
from telegram.ext import Application, CommandHandler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from hypercorn.asyncio import serve
from hypercorn.config import Config
import logging
from datetime import datetime

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# === Einstellungen ===
TWELVEDATA_API_KEY = os.getenv("TWELVEDATA_API_KEY")
SYMBOL = "EURUSD"
INTERVAL = "1min"   # 1-Minuten-Kerzen
TRAIN_INTERVAL_HOURS = 6

# === Telegram Bot Setup ===
TOKEN = os.getenv("BOT_TOKEN")
WEBHOOK_URL = os.getenv("RENDER_EXTERNAL_URL", "https://monica-option-bot.onrender.com") + "/webhook"
application = Application.builder().token(TOKEN).build()

# === Quart App ===
app = Quart(__name__)

# === KI-Modelleinstellungen ===
model = LogisticRegression()
scaler = StandardScaler()

# === Funktion: Hole Kursdaten ===
async def get_data():
    """Hole Kursdaten von TwelveData (primär) oder Yahoo (Fallback)."""
    logging.info("📡 Lade Kursdaten für %s...", SYMBOL)

    # --- Versuch 1: TwelveData ---
    if TWELVEDATA_API_KEY:
        try:
            url = f"https://api.twelvedata.com/time_series?symbol={SYMBOL}/USD&interval={INTERVAL}&apikey={TWELVEDATA_API_KEY}&outputsize=100"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()

            if "values" in data:
                df = pd.DataFrame(data["values"])
                df["datetime"] = pd.to_datetime(df["datetime"])
                df = df.sort_values("datetime")
                df = df.astype({"open": float, "high": float, "low": float, "close": float})
                logging.info("✅ Daten von TwelveData geladen (%d Einträge).", len(df))
                return df
            else:
                logging.warning("⚠️ TwelveData liefert keine gültigen Werte.")
        except Exception as e:
            logging.warning("⚠️ TwelveData Fehler: %s", e)

    # --- Versuch 2: Yahoo Finance ---
    try:
        df = yf.download(f"{SYMBOL}=X", period="1d", interval="1m", progress=False)
        if not df.empty:
            df = df.reset_index()
            logging.info("✅ Yahoo Finance als Fallback erfolgreich (%d Zeilen).", len(df))
            return df
        else:
            logging.warning("⚠️ Yahoo Finance leer.")
    except Exception as e:
        logging.warning("❌ Yahoo Finance Fehler: %s", e)

    return None

# === Funktion: Trainiere Modell ===
async def train_model():
    logging.info("📊 Starte Training...")
    df = await get_data()
    if df is None or df.empty:
        logging.warning("⚠️ Keine Daten zum Trainieren.")
        return

    # Feature Engineering
    df["change"] = df["close"].diff()
    df["target"] = (df["change"] > 0).astype(int)
    df = df.dropna()

    X = df[["open", "high", "low", "close"]]
    y = df["target"]

    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    logging.info(f"✅ Training abgeschlossen — Genauigkeit: {accuracy * 100:.2f}%")

# === Telegram Bot Handler ===
async def start(update: Update, context):
    await update.message.reply_text("👋 Willkommen beim Monica Option Bot! Gib /predict ein, um die aktuelle Prognose zu erhalten.")

async def predict(update: Update, context):
    df = await get_data()
    if df is None or df.empty:
        await update.message.reply_text("⚠️ Keine Kursdaten verfügbar.")
        return

    last = df.iloc[-1]
    X_last = scaler.transform([[last["open"], last["high"], last["low"], last["close"]]])
    prediction = model.predict(X_last)[0]
    signal = "📈 Kaufen" if prediction == 1 else "📉 Verkaufen"

    await update.message.reply_text(f"📊 EUR/USD Signal: {signal}\nLetzter Kurs: {last['close']}")

application.add_handler(CommandHandler("start", start))
application.add_handler(CommandHandler("predict", predict))

# === Webhook Setup ===
@app.route("/webhook", methods=["POST"])
async def webhook():
    data = await request.get_json()
    update = Update.de_json(data, application.bot)
    await application.process_update(update)
    return "ok", 200

@app.route("/")
async def home():
    return "✅ Monica Option Bot läuft stabil."

# === Automatisches Training ===
async def auto_train_loop():
    while True:
        await train_model()
        logging.info(f"⏰ Nächstes Training in {TRAIN_INTERVAL_HOURS} Stunden.")
        await asyncio.sleep(TRAIN_INTERVAL_HOURS * 3600)

# === Main Start ===
async def main():
    logging.info("🚀 Initialisiere Monica Option Bot...")
    await train_model()
    await application.bot.set_webhook(url=WEBHOOK_URL)
    logging.info("✅ Webhook gesetzt: %s", WEBHOOK_URL)

    asyncio.create_task(auto_train_loop())

    config = Config()
    config.bind = ["0.0.0.0:10000"]
    await serve(app, config)

if __name__ == "__main__":
    asyncio.run(main())
