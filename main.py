import os
import time
import json
import base64
import requests
import threading
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from telegram import Bot

# ==============================
#   CONFIG
# ==============================
TWELVEDATA_API_KEY = os.getenv("TWELVEDATA_API_KEY")
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")
ALPHAVANTAGE_API_KEY = os.getenv("ALPHAVANTAGE_API_KEY")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

GITHUB_REPO = "weimer87ew-glitch/monica-option-bot"
GITHUB_FILE_PATH = "backup/model.h5"
LOCAL_MODEL_PATH = "model.h5"

bot = Bot(token=TELEGRAM_TOKEN)

# ==============================
#   DATENLADER
# ==============================
def get_data_twelvedata(symbol="AAPL", interval="1h"):
    url = f"https://api.twelvedata.com/time_series?symbol={symbol}&interval={interval}&apikey={TWELVEDATA_API_KEY}&outputsize=100"
    r = requests.get(url)
    data = r.json()
    if "values" in data:
        df = pd.DataFrame(data["values"])
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.sort_values("datetime")
        df["close"] = df["close"].astype(float)
        return df[["datetime", "close"]]
    else:
        print("⚠️ Fehler bei TwelveData:", data)
        return pd.DataFrame()

def get_data_finnhub(symbol="AAPL"):
    url = f"https://finnhub.io/api/v1/quote?symbol={symbol}&token={FINNHUB_API_KEY}"
    r = requests.get(url).json()
    if "c" in r:
        df = pd.DataFrame([[time.time(), r["c"]]], columns=["datetime", "close"])
        return df
    return pd.DataFrame()

# ==============================
#   KI-MODELL
# ==============================
def create_model(input_shape):
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(32),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    return model

def train_model(df):
    print("🚀 Starte Training...")
    scaler = MinMaxScaler(feature_range=(0, 1))
    data = scaler.fit_transform(df["close"].values.reshape(-1, 1))

    X, y = [], []
    lookback = 10
    for i in range(lookback, len(data)):
        X.append(data[i - lookback:i, 0])
        y.append(data[i, 0])
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    model = create_model((X.shape[1], 1))
    model.fit(X, y, epochs=5, batch_size=16, verbose=1)
    model.save(LOCAL_MODEL_PATH)
    print("✅ Modell gespeichert:", LOCAL_MODEL_PATH)
    return model

# ==============================
#   GITHUB BACKUP & RESTORE
# ==============================
def upload_to_github():
    if not os.path.exists(LOCAL_MODEL_PATH):
        print("⚠️ Kein lokales Modell gefunden – Backup übersprungen.")
        return

    with open(LOCAL_MODEL_PATH, "rb") as f:
        content = f.read()

    url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{GITHUB_FILE_PATH}"
    headers = {"Authorization": f"token {GITHUB_TOKEN}"}
    sha = None
    get_res = requests.get(url, headers=headers)
    if get_res.status_code == 200:
        sha = get_res.json().get("sha")

    data = {
        "message": f"Auto-Backup {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "content": base64.b64encode(content).decode("utf-8"),
        "branch": "main"
    }
    if sha:
        data["sha"] = sha

    res = requests.put(url, headers=headers, json=data)
    if res.status_code in [200, 201]:
        print(f"✅ Backup erfolgreich nach GitHub ({GITHUB_FILE_PATH})")
    else:
        print(f"❌ Backup-Fehler: {res.status_code} {res.text}")

def restore_from_github():
    url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{GITHUB_FILE_PATH}"
    headers = {"Authorization": f"token {GITHUB_TOKEN}"}
    res = requests.get(url, headers=headers)
    if res.status_code == 200:
        data = res.json()
        content = base64.b64decode(data["content"])
        with open(LOCAL_MODEL_PATH, "wb") as f:
            f.write(content)
        print("✅ Modell aus GitHub wiederhergestellt:", LOCAL_MODEL_PATH)
    else:
        print("⚠️ Kein GitHub-Backup gefunden")

def schedule_backup(interval_hours=2):
    def loop():
        while True:
            try:
                upload_to_github()
            except Exception as e:
                print("❌ Fehler beim Backup:", e)
            time.sleep(interval_hours * 3600)
    threading.Thread(target=loop, daemon=True).start()

# ==============================
#   TRADING LOGIK
# ==============================
def generate_signal(model, df):
    scaler = MinMaxScaler(feature_range=(0, 1))
    data = scaler.fit_transform(df["close"].values.reshape(-1, 1))
    X_input = np.array([data[-10:]])
    X_input = np.reshape(X_input, (X_input.shape[0], X_input.shape[1], 1))
    pred = model.predict(X_input)[0][0]
    last_close = data[-1][0]
    return "BUY" if pred > last_close else "SELL"

def send_telegram_message(text):
    try:
        bot.send_message(chat_id=CHAT_ID, text=text)
    except Exception as e:
        print("⚠️ Telegram Fehler:", e)

# ==============================
#   HAUPTFUNKTION
# ==============================
def main():
    print("🤖 Starte Monica Option Bot v3.6")
    restore_from_github()
    schedule_backup(interval_hours=2)

    try:
        model = load_model(LOCAL_MODEL_PATH)
        print("✅ Modell geladen.")
    except Exception:
        print("⚠️ Kein Modell vorhanden – trainiere neu.")
        df = get_data_twelvedata()
        model = train_model(df)

    while True:
        df = get_data_twelvedata()
        if df.empty:
            time.sleep(300)
            continue

        signal = generate_signal(model, df)
        send_telegram_message(f"📊 Signal: {signal}")
        print("Signal:", signal)

        # alle 2 Stunden neu trainieren
        print("🔄 Warte 2 Stunden bis zum nächsten Training...")
        time.sleep(7200)
        df = get_data_twelvedata()
        model = train_model(df)
        upload_to_github()
        send_telegram_message("💾 Neues Modell trainiert und gesichert.")

if __name__ == "__main__":
    main()
