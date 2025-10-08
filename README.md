# Monica Option Bot 🚀

Ein vollautomatischer Trading- & Telegram-Bot (EURUSD) mit Quart & Watchdog.

## 📦 Installation (Lokal)

```bash
pip install -r requirements.txt
python main.py
```

## ☁️ Deployment (Render.com)

1. Erstelle ein neues **Web Service** Projekt
2. Wähle Python und lade dieses ZIP hoch
3. Setze diese Environment Variables:

```
BOT_TOKEN=dein_telegram_token
TELEGRAM_CHAT_ID=deine_chat_id
RENDER_EXTERNAL_URL=https://dein-service.onrender.com
PORT=10000
```

4. Deploy 🚀

## 🧠 Befehle

- `/start` – Startet Bot-Info
- `/train` – Trainiert das Modell
- `/status` – Zeigt Status & Genauigkeit
- `/predict` – Gibt aktuelle BUY/SELL Empfehlung

