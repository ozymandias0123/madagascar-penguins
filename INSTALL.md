# Penguin Squad — Installation Guide

## Quick Install (recommended)

```bash
pip install penguin-squad
penguin-squad
```

That's it. The setup wizard will open automatically on first run.

---

## Manual Install (from source)

```bash
git clone https://github.com/your-username/penguin-squad.git
cd penguin-squad
pip install -e .
penguin-squad
```

---

## Where are my API keys stored?

**All passwords and API keys are stored ONLY on your computer:**

```
Windows:  C:\Users\YourName\.penguin_squad\.env
Mac/Linux: /home/yourname/.penguin_squad/.env
```

- ✅ Never uploaded to the internet
- ✅ Never synced to OneDrive / iCloud / Dropbox
- ✅ Never committed to GitHub (`.gitignore` covers `.env`)
- ✅ Each user has their own separate key file

---

## TA-Lib installation (required for some strategies)

TA-Lib requires a C library. Install it **before** `pip install penguin-squad`:

**Windows:**
```bash
# Option 1 — pre-built wheel (easiest):
pip install TA-Lib-prebuilt

# Option 2 — official wheel from:
# https://github.com/cgohlke/talib-build/releases
```

**Mac:**
```bash
brew install ta-lib
pip install TA-Lib
```

**Linux:**
```bash
sudo apt-get install -y libta-lib-dev
pip install TA-Lib
```

---

## MetaTrader 5 (Windows only)

MT5 connector only works on Windows with MetaTrader 5 installed.

```bash
pip install penguin-squad[mt5]
```

For other exchanges (Bybit, Binance, OKX, Kraken…) no extra install needed — ccxt handles everything.

---

## First run

```
penguin-squad
```

The terminal wizard will ask for:
1. **Telegram Bot Token** — create at @BotFather
2. **Telegram Chat ID** — your personal chat ID
3. **LLM API keys** — Skipper (OpenAI), Kowalski (Anthropic), Rico (DeepSeek)

After setup, everything is controlled from Telegram. No terminal needed again.
